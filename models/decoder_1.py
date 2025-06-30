# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from tell.modules import (AdaptiveSoftmax, DynamicConv1dTBC, GehringLinear,
                          LightweightConv1dTBC, MultiHeadAttention)
from tell.modules.token_embedders import AdaptiveEmbedding
from tell.utils import eval_str_list, fill_with_neg_inf


class SimpleEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.embed_dim = embed_dim

    def forward(self, tokens, **kwargs):
        return self.embed(tokens)

    def get_output_dim(self):
        return self.embed_dim


class DecoderLayer(nn.Module):
    pass


class Decoder(nn.Module):
    pass


class DynamicConvDecoderLayer(DecoderLayer):
    def __init__(self, decoder_embed_dim, decoder_conv_dim, decoder_glu,
                 decoder_conv_type, weight_softmax, decoder_attention_heads,
                 weight_dropout, dropout, relu_dropout, input_dropout,
                 decoder_normalize_before, attention_dropout, decoder_ffn_embed_dim,
                 swap, kernel_size=0):
        super().__init__()
        self.embed_dim = decoder_embed_dim
        self.conv_dim = decoder_conv_dim
        self.normalize_before = decoder_normalize_before
        self.dropout = dropout
        self.relu_dropout = relu_dropout
        self.input_dropout = input_dropout
        self.need_attn = True

        if decoder_glu:
            self.linear1 = GehringLinear(self.embed_dim, 2 * self.conv_dim)
            self.act = nn.GLU()
        else:
            self.linear1 = GehringLinear(self.embed_dim, self.conv_dim)
            self.act = None

        conv_class = LightweightConv1dTBC if decoder_conv_type == 'lightweight' else DynamicConv1dTBC
        self.conv = conv_class(self.conv_dim, kernel_size, padding_l=kernel_size - 1,
                               weight_softmax=weight_softmax,
                               num_heads=decoder_attention_heads,
                               weight_dropout=weight_dropout)

        self.linear2 = GehringLinear(self.conv_dim, self.embed_dim)
        self.conv_layer_norm = nn.LayerNorm(self.embed_dim)

        self.context_attns = nn.ModuleDict({
            'image': MultiHeadAttention(self.embed_dim, decoder_attention_heads, kdim=2048, vdim=2048, dropout=attention_dropout),
            'article': MultiHeadAttention(self.embed_dim, decoder_attention_heads, kdim=1024, vdim=1024, dropout=attention_dropout),
            'faces': MultiHeadAttention(self.embed_dim, decoder_attention_heads, kdim=512, vdim=512, dropout=attention_dropout),
            'obj': MultiHeadAttention(self.embed_dim, decoder_attention_heads, kdim=2048, vdim=2048, dropout=attention_dropout)
        })
        self.context_attn_lns = nn.ModuleDict({k: nn.LayerNorm(self.embed_dim) for k in self.context_attns})

        self.context_fc = GehringLinear(self.embed_dim * 4, self.embed_dim)
        self.fc1 = GehringLinear(self.embed_dim, decoder_ffn_embed_dim)
        self.fc2 = GehringLinear(decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def maybe_layer_norm(self, layer_norm, X, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(X)
        return X

    def forward(self, X, contexts, incremental_state):
        residual = X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, before=True)
        X = F.dropout(X, p=self.input_dropout, training=self.training)
        X = self.linear1(X)
        if self.act:
            X = self.act(X)
        X = self.conv(X, incremental_state=incremental_state)
        X = self.linear2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.conv_layer_norm, X, after=True)

        X_contexts, attns = [], {}
        for k in ['image', 'article', 'faces', 'obj']:
            residual = X
            X_ctx = self.maybe_layer_norm(self.context_attn_lns[k], X, before=True)
            X_ctx, attn = self.context_attns[k](
                query=X_ctx, key=contexts[k], value=contexts[k],
                key_padding_mask=contexts.get(f"{k}_mask"),
                incremental_state=None, static_kv=True, need_weights=(not self.training and self.need_attn))
            X_ctx = F.dropout(X_ctx, p=self.dropout, training=self.training)
            X_ctx = residual + X_ctx
            X_ctx = self.maybe_layer_norm(self.context_attn_lns[k], X_ctx, after=True)
            X_contexts.append(X_ctx)
            if attn is not None:
                attns[k] = attn.cpu().detach().numpy()

        X_context = torch.cat(X_contexts, dim=-1)
        X = self.context_fc(X_context)

        residual = X
        X = self.maybe_layer_norm(self.final_layer_norm, X, before=True)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, p=self.relu_dropout, training=self.training)
        X = self.fc2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = residual + X
        X = self.maybe_layer_norm(self.final_layer_norm, X, after=True)

        return X, attns


class DynamicConvFacesObjectsDecoder(Decoder):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedder = SimpleEmbedding(vocab_size, config.decoder_output_dim, padding_idx=config.padding_idx)
        embed_dim = config.decoder_output_dim

        self.project_in_dim = GehringLinear(embed_dim, embed_dim, bias=False)
        self.layers = nn.ModuleList([
            DynamicConvDecoderLayer(
                embed_dim, config.decoder_conv_dim, config.decoder_glu,
                config.decoder_conv_type, config.weight_softmax, config.decoder_attention_heads,
                config.weight_dropout, config.dropout, config.relu_dropout, config.input_dropout,
                config.decoder_normalize_before, config.attention_dropout, config.decoder_ffn_embed_dim,
                config.swap, kernel_size=ks
            ) for ks in config.decoder_kernel_size_list
        ])

        self.project_out_dim = GehringLinear(embed_dim, embed_dim, bias=False)
        self.embed_out = nn.Parameter(torch.Tensor(vocab_size, embed_dim))
        nn.init.normal_(self.embed_out, mean=0, std=embed_dim ** -0.5)

        self.layer_norm = nn.LayerNorm(embed_dim) if config.decoder_normalize_before and config.final_norm else None
        self.dropout = config.dropout
        self.max_target_positions = config.max_target_positions

    def forward(self, prev_target, contexts, incremental_state=None, use_layers=None):
        X = self.embedder(prev_target)
        X = self.project_in_dim(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        X = X.transpose(0, 1)

        attns, inner_states = [], [X]
        for i, layer in enumerate(self.layers):
            if not use_layers or i in use_layers:
                X, attn = layer(X, contexts, incremental_state)
                inner_states.append(X)
            attns.append(attn)

        if self.layer_norm:
            X = self.layer_norm(X)
        X = X.transpose(0, 1)
        X = self.project_out_dim(X)
        X = F.linear(X, self.embed_out)

        return X, {'attn': attns, 'inner_states': inner_states}

    def max_positions(self):
        return self.max_target_positions

    def get_normalized_probs(self, net_output, log_probs=False, sample=None):
        logits = net_output[0].float()
        return F.log_softmax(logits, dim=-1) if log_probs else F.softmax(logits, dim=-1)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]
