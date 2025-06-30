import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# ----------------------
# Utility Functions (Identical to Original)
# ----------------------

def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)

# ----------------------
# Embedding Modules (Identical Interface to Original)
# ----------------------

class AdaptiveEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, 
                 cutoff, factor=1.0, scale_grad=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.cutoff = cutoff
        self.factor = factor
        self.scale_grad = scale_grad

        self.embeddings = nn.ModuleList()
        self.projections = nn.ModuleList()

        # Calculate cluster sizes
        cutoff_ends = [0] + self.cutoff + [self.num_embeddings]
        for i in range(len(cutoff_ends) - 1):
            cluster_size = cutoff_ends[i+1] - cutoff_ends[i]
            if cluster_size == 0:
                continue

            dim = int(self.embedding_dim // (self.factor ** i))
            emb = nn.Embedding(cluster_size, dim, padding_idx=self.padding_idx)
            self.embeddings.append(emb)

            if i > 0:
                projection = nn.Linear(dim, self.embedding_dim, bias=False)
                nn.init.normal_(projection.weight, mean=0, std=dim ** -0.5)
                self.projections.append(projection)
            else:
                self.projections.append(None)

    def forward(self, input):
        result = input.new_zeros(*input.size(), self.embedding_dim)
        for i in range(len(self.cutoff) + 1):
            if i == 0:
                l_idx, r_idx = 0, self.cutoff[0]
            elif i == len(self.cutoff):
                l_idx, r_idx = self.cutoff[-1], self.num_embeddings
            else:
                l_idx, r_idx = self.cutoff[i-1], self.cutoff[i]

            mask = (input >= l_idx) & (input < r_idx)
            indices = input[mask] - l_idx
            if indices.numel() == 0:
                continue

            emb = self.embeddings[i](indices)
            if self.projections[i] is not None:
                emb = self.projections[i](emb)
                if self.scale_grad:
                    emb = emb * (self.embedding_dim ** 0.5 / emb.size(-1))

            result[mask] = emb

        return result

# ----------------------
# Linear Modules (Identical Interface to Original)
# ----------------------

class GehringLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, weight_norm=True):
        super().__init__(in_features, out_features, bias)
        if weight_norm:
            self = nn.utils.weight_norm(self, name='weight')

# ----------------------
# Convolution Modules (Identical Interface to Original)
# ----------------------

class LightweightConv1dTBC(nn.Module):
    def __init__(self, input_size, kernel_size, padding_l, num_heads,
                 weight_softmax=True, bias=True, dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(input_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, x, incremental_state=None):
        """Input shape: Time x Batch x Channel
        Args:
            incremental_state: dict to store the state during incremental decoding
        """
        T, B, C = x.size()
        H = self.num_heads
        G = C // H

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)
        if self.dropout > 0:
            weight = F.dropout(weight, p=self.dropout, training=self.training)

        # Merge every G channels into a head
        x = x.view(T, B, H, G).transpose(0, 1)  # B x T x H x G
        x = x.contiguous().view(B * H, T, G)

        # Pad for causal convolution
        if self.padding_l > 0:
            x = F.pad(x, (0, 0, self.padding_l, 0))

        # Lightweight convolution
        x = x.unsqueeze(1)  # B*H x 1 x T x G
        weight = weight.view(1, H, 1, self.kernel_size).repeat(B, 1, 1, 1)
        weight = weight.view(B * H, 1, 1, self.kernel_size)
        x = F.conv2d(x, weight, groups=B * H)
        x = x.squeeze(1).transpose(1, 2)  # B*H x T x G

        # Reshape back
        x = x.view(B, H, T, G).transpose(1, 2).contiguous()  # B x T x H x G
        x = x.view(B, T, C).transpose(0, 1)  # T x B x C

        if self.bias is not None:
            x = x + self.bias

        return x

class DynamicConv1dTBC(nn.Module):
    def __init__(self, input_size, kernel_size, padding_l, num_heads,
                 weight_softmax=True, bias=True, dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.dropout = dropout

        self.weight_linear = GehringLinear(input_size, num_heads * kernel_size, bias=False)
        self.glu = GehringLinear(input_size, input_size * 2)
        self.out_linear = GehringLinear(input_size, input_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.glu.weight)
        if self.glu.bias is not None:
            nn.init.constant_(self.glu.bias, 0.)
        
        nn.init.xavier_uniform_(self.out_linear.weight, gain=1/math.sqrt(2))
        if self.out_linear.bias is not None:
            nn.init.constant_(self.out_linear.bias, 0.)

    def forward(self, x, incremental_state=None):
        """Input shape: Time x Batch x Channel"""
        T, B, C = x.size()
        H = self.num_heads
        K = self.kernel_size
        G = C // H

        # GLU gating
        gate = self.glu(x)  # T x B x 2C
        gate = gate.view(T, B, H, 2, G)
        gate = F.glu(gate, dim=3)  # T x B x H x G

        # Dynamic weights
        weights = self.weight_linear(x)  # T x B x H*K
        weights = weights.view(T, B, H, K)
        if self.weight_softmax:
            weights = F.softmax(weights, dim=-1)
        if self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout, training=self.training)

        # Reshape for convolution
        x = x.view(T, B, H, G).transpose(0, 1)  # B x T x H x G
        weights = weights.transpose(0, 1)  # B x T x H x K

        # Pad for causal convolution
        if self.padding_l > 0:
            x = F.pad(x, (0, 0, self.padding_l, 0))  # B x T+K-1 x H x G

        # Prepare for grouped convolution
        x = x.permute(0, 2, 1, 3).contiguous()  # B x H x T+K-1 x G
        x = x.view(B * H, 1, T + K - 1, G)  # B*H x 1 x T+K-1 x G
        weights = weights.permute(0, 2, 1, 3).contiguous()  # B x H x T x K
        weights = weights.view(B * H, 1, T, K)  # B*H x 1 x T x K

        # Apply convolution
        output = F.conv2d(x, weights, groups=B * H)  # B*H x 1 x T x G
        output = output.view(B, H, T, G).permute(1, 2, 0, 3).contiguous()  # H x T x B x G
        output = output.view(H * T, B, G).transpose(0, 1)  # B x H*T x G
        output = output.view(B, H, T, G).permute(2, 0, 1, 3).contiguous()  # T x B x H x G

        # Apply gate and output linear
        output = output.view(T, B, C) * gate
        output = self.out_linear(output)

        return output

# ----------------------
# Attention Modules (Identical Interface to Original)
# ----------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None,
                 dropout=0., bias=True, add_bias_kv=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = GehringLinear(embed_dim, embed_dim, bias=bias)
        self.k_proj = GehringLinear(self.kdim, embed_dim, bias=bias)
        self.v_proj = GehringLinear(self.vdim, embed_dim, bias=bias)
        self.out_proj = GehringLinear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

        if self.bias_k is not None:
            nn.init.xavier_uniform_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_uniform_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                incremental_state=None, need_weights=True, static_kv=False):
        """Input shape: Time x Batch x Channel"""
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)

        # Project to query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Add bias if needed
        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([
                    key_padding_mask,
                    key_padding_mask.new_zeros(bsz, 1)
                ], dim=1)

        # Reshape for multi-head attention
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # Compute attention weights
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, -1)

        # Apply softmax
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Compute attention output
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1)
            attn_weights = attn_weights.mean(dim=1)
        else:
            attn_weights = None

        return attn, attn_weights

# ----------------------
# Softmax Modules (Identical Interface to Original)
# ----------------------

class AdaptiveSoftmax(nn.Module):
    def __init__(self, vocab_size, input_dim, cutoff, factor=1.0,
                 dropout=0., adaptive_inputs=None, tie_proj=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.cutoff = cutoff
        self.factor = factor
        self.dropout = dropout
        self.tie_proj = tie_proj

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        cutoff_ends = [0] + list(cutoff) + [vocab_size]
        for i in range(len(cutoff_ends) - 1):
            l_idx, r_idx = cutoff_ends[i], cutoff_ends[i+1]
            cluster_size = r_idx - l_idx
            if cluster_size == 0:
                continue

            proj_dim = int(input_dim // (factor ** i))
            proj = nn.Linear(input_dim, proj_dim, bias=False)
            if i > 0 and tie_proj:
                proj.weight = adaptive_inputs.projections[i-1].weight
            self.out_projs.append(proj)

            layer = nn.Linear(proj_dim, cluster_size)
            if i == 0 and adaptive_inputs is not None:
                layer.weight = adaptive_inputs.embeddings[0].weight
            elif i > 0 and adaptive_inputs is not None and not tie_proj:
                layer.weight = adaptive_inputs.embeddings[i].weight
            self.out_layers.append(layer)

    def forward(self, input, target=None):
        if target is not None:
            # Training path
            input = input.contiguous().view(-1, self.input_dim)
            target = target.contiguous().view(-1)
            loss = 0
            nll = torch.zeros_like(target, dtype=input.dtype, device=input.device)

            for i in range(len(self.cutoff) + 1):
                l_idx = 0 if i == 0 else self.cutoff[i-1]
                r_idx = self.cutoff[i] if i < len(self.cutoff) else self.vocab_size

                mask = (target >= l_idx) & (target < r_idx)
                indices = mask.nonzero().squeeze()
                if indices.numel() == 0:
                    continue

                target_i = target.index_select(0, indices) - l_idx
                input_i = input.index_select(0, indices)

                if self.dropout > 0:
                    input_i = F.dropout(input_i, p=self.dropout, training=self.training)

                hidden_i = self.out_projs[i](input_i)
                logits_i = self.out_layers[i](hidden_i)

                loss += F.cross_entropy(logits_i, target_i, reduction='sum')
                nll.index_copy_(0, indices, F.cross_entropy(logits_i, target_i, reduction='none'))

            return loss, nll
        else:
            # Inference path
            input = input.contiguous().view(-1, self.input_dim)
            logits = []

            for i in range(len(self.cutoff) + 1):
                hidden_i = self.out_projs[i](input)
                logits_i = self.out_layers[i](hidden_i)
                logits.append(logits_i)

            return torch.cat(logits, dim=-1)

    def get_log_prob(self, input, target):
        logits = self.forward(input)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, target.unsqueeze(1)).squeeze(1)