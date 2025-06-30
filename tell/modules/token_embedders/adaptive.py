# import math
# from typing import List

# import torch
# import torch.nn as nn
# import torch.onnx.operators
# from allennlp.modules.token_embedders import TokenEmbedder
# from overrides import overrides


# @TokenEmbedder.register('adaptive')
# class AdaptiveEmbedding(TokenEmbedder):
#     """Adaptive input representation, proposed by Baevski & Auli (2019).

#     Adaptive input representations for neural language modeling extend the
#     adaptive softmax of Grave et al. (2017) to input representations of
#     variable capacity. See https://openreview.net/forum?id=ByxZX20qFQ.
#     """

#     def __init__(self, vocab, namespace, padding_idx: int,
#                  initial_dim: int, factor: float, output_dim: int,
#                  cutoff: List[int], vocab_size: int = None, scale_embeds=False):
#         super().__init__()

#         vocab_size = vocab_size or vocab.get_vocab_size(namespace)
#         if not cutoff or vocab_size > cutoff[-1]:
#             cutoff.append(vocab_size)

#         assert vocab_size == cutoff[-1], \
#             f'Cutoff {cutoff[-1]} is larger than vocab size {vocab_size}.'

#         self.cutoff = cutoff
#         self.embed_size = output_dim
#         self.padding_idx = padding_idx
#         self.embeddings = nn.ModuleList()
#         self.embed_scale = math.sqrt(output_dim) if scale_embeds else 1

#         for i in range(len(self.cutoff)):
#             prev = self.cutoff[i - 1] if i > 0 else 0
#             vocab_size = self.cutoff[i] - prev
#             embed_size = int(initial_dim // (factor ** i))
#             embed = nn.Embedding(vocab_size, embed_size, padding_idx)
#             projection = nn.Linear(embed_size, output_dim, bias=False)
#             seq = nn.Sequential(embed, projection)
#             self.embeddings.append(seq)

#         def init_weights(m):
#             if isinstance(m, nn.Embedding):
#                 std = math.sqrt(1 / m.weight.shape[1])
#                 m.weight.data.normal_(mean=0, std=std)
#                 m.weight.data[padding_idx].fill_(0)
#             elif hasattr(m, 'weight'):
#                 nn.init.xavier_uniform_(m.weight)

#         # Recursively initialize weights of all children
#         self.apply(init_weights)

#     def weights_for_band(self, band: int):
#         return self.embeddings[band][0].weight, self.embeddings[band][1].weight

#     def forward(self, X: torch.Tensor, incremental_state=None):
#         result_shape = X.shape + (self.embed_size,)
#         result = self.embeddings[0][0].weight.new_zeros(result_shape)

#         for i in range(len(self.cutoff)):
#             mask = X < self.cutoff[i]
#             if i > 0:
#                 mask.mul_(X >= self.cutoff[i - 1])
#                 chunk_input = X[mask] - self.cutoff[i - 1]
#             else:
#                 chunk_input = X[mask]
#             if mask.any():
#                 result[mask] = self.embeddings[i](chunk_input).type_as(result)

#         result = self.embed_scale * result
#         return result

#     @overrides
#     def get_output_dim(self) -> int:
#         return self.embed_size
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn

class AdaptiveEmbedding(nn.Module):
    """Pure PyTorch implementation of Adaptive Embedding.
    Original concept from Baevski & Auli (2019) - Adaptive Input Representations.
    """

    def __init__(self,
                 vocab_size: int,
                 padding_idx: int,
                 initial_dim: int,
                 factor: float,
                 output_dim: int,
                 cutoff: List[int],
                 scale_embeds: bool = False):
        super().__init__()
        # Ensure cutoff includes vocab_size if not already covered
        if not cutoff or vocab_size > cutoff[-1]:
            cutoff = cutoff + [vocab_size]
        assert cutoff[-1] == vocab_size, \
            f'Last cutoff must match vocab size ({cutoff[-1]} vs {vocab_size})'

        self.cutoff = cutoff
        self.embed_size = output_dim
        self.padding_idx = padding_idx
        self.embeddings = nn.ModuleList()
        self.embed_scale = math.sqrt(output_dim) if scale_embeds else 1.0

        # Create embedding layers for each frequency band
        for i in range(len(self.cutoff)):
            prev_bound = self.cutoff[i-1] if i > 0 else 0
            band_vocab_size = self.cutoff[i] - prev_bound
            band_dim = int(initial_dim // (factor ** i))
            
            seq = nn.Sequential(
                nn.Embedding(band_vocab_size, band_dim, padding_idx),
                nn.Linear(band_dim, output_dim, bias=False)
            )
            self.embeddings.append(seq)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            std = math.sqrt(1 / module.weight.shape[1])
            module.weight.data.normal_(mean=0, std=std)
            module.weight.data[self.padding_idx].fill_(0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def weights_for_band(self, band: int) -> Tuple[nn.Parameter, nn.Parameter]:
        """Return embedding and projection weights for a frequency band"""
        embed = self.embeddings[band][0].weight
        proj = self.embeddings[band][1].weight
        return embed, proj

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Initialize output tensor with zeros
        out_shape = input_ids.shape + (self.embed_size,)
        result = torch.zeros(out_shape, 
                            dtype=self.embeddings[0][0].weight.dtype,
                            device=input_ids.device)

        # Process each frequency band
        for i in range(len(self.cutoff)):
            # Create mask for current band
            if i == 0:
                mask = input_ids < self.cutoff[0]
                ids_band = input_ids[mask]
            else:
                mask = (input_ids >= self.cutoff[i-1]) & (input_ids < self.cutoff[i])
                ids_band = input_ids[mask] - self.cutoff[i-1]
            
            # Process elements in current band
            if mask.any():
                embeddings = self.embeddings[i](ids_band)
                result[mask] = embeddings

        return self.embed_scale * result

    def get_output_dim(self) -> int:
        return self.embed_size