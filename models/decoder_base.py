# Parameters pulled from config.yaml
d_model          = 1024  # adaptive.output_dim
nhead            = 8     # decoder_attention_heads
dim_feedforward  = 4096  # decoder_ffn_embed_dim
dropout          = 0.1   # attention_dropout
num_layers       = 6     # decoder_layers
norm_first       = True  # decoder_normalize_before
apply_final_norm = True  # final_norm

import torch.nn as nn

# 1) One decoder layer
DecoderLayer = nn.TransformerDecoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    norm_first=norm_first,
)

# 2) Stacked decoder block
Decoder = nn.TransformerDecoder(
    decoder_layer=DecoderLayer,
    num_layers=num_layers,
    norm=nn.LayerNorm(d_model) if apply_final_norm else None
)
