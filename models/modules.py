import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# ----------------------
# Utility Functions
# ----------------------

def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    return [type(i) for i in x]

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)

# ----------------------
# Embedding Modules
# ----------------------

class AdaptiveEmbedding(nn.Module):
    """Adaptive input embedding module"""
    def __init__(self, num_embeddings, embedding_dim, padding_idx, 
                 cutoff, factor=1.0, scale_grad=False):
        super().__init__()
        self.cutoff = cutoff
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.scale_grad = scale_grad
        
        # Create embeddings for each cluster
        self.embeddings = nn.ModuleList()
        self.projections = nn.ModuleList()
        
        # Calculate sizes for each cluster
        cluster_sizes = [cutoff[0]]
        for i in range(1, len(cutoff)):
            cluster_sizes.append(cutoff[i] - cutoff[i-1])
        cluster_sizes.append(num_embeddings - cutoff[-1])
        
        # Create embedding layers for each cluster
        for i, cluster_size in enumerate(cluster_sizes):
            # Skip empty clusters
            if cluster_size == 0:
                continue
                
            # Smaller dimension for rare words
            dim = embedding_dim // (factor ** i)
            
            # Create embedding layer
            emb = nn.Embedding(cluster_size, dim, padding_idx=padding_idx)
            self.embeddings.append(emb)
            
            # Create projection for rare words
            if i > 0:
                projection = nn.Linear(dim, embedding_dim)
                nn.init.normal_(projection.weight, mean=0, std=dim ** -0.5)
                nn.init.constant_(projection.bias, 0)
                self.projections.append(projection)
            else:
                self.projections.append(None)
    
    def forward(self, input):
        result = torch.zeros(
            (*input.size(), self.embedding_dim), 
            dtype=torch.float, device=input.device
        )
        
        # Process each cluster
        for i, emb in enumerate(self.embeddings):
            # Get indices for this cluster
            if i == 0:
                mask = input < self.cutoff[0]
                cluster_idx = input
            elif i == len(self.embeddings) - 1:
                mask = input >= self.cutoff[-1]
                cluster_idx = input - self.cutoff[-1]
            else:
                mask = (input >= self.cutoff[i-1]) & (input < self.cutoff[i])
                cluster_idx = input - self.cutoff[i-1]
            
            # Skip if no elements in this cluster
            if not mask.any():
                continue
                
            # Get embeddings
            cluster_input = cluster_idx[mask]
            cluster_emb = emb(cluster_input)
            
            # Project if necessary
            if self.projections[i] is not None:
                cluster_emb = self.projections[i](cluster_emb)
                
            # Apply gradient scaling trick
            if self.scale_grad and i > 0:
                cluster_emb = cluster_emb * (self.embedding_dim ** 0.5 / cluster_emb.size(-1))
            
            # Place in result tensor
            result[mask] = cluster_emb
        
        return result

# ----------------------
# Linear Modules
# ----------------------

class GehringLinear(nn.Linear):
    """Linear layer with weight normalization"""
    def __init__(self, in_features, out_features, bias=True, weight_norm=True):
        super().__init__(in_features, out_features, bias)
        if weight_norm:
            self = nn.utils.weight_norm(self, name='weight')
        
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# ----------------------
# Convolution Modules
# ----------------------

class LightweightConv1dTBC(nn.Module):
    """Lightweight convolution in TBC format (time, batch, channels)"""
    def __init__(self, input_size, kernel_size, padding, num_heads,
                 weight_softmax=True, bias=True, dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_heads = num_heads
        self.weight_softmax = weight_softmax
        self.dropout = dropout
        
        # Weight parameters
        self.weight = Parameter(torch.Tensor(num_heads, 1, kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)
    
    def forward(self, x, incremental_state=None):
        """Input shape: (time, batch, channels)"""
        T, B, C = x.size()
        H = self.num_heads
        
        # Reshape to (batch * heads, time, channels//heads)
        x = x.view(T, B, H, C // H)
        x = x.permute(1, 2, 0, 3).contiguous()
        x = x.view(B * H, T, C // H)
        
        # Compute weights
        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)
        if self.dropout:
            weight = F.dropout(weight, p=self.dropout, training=self.training)
        
        # Apply convolution
        x = F.conv1d(x, weight, padding=self.padding, groups=self.num_heads)
        
        # Reshape back to TBC format
        x = x.view(B, H, T, C // H)
        x = x.permute(2, 0, 1, 3).contiguous()
        x = x.view(T, B, C)
        
        # Add bias
        if self.bias is not None:
            x = x + self.bias
        
        return x

class DynamicConv1dTBC(nn.Module):
    """Dynamic convolution in TBC format (time, batch, channels)"""
    def __init__(self, input_size, kernel_size, padding_l, num_heads,
                 weight_softmax=True, weight_dropout=0.,bias=True, 
                 dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.dropout = dropout
        
        # Weight parameters
        self.weight_linear = GehringLinear(input_size, num_heads * kernel_size, bias=False)
        self.glu = GehringLinear(input_size, num_heads * 2)
        self.out_linear = GehringLinear(input_size, input_size, bias=bias)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # GLU gate initialization
        nn.init.xavier_uniform_(self.glu.weight)
        if self.glu.bias is not None:
            nn.init.constant_(self.glu.bias, 0.)
        
        # Output linear initialization
        nn.init.xavier_uniform_(self.out_linear.weight, gain=1/math.sqrt(2))
        if self.out_linear.bias is not None:
            nn.init.constant_(self.out_linear.bias, 0.)
    
    def forward(self, x, incremental_state=None):
        """Input shape: (time, batch, channels)"""
        T, B, C = x.size()
        H = self.num_heads
        K = self.kernel_size
        
        # Compute GLU gating
        gate = self.glu(x)
        gate = gate.view(T, B, H, 2)
        gate = F.glu(gate, dim=-1)  # (T, B, H, 1)
        
        # Compute weights
        weights = self.weight_linear(x)  # (T, B, H*K)
        weights = weights.view(T, B, H, K)
        
        if self.weight_softmax:
            weights = F.softmax(weights, dim=-1)
        
        if self.dropout:
            weights = F.dropout(weights, p=self.dropout, training=self.training)
        
        # Apply convolution
        x = x.view(T, B, H, C // H)
        x = x.permute(1, 2, 0, 3).contiguous()  # (B, H, T, C//H)
        weights = weights.permute(1, 2, 0, 3).contiguous()  # (B, H, T, K)
        
        # Apply convolution using einsum
        x = torch.nn.functional.pad(x, (0, 0, self.padding, 0))
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(2), 
            (K, 1), 
            dilation=(1, 1), 
            stride=(1, 1)
        )  # (B, H*C//H*K, T)
        
        unfolded = unfolded.view(B, H, C//H, K, T)
        unfolded = unfolded.permute(0, 1, 4, 3, 2)  # (B, H, T, K, C//H)
        weights = weights.unsqueeze(-1)  # (B, H, T, K, 1)
        
        output = (weights * unfolded).sum(dim=3)  # (B, H, T, C//H)
        output = output.permute(2, 0, 1, 3).contiguous()  # (T, B, H, C//H)
        output = output.view(T, B, C)  # (T, B, C)
        
        # Apply gate and output linear
        output = output * gate.view(T, B, H)
        output = self.out_linear(output)
        
        return output

# ----------------------
# Attention Modules
# ----------------------

class MultiHeadAttention(nn.Module):
    """Multi-headed attention"""
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, 
                 dropout=0., bias=True, add_bias_kv=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projection layers
        self.q_proj = GehringLinear(embed_dim, embed_dim, bias=bias)
        self.k_proj = GehringLinear(self.kdim, embed_dim, bias=bias)
        self.v_proj = GehringLinear(self.vdim, embed_dim, bias=bias)
        self.out_proj = GehringLinear(embed_dim, embed_dim, bias=bias)
        
        # Additional bias for keys/values
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Xavier initialization
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
        """Input shape: (time, batch, channels)"""
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
                    torch.zeros(key_padding_mask.size(0), 1).to(key_padding_mask)
                ], dim=1)
        
        # Reshape to (bsz * num_heads, seq_len, head_dim)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(k.size(0), bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(v.size(0), bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        # Compute attention scores
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute attention output
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn, attn_weights.mean(dim=1)  # average attention weights over heads
        else:
            return attn, None

# ----------------------
# Softmax Modules
# ----------------------

class AdaptiveSoftmax(nn.Module):
    """Efficient softmax approximation for large vocabularies"""
    def __init__(self, vocab_size, input_dim, cutoff, factor=1.0, 
                 dropout=0., adaptive_inputs=None, tie_proj=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.cutoff = cutoff
        self.dropout = dropout
        self.tie_proj = tie_proj
        
        # Initialize output layers for each cluster
        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()
        
        # Calculate cluster sizes
        cluster_sizes = [cutoff[0]]
        for i in range(1, len(cutoff)):
            cluster_sizes.append(cutoff[i] - cutoff[i-1])
        cluster_sizes.append(vocab_size - cutoff[-1])
        
        # Create output layers
        for i, cluster_size in enumerate(cluster_sizes):
            # Skip empty clusters
            if cluster_size == 0:
                continue
                
            # Smaller dimension for rare words
            dim = input_dim // (factor ** i)
            
            # Create projection layer
            proj = nn.Linear(input_dim, dim, bias=False)
            self.out_projs.append(proj.weight if tie_proj else proj)
            
            # Create output layer
            out_layer = nn.Linear(dim, cluster_size)
            if not tie_proj:
                nn.init.normal_(out_layer.weight, mean=0, std=dim ** -0.5)
                nn.init.constant_(out_layer.bias, 0)
            self.out_layers.append(out_layer)
        
        # Link to adaptive inputs if provided
        if adaptive_inputs is not None:
            for i in range(len(self.out_layers)):
                if tie_proj:
                    # Share projection weights
                    self.out_projs[i] = adaptive_inputs.projections[i].weight
                else:
                    # Share embedding weights
                    self.out_layers[i].weight = adaptive_inputs.embeddings[i].weight
                    if hasattr(self.out_layers[i], 'bias'):
                        nn.init.constant_(self.out_layers[i].bias, 0)
    
    def forward(self, input, target=None):
        if target is not None:
            # Training path
            # Flatten input and target
            input = input.contiguous().view(-1, self.input_dim)
            target = target.contiguous().view(-1)
            
            # Calculate loss for each cluster
            loss = 0
            nll = torch.zeros_like(target, dtype=input.dtype, device=input.device)
            
            # First cluster (common words)
            mask = target < self.cutoff[0]
            if mask.any():
                cluster_input = input[mask]
                cluster_target = target[mask]
                
                if self.dropout > 0:
                    cluster_input = F.dropout(cluster_input, p=self.dropout, training=self.training)
                
                # Project and compute loss
                proj = self.out_projs[0](cluster_input)
                out = self.out_layers[0](proj)
                loss += F.cross_entropy(out, cluster_target, reduction='sum')
                nll[mask] = F.cross_entropy(out, cluster_target, reduction='none')
            
            # Middle clusters
            for i in range(1, len(self.cutoff)):
                low = self.cutoff[i-1]
                high = self.cutoff[i]
                
                mask = (target >= low) & (target < high)
                if not mask.any():
                    continue
                
                cluster_input = input[mask]
                cluster_target = target[mask] - low
                
                if self.dropout > 0:
                    cluster_input = F.dropout(cluster_input, p=self.dropout, training=self.training)
                
                # Project and compute loss
                proj = self.out_projs[i](cluster_input)
                out = self.out_layers[i](proj)
                loss += F.cross_entropy(out, cluster_target, reduction='sum')
                nll[mask] = F.cross_entropy(out, cluster_target, reduction='none')
            
            # Last cluster (rare words)
            mask = target >= self.cutoff[-1]
            if mask.any():
                cluster_input = input[mask]
                cluster_target = target[mask] - self.cutoff[-1]
                
                if self.dropout > 0:
                    cluster_input = F.dropout(cluster_input, p=self.dropout, training=self.training)
                
                # Project and compute loss
                proj = self.out_projs[-1](cluster_input)
                out = self.out_layers[-1](proj)
                loss += F.cross_entropy(out, cluster_target, reduction='sum')
                nll[mask] = F.cross_entropy(out, cluster_target, reduction='none')
            
            # Return loss and NLL
            return loss, nll
        else:
            # Inference path
            # Compute logits for all clusters
            logits = []
            input = input.contiguous().view(-1, self.input_dim)
            
            # First cluster
            if self.out_projs[0] is not None:
                proj = self.out_projs[0](input)
                logits.append(self.out_layers[0](proj))
            else:
                logits.append(self.out_layers[0](input))
            
            # Middle clusters
            for i in range(1, len(self.cutoff)):
                if self.out_projs[i] is not None:
                    proj = self.out_projs[i](input)
                    logits.append(self.out_layers[i](proj))
                else:
                    logits.append(self.out_layers[i](input))
            
            # Last cluster
            if self.out_projs[-1] is not None:
                proj = self.out_projs[-1](input)
                logits.append(self.out_layers[-1](proj))
            else:
                logits.append(self.out_layers[-1](input))
            
            # Combine logits
            return torch.cat(logits, dim=-1)
    
    def get_log_prob(self, input, target):
        """Compute log probabilities"""
        # Compute logits
        logits = self.forward(input)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probabilities for target
        log_probs = log_probs.gather(1, target.unsqueeze(1))
        
        return log_probs.squeeze(1)