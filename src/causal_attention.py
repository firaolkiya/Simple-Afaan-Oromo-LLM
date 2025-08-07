import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_queries = nn.Parameter(torch.rand(d_in, d_out))
        self.W_keys = nn.Parameter(torch.rand(d_in, d_out))
        self.W_values = nn.Parameter(torch.rand(d_in, d_out))
        
        self.dropout = nn.Dropout(dropout)
        # Causal mask: upper triangle is masked out
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, vector):  # vector shape: [B, T, d_in]
        B, T, _ = vector.shape
        
        queries = vector @ self.W_queries    # [B, T, d_out]
        keys    = vector @ self.W_keys       # [B, T, d_out]
        values  = vector @ self.W_values     # [B, T, d_out]

        # Compute attention scores
        attention_scores = queries @ keys.transpose(1, 2)  # [B, T, T]

        # Apply causal mask
        attention_scores=attention_scores.masked_fill(self.mask.bool()[:T, :T], -torch.inf) # type: ignore

        # Softmax + dropout
        attention_weights = torch.softmax(attention_scores/(keys.shape[-1] ** 0.5), dim=-1)  # softmax over keys
        attention_weights = self.dropout(attention_weights)

        # Compute context
        context_vector = attention_weights @ values  # [B, T, d_out]

        return context_vector
