import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,num_heads, dropout, qkv_bias=False) -> None:
        super().__init__()
        
        assert(d_out%num_heads==0), \
            "d_out must divisible by num_heads"     
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        
        self.head_dim = d_out//num_heads
        
        self.W_queries = nn.Parameter(torch.rand(d_in, d_out))
        self.W_keys = nn.Parameter(torch.rand(d_in, d_out))
        self.W_values = nn.Parameter(torch.rand(d_in, d_out))
        
        self.dropout = nn.Dropout(dropout)
        # Causal mask: upper triangle is masked out
        self.register_buffer("mask", torch.triu(
            torch.ones(context_length, context_length),
            diagonal=1
            ))
    
    def forward(self,x):
        b,num_tokens,d_in = x.shape
        
        queries = x @ self.W_queries    # [B, T, d_out]
        keys    = x @ self.W_keys       # [B, T, d_out]
        values  = x @ self.W_values     # [B, T, d_out]
        
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)
        
        keys.transpose(1,2)
        values.transpose(1,2)
        queries.transpose(1,2)
        
        attention_score = queries @ keys.transpose(2,3)
        
        mask_bool = self.mask.bool()[:num_tokens,num_tokens] # type: ignore
        
        attention_score.masked_fille(mask_bool,-torch.inf)
        
        attention_weight = torch.softmax(attention_score/keys.shape[-1]**0.5,dim=-1)
        
        attention_weight = self.dropout(attention_weight)
        
        context_vec = (attention_weight @ values).transpose(1,2)
        
        context_vec.contiguous().view(b,num_tokens,self.d_out)
        
        return context_vec
        
        


