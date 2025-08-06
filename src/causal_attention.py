import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.W_queries = nn.Parameter(torch.rand(d_in,d_out))
        self.W_keys = nn.Parameter(torch.rand(d_in,d_out))
        self.W_values = nn.Parameter(torch.rand(d_in,d_out))
        
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("mask",torch.triu(torch.ones(context_length,context_length),diagonal=1))
        
    def forward(self,vector):
        b_num,num_token, d_out = vector.shape
        
        keys = vector @ self.W_keys
        values = vector @ self.W_values
        queries  = vector @ self.W_queries
        
        
        attention_score = queries @ keys.transpose(1,2)
        
        attention_weigths = attention_score.masked_fill(self.mask.bool()[:num_token,:num_token],-torch.inf) # type: ignore
        
        attention_weigths = torch.softmax(attention_weigths//(keys.shape[-1]**0.5),dim=1)
        
        attention_weigths = self.dropout(attention_weigths)
        
        context_vector = attention_weigths @ values
        
        return context_vector
        

