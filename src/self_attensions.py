import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.W_queries = nn.Parameter(torch.rand(d_in,d_out))
        self.W_keys = nn.Parameter(torch.rand(d_in,d_out))
        self.W_values = nn.Parameter(torch.rand(d_in,d_out))
        
        
    def forward(self,vector):
        keys = vector @ self.W_keys
        values = vector @ self.W_values
        queries  = vector @ self.W_queries
        
        attention_scores = queries @ keys.T
        
        attention_weight = torch.softmax(attention_scores//(values.shape[-1]**0.5),-1)
        
        context_vector = attention_weight @ values
        
        return context_vector
    