import pandas as pd
import torch.nn as nn
from vector_embedding import get_vector_embedding
from simplefied_self_attention import calculate_self_attention
from self_attensions import SelfAttention
from causal_attention import CausalAttention
if __name__ == "__main__":
    with open("data/oromic_data.txt", "r") as file:
        raw_data = file.read()

    input_vector = get_vector_embedding(raw_data)
    self_att = SelfAttention(4,2)

    causal_att = CausalAttention(4,4,4,0.5)
    
    cxt_vec = causal_att.forward(input_vector)
    
    print(cxt_vec)
    
    # result1 = self_att.forward(input_vector)
    
    # result2  = calculate_self_attention(input_vector)
    # print("Result from prev")
    # print(result2)
    
    # print("current_impl")
    # print(result1)
    
    
    
