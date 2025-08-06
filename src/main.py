import pandas as pd
from vector_embedding import get_vector_embedding
from simplefied_self_attention import calculate_self_attention
from self_attensions import SelfAttention
from causal_attention import causal_attention
if __name__ == "__main__":
    with open("data/oromic_data.txt", "r") as file:
        raw_data = file.read()

    input_vector = get_vector_embedding(raw_data)
    self_att = SelfAttention(4,2)
    result1 = self_att.forward(input_vector)
    
    result2  = calculate_self_attention(input_vector)
    print("Result from prev")
    print(result2)
    
    print("current_impl")
    print(result1)
    
    new = causal_attention(result2,4,4)
    
    print(new)