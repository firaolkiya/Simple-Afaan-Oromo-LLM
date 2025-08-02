import torch
import math

def compute_attention_score(embedding_vector):
    """
    Compute raw self-attention score via dot product.
    """
    dim = embedding_vector.size(-1)
    return (embedding_vector @ embedding_vector.T) / math.sqrt(dim)

def compute_attention_weight(attention_score):
    """
    Apply softmax to convert scores into weights.
    """
    return torch.softmax(attention_score, dim=-1)

def compute_context_vector(attention_weight, embedding_vector):
    """
    Use weights to compute context vectors.
    Shape: (seq_len, seq_len) @ (seq_len, dim) = (seq_len, dim)
    """
    return attention_weight @ embedding_vector

def calculate_self_attention(embedding_vector):
    """
    Main pipeline to get attention weights.
    """
    attention_score = compute_attention_score(embedding_vector)
    attention_weight = compute_attention_weight(attention_score)
    result = compute_context_vector(attention_weight,embedding_vector)
    return result