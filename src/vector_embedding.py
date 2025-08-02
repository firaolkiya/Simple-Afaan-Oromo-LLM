from torch.nn import Embedding
from my_data_loader import create_dataloader
import torch

def get_vector_embedding(raw_data):
    max_length = 4

    dataloader = create_dataloader(text=raw_data, batch_size=8, stride=4, max_length=max_length)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    vocabulary_size = 97
    vector_dimension = 4

    # token embedding
    token_embedding = Embedding(vocabulary_size, vector_dimension)
    token_embeddings_of_first_batch = token_embedding(inputs)

    print("Token Embedding Shape:", token_embeddings_of_first_batch.shape)
    print(token_embeddings_of_first_batch[0])

    return token_embeddings_of_first_batch[0]

if __name__ == "__main__":
    with open("data/oromic_data.txt", "r") as file:
        raw_data = file.read()

    get_vector_embedding(raw_data)
