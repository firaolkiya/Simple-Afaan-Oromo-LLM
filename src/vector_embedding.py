from torch.nn import Embedding
from my_data_loader import create_dataloader
from torch.nn import Embedding

with open("data/oromic_data.txt","r") as file:
    raw_data = file.read()
file.close()

max_length = 4

dataloader = create_dataloader(text=raw_data,batch_size=8,stride=4,max_length=max_length)

data_iter = iter(dataloader)
inputs,targets = next(data_iter)

vocabulary_size = 97
vector_dimension = 256

# token embedding
token_embedding = Embedding(vocabulary_size , vector_dimension)

# sample of the first batch of data set
token_embeddings = token_embedding(inputs)

print(token_embeddings.shape)

position_embedding = Embedding(max_length,vector_dimension)
print(position_embedding)