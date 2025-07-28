from collections import deque
from byte_bit_encoder import BPE
from torch.utils.data import DataLoader
from GPTdataset import GPTdataset
def pair_input_target(encoded,context_size):
    pairs = []
    inputs = deque()
    
    for i in range(len(encoded)-1):
        
        inputs.append(encoded[i])
        if len(inputs)>context_size:
            inputs.popleft()
        pairs.append((inputs,encoded[i+1]))
    
    return pairs
    
    
    
def create_dataloader(text,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_worker=0):
    with open("data/oromic_data.txt","r") as file:
        raw_data = file.read()

    tokenizer = BPE(vocab_size=100)
    tokenizer.loadVocab()
    
    dataset = GPTdataset(tokenizer=tokenizer,text=text,max_length=max_length,stride=stride
                         )
    
    

    