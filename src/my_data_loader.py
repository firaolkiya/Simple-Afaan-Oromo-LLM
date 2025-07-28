from collections import deque
from byte_bit_encoder import BPE
from torch.utils.data import DataLoader
from GPTdataset import GPTdataset
    
    
def create_dataloader(text,batch_size=4,max_length=256,stride=128,shuffle=True,drop_last=True,num_workers=0):
    # with open("data/oromic_data.txt","r") as file:
    #     raw_data = file.read()

    tokenizer = BPE(vocab_size=100)
    tokenizer.loadVocab()
    dataset = GPTdataset(tokenizer=tokenizer,
                         text=text,
                         max_length=max_length,
                         stride=stride
                         )
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
    
    )
    
    return dataloader


dataloader = create_dataloader("Afaan Oromoo dubbachuuf barachuu barbaada. Afaan keenya bareedaa fi aadaa qaba",max_length=4,stride=4)