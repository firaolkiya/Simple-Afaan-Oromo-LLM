from byte_bit_encoder import BPE
from torch import tensor
from torch.utils.data import Dataset
class GPTdataset(Dataset):
    
    def __init__(self,text,tokenizer:BPE,stride=128,max_length=256) -> None:
        self.input_ids  = []
        self.target_ids = []
        
        encoded = tokenizer.tokenize(text)
        
        for i in range(0,len(encoded)-max_length,stride):
            current_input = encoded[i:i+max_length]
            current_target = encoded[i+1:i+max_length+1]
            
            self.input_ids.append(tensor(current_input))
            self.target_ids.append(tensor(current_target))
            
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx], self.target_ids[idx]
        
        
        