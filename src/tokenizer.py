import re

class Tokinizer():
    
    def __init__(self,vocab):
        
        self.str_token = vocab
        self.token_str = {s:i for i,s in vocab.items()}
        
    def encode(self,text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        result = [item.strip() for item in result if item.strip()]
        encoded = [self.str_token[item] if item in self.str_token else "<|unk|>" for item in result]
        return encoded
    
    def decode(self,encoded):
        
        text = [self.token_str[token] for token in encoded]
        text = " ".join(text)
        text = re.sub(r'\s+([,.:;?_!"()\'])','\1',text)
        return text
        
    