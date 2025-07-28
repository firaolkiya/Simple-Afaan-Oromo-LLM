from collections import defaultdict
from typing import List, Dict, Tuple, Union
import pickle
import os

class BPE:
    def __init__(self, vocab_size: int, special_tokens: List[str]|None = None):
        self.vocab_size = vocab_size
        self.vocab: Dict[Union[str, Tuple], int] = {}
        self.inverse_vocab: Dict[int, Union[str, Tuple]] = {}
        self.merges: Dict[Tuple[int, int], int] = {}
        self.special_tokens = special_tokens or []
        self.next_id = 0

        # Initialize vocab with special tokens
        for token in self.special_tokens:
            self.vocab[(token,)] = self.next_id
            self.inverse_vocab[self.next_id] = (token,)
            self.next_id += 1

    def _add_token(self, token: Union[str, Tuple]) -> int:
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.inverse_vocab[self.next_id] = token
            self.next_id += 1
        return self.vocab[token]
    def saveVocab(self, path="data/vocab.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump({
                "vocab": self.vocab,
                "inverse_vocab": self.inverse_vocab,
                "merges": self.merges,
                "next_id": self.next_id,
                "special_tokens": self.special_tokens
            }, file)
    
    def loadVocab(self, path="data/vocab.pkl"):
        with open(path, "rb") as file:
            data = pickle.load(file)
            self.vocab = data["vocab"]
            self.inverse_vocab = data["inverse_vocab"]
            self.merges = data["merges"]
            self.next_id = data["next_id"]
            self.special_tokens = data["special_tokens"]
            
    def train(self, corpus: List[str]):
        # Add characters with </w> marker to initial vocab
        char_set = set("".join(corpus))
        for ch in sorted(char_set):
            self._add_token((ch,))
        self._add_token(("</w>",))  # end-of-word marker

        # Split words into character tokens + end-of-word
        word_splits = {}
        for word in corpus:
            split = [(c,) for c in word] + [("</w>",)]
            word_splits[word] = [self.vocab[tok] for tok in split]

        # Iteratively apply merges until vocab_size is reached
        while self.next_id < self.vocab_size:
            pair_freqs = defaultdict(int)
            for split in word_splits.values():
                for i in range(len(split) - 1):
                    pair_freqs[(split[i], split[i + 1])] += 1

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get) # type: ignore
            new_token_id = self.next_id
            self.merges[best_pair] = new_token_id
            self.inverse_vocab[new_token_id] = best_pair
            self.vocab[best_pair] = new_token_id
            self.next_id += 1

            for word, split in word_splits.items():
                new_split = []
                i = 0
                while i < len(split):
                    if i + 1 < len(split) and (split[i], split[i + 1]) == best_pair:
                        new_split.append(new_token_id)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                word_splits[word] = new_split

    def tokenize(self, text: str) -> List[int]:
        tokens = []

        for word in text.strip().split():
            # Special token detection
            if word in self.special_tokens:
                tokens.append(self.vocab.get((word,), self.vocab.get(("[UNK]",))))
                continue

            chars = [(c,) for c in word] + [("</w>",)]
            current = [self.vocab.get(ch, self.vocab.get(("[UNK]",))) for ch in chars]

            # Apply merges
            merge_applied = True
            while merge_applied:
                merge_applied = False
                i = 0
                new_current = []
                while i < len(current):
                    if i + 1 < len(current) and (current[i], current[i + 1]) in self.merges:
                        new_current.append(self.merges[(current[i], current[i + 1])]) # type: ignore
                        i += 2
                        merge_applied = True
                    else:
                        new_current.append(current[i])
                        i += 1
                current = new_current

            tokens.extend(current)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        def recursive_decode(token_id):
            token = self.inverse_vocab.get(token_id, ("[UNK]",))
            if isinstance(token, tuple) and len(token) == 1 and token[0] in self.special_tokens:
                return token[0]
            if isinstance(token, tuple) and all(isinstance(x, int) for x in token):
                return "".join(recursive_decode(tid) for tid in token)
            if isinstance(token, tuple):
                return "".join(token)
            return token

        text = "".join(recursive_decode(tid) for tid in token_ids)
        return text.replace("</w>", " ") 
