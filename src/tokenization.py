import re
from tokenizer import Tokinizer
from byte_bit_encoder import BPE
def split_all(text):
    return re.split(r'([,.:;?_!"()\']|--|\s)',text)

def preprocess(text):
    result = split_all(text)
    result = [item.strip() for item in result if item.strip()]
    return result

def assign_token(data):
    unique = sorted(set(data))
    mapped = {item:index for index,item in enumerate(unique)}
    return mapped
with open("data/oromic_data.txt","r")  as file:
    raw_text = file.read()
    corpus = preprocess(raw_text)
    # proccessed.extend(["<|endoftext|>","<|unk|>"])
    # token_mapped = assign_token(proccessed)
    
    # corpus = ["low", "lowest", "new", "wider", "newest"]
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    bpe = BPE(vocab_size=50, special_tokens=special_tokens)
    bpe.train(corpus)

    print("Vocabulary (token -> id):")
    for k, v in bpe.vocab.items():
        print(f"{v}: {k}")
    print("\nMerges (in order):")
    for k, v in bpe.merges.items():
        print(f"{k} -> {v}")

    # ---------------------
    # ✅ Tokenize & Decode
    # ---------------------
    test_data = """
    Har’a hojiiwwan ijoo bu’uura sirna daldalaa ammayyaa ta’an ifoomsinee jirra. Dr.Kaasuun Gofee.
    “Daldala seeraan alaa bittaafi gurgurtaa boba’aa irratti raawwatu to’achuuf labsii bahe hojiirra oolchuudha”- Aadde Ayishaa Yahyaa.
    Ministeerri Daldalaafi Walitti hidhamiinsa Naannawaa Daldala Al-ergiirraa Doolaara biil. 3.28 argadhe jedhe
    Pirezidaanti  Shimallis Abdiisaa Tarminaala Buufata Xiyyaara Jeneeraal Waaqoo Guutuu eebbisan.
    """

    encoded = bpe.tokenize(test_data)
    print("\nEncoded tokens:\n", encoded)

    decoded = bpe.decode(encoded)
    print("\nDecoded text:\n", decoded)

    # ✅ Example decode special token
    unk_id = bpe.vocab.get(("[UNK]",)) 
    print(f"\nSpecial token [UNK] id: {unk_id}")
    print(f"Decoded [UNK]: {bpe.decode([unk_id if unk_id else 3454])}")
    
    bpe.saveVocab()
  
    