import os.path

from llm import LLM
from datasets import load_dataset
from tokenizers import Tokenizer
import numpy as np

# Prepare dataset
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
ds_train = ds["train"]
ds_val = ds["validation"]
tok = Tokenizer.from_file("tokenizer.json")

def tokenize_split(split, path, chunk_size=10_000):
    if os.path.exists(path):
        print("Data already extracted")
        return
    texts = split["text"]
    chunks = []
    for i in range(0, len(texts), chunk_size):
        encs = tok.encode_batch(texts[i : i + chunk_size])
        ids = [t for enc in encs for t in enc.ids]
        chunks.append(np.array(ids, dtype=np.uint16))
    np.concatenate(chunks).tofile(path)

tokenize_split(ds_train, "train.bin")
tokenize_split(ds_val, "val.bin")
