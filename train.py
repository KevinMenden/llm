#%%
import os.path

import torch

from llm import LLM
from datasets import load_dataset
from tokenizers import Tokenizer
import numpy as np

config = {
    "batch_size": 256,
    "steps": 1000,
    "block_size": 128,
    "learning_rate": 0.001
}

DEVICE = "cuda"

def get_batch(data, batch_size: int, block_size: int):
    seq_limit = len(data) - block_size - 1
    starts = torch.randint(seq_limit, (batch_size,)).tolist()
    slices_x = [data[i : i + block_size] for i in starts]
    slices_y = [data[i+1 : i+1 + block_size] for i in starts]
    batch_x = torch.stack([torch.from_numpy(s.astype(np.int64)) for s in slices_x])
    batch_y = torch.stack([torch.from_numpy(s.astype(np.int64)) for s in slices_y])

    return batch_x.to(DEVICE), batch_y.to(DEVICE)


tok = Tokenizer.from_file("tokenizer.json")
model = LLM(vocab_size=tok.get_vocab_size()).to(DEVICE)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("val.bin", dtype=np.uint16, mode="r")

for step in range(config["steps"]):

    x, y = get_batch(train_data, config["batch_size"], config["block_size"])
    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step: {step}, loss: {loss.item()}")
