# %%
import time
from dataclasses import dataclass

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter

from llm import LLM


@dataclass
class Config:
    vocab_size: int = 16384

    # model
    n_dim: int = 1024
    n_heads: int = 8
    n_layers: int = 8
    context_size: int = 1024

    # training
    batch_size: int = 128
    steps: int = 10000
    block_size: int = 256
    lr: float = 0.001
    min_lr: float = 0.0001
    warmup_steps: int = 100
    val_steps: int = 100
    checkpoint_steps: int = 1000

    # logging
    run_name: str = "model_large_100m_rope_long"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


tok = Tokenizer.from_file("tokenizer.json")
model = LLM(
    vocab_size=tok.get_vocab_size(),
    n_dim=Config.n_dim,
    n_heads=Config.n_heads,
    n_layers=Config.n_layers,
).to(DEVICE)
model.eval()

weights = torch.load("/home/kevin/repos/llm/logs/checkpoints/model_large_100m_rope_long_9000.ckpt", weights_only=False)

model.load_state_dict(weights["model_state_dict"])


# Generate a few tokens
n_tokens = 100
text = "An elephant is"
ids = torch.tensor(tok.encode(text).ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

model.eval()
for i in range(n_tokens):
    ids = model.generate(ids)

new_text = tok.decode(ids[0].tolist())
print(new_text)
