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
    vocab_size: int = 2000

    # model
    n_dim: int = 512
    n_heads: int = 8
    n_layers: int = 8
    context_size: int = 1024

    # training
    batch_size: int = 256
    steps: int = 2000
    block_size: int = 256
    lr: float = 0.001
    min_lr: float = 0.0001
    warmup_steps: int = 0

    # logging
    run_name: str = "test_init"


DEVICE = "cuda"


def get_batch(data, batch_size: int, block_size: int):
    seq_limit = len(data) - block_size - 1
    starts = torch.randint(seq_limit, (batch_size,)).tolist()
    slices_x = [data[i : i + block_size] for i in starts]
    slices_y = [data[i + 1 : i + 1 + block_size] for i in starts]
    batch_x = torch.stack([torch.from_numpy(s.astype(np.int64)) for s in slices_x])
    batch_y = torch.stack([torch.from_numpy(s.astype(np.int64)) for s in slices_y])

    return batch_x.to(DEVICE), batch_y.to(DEVICE)


@torch.no_grad()
def estimate_loss(data, n_batches: int = 20) -> float:
    """Mean loss over a few random batches, with the model in eval mode"""
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, Config.batch_size, Config.block_size)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# Setup tensorboard logging
writer = SummaryWriter(f"logs/{Config.run_name}")

tok = Tokenizer.from_file("tokenizer.json")
model = LLM(
    vocab_size=tok.get_vocab_size(),
    n_dim=Config.n_dim,
    n_heads=Config.n_heads,
    n_layers=Config.n_layers,
).to(DEVICE)
model.train()
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} parameters")


# Optimization and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)

warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=Config.warmup_steps
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=Config.min_lr, T_max=Config.steps)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[Config.warmup_steps]
)

train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")


for step in range(Config.steps):
    # CUDA is asynchronous, so synchronize before reading the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    x, y = get_batch(train_data, Config.batch_size, Config.block_size)
    # run model with bf16 (80ms/s vs 180ms/s with f)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    torch.cuda.synchronize()
    step_time = time.perf_counter() - t0

    # logging
    writer.add_scalar("loss", loss.item(), step)
    writer.add_scalar("lr", scheduler.get_last_lr()[0], step)

    if step % 100 == 0:
        val_loss = estimate_loss(val_data)
        print(
            f"Step: {step}, loss: {loss.item()}, val loss: {val_loss}, "
            f"step time: {step_time * 1000:.1f} ms"
        )


# Generate a few tokens
n_tokens = 100
text = "this is the stat of"
ids = torch.tensor(tok.encode(text).ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

model.eval()
for i in range(n_tokens):
    ids = model.generate(ids)

new_text = tok.decode(ids[0].tolist())
print(new_text)
