# %%
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter

from llm import LLM


@dataclass
class Config:
    vocab_size: int = 16384

    # model
    n_dim: int = 768
    n_heads: int = 8
    n_layers: int = 12
    context_size: int = 1024

    # training
    batch_size: int = 64
    steps: int = 30000
    block_size: int = 256
    lr: float = 0.0001
    min_lr: float = 0.00001
    grad_clip = 1.0
    weight_decay = 0.1

    warmup_steps: int = 100
    val_steps: int = 100
    checkpoint_steps: int = 1000

    # logging
    run_name: str = "test_run"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    context_size=Config.context_size,
).to(DEVICE)

weights_path = "/home/kevin/repos/llm/logs/checkpoints/model_v2_v16k_rope_29000.ckpt"
if weights_path:
    weights = torch.load(weights_path, weights_only=False)
    model.load_state_dict(weights["model_state_dict"])


model.train()
n_params = sum(p.numel() for p in model.parameters())
print(f"{n_params:,} parameters")


# Optimization and learning rate
# Only decay params with more than 1 dimension
decay_params = [p for p in model.parameters() if p.dim() >= 2]
no_decay_params = [p for p in model.parameters() if p.dim() < 2]

optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": Config.weight_decay},
    {"params": no_decay_params, "weight_decay": 0.0}
], lr=Config.lr)

warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=Config.warmup_steps
)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, eta_min=Config.min_lr, T_max=Config.steps - Config.warmup_steps
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[Config.warmup_steps]
)

train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")

val_loss = None
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
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
    optimizer.step()
    scheduler.step()

    torch.cuda.synchronize()
    step_time = time.perf_counter() - t0

    # logging
    writer.add_scalar("loss", loss.item(), step)
    writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
    writer.add_scalar("grad_norm", grad_norm.item(), step)

    if step % Config.val_steps == 0:
        val_loss = estimate_loss(val_data)
        print(
            f"Step: {step}, loss: {loss.item()}, val loss: {val_loss}, "
            f"step time: {step_time * 1000:.1f} ms"
        )

    if step % Config.checkpoint_steps == 0:
        checkpoint = {
            "config": asdict(Config()),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "val_loss": val_loss,
        }
        checkpoint_name = f"logs/checkpoints/{Config.run_name}_{step}.ckpt"
        torch.save(checkpoint, checkpoint_name)


# Generate a few tokens
n_tokens = 100
text = "this is the stat of"
ids = torch.tensor(tok.encode(text).ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

model.eval()
for i in range(n_tokens):
    ids = model.generate(ids)

new_text = tok.decode(ids[0].tolist())
print(new_text)
