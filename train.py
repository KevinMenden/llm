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
    batch_size: int = 32
    grad_accum_steps: int = 2
    steps: int = 60000
    block_size: int = 512
    lr: float = 0.00015
    min_lr: float = 0.00001
    grad_clip: float = 1.0
    weight_decay: float = 0.1

    warmup_steps: int = 1000
    val_steps: int = 100
    checkpoint_steps: int = 10000

    # data
    textbook_fraction: float = 0.15

    # logging
    run_name: str = "model_v0.3"

    init_path: str | None = "logs/checkpoints/model_v0.2_99999.ckpt"
    resume_path: str | None = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

fineweb_train = np.memmap("data/fineweb_train.bin", dtype=np.uint16, mode="r")
textbooks_train = np.memmap("data/textbooks_train.bin", dtype=np.uint16, mode="r")
val_sets = {
    "fineweb": np.memmap("data/fineweb_val.bin", dtype=np.uint16, mode="r"),
    "textbooks": np.memmap("data/textbooks_val.bin", dtype=np.uint16, mode="r"),
}

val_losses = None

def get_batch(data, batch_size: int, block_size: int):
    starts = torch.randint(len(data) - block_size, (batch_size,)).tolist()
    batch = np.stack([data[i : i + block_size + 1] for i in starts]).astype(np.int64)
    batch = torch.from_numpy(batch).to(DEVICE)
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()

def get_train_batch():
    """Fixed share of textbook sequences in every batch, the rest is FineWeb"""
    n_textbook = round(Config.batch_size * Config.textbook_fraction)
    x_tb, y_tb = get_batch(textbooks_train, n_textbook, Config.block_size)
    x_fw, y_fw = get_batch(fineweb_train, Config.batch_size - n_textbook, Config.block_size)
    return torch.cat([x_tb, x_fw]), torch.cat([y_tb, y_fw])

@torch.no_grad()
def estimate_loss(data, n_batches: int = 20) -> float:
    """Mean loss over a few random batches, with the model in eval mode"""
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, Config.batch_size, Config.block_size)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# Setup tensorboard logging
writer = SummaryWriter(f"logs/{Config.run_name}")

tok = Tokenizer.from_file("dataset/tokenizer.json")

model = LLM(
    vocab_size=tok.get_vocab_size(),
    n_dim=Config.n_dim,
    n_heads=Config.n_heads,
    n_layers=Config.n_layers,
    context_size=Config.context_size,
).to(DEVICE)
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

# if we resume, restore all existing objects from their state dicts
start_step = 0
tokens_seen = 0
step_offset = 0


if Config.init_path and not Config.resume_path:
    checkpoint = torch.load(Config.init_path, weights_only=True, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    tokens_seen = checkpoint["tokens_seen"]
    step_offset = checkpoint["step"]

if Config.resume_path:
    checkpoint = torch.load(Config.resume_path, weights_only=True, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_step = checkpoint["step"] + 1
    tokens_seen = checkpoint["tokens_seen"]

for step in range(start_step, Config.steps):
    total_step = step + step_offset
    # CUDA is asynchronous, so synchronize before reading the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    optimizer.zero_grad()
    step_loss = 0.0
    for _ in range(Config.grad_accum_steps):
        x, y = get_train_batch()
        tokens_seen += x.numel()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
        (loss / Config.grad_accum_steps).backward()
        step_loss += loss.item() / Config.grad_accum_steps

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
    optimizer.step()
    scheduler.step()

    torch.cuda.synchronize()
    step_time = time.perf_counter() - t0

    # logging
    writer.add_scalar("loss", step_loss, total_step)
    writer.add_scalar("lr", scheduler.get_last_lr()[0], total_step)
    writer.add_scalar("grad_norm", grad_norm.item(), total_step)
    writer.add_scalar("loss_vs_tokens", step_loss, tokens_seen)

    if step % Config.val_steps == 0:
        val_losses = {name: estimate_loss(data) for name, data in val_sets.items()}
        for name, val_loss in val_losses.items():
            writer.add_scalar(f"val_loss/{name}", val_loss, total_step)
        print(
            f"Step: {total_step}, loss: {step_loss}, val losses: {val_losses}, "
            f"step time: {step_time * 1000:.1f} ms"
        )

    if step % Config.checkpoint_steps == 0 or step == Config.steps - 1:
        checkpoint = {
            "config": asdict(Config()),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "val_loss": val_losses,
            "tokens_seen": tokens_seen,
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
