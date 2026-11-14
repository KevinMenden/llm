"""Batches of whole conversations for SFT, from the arrays written by prepare_sft.py"""

import json
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path("data/sft")
PAD_ID = 4  # <|pad|>
IGNORE_INDEX = -100  # F.cross_entropy ignores these targets by default


class SFTData:
    def __init__(self, split: str, weights: dict[str, float] | None = None):
        """weights: sampling weight per source, e.g. {"tutorchat/dialogue": 3.0}.
        Sources that aren't listed get weight 1. Without weights, every conversation
        is equally likely."""
        self.ids = np.load(DATA_DIR / f"{split}_ids.npy", mmap_mode="r")
        self.mask = np.load(DATA_DIR / f"{split}_mask.npy", mmap_mode="r")
        self.offsets = np.load(DATA_DIR / f"{split}_offsets.npy")
        self.sources = np.load(DATA_DIR / f"{split}_sources.npy")
        with open(DATA_DIR / "sources.json") as f:
            self.source_names = json.load(f)

        # probability per conversation
        weights = weights or {}
        per_source = np.array([weights.get(name, 1.0) for name in self.source_names])
        p = per_source[self.sources]
        self.p = p / p.sum()

    def __len__(self):
        return len(self.sources)

    def get_batch(self, batch_size: int, device: str):
        """Random conversations, right padded to the longest one. Returns inputs x and
        targets y of shape (batch_size, T); y is IGNORE_INDEX where nothing is trained."""
        indices = np.random.choice(len(self), size=batch_size, p=self.p)
        starts, ends = self.offsets[indices], self.offsets[indices + 1]
        T = int((ends - starts).max()) - 1

        x = np.full((batch_size, T), PAD_ID, dtype=np.int64)
        y = np.full((batch_size, T), IGNORE_INDEX, dtype=np.int64)
        for row, (start, end) in enumerate(zip(starts, ends)):
            ids = self.ids[start:end].astype(np.int64)
            mask = self.mask[start:end].astype(bool)
            n = len(ids) - 1
            x[row, :n] = ids[:-1]
            # the target at position t is token t + 1, trained only where its mask is set
            y[row, :n] = np.where(mask[1:], ids[1:], IGNORE_INDEX)

        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
