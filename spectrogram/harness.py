"""Permuted-channel 3-way identification dataset."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from spectrogram.config import IMG_SIZE
from spectrogram.encode import encode_triplet

def permute_rows(rows: np.ndarray, recomb_idx: int, rng: np.random.Generator):
    perm = rng.permutation(3)
    new_rows = rows[perm]
    new_idx = int(np.where(perm == recomb_idx)[0][0])
    return new_rows, new_idx

def joint_normalize(img: np.ndarray) -> np.ndarray:
    m = float(img.mean()); s = float(img.std()) + 1e-6
    return ((img - m) / s).astype(np.float32)

def _scramble_positions(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    for c in range(out.shape[0]):
        out[c] = out[c][:, rng.permutation(out.shape[-1])]
    return out

class IdentificationDataset(Dataset):
    def __init__(self, triplets, arm, rng_seed=0, scramble=False):
        self.triplets = list(triplets)
        self.arm = arm
        self.scramble = scramble
        self.rng_seed = rng_seed

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        t = self.triplets[i]
        rng = np.random.default_rng(self.rng_seed * 1_000_003 + i)
        rows, label = permute_rows(t.rows, t.recomb_idx, rng)
        img = encode_triplet(rows, self.arm)          # (C, H, W)
        if self.scramble:
            img = _scramble_positions(img, rng)
        img = joint_normalize(img)
        x = torch.from_numpy(img).unsqueeze(0)        # (1,C,H,W) for interpolate
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                          align_corners=False).squeeze(0)
        return x, label
