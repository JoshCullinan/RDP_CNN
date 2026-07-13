"""Config-driven AMP training + prediction for the 3-way identification task."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from spectrogram.config import BATCH_SIZE, NUM_WORKERS

def _device(d=None):
    return torch.device(d or ("cuda" if torch.cuda.is_available() else "cpu"))

def train_model(model, train_ds, val_ds, *, epochs, batch_size=BATCH_SIZE, lr=1e-4,
                device=None, amp=True):
    dev = _device(device); model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")
    lossf = torch.nn.CrossEntropyLoss()
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    history, best = [], 0.0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp and dev.type == "cuda"):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        va = accuracy(predict(model, val_ds, dev), val_ds)
        history.append({"epoch": ep, "val_acc": va}); best = max(best, va)
    return {"model": model, "history": history, "best_val_acc": best}

@torch.no_grad()
def predict(model, ds, device=None):
    dev = _device(device); model.to(dev).eval()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    out = []
    for x, _ in dl:
        out.append(model(x.to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def accuracy(preds, ds) -> float:
    labels = np.array([ds[i][1] for i in range(len(ds))])
    return float((preds == labels).mean())
