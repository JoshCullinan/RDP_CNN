"""Confound diagnostics: own-parents leak gate, divergence analysis.
Positional-scramble is exercised via IdentificationDataset(scramble=True)."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from spectrogram.encode import encode_triplet, spectrogram_one
from spectrogram.harness import joint_normalize
from spectrogram.config import IMG_SIZE
import torch.nn.functional as F

def pairwise_divergence(triplet) -> float:
    r = triplet.rows
    pairs = [(0, 1), (0, 2), (1, 2)]
    mm = [(r[i] != r[j]).mean() for i, j in pairs]
    return float(np.mean(mm))

def divergence_vs_prediction(triplets, preds, correct) -> float:
    div = np.array([pairwise_divergence(t) for t in triplets])
    c = np.asarray(correct, float)
    if c.std() == 0 or div.std() == 0:
        return 0.0
    return float(np.corrcoef(div, c)[0, 1])

class _OwnParentsDataset(Dataset):
    """Single-sequence binary: recombinant (1) vs one of its parents (0)."""
    def __init__(self, triplets, rng_seed=0):
        self.items = []
        rng = np.random.default_rng(rng_seed)
        for t in triplets:
            self.items.append((t.rows[t.recomb_idx], 1))
            par = [k for k in range(3) if k != t.recomb_idx][rng.integers(2)]
            self.items.append((t.rows[par], 0))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        row, y = self.items[i]
        img = joint_normalize(spectrogram_one(row)[None])   # (1,F,T)
        x = F.interpolate(torch.from_numpy(img)[None], size=(IMG_SIZE, IMG_SIZE),
                          mode="bilinear", align_corners=False).squeeze(0)
        return x, y

def run_p2_gate(triplets, epochs=5, device=None) -> dict:
    """Fit on a train split of triplets, report AUC on a held-out split --
    a generalization metric, not a training-fit metric (I2)."""
    from spectrogram.models import SmallCNN
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_train = int(len(triplets) * 0.8)
    train_triplets = triplets[:n_train]
    heldout_triplets = triplets[n_train:]
    train_ds = _OwnParentsDataset(train_triplets)
    heldout_ds = _OwnParentsDataset(heldout_triplets)
    m = SmallCNN(in_ch=1, n_classes=2).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    for _ in range(epochs):
        m.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(m(x), y); loss.backward(); opt.step()
    m.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in DataLoader(heldout_ds, batch_size=16):
            ps.append(torch.softmax(m(x.to(dev)), 1)[:, 1].cpu().numpy()); ys.append(y.numpy())
    auc = float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))
    return {"auc": auc, "leak": auc > 0.6}
