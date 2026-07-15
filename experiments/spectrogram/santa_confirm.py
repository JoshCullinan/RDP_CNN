"""SANTA-only confirmation with PAPER-VALIDATED backbones (ResNet-50, VGG-16 —
the architectures the technique was shown to work on) plus the from-scratch
floor. Question: on the POWERED SANTA held-out identification metric, does the
positional control A0 beat the spectrogram arms A1/A2, robustly across seeds and
backbones? No LANL/LOCO/diagnostics. Uses real committed functions + filtered split."""
import json, time
import numpy as np
import torch

from cache_v2_reader import CacheV2
from spectrogram.data import load_santa_split
from spectrogram.harness import IdentificationDataset
from spectrogram.models import build_backbone, SmallCNN, in_channels_for
from spectrogram.train import train_model, predict, accuracy

SANTA_LIMIT = 8000
VAL_LIMIT = 2000
EPOCHS = 10
SEEDS = (0, 1)
ARMS = ("A0", "A1", "A2")
# (label, timm-name-or-None-for-floor)
BACKBONES = (("floor", None), ("resnet50", "resnet50"), ("vgg16", "vgg16"))

print("CUDA:", torch.cuda.is_available(), flush=True)
t0 = time.time()
cache = CacheV2()
santa_train = load_santa_split(cache, which="TRAIN", limit=SANTA_LIMIT)
santa_val = load_santa_split(cache, which="VAL", limit=VAL_LIMIT)
from collections import Counter
print(f"loaded {len(santa_train)} train {dict(Counter(t.group for t in santa_train))} | "
      f"{len(santa_val)} val {dict(Counter(t.group for t in santa_val))} in {time.time()-t0:.0f}s",
      flush=True)


def make_model(arm, bb_name):
    ch = in_channels_for(arm)
    if bb_name is None:
        return SmallCNN(in_ch=ch)
    return build_backbone(in_ch=ch, pretrained=True, name=bb_name)


results = {}   # "arm:backbone" -> list of val accs across seeds
for arm in ARMS:
    for bb_label, bb_name in BACKBONES:
        accs = []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            tr = IdentificationDataset(santa_train, arm, rng_seed=seed)
            va = IdentificationDataset(santa_val, arm, rng_seed=100 + seed)
            ts = time.time()
            fit = train_model(make_model(arm, bb_name), tr, va, epochs=EPOCHS)
            acc = accuracy(predict(fit["model"], va), va)
            accs.append(acc)
            print(f"  {arm}:{bb_label} seed{seed} val_acc={acc:.4f} "
                  f"[{(time.time()-ts)/60:.1f}m, elapsed {(time.time()-t0)/60:.0f}m]", flush=True)
        results[f"{arm}:{bb_label}"] = accs
        print(f"== {arm}:{bb_label}  mean={np.mean(accs):.4f} std={np.std(accs):.4f} "
              f"accs={[round(a,4) for a in accs]}", flush=True)

summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "accs": [float(x) for x in v]}
           for k, v in results.items()}
out = {"config": {"santa_limit": SANTA_LIMIT, "val_limit": VAL_LIMIT, "epochs": EPOCHS,
                  "seeds": list(SEEDS), "backbones": [b[0] for b in BACKBONES], "chance": 1/3},
       "santa_val": summary}
with open("results_santa_confirm.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n=== SANTA CONFIRMATION SUMMARY (val acc, chance=0.333) ===", flush=True)
for arm in ARMS:
    row = "  %-4s" % arm + " ".join(
        "%-9s=%.3f+-%.3f" % (b[0], summary[f"{arm}:{b[0]}"]["mean"], summary[f"{arm}:{b[0]}"]["std"])
        for b in BACKBONES)
    print(row, flush=True)
print(f"total {(time.time()-t0)/60:.0f} min | written to results_santa_confirm.json", flush=True)
