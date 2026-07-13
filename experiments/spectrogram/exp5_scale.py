"""Experiment 5: INVEST in the A4 (dot-plot) lead. Give it the tuning A0 already
had — more SANTA, more epochs, ResNet-50 — and put A0 through the SAME scramble
test at scale. Two questions:
  (1) Does A4 climb toward/past A0 when properly trained? (headroom)
  (2) Is A0's advantage really positional (drops on scramble at scale) or aggregate
      composition (survives scramble)? A4 already dropped at small scale.
Metric: powered SANTA val acc + scrambled val acc + splice_drop, per (arm, backbone)."""
import json, time
import numpy as np
import torch

from cache_v2_reader import CacheV2
from spectrogram.data import load_santa_split
from spectrogram.harness import IdentificationDataset
from spectrogram.models import SmallCNN, build_backbone, in_channels_for
from spectrogram.train import train_model, predict, accuracy

SANTA_LIMIT = 6000
VAL_LIMIT = 1500
EPOCHS = 12
SEEDS = (0, 1)
ARMS = ("A0", "A4")
BACKBONES = (("floor", None), ("resnet50", "resnet50"))

print("CUDA:", torch.cuda.is_available(), flush=True)
t0 = time.time()
cache = CacheV2()
tr = load_santa_split(cache, which="TRAIN", limit=SANTA_LIMIT)
vl = load_santa_split(cache, which="VAL", limit=VAL_LIMIT)
print(f"loaded train={len(tr)} val={len(vl)} in {time.time()-t0:.0f}s", flush=True)


def make(arm, bb):
    ch = in_channels_for(arm)
    return SmallCNN(in_ch=ch) if bb is None else build_backbone(in_ch=ch, pretrained=True, name=bb)


results = {}
for arm in ARMS:
    for bb_label, bb in BACKBONES:
        val, scr = [], []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            dtr = IdentificationDataset(tr, arm, rng_seed=seed)
            dva = IdentificationDataset(vl, arm, rng_seed=100 + seed)
            dsc = IdentificationDataset(vl, arm, rng_seed=100 + seed, scramble=True)
            ts = time.time()
            fit = train_model(make(arm, bb), dtr, dva, epochs=EPOCHS)
            a = accuracy(predict(fit["model"], dva), dva)
            s = accuracy(predict(fit["model"], dsc), dsc)
            val.append(a); scr.append(s)
            print(f"  {arm}:{bb_label} seed{seed} val={a:.4f} scrambled={s:.4f} "
                  f"[{(time.time()-ts)/60:.1f}m, elapsed {(time.time()-t0)/60:.0f}m]", flush=True)
        results[f"{arm}:{bb_label}"] = {"val": val, "scrambled": scr}
        print(f"== {arm}:{bb_label}  val={np.mean(val):.4f}  scrambled={np.mean(scr):.4f}  "
              f"drop={np.mean(val)-np.mean(scr):+.4f}", flush=True)

summary = {k: {"val": float(np.mean(v["val"])), "scrambled": float(np.mean(v["scrambled"])),
               "splice_drop": float(np.mean(v["val"]) - np.mean(v["scrambled"])),
               "val_runs": v["val"], "scr_runs": v["scrambled"]} for k, v in results.items()}
with open("results_exp5_scale.json", "w") as f:
    json.dump({"config": {"santa_limit": SANTA_LIMIT, "epochs": EPOCHS, "seeds": list(SEEDS),
                          "chance": 1/3}, "summary": summary}, f, indent=2)
print("\n=== EXP5 SCALE (val / scrambled / splice_drop, chance=0.333) ===", flush=True)
for k, s in summary.items():
    print(f"  {k:16s} val={s['val']:.3f} scrambled={s['scrambled']:.3f} drop={s['splice_drop']:+.3f}",
          flush=True)
print(f"total {(time.time()-t0)/60:.0f} min -> results_exp5_scale.json", flush=True)
