"""Race the splice-explicit arms against the positional control on the POWERED
SANTA metric, with a scramble-eval confound check (Fable-5 advisor's decision
experiment). Arms: A0 (positional control), A1 (summed spectrogram, reference),
A3 (difference-spectrogram), A4 (dot-plot). Small CNN, no big backbone needed.

Decision rule per arm: a frequency/2-D arm 'wins' only if it beats A0 on normal
val AND its accuracy COLLAPSES on position-scrambled val (proving it reads the
splice, not composition). If a model stays accurate when positions are scrambled,
it is reading composition -> confounded."""
import json, time
import numpy as np
import torch

from cache_v2_reader import CacheV2
from spectrogram.data import load_santa_split
from spectrogram.harness import IdentificationDataset
from spectrogram.models import SmallCNN, in_channels_for
from spectrogram.train import train_model, predict, accuracy

SANTA_LIMIT = 2000
VAL_LIMIT = 800
EPOCHS = 8
SEEDS = (0, 1)
ARMS = ("A0", "A1", "A3", "A4")

print("CUDA:", torch.cuda.is_available(), flush=True)
t0 = time.time()
cache = CacheV2()
santa_train = load_santa_split(cache, which="TRAIN", limit=SANTA_LIMIT)
santa_val = load_santa_split(cache, which="VAL", limit=VAL_LIMIT)
print(f"loaded train={len(santa_train)} val={len(santa_val)} in {time.time()-t0:.0f}s", flush=True)

results = {}
for arm in ARMS:
    normal, scrambled = [], []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        tr = IdentificationDataset(santa_train, arm, rng_seed=seed)
        va = IdentificationDataset(santa_val, arm, rng_seed=100 + seed)
        va_scr = IdentificationDataset(santa_val, arm, rng_seed=100 + seed, scramble=True)
        ts = time.time()
        fit = train_model(SmallCNN(in_ch=in_channels_for(arm)), tr, va, epochs=EPOCHS)
        acc = accuracy(predict(fit["model"], va), va)
        scr = accuracy(predict(fit["model"], va_scr), va_scr)
        normal.append(acc); scrambled.append(scr)
        print(f"  {arm} seed{seed} val={acc:.4f} scrambled={scr:.4f} "
              f"[{(time.time()-ts)/60:.1f}m, elapsed {(time.time()-t0)/60:.0f}m]", flush=True)
    results[arm] = {"val": normal, "scrambled": scrambled}
    print(f"== {arm}  val={np.mean(normal):.4f}+-{np.std(normal):.4f}  "
          f"scrambled={np.mean(scrambled):.4f}+-{np.std(scrambled):.4f}  "
          f"drop={np.mean(normal)-np.mean(scrambled):+.4f}", flush=True)

summary = {a: {"val_mean": float(np.mean(v["val"])), "val_std": float(np.std(v["val"])),
               "scr_mean": float(np.mean(v["scrambled"])), "scr_std": float(np.std(v["scrambled"])),
               "splice_drop": float(np.mean(v["val"]) - np.mean(v["scrambled"])),
               "val": [float(x) for x in v["val"]], "scrambled": [float(x) for x in v["scrambled"]]}
           for a, v in results.items()}
a0v = summary["A0"]["val_mean"]
verdict = {a: {"beats_A0_on_val": summary[a]["val_mean"] > a0v,
               "collapses_on_scramble": summary[a]["splice_drop"] > 0.1,
               "keep": (summary[a]["val_mean"] > a0v and summary[a]["splice_drop"] > 0.1)}
           for a in ("A1", "A3", "A4")}
out = {"config": {"santa_limit": SANTA_LIMIT, "val_limit": VAL_LIMIT, "epochs": EPOCHS,
                  "seeds": list(SEEDS), "chance": 1/3, "backbone": "SmallCNN"},
       "summary": summary, "verdict_vs_A0": verdict}
with open("results_splice_race.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n=== SPLICE-ARM RACE (val acc / scrambled acc, chance=0.333) ===", flush=True)
for a in ARMS:
    s = summary[a]
    print(f"  {a}: val={s['val_mean']:.3f} scrambled={s['scr_mean']:.3f} splice_drop={s['splice_drop']:+.3f}",
          flush=True)
print("verdict vs A0:", json.dumps(verdict), flush=True)
print(f"total {(time.time()-t0)/60:.0f} min -> results_splice_race.json", flush=True)
