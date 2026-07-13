"""Experiment 4 (Fable-5 §2.1 + §4): the 4-way {seq1, seq2, seq3, none} head —
the deployment-relevant product (RDP's job: which sequence, if ANY, is the
recombinant) AND a confound fix. 'none' examples are POSITION-SCRAMBLED triplets:
scrambling destroys the splice but preserves composition, so a model that reads
composition CANNOT learn to say 'none', while a splice-reader can. Key metric =
none-recall (does it catch scrambled/no-mosaic inputs).

ARM is set from exp3's best representation. Uses the committed encoders + the
scramble path in IdentificationDataset; defines a 4-class dataset/model inline
(no changes to committed code)."""
import json, sys, time
import numpy as np
import torch
from torch.utils.data import Dataset

from cache_v2_reader import CacheV2
from spectrogram.data import load_santa_split
from spectrogram.harness import IdentificationDataset
from spectrogram.models import SmallCNN, build_backbone, in_channels_for
from spectrogram.train import train_model, predict
from spectrogram.config import BATCH_SIZE

# args: arm [backbone=floor] [santa_limit=3000] [epochs=8]
ARM = sys.argv[1] if len(sys.argv) > 1 else "A0"
BACKBONE = sys.argv[2] if len(sys.argv) > 2 else "floor"
SANTA_LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
VAL_LIMIT = 1000
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
SEEDS = (0, 1)
NONE_FRAC = 0.5   # ~1/3 of examples end up 'none'


def make_fourway_model():
    ch = in_channels_for(ARM)
    if BACKBONE == "floor":
        return SmallCNN(in_ch=ch, n_classes=4)
    return build_backbone(in_ch=ch, pretrained=True, name=BACKBONE, n_classes=4)


class FourWayDataset(Dataset):
    """Real triplets -> label 0/1/2 (which channel is recombinant, permuted);
    scrambled triplets -> label 3 ('none', no coherent mosaic)."""
    def __init__(self, triplets, arm, seed, none_frac=NONE_FRAC):
        self.normal = IdentificationDataset(triplets, arm, rng_seed=seed)
        m = int(len(triplets) * none_frac)
        self.none = IdentificationDataset(triplets[:m], arm, rng_seed=seed + 7, scramble=True)

    def __len__(self):
        return len(self.normal) + len(self.none)

    def __getitem__(self, i):
        if i < len(self.normal):
            return self.normal[i]
        x, _ = self.none[i - len(self.normal)]
        return x, 3


def evaluate(model, ds, dev):
    from torch.utils.data import DataLoader
    model.to(dev).eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4):
            ps.append(model(x.to(dev)).argmax(1).cpu().numpy()); ys.append(np.asarray(y))
    y = np.concatenate(ys); p = np.concatenate(ps)
    overall = float((p == y).mean())
    none_mask = y == 3
    none_recall = float((p[none_mask] == 3).mean()) if none_mask.any() else float("nan")
    real_mask = ~none_mask
    real_acc = float((p[real_mask] == y[real_mask]).mean()) if real_mask.any() else float("nan")
    # how often does it FALSELY cry 'none' on a real recombinant triplet?
    false_none = float((p[real_mask] == 3).mean()) if real_mask.any() else float("nan")
    return {"overall": overall, "none_recall": none_recall, "real_acc": real_acc,
            "false_none_rate": false_none}


def main():
    print(f"CUDA: {torch.cuda.is_available()} | ARM={ARM}", flush=True)
    t0 = time.time()
    cache = CacheV2()
    tr_trip = load_santa_split(cache, which="TRAIN", limit=SANTA_LIMIT)
    va_trip = load_santa_split(cache, which="VAL", limit=VAL_LIMIT)
    print(f"loaded train={len(tr_trip)} val={len(va_trip)} in {time.time()-t0:.0f}s", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        tr = FourWayDataset(tr_trip, ARM, seed)
        va = FourWayDataset(va_trip, ARM, 100 + seed)
        ts = time.time()
        fit = train_model(make_fourway_model(), tr, va, epochs=EPOCHS)
        m = evaluate(fit["model"], va, dev)
        runs.append(m)
        print(f"  seed{seed} {m} [{(time.time()-ts)/60:.1f}m]", flush=True)
    agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    out = {"arm": ARM, "config": {"santa_limit": SANTA_LIMIT, "epochs": EPOCHS,
           "seeds": list(SEEDS), "none_frac": NONE_FRAC, "chance_4way": 0.25}, "mean": agg, "runs": runs}
    with open(f"results_fourway_{ARM}_{BACKBONE}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== 4-WAY {{seq1,seq2,seq3,none}} (arm={ARM} backbone={BACKBONE}) ===", flush=True)
    print(f"  overall={agg['overall']:.3f}  real_acc={agg['real_acc']:.3f}  "
          f"none_recall={agg['none_recall']:.3f}  false_none={agg['false_none_rate']:.3f}", flush=True)
    print("INTERP: high none_recall + low false_none => model uses the SPLICE (can tell "
          "'no coherent mosaic' apart), not just composition.", flush=True)
    print(f"total {(time.time()-t0)/60:.0f} min -> results_fourway_{ARM}.json", flush=True)


if __name__ == "__main__":
    main()
