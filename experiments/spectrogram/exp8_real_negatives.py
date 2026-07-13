"""Experiment 8 (the honest deployment test): 4-way {seq1,seq2,seq3,none} head
where 'none' = REAL non-recombinant triplets (three actual non-recombinant
sequences sampled from the same SANTA files), NOT position-scrambled ones.

This is the failure mode that collapsed the repo's M3 v3: adding real
non-recombinant negatives made the model over-learn 'high divergence -> none' and
crash real-recombinant recall. Question: does the positional (A0) 4-way head
detect 'no recombination here' on genuine negatives while KEEPING real-recombinant
identification (no collapse)?

Metrics: real_acc (which-of-3 on recombinants), none_recall (catches real
non-recomb), false_none (falsely cries none on a real recombinant = the collapse
signature)."""
import json, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from cache_v2_reader import CacheV2
from spectrogram.data import load_santa_split, split_file_set, fix_length, Triplet
from spectrogram.harness import IdentificationDataset, joint_normalize
from spectrogram.encode import encode_triplet
from spectrogram.models import build_backbone, in_channels_for
from spectrogram.train import train_model
from spectrogram.config import SANTA_SPLIT, IMG_SIZE, BATCH_SIZE

ARM = "A0"
BACKBONE = "resnet50"
SANTA_LIMIT = 6000
VAL_LIMIT = 1000
EPOCHS = 15
SEEDS = (0, 1)


def load_none_triplets(cache, which="TRAIN", limit=None, seed=0):
    """Three non-recombinant sequences from the same file = a real 'none' triplet."""
    sp = json.load(open(SANTA_SPLIT))
    keep = split_file_set(sp, which)
    rng = random.Random(seed)
    out = []
    for sh_name, shard in cache.shards.items():
        for rec in shard.align_idx:
            fi = int(rec["file_idx"])
            if (sh_name, shard.files[fi]) not in keep:
                continue
            ids = shard.sample_non_recombinant_ids(fi, 3, rng)
            if len(ids) < 3:
                continue
            align = shard.get_alignment(fi)
            rows = np.stack([fix_length(align[i - 1]) for i in ids]).astype(np.int8)
            out.append(Triplet(rows=rows, recomb_idx=-1, source="santa_none", group=sh_name))
            if limit and len(out) >= limit:
                return out
    return out


class NoneDataset(Dataset):
    """Encode a non-recombinant triplet (row order irrelevant) -> (img, label=3)."""
    def __init__(self, triplets, arm, seed=0):
        self.t = list(triplets); self.arm = arm; self.seed = seed
    def __len__(self):
        return len(self.t)
    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 7919 + i)
        rows = self.t[i].rows[rng.permutation(3)]
        img = joint_normalize(encode_triplet(rows, self.arm))
        x = F.interpolate(torch.from_numpy(img)[None], size=(IMG_SIZE, IMG_SIZE),
                          mode="bilinear", align_corners=False).squeeze(0)
        return x, 3


class Combined(Dataset):
    def __init__(self, recomb, none, arm, seed):
        self.real = IdentificationDataset(recomb, arm, rng_seed=seed)
        self.none = NoneDataset(none, arm, seed=seed)
    def __len__(self):
        return len(self.real) + len(self.none)
    def __getitem__(self, i):
        return self.real[i] if i < len(self.real) else self.none[i - len(self.real)]


def evaluate(model, real_ds, none_ds, dev):
    model.to(dev).eval()
    def preds(ds):
        ps, ys = [], []
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=BATCH_SIZE, num_workers=4):
                ps.append(model(x.to(dev)).argmax(1).cpu().numpy()); ys.append(np.asarray(y))
        return np.concatenate(ps), np.concatenate(ys)
    rp, ry = preds(real_ds); npd, _ = preds(none_ds)
    return {"real_acc": float((rp == ry).mean()),          # which-of-3 on recombinants
            "false_none_rate": float((rp == 3).mean()),      # collapse signature
            "none_recall": float((npd == 3).mean())}         # catches real non-recomb


def main():
    print(f"CUDA {torch.cuda.is_available()} | ARM={ARM} bb={BACKBONE} REAL non-recomb negatives", flush=True)
    t0 = time.time()
    cache = CacheV2()
    tr_r = load_santa_split(cache, which="TRAIN", limit=SANTA_LIMIT)
    va_r = load_santa_split(cache, which="VAL", limit=VAL_LIMIT)
    tr_n = load_none_triplets(cache, "TRAIN", limit=SANTA_LIMIT // 2, seed=1)
    va_n = load_none_triplets(cache, "VAL", limit=VAL_LIMIT // 2, seed=2)
    print(f"loaded recomb tr/va={len(tr_r)}/{len(va_r)} none tr/va={len(tr_n)}/{len(va_n)} "
          f"in {time.time()-t0:.0f}s", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        tr = Combined(tr_r, tr_n, ARM, seed)
        va_real = IdentificationDataset(va_r, ARM, rng_seed=100 + seed)
        va_none = NoneDataset(va_n, ARM, seed=200 + seed)
        m = build_backbone(in_ch=in_channels_for(ARM), pretrained=True, name=BACKBONE, n_classes=4)
        ts = time.time()
        fit = train_model(m, tr, va_real, epochs=EPOCHS)   # val monitor = real ids
        res = evaluate(fit["model"], va_real, va_none, dev)
        runs.append(res)
        print(f"  seed{seed} {res} [{(time.time()-ts)/60:.1f}m]", flush=True)
    agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    out = {"arm": ARM, "backbone": BACKBONE, "negatives": "REAL non-recombinant triplets",
           "config": {"santa_limit": SANTA_LIMIT, "epochs": EPOCHS, "seeds": list(SEEDS)},
           "mean": agg, "runs": runs}
    with open("results_exp8_real_negatives.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== EXP8: 4-WAY WITH REAL NON-RECOMB NEGATIVES (A0/resnet50) ===", flush=True)
    print(f"  real_acc(which-of-3)={agg['real_acc']:.3f}  none_recall={agg['none_recall']:.3f}  "
          f"false_none(collapse)={agg['false_none_rate']:.3f}", flush=True)
    print("INTERP: real_acc stays high (~exp7 0.59) + none_recall high + false_none LOW => NO v3 "
          "collapse; the positional 4-way head handles genuine negatives. High false_none = v3-style collapse.",
          flush=True)
    print(f"total {(time.time()-t0)/60:.0f} min -> results_exp8_real_negatives.json", flush=True)


if __name__ == "__main__":
    main()
