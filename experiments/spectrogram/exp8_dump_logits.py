"""Phase 0 Step 0.3 -- quantify the 'none is divergence-driven' confound.

Trains exp8's 4-way {seq1,seq2,seq3,none} head (ONE seed, exp8's exact scale)
and, on the SAME VAL real-negative set (and the positive/recombinant VAL set),
dumps PER-TRIPLET 4-way logits/softmax instead of discarding them at argmax
(the gap in exp8_real_negatives.evaluate()). It then correlates the model's
'none' evidence (none-logit AND none-softmax-prob) with each triplet's
`pairwise_divergence()` -- the scalar KPI a later phase must move.

Additive: imports exp8's functions/classes; does NOT modify exp8_real_negatives.py.
Obeys the repo OOM rule -- only per-triplet single-row reads (fix_length copies one
row); never copies the multi-GB cached alignment tensors.

Run from the worktree ROOT (relative cache/split paths require cwd=root):
    cd <worktree-root>
    PYTHONPATH=. python experiments/spectrogram/exp8_dump_logits.py

Smoke test (wiring only, ~2 min):
    DUMP_EPOCHS=1 DUMP_SANTA_LIMIT=200 DUMP_VAL_LIMIT=120 \
        PYTHONPATH=. python experiments/spectrogram/exp8_dump_logits.py
"""
import os, csv, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from cache_v2_reader import CacheV2
from spectrogram.data import load_santa_split
from spectrogram.harness import IdentificationDataset
from spectrogram.models import build_backbone, in_channels_for
from spectrogram.train import train_model
from spectrogram.probe import pairwise_divergence
from spectrogram.config import BATCH_SIZE

# exp8's own dataset construction -- reused verbatim (main() is __main__-guarded,
# so importing has no side effects).
from exp8_real_negatives import load_none_triplets, NoneDataset, Combined, ARM, BACKBONE

SEED = 0
SANTA_LIMIT = int(os.environ.get("DUMP_SANTA_LIMIT", 6000))   # exp8 scale
VAL_LIMIT = int(os.environ.get("DUMP_VAL_LIMIT", 1000))       # exp8 scale
EPOCHS = int(os.environ.get("DUMP_EPOCHS", 15))               # exp8 scale
OUT_CSV = "experiments/spectrogram/exp8_divergence_corr.csv"
OUT_JSON = "experiments/spectrogram/exp8_divergence_corr.json"

# exp8 seed-0 reference aggregates (results_exp8_real_negatives.json, runs[0]).
REF_SEED0 = {"real_acc": 0.441, "false_none_rate": 0.419, "none_recall": 0.610}


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def dump(model, ds, triplets, source, dev):
    """Order-preserving (shuffle=False) inference -> full 4-way logits per triplet,
    paired with the underlying Triplet's pairwise_divergence (same index order)."""
    model.to(dev).eval()
    logits_chunks, y_chunks = [], []
    for x, y in DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4):
        logits_chunks.append(model(x.to(dev)).float().cpu().numpy())
        y_chunks.append(np.asarray(y))
    logits = np.concatenate(logits_chunks, axis=0)     # (N, 4) -- tiny, no OOM risk
    labels = np.concatenate(y_chunks, axis=0)          # (N,)
    assert len(logits) == len(triplets), f"{source}: {len(logits)} logits vs {len(triplets)} triplets"
    probs = _softmax(logits)
    argmax = logits.argmax(axis=1)
    div = np.array([pairwise_divergence(t) for t in triplets], dtype=float)
    rows = []
    for i in range(len(triplets)):
        am = int(argmax[i])
        rows.append({
            "triplet_id": f"{source}_{i:05d}",
            "source": source,
            "none_logit": float(logits[i, 3]),
            "none_prob": float(probs[i, 3]),
            "argmax": am,
            "pairwise_divergence": float(div[i]),
            # exp8's collapse signature: a real RECOMBINANT (positive) called 'none'.
            # A negative called 'none' is CORRECT, so it is not a false-none error.
            "is_false_none": bool(source == "positive" and am == 3),
        })
    return rows, {"logits": logits, "labels": labels, "argmax": argmax,
                  "none_logit": logits[:, 3], "none_prob": probs[:, 3], "div": div}


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return {"pearson_r": float("nan"), "pearson_p": float("nan"),
                "spearman_r": float("nan"), "spearman_p": float("nan"), "n": int(len(x))}
    pr = pearsonr(x, y); sr = spearmanr(x, y)
    return {"pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
            "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue), "n": int(len(x))}


def main():
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA {torch.cuda.is_available()} | ARM={ARM} bb={BACKBONE} | "
          f"seed={SEED} santa={SANTA_LIMIT} val={VAL_LIMIT} epochs={EPOCHS}", flush=True)

    cache = CacheV2()
    # Mirror exp8.main() dataset construction exactly (seeds 1/2 for none sets).
    tr_r = load_santa_split(cache, which="TRAIN", limit=SANTA_LIMIT)
    va_r = load_santa_split(cache, which="VAL", limit=VAL_LIMIT)
    tr_n = load_none_triplets(cache, "TRAIN", limit=SANTA_LIMIT // 2, seed=1)
    va_n = load_none_triplets(cache, "VAL", limit=VAL_LIMIT // 2, seed=2)
    print(f"loaded recomb tr/va={len(tr_r)}/{len(va_r)} none tr/va={len(tr_n)}/{len(va_n)} "
          f"in {time.time()-t0:.0f}s", flush=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    tr = Combined(tr_r, tr_n, ARM, SEED)
    va_real = IdentificationDataset(va_r, ARM, rng_seed=100 + SEED)
    va_none = NoneDataset(va_n, ARM, seed=200 + SEED)

    m = build_backbone(in_ch=in_channels_for(ARM), pretrained=True, name=BACKBONE, n_classes=4)
    ts = time.time()
    fit = train_model(m, tr, va_real, epochs=EPOCHS)
    model = fit["model"]
    print(f"trained in {(time.time()-ts)/60:.1f}m  best_val_acc={fit['best_val_acc']:.3f}", flush=True)

    # --- inference dumps (positives = recombinants, negatives = real non-recomb) ---
    pos_rows, pos = dump(model, va_real, va_r, "positive", dev)
    neg_rows, neg = dump(model, va_none, va_n, "real_negative", dev)

    # DURABLE FIRST: write per-triplet CSV before any stats (advisor #1).
    all_rows = pos_rows + neg_rows
    cols = ["triplet_id", "source", "none_logit", "none_prob", "argmax",
            "pairwise_divergence", "is_false_none"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {OUT_CSV}", flush=True)

    # --- validation gate: recompute exp8's three aggregates from dumped argmaxes ---
    real_acc = float((pos["argmax"] == pos["labels"]).mean())
    false_none_rate = float((pos["argmax"] == 3).mean())
    none_recall = float((neg["argmax"] == 3).mean())
    gate = {"real_acc": real_acc, "false_none_rate": false_none_rate, "none_recall": none_recall,
            "reference_seed0": REF_SEED0,
            "note": "AMP/cudnn nondeterminism -> approximate, not exact, vs exp8 seed-0."}
    print(f"GATE recomputed real_acc={real_acc:.3f} false_none={false_none_rate:.3f} "
          f"none_recall={none_recall:.3f}  (ref {REF_SEED0})", flush=True)

    # --- correlations: 'none' evidence vs pairwise divergence ---
    div_all = np.concatenate([pos["div"], neg["div"]])
    nl_all = np.concatenate([pos["none_logit"], neg["none_logit"]])
    npb_all = np.concatenate([pos["none_prob"], neg["none_prob"]])

    correlations = {
        "none_logit_vs_divergence": {
            "negatives": corr(neg["none_logit"], neg["div"]),
            "positives": corr(pos["none_logit"], pos["div"]),
            "pooled": corr(nl_all, div_all),
        },
        "none_prob_vs_divergence": {
            "negatives": corr(neg["none_prob"], neg["div"]),
            "positives": corr(pos["none_prob"], pos["div"]),
            "pooled": corr(npb_all, div_all),
        },
    }

    # --- divergence gaps ---
    fn_mask = (pos["argmax"] == 3)                 # false-none errors (positives->none)
    corr_neg_mask = (neg["argmax"] == 3)           # correctly-classified negatives (->none)
    neg_recomb_mask = (neg["argmax"] != 3)         # negatives wrongly given a recombinant id
    pos_recomb_mask = (pos["argmax"] != 3)         # positives correctly given a recombinant id

    def _mean(a):
        return float(np.mean(a)) if len(a) else float("nan")

    mean_div_fn = _mean(pos["div"][fn_mask])
    mean_div_corrneg = _mean(neg["div"][corr_neg_mask])
    gaps = {
        # As specified: mean divergence of false-none errors minus correctly-classified negatives.
        "false_none_vs_correct_negatives": float(mean_div_fn - mean_div_corrneg),
        "mean_div_false_none_errors": mean_div_fn,
        "n_false_none_errors": int(fn_mask.sum()),
        "mean_div_correct_negatives": mean_div_corrneg,
        "n_correct_negatives": int(corr_neg_mask.sum()),
        # Cleaner within-source cuts (no cross-source mixing):
        "within_negatives_none_minus_recomb": float(
            _mean(neg["div"][corr_neg_mask]) - _mean(neg["div"][neg_recomb_mask])),
        "within_positives_none_minus_recomb": float(
            _mean(pos["div"][fn_mask]) - _mean(pos["div"][pos_recomb_mask])),
    }

    # headline KPI: negatives set is the task's primary set; none_logit + Spearman are
    # the robust measures (none_prob saturates on negatives, compressing Pearson).
    headline = {
        "set": "negatives",
        "none_logit_pearson_r": correlations["none_logit_vs_divergence"]["negatives"]["pearson_r"],
        "none_logit_spearman_r": correlations["none_logit_vs_divergence"]["negatives"]["spearman_r"],
        "none_prob_pearson_r": correlations["none_prob_vs_divergence"]["negatives"]["pearson_r"],
        "none_prob_spearman_r": correlations["none_prob_vs_divergence"]["negatives"]["spearman_r"],
    }

    out = {
        "task": "Phase 0 Step 0.3 -- 'none is divergence-driven' confound KPI",
        "seed": SEED,
        "config": {"arm": ARM, "backbone": BACKBONE, "santa_limit": SANTA_LIMIT,
                   "val_limit": VAL_LIMIT, "epochs": EPOCHS},
        "n": {"positives": len(pos_rows), "negatives": len(neg_rows), "pooled": len(all_rows)},
        "validation_gate": gate,
        "correlations": correlations,
        "divergence_gap": gaps,
        "headline_kpi": headline,
        "notes": ("exp8's false_none_rate (=0.407 mean / 0.419 seed-0) measures real "
                  "RECOMBINANTS (positives) wrongly called 'none' -- NOT non-recombinant "
                  "negatives, despite the step brief's phrasing. none_recall (0.610 seed-0) "
                  "is the negatives->none rate."),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== EXP8 DIVERGENCE-CONFOUND KPI ===", flush=True)
    print(f"NEG  none_logit vs div:  pearson={headline['none_logit_pearson_r']:.3f}  "
          f"spearman={headline['none_logit_spearman_r']:.3f}", flush=True)
    print(f"NEG  none_prob  vs div:  pearson={headline['none_prob_pearson_r']:.3f}  "
          f"spearman={headline['none_prob_spearman_r']:.3f}", flush=True)
    print(f"POS  none_logit vs div:  pearson={correlations['none_logit_vs_divergence']['positives']['pearson_r']:.3f}  "
          f"spearman={correlations['none_logit_vs_divergence']['positives']['spearman_r']:.3f}", flush=True)
    print(f"POOL none_logit vs div:  pearson={correlations['none_logit_vs_divergence']['pooled']['pearson_r']:.3f}  "
          f"spearman={correlations['none_logit_vs_divergence']['pooled']['spearman_r']:.3f}", flush=True)
    print(f"GAP  false-none({gaps['n_false_none_errors']}) div={gaps['mean_div_false_none_errors']:.4f}  "
          f"vs correct-neg({gaps['n_correct_negatives']}) div={gaps['mean_div_correct_negatives']:.4f}  "
          f"-> gap={gaps['false_none_vs_correct_negatives']:+.4f}", flush=True)
    print(f"-> {OUT_JSON}", flush=True)
    print(f"total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
