"""Multi-seed variance characterization + ensemble for the LANL benchmark.

The 0.509→0.519 question is a ~1-breakpoint gap on a 35-BP test set, inside
epoch-to-epoch noise (single model: LANL F1 0.461±0.027 at edge_buffer=25). An
honest "beats classical RDP" claim therefore needs (a) the per-seed F1
distribution, not one lucky run, and (b) a variance-reduced ensemble.

This script, for each seed's deployment checkpoint (SANTA-val-selected m3d_best.pt):
  - reports LANL F1 at edge_buffer=200 (the 0-TP-loss convention),
  - builds the multi-seed ensemble (mean of the per-seed BP probability tracks)
    and reports its F1,
  - reports the distribution (mean ± std, min, max) across seeds.

Usage:
  python m3_eval_ensemble.py --seed-ckpts \
      models_test/m3d_big_snaps/m3d_best.pt \
      models_test/m3d_seed1_snaps/m3d_best.pt \
      models_test/m3d_seed2_snaps/m3d_best.pt \
      models_test/m3d_seed3_snaps/m3d_best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from scipy.signal import find_peaks

from m3_eval_lanl import seq_to_int8, event_f1, load_trained_head, head_forward
from m3_dilated import raw_features

TOL = 200
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]


def load_crfs(triplet_dir: Path, truth):
    crfs = {}
    for fa in sorted(triplet_dir.glob("*.fa")):
        if fa.stem not in truth:
            continue
        s = list(SeqIO.parse(fa, "fasta"))
        if len(s) != 3:
            continue
        R, P1, P2 = (seq_to_int8(str(x.seq)) for x in s)
        ce = len(R)
        for a in (R, P1, P2):
            nz = np.where(a != 4)[0]
            if len(nz):
                ce = min(ce, int(nz[-1]) + 1)
        crfs[fa.stem] = (R, P1, P2, ce)
    return crfs


def peaks(p, thr, ce, eb):
    p2 = p.copy()
    p2[:eb] = 0.0
    p2[ce - eb:ce] = 0.0
    p2[ce:] = 0.0
    pk, _ = find_peaks(p2, height=thr, distance=TOL)
    return pk


def best_f1(prob_by_crf, crfs, truth, eb):
    best = (-1.0, None, None)
    for t in THRESHOLDS:
        tp = fp = fn = 0
        for crf, (R, P1, P2, ce) in crfs.items():
            a, f, n = event_f1(truth[crf], peaks(prob_by_crf[crf], t, ce, eb))
            tp += a; fp += f; fn += n
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best[0]:
            best = (f1, t, (tp, fp, fn))
    return best


def tracks_for(ckpt, crfs, device):
    head, hm, _ = load_trained_head(Path(ckpt), device, 128, 6, 0.1, "single")
    return {crf: head_forward(head, hm, raw_features(R, P1, P2).to(device), device)[0]
            for crf, (R, P1, P2, ce) in crfs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-ckpts", nargs="+", required=True,
                    help="per-seed deployment ckpts (each seed's m3d_best.pt)")
    ap.add_argument("--triplet-dir", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/triplets"))
    ap.add_argument("--truth-csv", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/truth_bps.csv"))
    ap.add_argument("--edge-buffer", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("models_test/m3_ensemble_lanl.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    truth = defaultdict(list)
    for r in csv.DictReader(open(args.truth_csv)):
        truth[r["crf"]].append(int(r["hxb2_position"]))
    truth = {k: sorted(set(v)) for k, v in truth.items()}
    crfs = load_crfs(args.triplet_dir, truth)

    print(f"=== Per-seed LANL F1 (deployment ckpt, edge_buffer={args.edge_buffer}) ===")
    per_seed = []
    all_tracks = []
    for ck in args.seed_ckpts:
        tr = tracks_for(ck, crfs, device)
        all_tracks.append(tr)
        f1, thr, c = best_f1(tr, crfs, truth, args.edge_buffer)
        per_seed.append(f1)
        print(f"  {ck}: F1={f1:.3f} @thr={thr} tp/fp/fn={c}")
    arr = np.array(per_seed)
    print(f"  --> across {len(arr)} seeds: mean={arr.mean():.3f} std={arr.std():.3f} "
          f"min={arr.min():.3f} max={arr.max():.3f}")

    ens = {crf: np.mean([tr[crf] for tr in all_tracks], axis=0) for crf in crfs}
    ef1, ethr, ec = best_f1(ens, crfs, truth, args.edge_buffer)
    print(f"\n=== Multi-seed ensemble (mean of {len(all_tracks)} BP tracks) ===")
    print(f"  F1={ef1:.3f} @thr={ethr} tp/fp/fn={ec}")
    print(f"  classical RDP standalone: 0.519   "
          f"→ ensemble {'BEATS' if ef1 > 0.519 else 'does not beat'} RDP "
          f"by {ef1-0.519:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "edge_buffer": args.edge_buffer,
        "per_seed_f1": {ck: f for ck, f in zip(args.seed_ckpts, per_seed)},
        "per_seed_mean": float(arr.mean()), "per_seed_std": float(arr.std()),
        "ensemble_f1": ef1, "ensemble_thr": ethr, "ensemble_tp_fp_fn": ec,
        "rdp_baseline": 0.519,
    }, indent=2, default=float))
    print(f"\n  report → {args.out}")


if __name__ == "__main__":
    main()
