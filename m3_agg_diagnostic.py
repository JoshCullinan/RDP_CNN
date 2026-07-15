"""DIAGNOSTIC: does robust aggregation of the 4 per-seed BP-probability tracks
recover the ensemble F1 lost to two noisy (low-precision/FP-heavy) seeds
(seed0, seed3)?

Context: `m3_eval_ensemble.py`'s deployed aggregation is a plain MEAN of the 4
per-seed per-position BP tracks, then peak-extract + single-global-threshold
scoring. That mean-of-tracks ensemble scores F1=0.512 (eb=200) / 0.477 (eb=25)
-- BELOW the per-seed mean (0.533) -- because seed0 and seed3's noisy tracks
dilute otherwise-good consensus peaks from seed1/seed2.

This script tests whether a robust (order-statistic) aggregator recovers that
loss by adding two alternatives:
  - MEDIAN of tracks (element-wise median across the 4 seed tracks/position)
  - RANK-MEAN (average of per-seed percentile-rank-transformed tracks, via
    scipy.stats.rankdata -- makes the aggregate invariant to any one seed's
    absolute probability scale/calibration, only relative ordering matters)

IMPORTANT FRAMING: this is a DIAGNOSTIC of aggregation fragility, NOT a new
headline number. The pre-registered baseline stays the mean-of-tracks result
above (produced by m3_eval_ensemble.py, unmodified by this script). Report
median/rank-mean as "does/doesn't indicate the 2 noisy seeds are the
mechanism", not as "the" ensemble result.

Reuses m3_eval_ensemble.load_crfs / tracks_for / best_f1 -- same track
loading, same pooled tp/fp/fn-at-single-global-threshold scoring, same 4 LANL
CRFs. Purely additive: does not alter m3_eval_ensemble.py or its outputs.

Usage:
  python m3_agg_diagnostic.py --seed-ckpts \
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
from scipy.stats import rankdata

from m3_eval_ensemble import load_crfs, tracks_for, best_f1

MEAN_BASELINE = {200: 0.512, 25: 0.477}  # pre-registered, from m3_eval_ensemble.py


def rank_transform(track: np.ndarray) -> np.ndarray:
    """Percentile-rank transform to [0, 1], average rank for ties."""
    ranks = rankdata(track, method="average")  # 1..N
    return (ranks - 1.0) / max(1, len(track) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-ckpts", nargs="+", required=True,
                    help="per-seed deployment ckpts (each seed's m3d_best.pt)")
    ap.add_argument("--triplet-dir", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/triplets"))
    ap.add_argument("--truth-csv", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/truth_bps.csv"))
    ap.add_argument("--edge-buffers", nargs="+", type=int, default=[200, 25])
    ap.add_argument("--agg", choices=["median", "rankmean", "both"], default="both")
    ap.add_argument("--out", type=Path,
                    default=Path("models_test/m3_agg_diagnostic.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    truth = defaultdict(list)
    for r in csv.DictReader(open(args.truth_csv)):
        truth[r["crf"]].append(int(r["hxb2_position"]))
    truth = {k: sorted(set(v)) for k, v in truth.items()}
    crfs = load_crfs(args.triplet_dir, truth)

    print(f"=== Loading {len(args.seed_ckpts)} seed tracks ===")
    all_tracks = []
    for ck in args.seed_ckpts:
        all_tracks.append(tracks_for(ck, crfs, device))
        print(f"  loaded: {ck}")

    median_tracks = {crf: np.median([tr[crf] for tr in all_tracks], axis=0)
                      for crf in crfs} if args.agg in ("median", "both") else None

    rank_tracks = None
    if args.agg in ("rankmean", "both"):
        per_seed_ranked = [{crf: rank_transform(tr[crf]) for crf in crfs}
                            for tr in all_tracks]
        rank_tracks = {crf: np.mean([rt[crf] for rt in per_seed_ranked], axis=0)
                        for crf in crfs}

    print("\n=== DIAGNOSTIC: robust aggregation vs pre-registered mean-of-tracks ===")
    print("(mean-of-tracks ensemble is NOT recomputed/changed here -- values below "
          "are the frozen numbers from m3_eval_ensemble.py)")
    results = {}
    for eb in args.edge_buffers:
        results[eb] = {"mean_baseline_f1": MEAN_BASELINE.get(eb)}
        print(f"\n  --- edge_buffer={eb} ---")
        base = MEAN_BASELINE.get(eb)
        if base is not None:
            print(f"    mean-of-tracks (baseline, frozen) : F1={base:.3f}")
        if median_tracks is not None:
            f1, thr, c = best_f1(median_tracks, crfs, truth, eb)
            results[eb]["median"] = {"f1": f1, "thr": thr, "tp_fp_fn": list(c)}
            print(f"    median-of-tracks                  : F1={f1:.3f} @thr={thr} "
                  f"tp/fp/fn={c}")
        if rank_tracks is not None:
            f1, thr, c = best_f1(rank_tracks, crfs, truth, eb)
            results[eb]["rankmean"] = {"f1": f1, "thr": thr, "tp_fp_fn": list(c)}
            print(f"    rank-mean-of-tracks                : F1={f1:.3f} @thr={thr} "
                  f"tp/fp/fn={c}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "framing": "DIAGNOSTIC of aggregation fragility, not a new headline. "
                   "The pre-registered baseline (mean-of-tracks ensemble, "
                   "produced by m3_eval_ensemble.py) is unchanged by this script.",
        "mean_baseline": MEAN_BASELINE,
        "seed_ckpts": args.seed_ckpts,
        "results": results,
    }, indent=2, default=float))
    print(f"\n  report → {args.out}")


if __name__ == "__main__":
    main()
