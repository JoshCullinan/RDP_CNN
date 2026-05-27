"""Multi-virus eval v2 — enumerate triplets, measure divergence vs FP rate.

The v1 eval (m3_eval_multivirus.py) ran one cherry-picked triplet per
panel and surfaced the qualitative pattern:
  - SARS-CoV-2 XBB ✓ hit at Δ=293 bp from truth
  - Zika ✓ clean
  - Ebola ✗ 13 false peaks (cross-species triplet)

This v2 runs ALL non-recombinant triplets per panel (C(n,3) * 3 orderings,
filtered to one canonical ordering per triplet) and records:

  - Pairwise divergence: |R-P1|, |R-P2|, |P1-P2|  (mismatch fraction over non-gap)
  - Number of peaks at deployment threshold (0.50)
  - Mean / max predicted probability across the genome

The output answers: at what divergence level does M3 start over-predicting?
This is the precise failure-mode characterization for the model.

For HIV the gold-standard truth is LANL CRFs; we already have F1 0.509 there.
This script is about *non-HIV* false-positive characterization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_dilated import DilatedHead, raw_features


NT_TO_INT = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}


def seq_to_int8(s: str) -> np.ndarray:
    arr = np.empty(len(s), dtype=np.int8)
    su = s.upper()
    for i, ch in enumerate(su):
        arr[i] = NT_TO_INT.get(ch, 4)
    return arr


def pairwise_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Mismatch fraction over positions where BOTH are non-gap."""
    non_gap = (a != 4) & (b != 4)
    if not non_gap.any():
        return float("nan")
    diff = (a != b) & non_gap
    return float(diff.sum() / non_gap.sum())


TOLERANCE = 500
EDGE_BUFFER = 25


def extract_peaks(p: np.ndarray, threshold: float, edge_buffer: int = EDGE_BUFFER
                  ) -> np.ndarray:
    L = len(p)
    p2 = p.copy()
    p2[:edge_buffer] = 0.0
    p2[L - edge_buffer:] = 0.0
    peaks, _ = find_peaks(p2, height=threshold, distance=TOLERANCE)
    return peaks


def load_panel(panel: str) -> dict[str, np.ndarray]:
    fa = Path(f"/home/joshc/Dev/RDP_CNN/data/real_recombinants/{panel}/aligned.fa")
    return {r.id: seq_to_int8(str(r.seq)) for r in SeqIO.parse(fa, "fasta")}


# Sequences in each panel that ARE known recombinants — exclude these from
# negative-control triplet enumeration.
KNOWN_RECOMBINANTS = {
    "sarscov2_full": {"XBB_1_5_recombinant"},
    "sarscov2_orf1ab": {"XBB_1_5_recombinant"},
    "sarscov2_spike": {"XBB_1_5_recombinant"},
    "ebola": set(),
    "zika": set(),
}


def evaluate_panel(panel: str, head, device, amp_dtype,
                   exclude_recombinants: bool = True) -> list[dict]:
    seqs = load_panel(panel)
    ids = list(seqs.keys())
    if exclude_recombinants:
        ids = [i for i in ids if i not in KNOWN_RECOMBINANTS.get(panel, set())]
    print(f"\n[{time.strftime('%H:%M:%S')}] panel={panel}  n_seqs={len(ids)}",
          flush=True)
    n_triplets = len(list(combinations(ids, 3)))
    print(f"  enumerating {n_triplets} unordered triplets", flush=True)

    results = []
    for trio in combinations(ids, 3):
        # One canonical ordering: (id_a, id_b, id_c) treated as (R, P1, P2).
        # We don't permute since each sequence has equal "recombinant probability"
        # under the negative-control assumption.
        R_id, P1_id, P2_id = trio
        R, P1, P2 = seqs[R_id], seqs[P1_id], seqs[P2_id]

        div_r_p1 = pairwise_divergence(R, P1)
        div_r_p2 = pairwise_divergence(R, P2)
        div_p1_p2 = pairwise_divergence(P1, P2)

        feats = raw_features(R, P1, P2).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                logits = head(feats[None]).squeeze(0)
        p = torch.sigmoid(logits.float()).cpu().numpy()

        peaks_by_thr = {}
        for thr in [0.30, 0.50, 0.70, 0.80, 0.90]:
            pk = extract_peaks(p, thr)
            peaks_by_thr[thr] = len(pk)

        results.append({
            "panel": panel,
            "R": R_id, "P1": P1_id, "P2": P2_id,
            "L": len(R),
            "div_R_P1": div_r_p1,
            "div_R_P2": div_r_p2,
            "div_P1_P2": div_p1_p2,
            "div_mean": (div_r_p1 + div_r_p2 + div_p1_p2) / 3,
            "div_max": max(div_r_p1, div_r_p2, div_p1_p2),
            "mean_prob": float(p.mean()),
            "p95_prob": float(np.quantile(p, 0.95)),
            "max_prob": float(p.max()),
            "n_peaks_30": peaks_by_thr[0.30],
            "n_peaks_50": peaks_by_thr[0.50],
            "n_peaks_70": peaks_by_thr[0.70],
            "n_peaks_80": peaks_by_thr[0.80],
            "n_peaks_90": peaks_by_thr[0.90],
        })

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("models_test/m3d_big_snaps/m3d_best.pt"))
    ap.add_argument("--panels", nargs="+",
                    default=["sarscov2_full", "ebola", "zika"])
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--head-blocks", type=int, default=6)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--out", type=Path,
                    default=Path("models_test/m3_multivirus_v2.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16

    head = DilatedHead(in_channels=22, hidden=args.head_hidden,
                       n_blocks=args.head_blocks, dropout=args.head_dropout).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    head.load_state_dict(ck["head_state"])
    head.eval()
    print(f"[{time.strftime('%H:%M:%S')}] device={device}  ckpt={args.ckpt}", flush=True)

    all_results = []
    for panel in args.panels:
        try:
            r = evaluate_panel(panel, head, device, amp_dtype)
        except Exception as exc:
            print(f"  FAIL on {panel}: {exc}")
            continue
        all_results.extend(r)

    # Per-panel summary
    print(f"\n=== Per-panel summary (non-recombinant triplets) ===")
    print(f"{'panel':>16}  {'n':>3}  {'div_mean':>9}  {'mean_prob':>9}  "
          f"{'peaks@0.5':>10}  {'peaks@0.8':>10}")
    for panel in args.panels:
        rows = [r for r in all_results if r["panel"] == panel]
        if not rows:
            continue
        div = np.mean([r["div_mean"] for r in rows])
        mp = np.mean([r["mean_prob"] for r in rows])
        pk50 = np.mean([r["n_peaks_50"] for r in rows])
        pk80 = np.mean([r["n_peaks_80"] for r in rows])
        print(f"{panel:>16}  {len(rows):>3}  {div:>9.3f}  {mp:>9.4f}  "
              f"{pk50:>10.2f}  {pk80:>10.2f}")

    # Divergence vs peaks scatter (binned)
    print(f"\n=== Divergence vs false-peak rate ===")
    print(f"  Across ALL non-recombinant triplets:")
    bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 1.0)]
    for lo, hi in bins:
        rows = [r for r in all_results if lo <= r["div_max"] < hi]
        if not rows:
            print(f"  div_max [{lo:.2f}, {hi:.2f}): no triplets")
            continue
        avg_peaks_50 = np.mean([r["n_peaks_50"] for r in rows])
        avg_peaks_80 = np.mean([r["n_peaks_80"] for r in rows])
        n_with_any_peak_80 = sum(1 for r in rows if r["n_peaks_80"] > 0)
        print(f"  div_max [{lo:.2f}, {hi:.2f}): n={len(rows):>3}  "
              f"avg_peaks@0.5={avg_peaks_50:>5.2f}  avg_peaks@0.8={avg_peaks_80:>5.2f}  "
              f"frac_with_peak@0.8={n_with_any_peak_80/len(rows):.2f}")

    # Worst offenders (highest peak counts on negatives)
    print(f"\n=== Worst false-positive triplets (top 10 by peaks@0.80) ===")
    sorted_rows = sorted(all_results, key=lambda r: -r["n_peaks_80"])
    for r in sorted_rows[:10]:
        print(f"  {r['panel']:>16}  div={r['div_max']:.3f}  peaks@0.8={r['n_peaks_80']:>2}  "
              f"({r['R'][:25]} / {r['P1'][:20]} / {r['P2'][:20]})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_results, indent=2))
    print(f"\n  report → {args.out}  ({len(all_results)} triplets)")


if __name__ == "__main__":
    main()
