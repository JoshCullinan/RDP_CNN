"""Multi-virus generalization eval for M3.

The deeper project goal is sequence-only breakpoint detection across virus
families, not just HIV. We have LANL F1 0.509 on HIV. This script tests
whether the same M3 ckpt — trained ONLY on SANTA (HIV-flavored) data —
transfers to:

  - SARS-CoV-2 XBB.1.5 (known cross-lineage recombinant of BA.2.10 + BJ.1)
  - Ebola, Zika, SARS-CoV-2 non-recombinant lineages (negative controls)

For the XBB positive test, the literature documents the recombination
breakpoint at roughly nucleotide 22577 (in Spike, codon S:339), the
junction between BA.2.10's N-terminal half and BJ.1's C-terminal half.
A successful detection picks up at least one peak within ±500 bp of this
position.

For the negative tests, we expect FEW peaks — the parents are distinct
lineages but not in a recombinant relationship with the "recombinant"
slot. High peak counts on negatives means the model is over-detecting
on real virus data (false positive rate concern).

Reports per-test: peak count, peak positions, hit/miss vs known BP,
peak score distribution. Threshold-swept to find best operating point.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_dilated import raw_features
from m3_eval_lanl import load_trained_head, head_forward


NT_TO_INT = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}


def seq_to_int8(s: str) -> np.ndarray:
    arr = np.empty(len(s), dtype=np.int8)
    su = s.upper()
    for i, ch in enumerate(su):
        arr[i] = NT_TO_INT.get(ch, 4)
    return arr


TOLERANCE = 500          # Wider for cross-virus eval; real BP positions
                         # are less precisely known than SANTA truth
EDGE_BUFFER = 25
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60,
              0.70, 0.75, 0.80, 0.85, 0.90]


def extract_peaks(p: np.ndarray, threshold: float, edge_buffer: int = EDGE_BUFFER
                  ) -> np.ndarray:
    L = len(p)
    p2 = p.copy()
    p2[:edge_buffer] = 0.0
    p2[L - edge_buffer:] = 0.0
    peaks, _ = find_peaks(p2, height=threshold, distance=TOLERANCE)
    return peaks


def f1_at(true_bps: list[int], pred_peaks: np.ndarray) -> tuple[float, float, float, int, int, int]:
    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    pairs = []
    for ti, tbp in enumerate(true_bps):
        for pi, ppk in enumerate(pred_peaks):
            d = abs(int(ppk) - int(tbp))
            if d <= TOLERANCE:
                pairs.append((d, ti, pi))
    pairs.sort()
    for d, ti, pi in pairs:
        if ti in matched_true or pi in matched_pred:
            continue
        matched_true.add(ti)
        matched_pred.add(pi)
    tp = len(matched_true)
    fp = len(pred_peaks) - tp
    fn = len(true_bps) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return f1, prec, rec, tp, fp, fn


def load_triplet(panel: str, ids: tuple[str, str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fa = Path(f"/home/joshc/Dev/RDP_CNN/data/real_recombinants/{panel}/aligned.fa")
    recs = {r.id: str(r.seq) for r in SeqIO.parse(fa, "fasta")}
    missing = [i for i in ids if i not in recs]
    if missing:
        raise SystemExit(f"missing in {panel}/aligned.fa: {missing}\n"
                         f"available: {sorted(recs)}")
    return tuple(seq_to_int8(recs[i]) for i in ids)


def evaluate_triplet(name: str, R: np.ndarray, P1: np.ndarray, P2: np.ndarray,
                     known_bps: list[int], is_recombinant: bool,
                     head, head_mode, device, amp_dtype) -> dict:
    feats = raw_features(R, P1, P2).to(device)
    p, aux_prob = head_forward(head, head_mode, feats, device)

    L = len(R)
    # peaks across the threshold sweep
    per_thr = {}
    for thr in THRESHOLDS:
        peaks = extract_peaks(p, thr)
        if known_bps:
            f1, prec, rec, tp, fp, fn = f1_at(known_bps, peaks)
            per_thr[thr] = {"n_peaks": len(peaks), "peaks": peaks.tolist(),
                            "f1": f1, "prec": prec, "rec": rec,
                            "tp": tp, "fp": fp, "fn": fn}
        else:
            per_thr[thr] = {"n_peaks": len(peaks), "peaks": peaks.tolist()}

    # summary stats: average prob, max prob, score distribution
    out = {
        "name": name,
        "is_recombinant": is_recombinant,
        "known_bps": known_bps,
        "L": L,
        "mean_prob": float(p.mean()),
        "median_prob": float(np.median(p)),
        "max_prob": float(p.max()),
        "p95_prob": float(np.quantile(p, 0.95)),
        "aux_prob": aux_prob,
        "per_thr": per_thr,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("models_test/m3d_big_snaps/m3d_best.pt"))
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--head-blocks", type=int, default=6)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--head-mode", choices=["auto", "single", "multi"], default="auto")
    ap.add_argument("--gate-thr", type=float, default=0.5,
                    help="aux_prob gate: a triplet below this is judged "
                         "non-recombinant (multi head)")
    ap.add_argument("--out", type=Path, default=Path("models_test/m3_multivirus.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16

    head, head_mode, ck = load_trained_head(
        args.ckpt, device, args.head_hidden, args.head_blocks, args.head_dropout,
        args.head_mode)
    print(f"[{time.strftime('%H:%M:%S')}] device={device}  head_mode={head_mode}  "
          f"gate_thr={args.gate_thr}", flush=True)
    print(f"  ckpt: {args.ckpt}", flush=True)
    print(f"  trained epoch={ck.get('epoch')}  gs={ck.get('global_step', 0):,}", flush=True)

    # --- Test plan -------------------------------------------------------
    # Positive: SARS-CoV-2 XBB.1.5 = BA.2.10 + BJ.1 recombinant.
    #   Documented BP near nt 22577 (Tamura et al. 2023 Nat Comm).
    # Negatives: triplets of distinct-lineage sequences with NO
    #   recombination relationship.
    tests = [
        {
            "name": "SARS-CoV-2 XBB.1.5 (POSITIVE)",
            "panel": "sarscov2_full",
            "ids": ("XBB_1_5_recombinant", "BA_2_10_XBB_parent", "BJ_1_XBB_parent"),
            "known_bps": [22577],
            "is_recombinant": True,
        },
        {
            "name": "SARS-CoV-2 distinct lineages (NEGATIVE)",
            "panel": "sarscov2_full",
            "ids": ("Wuhan_Hu_1_ref", "Alpha_B_1_1_7", "Delta_B_1_617_2"),
            "known_bps": [],
            "is_recombinant": False,
        },
        {
            "name": "Ebola Zaire (NEGATIVE)",
            "panel": "ebola",
            "ids": ("Zaire_Mayinga_76", "Sudan_Boniface_76", "Bundibugyo_Uganda_2007"),
            "known_bps": [],
            "is_recombinant": False,
        },
        {
            "name": "Zika Asian (NEGATIVE)",
            "panel": "zika",
            "ids": ("Asian_PRVABC59_PR_2015", "African_MR766", "Asian_FrPolynesia_2013"),
            "known_bps": [],
            "is_recombinant": False,
        },
    ]

    results = []
    for t in tests:
        print(f"\n=== {t['name']} ===", flush=True)
        try:
            R, P1, P2 = load_triplet(t["panel"], t["ids"])
        except SystemExit as exc:
            print(f"  FAIL to load: {exc}")
            continue
        print(f"  ids: {t['ids']}")
        print(f"  seq_len: {len(R)}")
        r = evaluate_triplet(t["name"], R, P1, P2, t["known_bps"],
                              t["is_recombinant"], head, head_mode, device, amp_dtype)
        results.append(r)

        # Report
        print(f"  prob stats: mean={r['mean_prob']:.4f}  median={r['median_prob']:.4f}  "
              f"p95={r['p95_prob']:.4f}  max={r['max_prob']:.4f}")
        if r["aux_prob"] is not None:
            gated = r["aux_prob"] < args.gate_thr
            want = "want HIGH" if t["is_recombinant"] else "want LOW"
            print(f"  aux_prob (recombinant gate): {r['aux_prob']:.3f}  "
                  f"[{want}; gate@{args.gate_thr} → "
                  f"{'GATED (peaks suppressed)' if gated else 'pass'}]")
        if t["is_recombinant"]:
            best_thr = max(r["per_thr"].items(), key=lambda kv: kv[1]["f1"])
            thr, b = best_thr
            print(f"  KNOWN BP: {t['known_bps']}  (tolerance ±{TOLERANCE})")
            print(f"  best F1 {b['f1']:.3f} @ thr={thr:.2f}  "
                  f"P {b['prec']:.3f}  R {b['rec']:.3f}  "
                  f"(tp/fp/fn = {b['tp']}/{b['fp']}/{b['fn']})")
            print(f"  peaks at best thr: {b['peaks'][:20]}")
            # Distance from known BP for nearest peak
            if b["peaks"]:
                dists = [abs(p - t["known_bps"][0]) for p in b["peaks"]]
                nearest_p = b["peaks"][int(np.argmin(dists))]
                print(f"  nearest peak to known BP {t['known_bps'][0]}: "
                      f"{nearest_p} (Δ={min(dists)} bp)")
        else:
            print(f"  Peak counts vs threshold:")
            for thr in [0.10, 0.30, 0.50, 0.70, 0.80, 0.90]:
                b = r["per_thr"][thr]
                print(f"    thr={thr:.2f}: {b['n_peaks']:>2} peaks")
            # Show peaks at thr=0.5 (deployment-style threshold)
            b50 = r["per_thr"][0.50]
            print(f"  peaks at thr=0.50: {b50['peaks']}")

    # --- Summary --------------------------------------------------------
    print(f"\n=== Multi-virus summary ===")
    for r in results:
        ax = (f"  aux={r['aux_prob']:.3f}" if r["aux_prob"] is not None else "")
        if r["is_recombinant"]:
            best_thr = max(r["per_thr"].items(), key=lambda kv: kv[1]["f1"])
            thr, b = best_thr
            verdict = "✓ HIT" if b["tp"] >= 1 else "✗ MISS"
            print(f"  {r['name']}: {verdict} — best F1 {b['f1']:.3f} @ thr={thr:.2f}, "
                  f"{b['tp']}/{len(r['known_bps'])} known BPs hit, {b['fp']} false peaks{ax}")
        else:
            # For negatives, report number of high-confidence peaks
            n50 = r["per_thr"][0.50]["n_peaks"]
            n80 = r["per_thr"][0.80]["n_peaks"]
            print(f"  {r['name']}: {n50} peaks @ thr=0.50, {n80} @ thr=0.80  "
                  f"(max p={r['max_prob']:.3f}){ax}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n  report → {args.out}")


if __name__ == "__main__":
    main()
