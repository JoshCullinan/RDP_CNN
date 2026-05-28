"""Evaluate M3 dilated head on real-HIV LANL CRF triplets.

The deployment benchmark. Loads our trained M3 ckpt (raw 22-channel
dilated head) and runs find_peaks F1 evaluation against curated LANL
ground-truth breakpoints across the 4 available CRFs:

  CRF02_AG, CRF07_BC, CRF08_BC, CRF12_BF.

Baselines on this same evaluation:
  Classical RDP standalone        : F1 0.519
  Legacy CNN runB2 (with RDP-input fusion): F1 ~0.430 (some reports 0.533 w/ sig10)
  Legacy CNN runOH (sequence-only): F1 0.000 (collapse)

Our M3 is sequence-only (no RustRDP inputs), so the *meaningful* comparison
is against runOH=0.000 — anything above zero is improvement over the legacy
sequence-only architecture.

LANL alignments are pre-aligned in HXB2 coordinates, so:
  - The 3 sequences in each .fa are all the same length, aligned column-wise.
  - truth_bps.csv `hxb2_position` is the alignment column index.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_dilated import DilatedHead, M3MultiHead, build_head, raw_features


def load_trained_head(ckpt: Path, device: str, head_hidden: int,
                      head_blocks: int, head_dropout: float,
                      head_mode: str = "auto"):
    """Load a trained BP head, auto-detecting single (v2) vs multi (v4) from
    the checkpoint's saved cfg. Returns (head, head_mode)."""
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    if head_mode == "auto":
        head_mode = ck.get("cfg", {}).get("head_mode", "single")
    head = build_head(head_mode, 22, head_hidden, head_blocks, head_dropout).to(device)
    head.load_state_dict(ck["head_state"])
    head.eval()
    return head, head_mode, ck


def head_forward(head, head_mode: str, feats: torch.Tensor, device: str):
    """Run the head; returns (bp_prob (L,) np.ndarray, aux_prob float|None)."""
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(device == "cuda")):
            out = head(feats[None])
    if head_mode == "multi":
        bp_logits, aux_logit = out
        aux_prob = float(torch.sigmoid(aux_logit.float())[0])
        p = torch.sigmoid(bp_logits[0].float()).cpu().numpy()
        return p, aux_prob
    p = torch.sigmoid(out.squeeze(0).float()).cpu().numpy()
    return p, None


# v2 cache encoding: {A:0, T:1, G:2, C:3, gap:4}
NT_TO_INT = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}


def seq_to_int8(s: str) -> np.ndarray:
    arr = np.empty(len(s), dtype=np.int8)
    su = s.upper()
    for i, ch in enumerate(su):
        arr[i] = NT_TO_INT.get(ch, 4)
    return arr


TOLERANCE = 200
EDGE_BUFFER = 25


def extract_peaks(p: np.ndarray, threshold: float, edge_buffer: int = EDGE_BUFFER,
                  content_end: int | None = None) -> np.ndarray:
    L = len(p)
    p2 = p.copy()
    p2[:edge_buffer] = 0.0
    if content_end is not None:
        p2[content_end - edge_buffer:content_end] = 0.0
        p2[content_end:] = 0.0
    else:
        p2[L - edge_buffer:] = 0.0
    peaks, _ = find_peaks(p2, height=threshold, distance=TOLERANCE)
    return peaks


def event_f1(true_bps: list[int], pred_peaks: np.ndarray,
              tolerance: int = TOLERANCE) -> tuple[int, int, int]:
    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    pairs = []
    for ti, tbp in enumerate(true_bps):
        for pi, ppk in enumerate(pred_peaks):
            d = abs(int(ppk) - int(tbp))
            if d <= tolerance:
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
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("models_test/m3d_legacy_snaps/m3d_best.pt"))
    ap.add_argument("--triplet-dir", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/triplets"))
    ap.add_argument("--truth-csv", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/truth_bps.csv"))
    ap.add_argument("--meta-csv", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/crf_meta.csv"))
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--head-blocks", type=int, default=6)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--head-mode", choices=["auto", "single", "multi"], default="auto")
    ap.add_argument("--out", type=Path, default=Path("models_test/m3_eval_lanl.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load truth
    truth: dict[str, list[int]] = defaultdict(list)
    with args.truth_csv.open() as f:
        for r in csv.DictReader(f):
            truth[r["crf"]].append(int(r["hxb2_position"]))
    truth = {k: sorted(set(v)) for k, v in truth.items()}

    # Load model (auto-detects single v2 vs multi v4 from the ckpt cfg)
    head, head_mode, ck = load_trained_head(
        args.ckpt, device, args.head_hidden, args.head_blocks, args.head_dropout,
        args.head_mode)
    print(f"[{time.strftime('%H:%M:%S')}] device={device}  head_mode={head_mode}",
          flush=True)
    print(f"  ckpt: {args.ckpt}  epoch={ck.get('epoch')}  "
          f"gs={ck.get('global_step', 0):,}", flush=True)
    print(f"  head params: {sum(p.numel() for p in head.parameters()):,}", flush=True)

    # Run on each CRF triplet
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60,
                  0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    per_crf = {}
    print(f"\n=== Per-CRF evaluation ===", flush=True)
    for fa in sorted(args.triplet_dir.glob("*.fa")):
        crf = fa.stem
        if crf not in truth:
            print(f"  {crf}: SKIP — no truth BPs")
            continue
        seqs = list(SeqIO.parse(fa, "fasta"))
        if len(seqs) != 3:
            print(f"  {crf}: SKIP — expected 3 sequences, got {len(seqs)}")
            continue
        R = seq_to_int8(str(seqs[0].seq))
        P1 = seq_to_int8(str(seqs[1].seq))
        P2 = seq_to_int8(str(seqs[2].seq))
        L = len(R)
        # content_end = last non-gap position across the three sequences
        content_end = L
        for arr in (R, P1, P2):
            non_gap = np.where(arr != 4)[0]
            if len(non_gap):
                content_end = min(content_end, int(non_gap[-1]) + 1)

        feats = raw_features(R, P1, P2).to(device)
        p, aux_prob = head_forward(head, head_mode, feats, device)

        tbps = truth[crf]
        best_per_thr = []
        for thr in thresholds:
            peaks = extract_peaks(p, thr, edge_buffer=EDGE_BUFFER,
                                  content_end=content_end)
            tp, fp, fn = event_f1(tbps, peaks)
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            best_per_thr.append({"thr": thr, "f1": f1, "precision": prec,
                                  "recall": rec, "tp": tp, "fp": fp, "fn": fn,
                                  "peaks": [int(x) for x in peaks]})
        best_per_thr.sort(key=lambda d: -d["f1"])
        best = best_per_thr[0]
        per_crf[crf] = {
            "L": L, "content_end": content_end,
            "truth_bps": tbps,
            "best": best, "all_thresholds": best_per_thr,
            "aux_prob": aux_prob,
        }
        print(f"  {crf} (L={L} content_end={content_end} n_true_bps={len(tbps)})")
        aux_str = f"  aux_prob={aux_prob:.3f}" if aux_prob is not None else ""
        print(f"    BEST F1 {best['f1']:.3f} @ thr={best['thr']:.2f}  "
              f"P {best['precision']:.3f}  R {best['recall']:.3f}  "
              f"(tp/fp/fn = {best['tp']}/{best['fp']}/{best['fn']}){aux_str}")
        print(f"    truth: {tbps}")
        print(f"    peaks: {best['peaks'][:20]}")

    # Aggregate across CRFs at each shared threshold
    print(f"\n=== Aggregate (single global threshold) ===", flush=True)
    agg = []
    for thr in thresholds:
        total_tp = total_fp = total_fn = 0
        for crf, r in per_crf.items():
            for t in r["all_thresholds"]:
                if t["thr"] == thr:
                    total_tp += t["tp"]; total_fp += t["fp"]; total_fn += t["fn"]
                    break
        prec = total_tp / max(1, total_tp + total_fp)
        rec = total_tp / max(1, total_tp + total_fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        agg.append({"thr": thr, "f1": f1, "precision": prec, "recall": rec,
                     "tp": total_tp, "fp": total_fp, "fn": total_fn})
    agg.sort(key=lambda d: -d["f1"])
    best_agg = agg[0]
    print(f"  Best aggregate F1 {best_agg['f1']:.3f} @ thr={best_agg['thr']:.2f}  "
          f"P {best_agg['precision']:.3f}  R {best_agg['recall']:.3f}  "
          f"(tp/fp/fn = {best_agg['tp']}/{best_agg['fp']}/{best_agg['fn']})")

    print(f"\n=== Comparison ===")
    print(f"  Classical RDP standalone         : F1 0.519")
    print(f"  Legacy CNN runB2 (with RDP fusion): F1 ~0.430-0.533")
    print(f"  Legacy CNN runOH (sequence-only)  : F1 0.000")
    print(f"  Our M3 (sequence-only)            : F1 {best_agg['f1']:.3f}")

    if head_mode == "multi":
        aux_vals = [r["aux_prob"] for r in per_crf.values() if r["aux_prob"] is not None]
        if aux_vals:
            print(f"\n=== Aux gate (LANL = TRUE recombinants → want HIGH) ===")
            print(f"  mean aux_prob over CRFs: {np.mean(aux_vals):.3f}  "
                  f"min {min(aux_vals):.3f}  max {max(aux_vals):.3f}")
            print(f"  per-CRF: " + ", ".join(
                f"{crf}={r['aux_prob']:.3f}" for crf, r in per_crf.items()
                if r["aux_prob"] is not None))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "head_mode": head_mode,
        "per_crf": {crf: {"best": r["best"], "L": r["L"],
                           "content_end": r["content_end"],
                           "truth_bps": r["truth_bps"],
                           "aux_prob": r["aux_prob"]}
                     for crf, r in per_crf.items()},
        "aggregate": agg[:5],
    }, indent=2))
    print(f"  report → {args.out}")


if __name__ == "__main__":
    main()
