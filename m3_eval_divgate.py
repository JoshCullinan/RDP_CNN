"""M3 v4 FINAL validation: deployed v2 BP detector + unsupervised divergence gate.

This is the v4 deliverable. The learned aux gate failed (memory
project_m3_v4_multihead); v4 instead wraps the unchanged, deployed v2 detector
(models_test/m3d_big_snaps/m3d_best.pt, LANL F1 0.509) with the divergence-anomaly
gate from m3_divergence_gate.py. A triplet with div_max above the threshold is
flagged cross-species and its peaks are suppressed.

Prints the four-criteria scorecard:
  1. LANL F1 (gated)            >= 0.49   — real HIV recombinants not regressed
  2. Ebola peaks@0.8 (gated)    <= 1.5    — cross-species false positives killed
  3. SARS-CoV-2 XBB kept + localised <=500 bp of 22577
  4. gate AUROC (LANL vs Ebola) >= 0.85   — separation quality of the gate
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_dilated import raw_features
from m3_eval_lanl import (load_trained_head, head_forward, seq_to_int8,
                          extract_peaks, event_f1, EDGE_BUFFER, TOLERANCE)
from m3_divergence_gate import divergence_gate, triplet_div_max, DEFAULT_DIV_GATE

PANEL_DIR = Path("/home/joshc/Dev/RDP_CNN/data/real_recombinants")
# wider edge buffer for whole-genome panels (5'/3' UTR artifacts), matching
# m3_eval_multivirus_v2.
PANEL_EDGE_BUFFER = 200
PANEL_TOLERANCE = 500
KNOWN_RECOMB = {"sarscov2_full": {"XBB_1_5_recombinant"}}


def bp_prob(head, head_mode, R, P1, P2, device):
    feats = raw_features(R, P1, P2).to(device)
    p, _ = head_forward(head, head_mode, feats, device)
    return p


def panel_peaks(p, thr, edge=PANEL_EDGE_BUFFER):
    L = len(p); p2 = p.copy(); p2[:edge] = 0.0; p2[L - edge:] = 0.0
    from scipy.signal import find_peaks
    pk, _ = find_peaks(p2, height=thr, distance=PANEL_TOLERANCE)
    return pk


def auroc(pos, neg):  # P(pos ranks above neg); here "recombinant-like" = LOW div
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    if not len(p) or not len(n):
        return float("nan")
    c = p[:, None] - n[None, :]
    return (float((c > 0).sum()) + 0.5 * float((c == 0).sum())) / (len(p) * len(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("models_test/m3d_big_snaps/m3d_best.pt"))
    ap.add_argument("--div-thr", type=float, default=DEFAULT_DIV_GATE)
    ap.add_argument("--triplet-dir", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/triplets"))
    ap.add_argument("--truth-csv", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/truth_bps.csv"))
    ap.add_argument("--out", type=Path, default=Path("models_test/m3_v4_divgate.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head, head_mode, ck = load_trained_head(args.ckpt, device, 128, 6, 0.1, "auto")
    print(f"[{time.strftime('%H:%M:%S')}] device={device}  BP ckpt={args.ckpt} "
          f"(head_mode={head_mode})  div_gate_thr={args.div_thr}", flush=True)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60,
                  0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    # ---------- Criterion 1: LANL F1 (gated) ----------
    truth = defaultdict(list)
    with args.truth_csv.open() as f:
        for r in csv.DictReader(f):
            truth[r["crf"]].append(int(r["hxb2_position"]))
    truth = {k: sorted(set(v)) for k, v in truth.items()}

    lanl_div = {}
    lanl_perthr = {}   # crf -> {thr -> (tp,fp,fn)}
    for fa in sorted(args.triplet_dir.glob("*.fa")):
        crf = fa.stem
        if crf not in truth:
            continue
        seqs = list(SeqIO.parse(fa, "fasta"))
        if len(seqs) != 3:
            continue
        R, P1, P2 = (seq_to_int8(str(s.seq)) for s in seqs)
        keep, dm = divergence_gate(R, P1, P2, args.div_thr)
        lanl_div[crf] = dm
        L = len(R)
        content_end = L
        for arr in (R, P1, P2):
            nz = np.where(arr != 4)[0]
            if len(nz):
                content_end = min(content_end, int(nz[-1]) + 1)
        p = bp_prob(head, head_mode, R, P1, P2, device)
        lanl_perthr[crf] = {}
        for thr in thresholds:
            peaks = extract_peaks(p, thr, EDGE_BUFFER, content_end) if keep \
                else np.array([], dtype=int)            # gated → no peaks
            lanl_perthr[crf][thr] = event_f1(truth[crf], peaks)
    # aggregate at best global threshold
    best = {"f1": -1}
    for thr in thresholds:
        tp = fp = fn = 0
        for crf in lanl_perthr:
            a, b, c = lanl_perthr[crf][thr]
            tp += a; fp += b; fn += c
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        if f1 > best["f1"]:
            best = {"f1": f1, "thr": thr, "prec": prec, "rec": rec,
                    "tp": tp, "fp": fp, "fn": fn}
    c1 = best["f1"]
    print(f"\n[C1] LANL F1 (gated) = {c1:.3f} @ thr={best['thr']:.2f} "
          f"(P {best['prec']:.3f} R {best['rec']:.3f})  "
          f"div_max per CRF: " + ", ".join(f"{k}={v:.3f}" for k, v in lanl_div.items()))

    # ---------- Criteria 2 + 4: enumerated panels ----------
    panel_stats = {}
    ebola_div = []
    for panel in ["ebola", "zika", "sarscov2_full"]:
        fa = PANEL_DIR / panel / "aligned.fa"
        seqs = {r.id: seq_to_int8(str(r.seq)) for r in SeqIO.parse(fa, "fasta")}
        ids = [i for i in seqs if i not in KNOWN_RECOMB.get(panel, set())]
        gated_pk80, raw_pk80, divs = [], [], []
        for trio in combinations(ids, 3):
            R, P1, P2 = (seqs[i] for i in trio)
            keep, dm = divergence_gate(R, P1, P2, args.div_thr)
            divs.append(dm)
            p = bp_prob(head, head_mode, R, P1, P2, device)
            n80 = len(panel_peaks(p, 0.80))
            raw_pk80.append(n80)
            gated_pk80.append(0 if not keep else n80)
        panel_stats[panel] = {
            "n": len(gated_pk80),
            "div_mean": float(np.mean(divs)),
            "raw_peaks80": float(np.mean(raw_pk80)),
            "gated_peaks80": float(np.mean(gated_pk80)),
            "frac_gated": float(np.mean([d > args.div_thr for d in divs])),
        }
        if panel == "ebola":
            ebola_div = divs
    c2 = panel_stats["ebola"]["gated_peaks80"]
    print(f"\n[C2] Ebola peaks@0.8: raw {panel_stats['ebola']['raw_peaks80']:.2f} "
          f"→ gated {c2:.2f}  ({panel_stats['ebola']['frac_gated']*100:.0f}% of "
          f"Ebola triplets gated)")
    for pn, s in panel_stats.items():
        print(f"     {pn:>14}: div {s['div_mean']:.3f}  raw_pk80 {s['raw_peaks80']:.2f}"
              f"  gated_pk80 {s['gated_peaks80']:.2f}  ({s['frac_gated']*100:.0f}% gated)")

    # ---------- Criterion 3: XBB positive ----------
    sc = {r.id: seq_to_int8(str(r.seq))
          for r in SeqIO.parse(PANEL_DIR / "sarscov2_full" / "aligned.fa", "fasta")}
    R, P1, P2 = (sc["XBB_1_5_recombinant"], sc["BA_2_10_XBB_parent"], sc["BJ_1_XBB_parent"])
    keep_xbb, dm_xbb = divergence_gate(R, P1, P2, args.div_thr)
    p = bp_prob(head, head_mode, R, P1, P2, device)
    xbb_peaks = panel_peaks(p, 0.50)
    xbb_delta = min((abs(int(pk) - 22577) for pk in xbb_peaks), default=10**9)
    c3 = keep_xbb and xbb_delta <= 500
    print(f"\n[C3] XBB: div_max={dm_xbb:.3f} → {'KEPT' if keep_xbb else 'GATED'};  "
          f"nearest peak Δ={xbb_delta} bp from 22577")

    # ---------- Criterion 4: gate AUROC (LANL low-div vs Ebola high-div) ----------
    # "recombinant-like" = LOW div. AUROC that Ebola div > LANL div (i.e. gate
    # ranks cross-species above recombinants).
    lanl_divs = list(lanl_div.values())
    c4 = auroc(ebola_div, lanl_divs)   # P(Ebola_div > LANL_div)
    print(f"\n[C4] gate AUROC (Ebola-div > LANL-div) = {c4:.3f}")

    # ---------- Scorecard ----------
    rows = [
        ("1. LANL F1 (gated) ≥ 0.49", c1, c1 >= 0.49, f"{c1:.3f}"),
        ("2. Ebola peaks@0.8 (gated) ≤ 1.5", c2, c2 <= 1.5, f"{c2:.2f}"),
        ("3. XBB kept & Δ≤500 bp", None, bool(c3), f"kept={keep_xbb} Δ={xbb_delta}"),
        ("4. gate AUROC ≥ 0.85", c4, c4 >= 0.85, f"{c4:.3f}"),
    ]
    print(f"\n{'='*56}\n M3 v4 (v2 detector + divergence gate @ {args.div_thr}) "
          f"SCORECARD\n{'='*56}")
    allpass = True
    for name, _, ok, val in rows:
        allpass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<36} {val}")
    print(f"{'='*56}\n  {'ALL CRITERIA PASS' if allpass else 'SOME CRITERIA FAIL'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "div_thr": args.div_thr, "bp_ckpt": str(args.ckpt),
        "c1_lanl_f1_gated": c1, "c1_detail": best, "lanl_div_max": lanl_div,
        "c2_ebola_gated_peaks80": c2, "panel_stats": panel_stats,
        "c3_xbb": {"kept": keep_xbb, "div_max": dm_xbb, "delta_bp": xbb_delta},
        "c4_gate_auroc": c4, "all_pass": bool(allpass),
    }, indent=2, default=float))
    print(f"  report → {args.out}")


if __name__ == "__main__":
    main()
