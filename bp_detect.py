#!/usr/bin/env python3
"""bp_detect — sequence-only viral recombination breakpoint detector (M3 v4).

Takes a 3-sequence ALIGNED FASTA (recombinant + 2 candidate parents) and reports
per-position breakpoint probabilities, breakpoint calls, and a recombinant-
confidence / out-of-distribution warning. Sequence-only: no RDP/GeneConv/MaxChi
method outputs required.

  $ python bp_detect.py triplet.fa
  $ python bp_detect.py triplet.fa --recomb-id XBB_1_5 --out-prefix xbb --json

INPUT CONTRACT
  - Exactly 3 sequences, mutually ALIGNED (same length, column-wise; gaps '-' ok).
    bp_detect does not align — pre-align with MAFFT/MUSCLE if needed.
  - By default the FIRST record is the recombinant and the other two are the
    candidate parents (the project's current "breakpoints in a known recombinant"
    framing). Override with --recomb-id to name the recombinant record.

WHAT IT DOES
  - BP head (frozen deployed M3 v2 detector): per-position P(breakpoint).
  - Aux head (divergence-anomaly): graded confidence that the triplet is a
    within-distribution recombinant worth trusting. If the triplet looks
    cross-species (max pairwise divergence beyond the trained regime), breakpoint
    calls are SUPPRESSED and a warning is emitted rather than reporting false
    positives. See WRITEUP_M3 §4.5 / m3_gated_detector.py.

OUTPUTS (to stdout always; files if --out-prefix given)
  <prefix>.track.tsv  position<TAB>bp_probability     (per-position track, plottable)
  <prefix>.peaks.tsv  peak_position<TAB>bp_probability (called breakpoints)
  <prefix>.json       full machine-readable result
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_gated_detector import M3GatedDetector, seq_to_int8, DEFAULT_BP_CKPT
from m3_divergence_gate import DEFAULT_DIV_GATE


def load_triplet(fa: Path, recomb_id: str | None):
    """Return (recomb_record, [parent_records]) and validate alignment."""
    recs = list(SeqIO.parse(fa, "fasta"))
    if len(recs) != 3:
        raise SystemExit(f"error: expected exactly 3 sequences in {fa}, "
                         f"found {len(recs)}")
    lengths = {len(r.seq) for r in recs}
    if len(lengths) != 1:
        raise SystemExit(
            f"error: the 3 sequences are not aligned (lengths {sorted(lengths)}). "
            f"Pre-align them (e.g. MAFFT) so all 3 are the same length.")
    if recomb_id is not None:
        match = [r for r in recs if r.id == recomb_id]
        if not match:
            raise SystemExit(f"error: --recomb-id '{recomb_id}' not found. "
                             f"Records: {[r.id for r in recs]}")
        R = match[0]
        parents = [r for r in recs if r.id != recomb_id]
    else:
        R, parents = recs[0], recs[1:]
    return R, parents


def main():
    ap = argparse.ArgumentParser(
        description="Sequence-only viral recombination breakpoint detector (M3 v4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("fasta", type=Path, help="aligned 3-sequence FASTA (R, P1, P2)")
    ap.add_argument("--recomb-id", default=None,
                    help="record id of the recombinant (default: first record)")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_BP_CKPT,
                    help="BP detector checkpoint")
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="probability threshold for calling a breakpoint peak")
    ap.add_argument("--div-thr", type=float, default=DEFAULT_DIV_GATE,
                    help="divergence regime boundary for the OOD gate")
    ap.add_argument("--edge-buffer", type=int, default=200,
                    help="ignore peaks within this many bp of the alignment ends")
    ap.add_argument("--min-peak-distance", type=int, default=500,
                    help="minimum bp separation between called peaks")
    ap.add_argument("--no-gate", action="store_true",
                    help="report raw BP peaks without the OOD divergence gate")
    ap.add_argument("--out-prefix", type=Path, default=None,
                    help="write <prefix>.track.tsv/.peaks.tsv/.json")
    ap.add_argument("--json", action="store_true",
                    help="print the full result as JSON to stdout")
    args = ap.parse_args()

    R_rec, parents = load_triplet(args.fasta, args.recomb_id)
    R = seq_to_int8(str(R_rec.seq))
    P1 = seq_to_int8(str(parents[0].seq))
    P2 = seq_to_int8(str(parents[1].seq))

    det = M3GatedDetector(bp_ckpt=args.ckpt, div_thr=args.div_thr)
    res = det.predict(R, P1, P2, threshold=args.threshold,
                      edge_buffer=args.edge_buffer,
                      peak_distance=args.min_peak_distance)

    peaks = res["raw_peaks"] if args.no_gate else res["peaks"]
    p = res["bp_prob"]
    peak_probs = [round(float(p[i]), 4) for i in peaks]

    # ---- human-readable report ----
    print(f"bp_detect — M3 v4 sequence-only breakpoint detector", flush=True)
    print(f"  input        : {args.fasta}  (3 seqs, aligned length {len(R)})")
    print(f"  recombinant  : {R_rec.id}")
    print(f"  parents      : {parents[0].id}, {parents[1].id}")
    print(f"  max pairwise divergence : {res['div_max']:.3f}")
    verdict = "TRUSTED (within-species regime)" if res["trusted"] \
        else "OUT-OF-DISTRIBUTION"
    print(f"  recombinant-confidence  : {res['aux_confidence']:.3f}  → {verdict}")
    if res["warning"] and not args.no_gate:
        print(f"  WARNING: {res['warning']}")
    if peaks:
        print(f"  breakpoint calls (thr={args.threshold}, n={len(peaks)}):")
        for pos, pr in zip(peaks, peak_probs):
            print(f"      position {pos:>7}   p={pr:.3f}")
    else:
        reason = "gated (out-of-distribution)" if (not res["trusted"] and not args.no_gate) \
            else "none above threshold"
        print(f"  breakpoint calls: none ({reason})")

    # ---- files ----
    if args.out_prefix is not None:
        pref = args.out_prefix
        pref.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{pref}.track.tsv", "w") as f:
            f.write("position\tbp_probability\n")
            for i, v in enumerate(p):
                f.write(f"{i}\t{v:.6f}\n")
        with open(f"{pref}.peaks.tsv", "w") as f:
            f.write("peak_position\tbp_probability\n")
            for pos, pr in zip(peaks, peak_probs):
                f.write(f"{pos}\t{pr:.4f}\n")
        result_json = {
            "input": str(args.fasta), "recombinant": R_rec.id,
            "parents": [parents[0].id, parents[1].id],
            "aligned_length": len(R),
            "div_max": res["div_max"], "aux_confidence": res["aux_confidence"],
            "trusted": res["trusted"], "gated": (not res["trusted"]) and (not args.no_gate),
            "threshold": args.threshold, "div_thr": args.div_thr,
            "peaks": peaks, "peak_probs": peak_probs,
            "warning": res["warning"] if not args.no_gate else None,
        }
        with open(f"{pref}.json", "w") as f:
            json.dump(result_json, f, indent=2)
        print(f"  wrote: {pref}.track.tsv  {pref}.peaks.tsv  {pref}.json")

    if args.json:
        print(json.dumps({
            "recombinant": R_rec.id, "parents": [parents[0].id, parents[1].id],
            "div_max": res["div_max"], "aux_confidence": res["aux_confidence"],
            "trusted": res["trusted"], "peaks": peaks, "peak_probs": peak_probs,
            "warning": res["warning"] if not args.no_gate else None,
        }, indent=2))


if __name__ == "__main__":
    main()
