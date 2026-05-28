"""Aggregate M3 peak positions across all SARS-CoV-2 non-recombinant triplets.

Investigates whether the ~3 peaks/triplet at thr=0.80 cluster near known
SARS-CoV-2 recombination hotspots (Spike gene, S1/S2 boundary), which
would mean they're not really false positives.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_dilated import DilatedHead, raw_features

NT_TO_INT = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
TOLERANCE = 500
EDGE_BUFFER = 25


def seq_to_int8(s: str) -> np.ndarray:
    arr = np.empty(len(s), dtype=np.int8)
    su = s.upper()
    for i, ch in enumerate(su):
        arr[i] = NT_TO_INT.get(ch, 4)
    return arr


def extract_peaks(p: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    L = len(p)
    p2 = p.copy()
    p2[:EDGE_BUFFER] = 0.0
    p2[L - EDGE_BUFFER:] = 0.0
    peaks, props = find_peaks(p2, height=thr, distance=TOLERANCE)
    heights = props["peak_heights"] if peaks.size else np.array([])
    return peaks, heights


# SARS-CoV-2 known landmarks
SARSCOV2_LANDMARKS = {
    "5'UTR end":         265,
    "ORF1a start":       266,
    "ORF1a/b junction":  13468,
    "ORF1ab end":        21555,
    "Spike start":       21563,
    "Spike NTD ~":       22000,
    "Spike RBD start":   22517,
    "XBB BP (BA.2/BJ.1)": 22577,
    "Spike S1/S2":       23618,
    "Spike end":         25384,
    "ORF3a":             25393,
    "M (envelope mem)":  26523,
    "N (nucleocapsid)":  28274,
    "3'UTR start":       29675,
}


def nearest_landmark(pos: int) -> tuple[str, int]:
    nearest = min(SARSCOV2_LANDMARKS.items(), key=lambda kv: abs(kv[1] - pos))
    return nearest[0], abs(nearest[1] - pos)


def main():
    ckpt_path = Path("models_test/m3d_big_snaps/m3d_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = DilatedHead(in_channels=22, hidden=128, n_blocks=6, dropout=0.1).to(device)
    head.load_state_dict(torch.load(ckpt_path, map_location=device,
                                    weights_only=False)["head_state"])
    head.eval()
    print(f"[{time.strftime('%H:%M:%S')}] device={device} ckpt={ckpt_path}")

    fa = Path("/home/joshc/Dev/RDP_CNN/data/real_recombinants/sarscov2_full/aligned.fa")
    seqs = {r.id: seq_to_int8(str(r.seq)) for r in SeqIO.parse(fa, "fasta")}
    ids = [i for i in seqs if i != "XBB_1_5_recombinant"]
    print(f"  panel: {len(ids)} sequences, {len(list(combinations(ids, 3)))} triplets")

    all_peaks: list[dict] = []
    pos_counter: Counter[int] = Counter()
    bin_counter: Counter[int] = Counter()      # 500 bp bins
    BIN_W = 500

    for trio in combinations(ids, 3):
        R, P1, P2 = (seqs[trio[0]], seqs[trio[1]], seqs[trio[2]])
        feats = raw_features(R, P1, P2).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device == "cuda")):
                logits = head(feats[None]).squeeze(0)
        p = torch.sigmoid(logits.float()).cpu().numpy()
        peaks, heights = extract_peaks(p, thr=0.80)
        for pk, h in zip(peaks, heights):
            all_peaks.append({"trio": trio, "pos": int(pk), "height": float(h)})
            pos_counter[int(pk)] += 1
            bin_counter[int(pk) // BIN_W] += 1

    print(f"\n  total peaks at thr=0.80: {len(all_peaks)}")
    print(f"  unique positions: {len(pos_counter)}")
    print(f"\n=== Peak position hotspots (binned 500bp) ===")
    print(f"{'bin start':>10}  {'bin end':>8}  {'count':>5}  nearest landmark")
    for bin_idx, count in sorted(bin_counter.items(), key=lambda kv: -kv[1])[:25]:
        bin_start = bin_idx * BIN_W
        bin_end = bin_start + BIN_W
        lm_name, lm_dist = nearest_landmark(bin_start + BIN_W // 2)
        print(f"  {bin_start:>10}  {bin_end:>6}  {count:>5}  "
              f"{lm_name} (Δ={lm_dist})")

    print(f"\n=== Top 20 most-recurring exact peak positions ===")
    for pos, count in pos_counter.most_common(20):
        lm_name, lm_dist = nearest_landmark(pos)
        print(f"  pos {pos:>6}: hit in {count:>2} triplets  → near {lm_name} (Δ={lm_dist})")

    print(f"\n=== SARS-CoV-2 Spike region (21563–25384) coverage ===")
    spike_peaks = [p for p in all_peaks if 21563 <= p["pos"] <= 25384]
    print(f"  peaks in Spike: {len(spike_peaks)} / {len(all_peaks)} "
          f"({100*len(spike_peaks)/max(1,len(all_peaks)):.1f}%)")
    spike_frac_genome = (25384 - 21563) / 29903
    print(f"  Spike is {100*spike_frac_genome:.1f}% of the 29.9 kb genome")
    print(f"  enrichment ratio: "
          f"{(len(spike_peaks)/max(1,len(all_peaks))) / spike_frac_genome:.2f}× "
          f"(>1.0 = peaks concentrate in Spike)")

    print(f"\n=== Per-triplet peak detail (first 10 triplets) ===")
    last_trio = None
    triplet_idx = 0
    for r in all_peaks:
        if r["trio"] != last_trio:
            triplet_idx += 1
            if triplet_idx > 10:
                break
            print(f"\n  triplet #{triplet_idx}: {r['trio'][0][:22]} / "
                  f"{r['trio'][1][:22]} / {r['trio'][2][:22]}")
            last_trio = r["trio"]
        lm, dist = nearest_landmark(r["pos"])
        print(f"    pos {r['pos']:>6} (h={r['height']:.3f})  near {lm} Δ={dist}")

    out = Path("models_test/m3_sars_peaks.json")
    out.write_text(json.dumps({
        "all_peaks": all_peaks,
        "spike_enrichment": {
            "spike_peaks": len(spike_peaks),
            "total_peaks": len(all_peaks),
            "spike_frac_genome": spike_frac_genome,
            "enrichment_ratio": (len(spike_peaks)/max(1, len(all_peaks))) /
                                  spike_frac_genome,
        },
        "landmarks": SARSCOV2_LANDMARKS,
    }, indent=2, default=lambda x: list(x) if isinstance(x, tuple) else x))
    print(f"\n  report → {out}")


if __name__ == "__main__":
    main()
