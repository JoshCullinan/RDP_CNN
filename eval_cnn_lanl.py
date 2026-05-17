#!/usr/bin/env python3
"""Run runB2 CNN on LANL CRF triplets — the missing half of the sim-to-real comparison.

runB2's 33-channel encoding:
  [0..14]  one-hot triplet  (compute from sequence)
  [15..21] match/info/MaxChi — ZEROED at inference per runB2 convention
  [22..23] Gaussian peaks at RDP-consensus PredBPStart/PredBPEnd
  [24..32] 9 method confidences (-log10(p_min) clipped/normalised), in order:
           RDP, GENECONV, Bootscan, MaxChi, Chimaera, SiScan, PhylPro, LARD, 3Seq

For LANL we have no precomputed RDP method outputs (the .fa.csv files only
exist for SANTA sims). We synthesise the inputs by running RustRDP on the
triplet and using:
  - Per-method min p-value across event rows → channel 24..32 slot
    (RustRDP supplies 5 of the 9 methods; the other 4 stay zero — that's
    out-of-distribution for the model but the channels were zero-init in
    training too via RDP_DROPOUT_P=0.1, so the model should degrade gracefully)
  - Across all events emitted by the RDP method, take the leftmost Start and
    rightmost End as the consensus BP pair → Gaussian peaks at channels 22-23

Then standard runB2 inference, find_peaks at threshold matching the
deployment convention (we use 0.7 — the SUB-best threshold at EB=25), score
predicted peaks against scraped LANL truth BPs at TOLERANCE=200.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from Bio import SeqIO
from scipy.signal import find_peaks

# Memory growth — same hygiene as everywhere else
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"WARN memory_growth({gpu}): {e}", flush=True)

# ---------------- Constants matching train_diagnostic.py / cell-3 ----------
MAX_SEQ_LEN = 32000
N_INPUT_CHANNELS = 33
N_CHANNELS_OH = 5
N_MAXCHI_WINDOWS = 4
RDP_PRED_SIGMA = 50
RDP_METHOD_LOGP_CLIP = 30.0
RDP_METHOD_NAMES = ('RDP', 'GENECONV', 'Bootscan', 'Maxchi', 'Chimaera',
                    'SiSscan', 'PhylPro', 'LARD', '3Seq')
# Maps RustRDP output method names to our channel slot:
RUSTRDP_TO_CHANNEL = {
    'RDP': 0, 'Bootscan': 2, 'MaxChi': 3, 'Chimaera': 4, 'SiScan': 5,
}
B2_MASK_LO, B2_MASK_HI = 15, 22  # zero these at inference too
TOLERANCE = 200
THRESHOLD = 0.7  # SUB-best at EB=25 per results_rustrdp_santa_subset (for runB2)
EDGE_BUFFER = 25  # honest convention

# ---------------- Paths ----------------------------------------------------
RUSTRDP = '/home/joshc/Dev/RustRDP/target/release/rustrdp'
MODEL_PATH = 'models_test/cnn_breakpoint_runB2_no_rdp_channels_final.keras'
META_CSV = Path('data/lanl_crf/crf_meta.csv')
BPS_CSV = Path('data/lanl_crf/truth_bps.csv')
TRIPLET_DIR = Path('data/lanl_crf/triplets')
TMP_DIR = Path('/tmp/rustrdp_lanl_cnn')
OUT_JSON = Path('results_cnn_lanl.json')


# ---------------- Encoding helpers ---------------------------------------
def one_hot(seq):
    nuc_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
    out = np.zeros((MAX_SEQ_LEN, N_CHANNELS_OH), dtype=np.float32)
    s = str(seq).upper()
    for i, nt in enumerate(s[:MAX_SEQ_LEN]):
        out[i, nuc_idx.get(nt, 4)] = 1.0
    return out


def gaussian_peak(center, sigma=RDP_PRED_SIGMA, length=MAX_SEQ_LEN):
    out = np.zeros(length, dtype=np.float32)
    if center is None:
        return out
    try:
        c = int(center)
    except (TypeError, ValueError):
        return out
    if c < 0 or c >= length:
        return out
    pos = np.arange(length, dtype=np.float32)
    return np.exp(-0.5 * ((pos - c) / float(sigma)) ** 2).astype(np.float32)


def run_rustrdp(fasta, out_csv, max_pvalue=1.0):
    cmd = [RUSTRDP, '-i', str(fasta),
           '-m', 'maxchi,rdp,chimaera,siscan,bootscan',
           '-o', str(out_csv), '--max-pvalue', str(max_pvalue)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def parse_rustrdp(out_csv):
    """Return dict method_name -> list of (Start, End, Pvalue)."""
    by_method = defaultdict(list)
    with Path(out_csv).open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Method,'):
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                by_method[parts[0]].append((int(parts[1]), int(parts[2]), float(parts[6])))
            except ValueError:
                continue
    return dict(by_method)


def encode_lanl_triplet(seqs, by_method):
    """seqs: list of 3 SeqRecord (recomb, parent1, parent2) - alignment columns
    in HXB2 coordinates. Returns X with shape (MAX_SEQ_LEN, 33)."""
    X = np.zeros((MAX_SEQ_LEN, N_INPUT_CHANNELS), dtype=np.float32)
    # 0..14: one-hot recomb / parent1 / parent2
    X[:, 0:5]  = one_hot(seqs[0].seq)
    X[:, 5:10] = one_hot(seqs[1].seq)
    X[:, 10:15] = one_hot(seqs[2].seq)
    # 15..21: ZERO (runB2 convention)
    # X[:, 15:22] = 0  (already zero)

    # 22..23: max-pooled Gaussians at EVERY event row's Start (ch22) and End (ch23)
    # across all 5 RustRDP methods. Earlier "leftmost-Start / rightmost-End" of
    # RDP alone produced Gaussians spanning the genome boundaries, which
    # collapsed CNN predictions to the edges (the model trained on tight
    # per-event Gaussians); sum-all-events lifts aggregate F1 from 0.048 → 0.430.
    for _mname, rows in by_method.items():
        for s, e, _p in rows:
            X[:, 22] = np.maximum(X[:, 22], gaussian_peak(s))
            X[:, 23] = np.maximum(X[:, 23], gaussian_peak(e))

    # 24..32: 9 method confidences (-log10(min_p) / 30, clipped)
    for method_name, slot in RUSTRDP_TO_CHANNEL.items():
        rows = by_method.get(method_name, [])
        if not rows:
            continue
        p_min = min(r[2] for r in rows)
        if p_min <= 0 or not math.isfinite(p_min):
            continue
        w = -math.log10(p_min) / RDP_METHOD_LOGP_CLIP
        w = max(0.0, min(1.0, w))
        X[:, 24 + slot] = w
    # Missing slots (GENECONV=1, PhylPro=6, LARD=7, 3Seq=8) stay zero.
    return X


def score(pred_bps, true_bps, tol=TOLERANCE):
    used = set(); n_match = 0
    for tb in true_bps:
        best = None; best_d = tol + 1
        for j, p in enumerate(pred_bps):
            if j in used: continue
            d = abs(int(p) - int(tb))
            if d <= tol and d < best_d:
                best_d = d; best = j
        if best is not None:
            used.add(best); n_match += 1
    tp = n_match
    fp = len(pred_bps) - len(used)
    fn = len(true_bps) - n_match
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return tp, fp, fn, p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MODEL_PATH)
    ap.add_argument('--threshold', type=float, default=THRESHOLD)
    ap.add_argument('--edge-buffer', type=int, default=EDGE_BUFFER)
    ap.add_argument('--out', default=str(OUT_JSON))
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)
    print(f"  params={model.count_params():,}, input_shape={model.input_shape}")

    # Truth
    truth = defaultdict(list)
    with BPS_CSV.open() as f:
        for r in csv.DictReader(f):
            truth[r['crf']].append(int(r['hxb2_position']))
    truth = {k: sorted(set(v)) for k, v in truth.items()}

    crfs = []
    with META_CSV.open() as f:
        for r in csv.DictReader(f):
            crfs.append(r)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    agg_tp = agg_fp = agg_fn = 0

    for r in crfs:
        crf = r['crf']
        fa = TRIPLET_DIR / f"{crf}.fa"
        if not fa.exists():
            print(f"\n  {crf}: SKIP — no triplet at {fa}")
            results[crf] = {'skipped': True}
            continue

        seqs = list(SeqIO.parse(fa, 'fasta'))
        if len(seqs) != 3:
            print(f"\n  {crf}: SKIP — expected 3 sequences, found {len(seqs)}")
            continue

        # Run RustRDP for the channel-22..32 features
        out_csv = TMP_DIR / f"{crf}_rdp.csv"
        proc = run_rustrdp(fa, out_csv)
        if proc.returncode != 0:
            print(f"\n  {crf}: RustRDP FAIL: {proc.stderr[:200]}")
            continue
        by_method = parse_rustrdp(out_csv)

        # Encode + predict
        X = encode_lanl_triplet(seqs, by_method)
        X_batch = X[None, :, :]
        t0 = time.time()
        y_pred = model.predict(X_batch, verbose=0)[0]  # (MAX_SEQ_LEN,)
        pred_time = time.time() - t0

        # Apply edge-suppression (matches deployment convention) at the
        # triplet's content end. Content end = HXB2 length used = first
        # column where all 3 sequences are gap or beyond original seq.
        content_end = min(len(seqs[0].seq), len(seqs[1].seq), len(seqs[2].seq))
        content_end = min(content_end, MAX_SEQ_LEN)
        pred = y_pred.copy()
        pred[content_end:] = 0.0
        if args.edge_buffer > 0:
            pred[:args.edge_buffer] = 0.0
            if content_end > args.edge_buffer:
                pred[content_end - args.edge_buffer: content_end] = 0.0

        peaks, _ = find_peaks(pred, height=args.threshold, distance=TOLERANCE)
        true_bps = truth.get(crf, [])
        tp, fp, fn, p, rcl, f = score(list(peaks), true_bps)
        agg_tp += tp; agg_fp += fp; agg_fn += fn

        results[crf] = {
            'truth_bps': true_bps,
            'pred_peaks': [int(x) for x in peaks],
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': p, 'recall': rcl, 'f1': f,
            'pred_time_s': pred_time,
            'content_end': content_end,
        }

        print(f"\n  {crf} (truth={len(true_bps)} BPs, content_end={content_end})")
        print(f"    runB2 CNN peaks (thr={args.threshold}, EB={args.edge_buffer}): "
              f"{[int(x) for x in peaks]}")
        print(f"    truth: {true_bps}")
        print(f"    tp={tp}  fp={fp}  fn={fn}  P={p:.3f}  R={rcl:.3f}  F1={f:.3f}  "
              f"(predict {pred_time:.1f}s)")

    # Aggregate
    P = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else 0.0
    R = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else 0.0
    F = 2*P*R/(P+R) if (P+R) else 0.0
    print(f"\n{'='*72}")
    print(f"  CNN runB2 on LANL CRFs (threshold={args.threshold}, EB={args.edge_buffer})")
    print(f"{'='*72}")
    print(f"  AGGREGATE: tp={agg_tp}/{agg_tp+agg_fn}, fp={agg_fp}, "
          f"P={P:.3f}  R={R:.3f}  F1={F:.3f}")
    print(f"  For comparison: RustRDP best method (RDP) on same set: F1=0.519")
    print(f"{'='*72}")

    out = {
        'model': str(args.model),
        'threshold': args.threshold,
        'edge_buffer': args.edge_buffer,
        'tolerance': TOLERANCE,
        'per_crf': results,
        'aggregate': {'tp': agg_tp, 'fp': agg_fp, 'fn': agg_fn,
                      'precision': P, 'recall': R, 'f1': F},
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
