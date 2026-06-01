#!/usr/bin/env python3
"""Ensemble of σ-variant runB2 models on LANL CRFs.

The σ-sweep showed each CRF prefers a different σ:
  CRF02_AG, CRF08_BC → σ=10
  CRF07_BC           → σ=5
  CRF12_BF           → σ=20 (original)

A mean-ensemble over the per-position prediction tensors should produce
robust peaks across all four CRFs. Since the channel encoding for runB2
inference is identical for all three σ variants (they only differ in the
labels used during training), we can encode once per CRF and predict
three times.

Outputs F1 per CRF for the ensemble + each individual model, plus an
aggregate comparison.
"""
import argparse
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from Bio import SeqIO
from scipy.signal import find_peaks

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

# Reuse the encoding pipeline from eval_cnn_lanl.py
import importlib.util
spec = importlib.util.spec_from_file_location("ecl", "eval_cnn_lanl.py")
ecl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ecl)

MAX_SEQ_LEN = ecl.MAX_SEQ_LEN
TOLERANCE = 200

DEFAULT_MODELS = {
    'sig20': 'models_test/cnn_breakpoint_runB2_no_rdp_channels_final.keras',
    'sig10': 'models_test/cnn_breakpoint_runB2_sig10_final.keras',
    'sig5':  'models_test/cnn_breakpoint_runB2_sig5_final.keras',
}


def score(pred_bps, true_bps, tol=TOLERANCE):
    used = set(); m = 0
    for tb in true_bps:
        best = None; bd = tol + 1
        for j, p in enumerate(pred_bps):
            if j in used: continue
            d = abs(int(p) - int(tb))
            if d <= tol and d < bd: bd = d; best = j
        if best is not None: used.add(best); m += 1
    tp = m; fp = len(pred_bps) - len(used); fn = len(true_bps) - m
    P = tp/(tp+fp) if (tp+fp) else 0
    R = tp/(tp+fn) if (tp+fn) else 0
    F = 2*P*R/(P+R) if (P+R) else 0
    return tp, fp, fn, P, R, F


def run_rustrdp(fasta, out_csv):
    cmd = [ecl.RUSTRDP, '-i', str(fasta), '-m', 'maxchi,rdp,chimaera,siscan,bootscan',
           '-o', str(out_csv), '--max-pvalue', '1.0']
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--edge-buffer', type=int, default=25)
    ap.add_argument('--min-distance', type=int, default=TOLERANCE)
    ap.add_argument('--out', default='results_cnn_lanl_ensemble.json')
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] Loading {len(DEFAULT_MODELS)} models...")
    models = {}
    for k, p in DEFAULT_MODELS.items():
        print(f"  {k}: {p}")
        models[k] = tf.keras.models.load_model(p, compile=False)
        print(f"    params={models[k].count_params():,}")

    # Truth
    import csv as _csv
    truth = defaultdict(list)
    with open('data/lanl_crf/truth_bps.csv') as f:
        for r in _csv.DictReader(f):
            truth[r['crf']].append(int(r['hxb2_position']))
    truth = {k: sorted(set(v)) for k, v in truth.items()}

    crfs = []
    with open('data/lanl_crf/crf_meta.csv') as f:
        for r in _csv.DictReader(f):
            crfs.append(r)

    TMP = Path('/tmp/rustrdp_ensemble')
    TMP.mkdir(parents=True, exist_ok=True)

    results = {}
    agg_tp = defaultdict(int); agg_fp = defaultdict(int); agg_fn = defaultdict(int)
    methods = list(models.keys()) + ['ensemble_mean', 'ensemble_max', 'ensemble_w10']

    for r in crfs:
        crf = r['crf']
        fa = Path(f"data/lanl_crf/triplets/{crf}.fa")
        if not fa.exists():
            print(f"\n  {crf}: SKIP — no triplet")
            continue
        seqs = list(SeqIO.parse(fa, 'fasta'))
        if len(seqs) != 3:
            print(f"\n  {crf}: SKIP — expected 3 sequences, got {len(seqs)}")
            continue

        # RustRDP for channels 22-32 (same for all models)
        out_csv = TMP / f"{crf}.csv"
        proc = run_rustrdp(fa, out_csv)
        if proc.returncode != 0:
            print(f"\n  {crf}: RustRDP FAIL")
            continue
        by_method = ecl.parse_rustrdp(out_csv)
        X = ecl.encode_lanl_triplet(seqs, by_method, variant='B2')
        X_b = X[None, :, :]

        content_end = min(len(seqs[0].seq), len(seqs[1].seq),
                          len(seqs[2].seq), MAX_SEQ_LEN)

        # Predict with each model
        preds = {}
        for k, m in models.items():
            preds[k] = m.predict(X_b, verbose=0)[0]
        preds['ensemble_mean'] = np.mean([preds[k] for k in models], axis=0)
        preds['ensemble_max']  = np.max([preds[k] for k in models], axis=0)
        # Weighted: σ=10 is the known best; weight it 2× the others.
        weights = {'sig20': 1.0, 'sig10': 2.0, 'sig5': 1.0}
        wsum = sum(weights.values())
        preds['ensemble_w10'] = sum(weights[k] * preds[k] for k in models) / wsum

        true_bps = truth.get(crf, [])
        print(f"\n  {crf} (truth={len(true_bps)} BPs, content_end={content_end})")
        per_crf = {}
        for mname in methods:
            p = preds[mname].copy()
            p[content_end:] = 0
            if args.edge_buffer > 0:
                p[:args.edge_buffer] = 0
                if content_end > args.edge_buffer:
                    p[content_end - args.edge_buffer: content_end] = 0
            peaks, _ = find_peaks(p, height=args.threshold, distance=args.min_distance)
            tp, fp, fn, P, R, F = score(list(peaks), true_bps)
            agg_tp[mname] += tp; agg_fp[mname] += fp; agg_fn[mname] += fn
            per_crf[mname] = dict(tp=tp, fp=fp, fn=fn, precision=P, recall=R, f1=F,
                                  pred_peaks=[int(x) for x in peaks])
            print(f"    {mname:>14}: peaks={len(peaks):>2}  tp={tp}/{len(true_bps)}  "
                  f"fp={fp:>2}  P={P:.3f}  R={R:.3f}  F1={F:.3f}")
        results[crf] = {'truth_bps': true_bps, 'content_end': content_end,
                        'per_method': per_crf}

    # Aggregate
    print(f"\n{'='*78}")
    print(f"  Ensemble eval on 4 LANL CRFs (thr={args.threshold}, EB={args.edge_buffer}, "
          f"min_distance={args.min_distance})")
    print(f"{'='*78}")
    print(f"  {'method':>14} | {'precision':>9} | {'recall':>7} | {'F1':>7} | tp/fp/fn")
    print(f"  {'-'*14}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*15}")
    summary = {}
    for mname in methods:
        P = agg_tp[mname]/(agg_tp[mname]+agg_fp[mname]) if (agg_tp[mname]+agg_fp[mname]) else 0
        R = agg_tp[mname]/(agg_tp[mname]+agg_fn[mname]) if (agg_tp[mname]+agg_fn[mname]) else 0
        F = 2*P*R/(P+R) if (P+R) else 0
        summary[mname] = dict(tp=agg_tp[mname], fp=agg_fp[mname], fn=agg_fn[mname],
                              precision=P, recall=R, f1=F)
        print(f"  {mname:>14} | {P:>9.3f} | {R:>7.3f} | {F:>7.3f} | "
              f"{agg_tp[mname]}/{agg_fp[mname]}/{agg_fn[mname]}")
    print(f"{'='*78}")

    out = {
        'models': DEFAULT_MODELS,
        'threshold': args.threshold,
        'edge_buffer': args.edge_buffer,
        'min_distance': args.min_distance,
        'tolerance': TOLERANCE,
        'per_crf': results,
        'aggregate': summary,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
