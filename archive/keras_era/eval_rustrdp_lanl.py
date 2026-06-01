#!/usr/bin/env python3
"""Real-HIV CRF evaluation: RustRDP on LANL-aligned triplets vs scraped LANL
breakpoint truth.

Per CRF in data/lanl_crf/crf_meta.csv:
- Looks for data/lanl_crf/triplets/<CRF>.fa (a 3-sequence, equal-length,
  HXB2-coordinate-aligned FASTA of [parent1, parent2, recombinant] in any
  order — labels don't have to match the LANL parental subtypes since
  RustRDP infers identity itself).
- Runs RustRDP with the requested methods at the requested p-value.
- Compares (Start, End) of each event row to the scraped LANL truth BPs in
  data/lanl_crf/truth_bps.csv, using the same TOLERANCE/greedy-matching
  as eval_rustrdp_santa.py.

Note: CNN evaluation on LANL is a separate task — runB2's 33-channel encoding
includes 11 RDP-derived signal channels (positions 22..32) that need to be
precomputed by an external RDP pipeline, which we don't have for LANL yet.
This script handles only the classical-method side for now; CNN-on-LANL is
a follow-up. Per-method numbers here are still informative even alone.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

RUSTRDP = '/home/joshc/Dev/RustRDP/target/release/rustrdp'
META_CSV = Path('data/lanl_crf/crf_meta.csv')
BPS_CSV = Path('data/lanl_crf/truth_bps.csv')
TRIPLET_DIR = Path('data/lanl_crf/triplets')
TMP_DIR = Path('/tmp/rustrdp_lanl')
OUT_JSON = Path('results_rustrdp_lanl.json')

DEFAULT_METHODS = ('maxchi', 'rdp', 'chimaera', 'siscan', 'bootscan')
TOLERANCE = 200


def load_truth():
    """crf -> sorted list of HXB2 BP positions."""
    by_crf = defaultdict(list)
    with BPS_CSV.open() as f:
        for r in csv.DictReader(f):
            by_crf[r['crf']].append(int(r['hxb2_position']))
    return {k: sorted(set(v)) for k, v in by_crf.items()}


def run_rustrdp(fasta, out_csv, methods, max_pvalue):
    cmd = [RUSTRDP, '-i', str(fasta), '-m', ','.join(methods),
           '-o', str(out_csv), '--max-pvalue', str(max_pvalue)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def parse_rustrdp_csv(out_csv):
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


def score(pred_bps, true_bps, tol=TOLERANCE):
    """Greedy nearest-first matching. Each true BP can match at most one pred."""
    used = set(); n_match = 0
    for tb in true_bps:
        best = None; best_d = tol + 1
        for j, p in enumerate(pred_bps):
            if j in used:
                continue
            d = abs(p - tb)
            if d <= tol and d < best_d:
                best_d = d; best = j
        if best is not None:
            used.add(best); n_match += 1
    tp = n_match
    fp = len(pred_bps) - len(used)
    fn = len(true_bps) - n_match
    return tp, fp, fn


def f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0, p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default=','.join(DEFAULT_METHODS))
    ap.add_argument('--max-pvalue', type=float, default=0.5)
    ap.add_argument('--out', default=str(OUT_JSON))
    args = ap.parse_args()
    methods = tuple(args.methods.split(','))
    method_names = {'maxchi': 'MaxChi', 'rdp': 'RDP', 'chimaera': 'Chimaera',
                    'siscan': 'SiScan', 'bootscan': 'Bootscan'}

    truth = load_truth()
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    crfs = []
    with META_CSV.open() as f:
        for r in csv.DictReader(f):
            crfs.append(r)

    print(f"[{time.strftime('%H:%M:%S')}] RustRDP on LANL CRFs (tol={TOLERANCE})")
    print(f"  methods={methods}, max-pvalue={args.max_pvalue}")

    results = {}
    # Aggregate across CRFs
    agg_tp = {m: 0 for m in methods}
    agg_fp = {m: 0 for m in methods}
    agg_fn = {m: 0 for m in methods}

    for r in crfs:
        crf = r['crf']
        fa = TRIPLET_DIR / f"{crf}.fa"
        if not fa.exists():
            print(f"\n  {crf}: SKIP — no triplet FASTA at {fa}")
            results[crf] = {'skipped': True, 'reason': f'missing {fa}'}
            continue

        out_csv = TMP_DIR / f"{crf}.csv"
        proc = run_rustrdp(fa, out_csv, methods, args.max_pvalue)
        if proc.returncode != 0:
            print(f"\n  {crf}: FAIL rc={proc.returncode}\n    {proc.stderr.strip()[:300]}")
            results[crf] = {'failed': True, 'stderr': proc.stderr[:1000]}
            continue
        by_method = parse_rustrdp_csv(out_csv)

        true_bps = truth.get(crf, [])
        print(f"\n  {crf} (prototype {r['prototype']}/{r['accession']}, {len(true_bps)} truth BPs)")
        per_method = {}
        for m in methods:
            mname = method_names.get(m, m)
            rows = by_method.get(mname, [])
            pred_bps = []
            for s, e, _p in rows:
                pred_bps.extend([s, e])
            tp, fp, fn = score(pred_bps, true_bps)
            f, prec, rec = f1(tp, fp, fn)
            per_method[m] = dict(n_rows=len(rows), n_pred_bps=len(pred_bps),
                                 tp=tp, fp=fp, fn=fn,
                                 precision=prec, recall=rec, f1=f)
            agg_tp[m] += tp; agg_fp[m] += fp; agg_fn[m] += fn
            print(f"    {m:>10}: rows={len(rows):>3}  pred_bps={len(pred_bps):>4}  "
                  f"tp={tp:>3}/{len(true_bps)}  fp={fp:>3}  fn={fn:>3}  "
                  f"P={prec:.3f}  R={rec:.3f}  F1={f:.3f}")
        results[crf] = {'truth_bps': true_bps, 'per_method': per_method}

    # Aggregate
    print(f"\n{'='*72}")
    print(f"  AGGREGATE across {sum(1 for v in results.values() if 'per_method' in v)} CRFs")
    print(f"{'='*72}")
    print(f"  {'method':>10} | {'precision':>9} | {'recall':>7} | {'F1':>7} | tp/fp/fn")
    print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*15}")
    summary = {}
    for m in methods:
        f, prec, rec = f1(agg_tp[m], agg_fp[m], agg_fn[m])
        summary[m] = dict(tp=agg_tp[m], fp=agg_fp[m], fn=agg_fn[m],
                          precision=prec, recall=rec, f1=f)
        print(f"  {m:>10} | {prec:>9.3f} | {rec:>7.3f} | {f:>7.3f} | "
              f"{agg_tp[m]}/{agg_fp[m]}/{agg_fn[m]}")
    print(f"{'='*72}")

    out = {
        'methods': list(methods),
        'max_pvalue': args.max_pvalue,
        'tolerance': TOLERANCE,
        'per_crf': results,
        'aggregate': summary,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
