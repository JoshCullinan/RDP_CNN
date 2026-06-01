#!/usr/bin/env python3
"""SANTA-baseline F1 for RustRDP classical methods on the 11-file UnseenTestSet
subset.

For each event in the held-out subset, extracts the triplet (recombinant +
2 parents) per the SANTA truth CSV, writes a 3-seq FASTA, runs RustRDP, and
scores the resulting (Start, End) event-level predictions against the true
(SimBPStart, SimBPEnd) breakpoints using the same TOLERANCE=200 / greedy
nearest-first matching as evaluate_peaks in the CNN pipeline.

Apples-to-apples comparison target: runB2 (CNN) sub F1 @EB=25 = 0.448 on the
same 584-event subset.

Defensive: per-event temp FASTAs in /tmp, bounded parallelism, runs RustRDP
as a subprocess (its own process memory; we don't load alignments into Python
RAM beyond one source FASTA at a time).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from Bio import SeqIO

RUSTRDP = '/home/joshc/Dev/RustRDP/target/release/rustrdp'
SUBSET_FILE = Path('splits/honest_eval_subset_11.txt')
TOLERANCE = 200
METHODS_DEFAULT = ('maxchi', 'rdp')
OUT_JSON = Path('results_rustrdp_santa_subset.json')
TMP_ROOT = Path('/tmp/rustrdp_eval')


def parse_iseqs(s):
    return [int(t.strip()) for t in str(s).split('$') if t.strip().isdigit()]


def extract_events_from_file(fa_path):
    """Yield (event_id, recomb_id, parent_id_1, parent_id_2, bp_start, bp_end,
             seqs_by_id) per event in the file. seqs_by_id is shared per file."""
    sim_csv = Path(f"{fa_path}SimVSRealCompare.csv")
    stats_csv = Path(f"{fa_path}RecombIdentifyStats.csv")
    if not sim_csv.exists() or not stats_csv.exists():
        return []

    try:
        seqs = {int(r.id): str(r.seq) for r in SeqIO.parse(str(fa_path), 'fasta')}
        sim = pd.read_csv(sim_csv, skipinitialspace=True, index_col=False)
        stats = pd.read_csv(stats_csv, skipinitialspace=True, index_col=False)
    except Exception as e:
        print(f"  parse fail {fa_path}: {e}", flush=True)
        return []

    out = []
    for _, row in sim.iterrows():
        event = row['RDPEvent']
        recomb_id = int(row['ActualRecomb'])
        bp_start = int(row['SimBPStart'])
        bp_end = int(row['SimBPEnd'])

        ev_rows = stats[stats['Event'] == event]
        if len(ev_rows) != 3:
            continue
        parent_ids = []
        for _, sr in ev_rows.iterrows():
            ids = parse_iseqs(sr['ISeqs(A)'])
            if recomb_id in ids:
                continue
            if ids:
                parent_ids.append(ids[0])
        if len(parent_ids) < 2:
            continue
        if not all(sid in seqs for sid in [recomb_id, parent_ids[0], parent_ids[1]]):
            continue
        out.append({
            'file': fa_path.name,
            'event': int(event),
            'recomb_id': recomb_id,
            'parent1_id': parent_ids[0],
            'parent2_id': parent_ids[1],
            'bp_start': bp_start,
            'bp_end': bp_end,
            'recomb_seq': seqs[recomb_id],
            'parent1_seq': seqs[parent_ids[0]],
            'parent2_seq': seqs[parent_ids[1]],
        })
    return out


def write_triplet_fasta(ev, path):
    with open(path, 'w') as f:
        f.write(f">recomb_{ev['recomb_id']}\n{ev['recomb_seq']}\n")
        f.write(f">parent1_{ev['parent1_id']}\n{ev['parent1_seq']}\n")
        f.write(f">parent2_{ev['parent2_id']}\n{ev['parent2_seq']}\n")


def run_rustrdp(fasta_path, out_csv, methods, max_pvalue):
    cmd = [
        RUSTRDP,
        '-i', str(fasta_path),
        '-m', ','.join(methods),
        '-o', str(out_csv),
        '--max-pvalue', str(max_pvalue),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        return None, res.stderr
    return out_csv, None


def parse_rustrdp_csv(out_csv):
    """Return dict method -> list of (Start, End, Pvalue)."""
    by_method = {}
    if not Path(out_csv).exists():
        return by_method
    with open(out_csv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Method,'):
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            method = parts[0]
            try:
                start = int(parts[1]); end = int(parts[2])
                pvalue = float(parts[6])
            except ValueError:
                continue
            by_method.setdefault(method, []).append((start, end, pvalue))
    return by_method


def score_event(pred_bps, true_bps, tol=TOLERANCE):
    """Greedy nearest-first matching. Each true BP can match at most one pred BP.
    Returns (tp, fp, fn, matched_set_indicator)."""
    used = set()
    n_match = 0
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
    return tp, fp, fn, n_match


def process_event(ev, tmpdir, methods, max_pvalue):
    """Run RustRDP on one event triplet, return per-method scoring records."""
    key = f"{ev['file']}_ev{ev['event']}"
    fa = Path(tmpdir) / f"{key}.fa"
    csv_out = Path(tmpdir) / f"{key}.csv"
    write_triplet_fasta(ev, fa)
    _, err = run_rustrdp(fa, csv_out, methods, max_pvalue)
    by_method = parse_rustrdp_csv(csv_out)

    # Clean up temp files immediately to bound disk
    try: fa.unlink()
    except OSError: pass
    try: csv_out.unlink()
    except OSError: pass

    true_bps = [ev['bp_start'], ev['bp_end']]
    out = {}
    for m in methods:
        method_name = {'maxchi': 'MaxChi', 'rdp': 'RDP', 'chimaera': 'Chimaera',
                       'siscan': 'SiScan', 'bootscan': 'Bootscan'}.get(m, m)
        rows = by_method.get(method_name, [])
        # Flatten event-level (Start, End) into 2*N predicted breakpoints
        pred_bps = []
        for s, e, _p in rows:
            pred_bps.extend([s, e])
        tp, fp, fn, nmatch = score_event(pred_bps, true_bps)
        out[m] = dict(tp=tp, fp=fp, fn=fn, n_pred=len(pred_bps),
                      n_events_rows=len(rows), n_match=nmatch)
    out['_meta'] = dict(file=ev['file'], event=ev['event'],
                        bp_start=ev['bp_start'], bp_end=ev['bp_end'])
    return out


def f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0, p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default=','.join(METHODS_DEFAULT),
                    help='Comma-separated list (maxchi,rdp,chimaera,siscan,bootscan)')
    ap.add_argument('--max-pvalue', type=float, default=0.05)
    ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1),
                    help='Process pool size. Capped low for RAM safety.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Stop after this many events (debug).')
    ap.add_argument('--out', default=str(OUT_JSON))
    args = ap.parse_args()

    methods = tuple(args.methods.split(','))
    print(f"[{time.strftime('%H:%M:%S')}] RustRDP SANTA-baseline eval")
    print(f"  methods: {methods}")
    print(f"  max_pvalue: {args.max_pvalue}")
    print(f"  workers: {args.workers}")
    print(f"  subset file: {SUBSET_FILE}")

    files = [Path(p) for p in SUBSET_FILE.read_text().splitlines() if p.strip()]
    print(f"  files: {len(files)}")

    print(f"\n[{time.strftime('%H:%M:%S')}] extracting events...")
    all_events = []
    for fa in files:
        evs = extract_events_from_file(fa)
        print(f"    {fa.name}: {len(evs)} events")
        all_events.extend(evs)
    print(f"  total events: {len(all_events)}")

    if args.limit:
        all_events = all_events[:args.limit]
        print(f"  limited to first {len(all_events)} events")

    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix='rrdp_', dir=TMP_ROOT)

    print(f"\n[{time.strftime('%H:%M:%S')}] running RustRDP in pool (workers={args.workers})...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_event, ev, tmpdir, methods, args.max_pvalue): i
                for i, ev in enumerate(all_events)}
        for done_count, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as e:
                print(f"    event {futs[fut]} failed: {e}", flush=True)
            if done_count % 50 == 0:
                print(f"    {done_count}/{len(all_events)} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"  all events done in {time.time()-t0:.0f}s")

    # Per-method aggregate
    agg = {}
    for m in methods:
        tot_tp = sum(r[m]['tp'] for r in results)
        tot_fp = sum(r[m]['fp'] for r in results)
        tot_fn = sum(r[m]['fn'] for r in results)
        events_both = sum(1 for r in results if r[m]['n_match'] == 2)
        events_one = sum(1 for r in results if r[m]['n_match'] == 1)
        events_missed = sum(1 for r in results if r[m]['n_match'] == 0)
        f, p, r = f1(tot_tp, tot_fp, tot_fn)
        agg[m] = dict(tp=tot_tp, fp=tot_fp, fn=tot_fn,
                      precision=p, recall=r, f1=f,
                      events_both=events_both,
                      events_one=events_one,
                      events_missed=events_missed,
                      events_total=len(results))

    # Combined: union of all methods' predicted breakpoints
    print(f"\n[{time.strftime('%H:%M:%S')}] computing union-of-methods F1...")
    union_tp = union_fp = union_fn = 0
    union_both = union_one = union_missed = 0
    for r in results:
        # Reconstruct union of pred BPs across methods is hard from aggregated counts;
        # store the per-method n_pred + n_match in the JSON, and compute union below
        # from the raw scoring re-done. For now: report a "best of methods" lower
        # bound on union F1 (max recall across methods at sum of FPs).
        pass
    # We don't have raw pred_bps stored per event; the "combined" view requires
    # re-evaluation. Skip in v1 — per-method numbers are the headline.

    out = {
        'methods': list(methods),
        'max_pvalue': args.max_pvalue,
        'tolerance': TOLERANCE,
        'subset_file': str(SUBSET_FILE),
        'n_events': len(results),
        'per_method': agg,
        'cnn_runB2_sub_f1_at_eb25': 0.448,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {args.out}")

    # Compact table
    print(f"\n{'='*82}")
    print(f"  RustRDP SANTA-baseline on 11-file subset ({len(results)} events, tol={TOLERANCE}bp)")
    print(f"{'='*82}")
    print(f"  {'method':>10} | {'precision':>9} | {'recall':>7} | {'F1':>7} | "
          f"{'both':>5} | {'one':>5} | {'miss':>5}")
    print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")
    for m in methods:
        a = agg[m]
        print(f"  {m:>10} | {a['precision']:>9.4f} | {a['recall']:>7.4f} | "
              f"{a['f1']:>7.4f} | {a['events_both']:>5} | "
              f"{a['events_one']:>5} | {a['events_missed']:>5}")
    print(f"  CNN runB2  | (n/a)     | (n/a)   | {0.448:>7.4f} | (subset)")
    print(f"{'='*82}")


if __name__ == '__main__':
    main()
