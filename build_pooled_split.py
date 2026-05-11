#!/usr/bin/env python3
"""Build a pooled stratified split across XML-1..6 + UnseenTestSet.

For each .fa, parse its .faSimVSRealCompare.csv to compute:
  - n_events
  - max BP coordinate across all events

Stratify files into length buckets [0-10k, 10-20k, 20-30k+] and split each
bucket 70/15/15 into train/val/test, file-level so events from the same
SANTA run stay together.

Writes:
  splits/run41_index.tsv  -- per-file metadata
  splits/run41_train.txt  -- one path per line (relative to dataRaw/)
  splits/run41_val.txt
  splits/run41_test.txt
"""
import os, csv, random
from pathlib import Path

DATA_ROOT = Path('dataRaw')
OUT_DIR = Path('splits'); OUT_DIR.mkdir(exist_ok=True)
DIRS = ['XML-1', 'XML-2', 'XML-3', 'XML-4', 'XML-5', 'XML-6', 'UnseenTestSet']
SEED = 42
random.seed(SEED)

BUCKETS = [(0, 10_000), (10_000, 20_000), (20_000, 10**9)]


def file_max_bp(csv_path: Path):
    """Return (n_events, max_bp). max_bp = max across all events of
    max(SimBPStart, SimBPEnd). NaN/blank treated as 0."""
    if not csv_path.exists():
        return 0, 0
    n = 0; mx = 0
    with csv_path.open() as f:
        r = csv.DictReader(f, skipinitialspace=True)
        # tolerate the leading-space header form ' PredBPStart'
        for row in r:
            try:
                a = int(float(row.get('SimBPStart') or 0))
                b = int(float(row.get('SimBPEnd') or 0))
                mx = max(mx, a, b)
                n += 1
            except (TypeError, ValueError):
                continue
    return n, mx


index = []
for d in DIRS:
    for fa in sorted((DATA_ROOT / d).glob('*.fa')):
        csv_path = fa.with_name(fa.name + 'SimVSRealCompare.csv')
        if not csv_path.exists():
            continue
        n, mx = file_max_bp(csv_path)
        if n == 0:
            continue
        index.append({
            'rel_path': str(fa.relative_to(DATA_ROOT)),
            'dir': d,
            'n_events': n,
            'max_bp': mx,
        })

print(f"Indexed {len(index)} .fa files with paired SimVSReal CSV")

# Per-bucket counts before split
def bucket_of(mx):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= mx < hi:
            return i
    return len(BUCKETS) - 1

for i, (lo, hi) in enumerate(BUCKETS):
    n = sum(1 for r in index if bucket_of(r['max_bp']) == i)
    e = sum(r['n_events'] for r in index if bucket_of(r['max_bp']) == i)
    print(f"  bucket {lo:>6} - {hi:>6}: {n:>5} files / {e:>7} events")

# Per-dir × per-bucket distribution (sanity)
by_dir = {}
for r in index:
    by_dir.setdefault(r['dir'], [0, 0, 0])
    by_dir[r['dir']][bucket_of(r['max_bp'])] += 1
print("\nPer-dir bucket counts (0-10k, 10-20k, 20k+):")
for d, b in by_dir.items():
    print(f"  {d:<14} {b}")

# Stratified split: within each bucket, shuffle and 70/15/15
splits = {'train': [], 'val': [], 'test': []}
for i in range(len(BUCKETS)):
    rows = [r for r in index if bucket_of(r['max_bp']) == i]
    random.shuffle(rows)
    n_train = int(len(rows) * 0.70)
    n_val = int(len(rows) * 0.15)
    splits['train'] += rows[:n_train]
    splits['val'] += rows[n_train:n_train + n_val]
    splits['test'] += rows[n_train + n_val:]

# Re-shuffle within each split for diversity
for k in splits:
    random.shuffle(splits[k])

# Write files
with (OUT_DIR / 'run41_index.tsv').open('w') as f:
    f.write('rel_path\tdir\tn_events\tmax_bp\n')
    for r in index:
        f.write(f"{r['rel_path']}\t{r['dir']}\t{r['n_events']}\t{r['max_bp']}\n")

for k, rows in splits.items():
    with (OUT_DIR / f'run41_{k}.txt').open('w') as f:
        for r in rows:
            f.write(r['rel_path'] + '\n')

# Summary
print("\n=== Split summary ===")
for k, rows in splits.items():
    by_b = [0, 0, 0]
    by_d = {}
    n_ev = 0
    for r in rows:
        by_b[bucket_of(r['max_bp'])] += 1
        by_d[r['dir']] = by_d.get(r['dir'], 0) + 1
        n_ev += r['n_events']
    print(f"\n{k}: {len(rows)} files / {n_ev} events")
    print(f"  buckets (0-10k / 10-20k / 20k+): {by_b}")
    print(f"  dirs: {by_d}")
