#!/usr/bin/env python3
"""Build run42 pooled stratified split across XML-1..5 + long_content_30k_{001,002}
+ UnseenTestSet.

XML-6 (the failed 10k-bounded dump) is excluded — it has no parent-mapping
file so parse_simulation() returns 0 events for every .fa.

For each .fa, parse its .faSimVSRealCompare.csv to compute:
  - n_events
  - max BP coordinate across all events
We also require *either* a `.faRecombIdentifyStats.csv` (legacy) or a
`.faParents.csv` (long-content) so the file actually yields events.

Stratify files into length buckets [0-10k, 10-20k, 20-30k+] and split each
bucket 70/15/15. Preserve run41 lists; write run42 lists.
"""
import csv, random
from pathlib import Path

DATA_ROOT = Path('dataRaw')
OUT_DIR = Path('splits'); OUT_DIR.mkdir(exist_ok=True)
DIRS = ['XML-1', 'XML-2', 'XML-3', 'XML-4', 'XML-5',
        'long_content_30k_001', 'long_content_30k_002',
        'UnseenTestSet']
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
        for row in r:
            try:
                a = int(float(row.get('SimBPStart') or 0))
                b = int(float(row.get('SimBPEnd') or 0))
                mx = max(mx, a, b)
                n += 1
            except (TypeError, ValueError):
                continue
    return n, mx


def has_parent_mapping(fa: Path) -> bool:
    """A file is parseable if it has either a stats CSV (legacy) or a parents CSV."""
    return (fa.with_name(fa.name + 'RecombIdentifyStats.csv').exists()
            or fa.with_name(fa.name + 'Parents.csv').exists())


index = []
for d in DIRS:
    if not (DATA_ROOT / d).exists():
        print(f"  skip {d} (directory missing)")
        continue
    n_total = n_indexed = 0
    for fa in sorted((DATA_ROOT / d).glob('*.fa')):
        n_total += 1
        csv_path = fa.with_name(fa.name + 'SimVSRealCompare.csv')
        if not csv_path.exists():
            continue
        if not has_parent_mapping(fa):
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
        n_indexed += 1
    print(f"  {d}: {n_indexed}/{n_total} .fa files parseable")

print(f"\nIndexed {len(index)} parseable .fa files total")


def bucket_of(mx):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= mx < hi:
            return i
    return len(BUCKETS) - 1

print("\nPer-bucket counts (across all dirs):")
for i, (lo, hi) in enumerate(BUCKETS):
    n = sum(1 for r in index if bucket_of(r['max_bp']) == i)
    e = sum(r['n_events'] for r in index if bucket_of(r['max_bp']) == i)
    print(f"  bucket {lo:>6} - {hi:>6}: {n:>5} files / {e:>8} events")

by_dir = {}
for r in index:
    by_dir.setdefault(r['dir'], [0, 0, 0])
    by_dir[r['dir']][bucket_of(r['max_bp'])] += 1
print("\nPer-dir bucket counts (0-10k, 10-20k, 20k+):")
for d, b in by_dir.items():
    print(f"  {d:<24} {b}")

splits = {'train': [], 'val': [], 'test': []}
for i in range(len(BUCKETS)):
    rows = [r for r in index if bucket_of(r['max_bp']) == i]
    random.shuffle(rows)
    n_train = int(len(rows) * 0.70)
    n_val = int(len(rows) * 0.15)
    splits['train'] += rows[:n_train]
    splits['val'] += rows[n_train:n_train + n_val]
    splits['test'] += rows[n_train + n_val:]

for k in splits:
    random.shuffle(splits[k])

with (OUT_DIR / 'run42_index.tsv').open('w') as f:
    f.write('rel_path\tdir\tn_events\tmax_bp\n')
    for r in index:
        f.write(f"{r['rel_path']}\t{r['dir']}\t{r['n_events']}\t{r['max_bp']}\n")

for k, rows in splits.items():
    with (OUT_DIR / f'run42_{k}.txt').open('w') as f:
        for r in rows:
            f.write(r['rel_path'] + '\n')

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
