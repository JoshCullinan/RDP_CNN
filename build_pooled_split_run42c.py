#!/usr/bin/env python3
"""Build run42c pooled stratified split — same as run42 but filters out
long_content files whose mean informative% is below a quality threshold.

Reads long_content_informative_scores.tsv (output of
score_long_content_informative.py) to pick which long_content files are
high enough quality to include.

Threshold: 5% mean informative% (the legacy lower bound — XML-1 sits at
6.8% on average; below 5% suggests parents are too genetically close to
the recombinant and the parental-signal channels collapse).

XML-1..5 and UnseenTestSet are NOT filtered — they use RDP5's
.faRecombIdentifyStats.csv which is already known-good.
"""
import csv, random
from pathlib import Path

DATA_ROOT = Path('dataRaw')
OUT_DIR = Path('splits'); OUT_DIR.mkdir(exist_ok=True)
DIRS = ['XML-1', 'XML-2', 'XML-3', 'XML-4', 'XML-5',
        'long_content_30k_001', 'long_content_30k_002', 'long_content_30k_003',
        'UnseenTestSet']
SEED = 42
random.seed(SEED)

BUCKETS = [(0, 10_000), (10_000, 20_000), (20_000, 10**9)]

# Informative% filter (long_content only)
INFORMATIVE_THRESHOLD = 5.0  # mean informative% per file


def load_informative_scores():
    """Load long_content informative% scores. Returns dict[rel_path]->float."""
    scores_path = Path('long_content_informative_scores.tsv')
    scores = {}
    if not scores_path.exists():
        print(f"  WARNING: {scores_path} not found — long_content files will NOT be filtered")
        return scores
    with scores_path.open() as f:
        rdr = csv.DictReader(f, delimiter='\t')
        for row in rdr:
            try:
                scores[row['path']] = float(row['mean_inf_pct'])
            except (KeyError, ValueError):
                continue
    print(f"  Loaded {len(scores)} informative% scores")
    return scores


def file_max_bp(csv_path: Path):
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
    return (fa.with_name(fa.name + 'RecombIdentifyStats.csv').exists()
            or fa.with_name(fa.name + 'Parents.csv').exists())


def bucket_of(mx):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= mx < hi:
            return i
    return len(BUCKETS) - 1


scores = load_informative_scores()

index = []
n_filtered_weak = 0
n_no_score = 0
for d in DIRS:
    if not (DATA_ROOT / d).exists():
        print(f"  skip {d}")
        continue
    n_kept = n_total = 0
    is_long_content = d.startswith('long_content_')
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
        # Apply informative% filter on long_content only
        if is_long_content:
            rel = str(fa.relative_to(DATA_ROOT))
            if rel not in scores:
                n_no_score += 1
                continue  # don't include files without a score (regen incomplete or errored)
            if scores[rel] < INFORMATIVE_THRESHOLD:
                n_filtered_weak += 1
                continue
        index.append({
            'rel_path': str(fa.relative_to(DATA_ROOT)),
            'dir': d,
            'n_events': n,
            'max_bp': mx,
        })
        n_kept += 1
    print(f"  {d}: kept {n_kept}/{n_total}")

print(f"\nTotal indexed: {len(index)}")
print(f"Long_content filtered for informative% < {INFORMATIVE_THRESHOLD}: {n_filtered_weak}")
print(f"Long_content with no score (regen incomplete?): {n_no_score}")

print("\nPer-bucket counts:")
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

with (OUT_DIR / 'run42c_index.tsv').open('w') as f:
    f.write('rel_path\tdir\tn_events\tmax_bp\n')
    for r in index:
        f.write(f"{r['rel_path']}\t{r['dir']}\t{r['n_events']}\t{r['max_bp']}\n")

for k, rows in splits.items():
    with (OUT_DIR / f'run42c_{k}.txt').open('w') as f:
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
