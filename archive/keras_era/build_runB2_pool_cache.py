#!/usr/bin/env python3
"""Build a 7000-event train cache from splits/run41_train.txt for runB2_7k.

Mirrors cache_pooled_test.py but for the TRAIN split with max_events=7000.
The cache key includes max_events, so the resulting filename differs from
the existing 4000-event cache (ds_pool_run41_train_13767833f56c8d8d.npz).

Note: cell-10's parse_simulation has gained .faParents.csv support and a
max_events_per_file cap since run41 (commit 4ed2e33). For run41_train.txt
files (all XML-{N}), the legacy .faRecombIdentifyStats.csv path is taken
and max_events_per_file=60 is a no-op (XML files are <60 events each).
So this 7000-event cache produces parsing-identical events to the existing
run41 4000-event cache (just more of them, drawn from the same file order).

Output: cache/ds_pool_run41_train_<key>.npz  (key differs from existing)
        cache/ds_pool_run41_train_<key>.pkl
"""
import json, time
from pathlib import Path

NB_PATH = Path('CNN.ipynb')
CELLS_TO_RUN = ['cell-1', 'cell-2', 'cell-3', 'cell-4', 'cell-5', 'cell-6',
                'cell-7', 'cell-8', 'cell-9', 'cell-10', 'cell-11']
TRAIN_LIST = Path('splits/run41_train.txt')
VAL_LIST   = Path('splits/run41_val.txt')
TRAIN_MAX_EVENTS = 7000
VAL_MAX_EVENTS   = 1500  # match existing run41 val cache size for fair comparison


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = {c.get('id'): ''.join(c['source']) for c in nb['cells']
             if c.get('cell_type') == 'code'}

    g = {'__name__': '__main__'}
    for cid in CELLS_TO_RUN:
        if cid not in cells:
            print(f"WARN: {cid} missing", flush=True)
            continue
        src = cells[cid]
        print(f"--- exec {cid} ({len(src)} chars) ---", flush=True)
        exec(compile(src, f'CNN.ipynb#{cid}', 'exec'), g)

    DATA_ROOT = g['DATA_ROOT']
    load_filelist_dataset = g['load_filelist_dataset']

    # Read train split list
    paths = []
    with TRAIN_LIST.open() as f:
        for ln in f:
            rel = ln.strip()
            if rel:
                paths.append(DATA_ROOT / rel)
    print(f"\n=== loading pool/run41_train ({len(paths)} files, max_events={TRAIN_MAX_EVENTS}) ===", flush=True)
    t0 = time.time()
    X, y, mask, meta = load_filelist_dataset(
        'run41_train', paths, label_mode='gaussian',
        max_events=TRAIN_MAX_EVENTS,
    )
    print(f"\nTRAIN cache built in {time.time()-t0:.0f}s", flush=True)
    print(f"  X.shape={X.shape} dtype={X.dtype}  meta_n={len(meta)}", flush=True)

    # Also ensure val cache exists (should already be built from run41)
    val_paths = []
    with VAL_LIST.open() as f:
        for ln in f:
            rel = ln.strip()
            if rel:
                val_paths.append(DATA_ROOT / rel)
    print(f"\n=== loading pool/run41_val ({len(val_paths)} files, max_events={VAL_MAX_EVENTS}) ===", flush=True)
    Xv, yv, mv, mev = load_filelist_dataset(
        'run41_val', val_paths, label_mode='gaussian',
        max_events=VAL_MAX_EVENTS,
    )
    print(f"  val X.shape={Xv.shape}, meta_n={len(mev)}", flush=True)

    import glob
    caches = sorted(glob.glob('cache/ds_pool_run41_train_*.npz'))
    print(f"\nall run41_train caches: {caches}", flush=True)


if __name__ == '__main__':
    main()
