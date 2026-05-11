#!/usr/bin/env python3
"""Cache the run #41 pooled test split via the notebook's canonical parser.

Mirrors cache_test_set.py but for splits/run41_test.txt instead of the
UnseenTestSet directory. Output: cache/ds_pool_run41_test_<key>.npz/.pkl.

Run after training under cgroup:
    systemd-run --user --scope --quiet -p MemoryMax=20G \
      bash -c "source .venv/bin/activate && python3 -u cache_pooled_test.py"
"""
import json
from pathlib import Path

NB_PATH = Path('CNN.ipynb')
CELLS_TO_RUN = ['cell-1', 'cell-2', 'cell-3', 'cell-4', 'cell-5', 'cell-6',
                'cell-7', 'cell-8', 'cell-9', 'cell-10', 'cell-11']
TEST_LIST = Path('splits/run41_test.txt')
TEST_MAX_EVENTS = 2500  # roughly matches val cache scale; safe under 20G cgroup


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = {c.get('id'): ''.join(c['source']) for c in nb['cells']
             if c.get('cell_type') == 'code'}

    g = {'__name__': '__main__'}
    for cid in CELLS_TO_RUN:
        if cid not in cells:
            print(f"WARN: {cid} missing")
            continue
        src = cells[cid]
        print(f"--- exec {cid} ({len(src)} chars) ---")
        exec(compile(src, f'CNN.ipynb#{cid}', 'exec'), g)

    DATA_ROOT = g['DATA_ROOT']
    load_filelist_dataset = g['load_filelist_dataset']

    # Read split list
    paths = []
    with TEST_LIST.open() as f:
        for ln in f:
            rel = ln.strip()
            if rel:
                paths.append(DATA_ROOT / rel)
    print(f"\n=== loading pool/run41_test ({len(paths)} files, max_events={TEST_MAX_EVENTS}) ===")
    X, y, mask, meta = load_filelist_dataset(
        'run41_test', paths, label_mode='breakpoint',
        max_events=TEST_MAX_EVENTS,
    )
    print(f"DONE. X.shape={X.shape} dtype={X.dtype}  meta_n={len(meta)}")

    import glob
    test_caches = sorted(glob.glob('cache/ds_pool_run41_test_*.npz'))
    print(f"cache files: {test_caches}")


if __name__ == '__main__':
    main()
