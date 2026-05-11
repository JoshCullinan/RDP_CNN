#!/usr/bin/env python3
"""Cache UnseenTestSet using the notebook's canonical parser.

eval_only.py originally reimplemented parse_simulation, but it lost
candidate ordering (set() cast) and produced wrong triplets. The val
numbers from eval_only.py are still valid (they used the cached canonical
pkl), but the test numbers were garbage.

This script exec()s the notebook's encoder + parser + cache cells
(1,3,6,10,11) directly so the cache uses the same code path as training.
After it runs, cache/ds_UnseenTestSet_*.npz/.pkl will exist and eval_only.py
can read them with no parser involvement.
"""
import json
from pathlib import Path

NB_PATH = Path('CNN.ipynb')
CELLS_TO_RUN = ['cell-1', 'cell-2', 'cell-3', 'cell-4', 'cell-5', 'cell-6',
                'cell-7', 'cell-8', 'cell-9', 'cell-10', 'cell-11']

def main():
    nb = json.loads(NB_PATH.read_text())
    cells = {c.get('id'): ''.join(c['source']) for c in nb['cells']
             if c.get('cell_type') == 'code'}

    # Build a single string of all the cells we want to run, in order.
    g = {}
    g['__name__'] = '__main__'
    for cid in CELLS_TO_RUN:
        if cid not in cells:
            print(f"WARN: {cid} missing")
            continue
        src = cells[cid]
        # Skip the test-load triggers in cell-9 etc — those run at notebook
        # bottom of cell. Let's just exec each cell's source. Cell-9 has a
        # parse on a sample file; that's harmless.
        print(f"--- exec {cid} ({len(src)} chars) ---")
        try:
            exec(compile(src, f'CNN.ipynb#{cid}', 'exec'), g)
        except Exception as e:
            print(f"  exec error in {cid}: {type(e).__name__}: {e}")
            raise

    print("\n=== now calling load_dataset(['UnseenTestSet'], label_mode='breakpoint') ===")
    load_dataset = g['load_dataset']
    X, y, mask, meta = load_dataset(['UnseenTestSet'], label_mode='breakpoint')
    print(f"DONE. X.shape={X.shape} dtype={X.dtype}  meta_n={len(meta)}")

    # Verify cache files appeared
    import glob
    test_caches = sorted(glob.glob('cache/ds_UnseenTestSet_*.npz'))
    print(f"cache files: {test_caches}")


if __name__ == '__main__':
    main()
