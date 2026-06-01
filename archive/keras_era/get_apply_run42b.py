"""Apply patches for run #42b (post-regen, filtered for informative% >= 5%):
  - cell-11: bump CACHE_VERSION to v7_oracle_parents (invalidate v6 train/val caches)
  - cell-12: switch to splits/run42b_{train,val}.txt
  - cell-22: save to cnn_breakpoint_run42b_final.keras

Note: UnseenTestSet cache from v6 still has 33 channels matching this run,
but the CACHE_VERSION change invalidates it too. We'll need to rebuild that
cache once. Plan: this script does ONLY the train/val swap; UnseenTestSet
cache rebuild is a separate step.
"""
import json
from pathlib import Path

NB = Path('/home/joshcullinan/RDP_CNN/CNN.ipynb')
with NB.open() as f:
    nb = json.load(f)


def patch(cid, old, new):
    for c in nb['cells']:
        if c.get('id') == cid:
            src = ''.join(c['source'])
            assert old in src, f"In {cid}: missing:\n{old}"
            src = src.replace(old, new)
            c['source'] = src.splitlines(keepends=True)
            c['outputs'] = []
            c['execution_count'] = None
            print(f"  [{cid}] patched")
            return
    raise SystemExit(f"Cell {cid} not found")

# cell-11: CACHE_VERSION
patch('cell-11',
    "CACHE_VERSION = 'v6_long30k'  # run #42: + long_content_30k_{001,002} via .faParents.csv",
    "CACHE_VERSION = 'v7_oracle_parents'  # run #42b: long_content parents regenerated via event_classifier port + informative%-filtered")

# cell-12: split lists
patch('cell-12',
    "TRAIN_LIST = SPLITS_DIR / 'run42_train.txt'",
    "TRAIN_LIST = SPLITS_DIR / 'run42b_train.txt'")
patch('cell-12',
    "VAL_LIST   = SPLITS_DIR / 'run42_val.txt'",
    "VAL_LIST   = SPLITS_DIR / 'run42b_val.txt'")
patch('cell-12',
    "'run42_train', train_files, max_events=TRAIN_MAX_EVENTS,",
    "'run42b_train', train_files, max_events=TRAIN_MAX_EVENTS,")
patch('cell-12',
    "'run42_val', val_files, max_events=VAL_MAX_EVENTS,",
    "'run42b_val', val_files, max_events=VAL_MAX_EVENTS,")
patch('cell-12',
    'print("\\n=== TRAIN: pool/run42_train ===")',
    'print("\\n=== TRAIN: pool/run42b_train ===")')
patch('cell-12',
    'print("\\n=== VAL: pool/run42_val ===")',
    'print("\\n=== VAL: pool/run42b_val ===")')
patch('cell-13',
    'print("Pooled split (run #42): XML-1..5 + long_content_30k_001/002 + UnseenTestSet stratified by max-BP")',
    'print("Pooled split (run #42b): XML-1..5 + long_content_30k_001/002 (informative%>=5 filtered) + UnseenTestSet")')

# cell-22: save name
patch('cell-22',
    "_versioned = 'models_test/cnn_breakpoint_run42_final.keras'",
    "_versioned = 'models_test/cnn_breakpoint_run42b_final.keras'")

with NB.open('w') as f:
    json.dump(nb, f, indent=1)
print("Saved CNN.ipynb")
