#!/usr/bin/env python3
"""Apply run #41 changes to CNN.ipynb: pooled stratified split + bumped cache.

Changes:
  cell-11: bump CACHE_VERSION to 'v5_pool', add load_filelist_dataset()
  cell-12: read splits/run41_{train,val}.txt, load via load_filelist_dataset
           with max_events caps (preserving the existing memory budget)
  cell-13: keep same variable plumbing (no real change needed)
  cell-22: bump versioned save name to run41_final.keras

Architecture (cell-3) is untouched: 33-channel run #39 backbone. XML-6
events have channels [22, 33) all zero (no PredBP, no .fa.csv); the
RDP-dropout regularizer (run #38 trained on 10% zeroed inputs) handles
the asymmetry. UnseenTestSet events keep RDP signal populated.
"""
import json
from pathlib import Path

NB = Path('CNN.ipynb')
nb = json.loads(NB.read_text())

# ---- new cell-11 source: bump cache version + add load_filelist_dataset ----
CELL_11_NEW = r'''# Disk cache for parsed datasets. The MaxChi computation in
# encode_triplet is the bottleneck of data prep (per-file ~tens of ms;
# at full XML-1..4 it's the dominant wall-time cost on a single CPU).
# We cache the full per-directory `(X, y_gaussian, y_breakpoint,
# y_region, mask)` and the meta list once parsed, keyed on the config
# knobs that change the encoding/labels.
#
# Run #41: pooled-split branch added. `load_filelist_dataset` parses an
# explicit list of .fa paths (rather than a directory glob) and caches
# under `ds_pool_<split_name>_<key>.npz`. Used for the run #41 stratified
# pooled split across XML-1..6 + UnseenTestSet.
import hashlib, pickle

CACHE_DIR = Path('cache')
CACHE_DIR.mkdir(exist_ok=True)
CACHE_VERSION = 'v5_pool'  # run #41: pooled stratified split (XML-1..6 + UnseenTestSet)

X_DTYPE = np.float16  # storage dtype for X; cast to float32 in tf.data


def _dataset_cache_key(directory_name, fa_files, max_files):
    payload = (
        CACHE_VERSION,
        directory_name,
        tuple(f.name for f in fa_files),
        int(MAX_SEQ_LEN),
        int(N_INPUT_CHANNELS),
        tuple(MAXCHI_WINDOWS),
        float(LABEL_SIGMA),
        int(BP_WINDOW),
        max_files if max_files is not None else 'all',
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


def load_dataset(directories, label_mode=None, max_files=None):
    """Per-directory loader (legacy path for XML-N caches)."""
    if label_mode is None:
        label_mode = LABEL_MODE
    label_keys = {
        'gaussian':   'labels_gaussian',
        'breakpoint': 'labels_bp',
        'region':     'labels_region',
    }
    if label_mode not in label_keys:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    inputs, labels_g, labels_bp, labels_rg, meta = [], [], [], [], []

    for d in directories:
        fa_files = sorted((DATA_ROOT / d).glob("*.fa"))
        if max_files:
            fa_files = fa_files[:max_files]

        cache_key = _dataset_cache_key(d, fa_files, max_files)
        npz_path = CACHE_DIR / f"ds_{d}_{cache_key}.npz"
        pkl_path = CACHE_DIR / f"ds_{d}_{cache_key}.pkl"

        if npz_path.exists() and pkl_path.exists():
            print(f"\nCache HIT: {d}  ({npz_path.name}, {npz_path.stat().st_size/1e6:.1f} MB)")
            data = np.load(npz_path)
            X_raw = data['X']
            X_d = X_raw.astype(X_DTYPE) if X_raw.dtype != X_DTYPE else X_raw
            del X_raw
            yg_d  = data['y_g']
            ybp_d = data['y_bp']
            yrg_d = data['y_rg']
            with open(pkl_path, 'rb') as f:
                meta_d = pickle.load(f)
            print(f"  Loaded {X_d.shape[0]} samples from cache (X dtype={X_d.dtype}, {X_d.nbytes/1e9:.2f} GB)")
        else:
            print(f"\nCache MISS: {d}  ({len(fa_files)} files; will parse and cache)")
            X_l, yg_l, ybp_l, yrg_l, meta_l = [], [], [], [], []
            for fa in tqdm(fa_files, desc=d):
                for t in parse_simulation(fa):
                    X_l.append(t['input'].astype(X_DTYPE))
                    yg_l.append(t['labels_gaussian'])
                    ybp_l.append(t['labels_bp'])
                    yrg_l.append(t['labels_region'])
                    meta_l.append(t['meta'])
            X_d   = np.array(X_l)                     if X_l   else np.zeros((0, MAX_SEQ_LEN, N_INPUT_CHANNELS), dtype=X_DTYPE)
            yg_d  = np.array(yg_l,  dtype=np.float32) if yg_l  else np.zeros((0, MAX_SEQ_LEN), dtype=np.float32)
            ybp_d = np.array(ybp_l, dtype=np.float32) if ybp_l else np.zeros((0, MAX_SEQ_LEN), dtype=np.float32)
            yrg_d = np.array(yrg_l, dtype=np.float32) if yrg_l else np.zeros((0, MAX_SEQ_LEN), dtype=np.float32)
            meta_d = meta_l
            print(f"  Parsed {X_d.shape[0]} samples; saving cache ...")
            np.savez_compressed(npz_path, X=X_d, y_g=yg_d, y_bp=ybp_d, y_rg=yrg_d)
            with open(pkl_path, 'wb') as f:
                pickle.dump(meta_d, f)
            print(f"  Cache written: {npz_path.name} ({npz_path.stat().st_size/1e6:.1f} MB)")

        inputs.append(X_d)
        labels_g.append(yg_d)
        labels_bp.append(ybp_d)
        labels_rg.append(yrg_d)
        meta.extend(meta_d)

    X = np.concatenate(inputs, axis=0) if inputs else np.zeros((0, MAX_SEQ_LEN, N_INPUT_CHANNELS), dtype=X_DTYPE)
    del inputs
    if label_mode == 'gaussian':
        y = np.concatenate(labels_g, axis=0) if labels_g else np.zeros((0, MAX_SEQ_LEN), dtype=np.float32)
    elif label_mode == 'breakpoint':
        y = np.concatenate(labels_bp, axis=0) if labels_bp else np.zeros((0, MAX_SEQ_LEN), dtype=np.float32)
    else:
        y = np.concatenate(labels_rg, axis=0) if labels_rg else np.zeros((0, MAX_SEQ_LEN), dtype=np.float32)
    del labels_g, labels_bp, labels_rg

    mask = (X.sum(axis=-1) > 0).astype(np.float32)
    pos_mass = float(np.sum(y))
    valid_frac = float(mask.mean()) if mask.size else 0.0
    print(f"\n{'='*60}")
    print(f"Loaded {X.shape[0]} samples (label_mode={label_mode})")
    print(f"X shape: {X.shape}  dtype={X.dtype}  ({X.nbytes/1e9:.2f} GB)  |  y shape: {y.shape}  |  mask shape: {mask.shape}")
    if y.size:
        print(f"Label mass (sum y): {pos_mass:.1f} / {y.size}  "
              f"(mean={y.mean():.5f}, max={y.max():.3f})")
    print(f"Valid (non-padded) fraction: {valid_frac:.3f}")
    print(f"{'='*60}")

    return X, y, mask, meta


def load_filelist_dataset(split_name, fa_paths, label_mode=None,
                          max_events=None, max_files=None):
    """Load a pooled-split dataset from an explicit list of .fa Paths.

    Run #41+ entry point. Files are loaded in the order provided (caller
    is expected to have shuffled with stratification preserved). Stops
    accumulating events once `max_events` is reached.

    Cache is keyed on the sorted file list + config; safe to share across
    runs that pass the same list.
    """
    if label_mode is None:
        label_mode = LABEL_MODE
    if max_files is not None:
        fa_paths = fa_paths[:max_files]

    fa_paths = [Path(p) for p in fa_paths]

    payload = (
        CACHE_VERSION,
        'pool',
        split_name,
        tuple(p.as_posix() for p in fa_paths),
        int(MAX_SEQ_LEN),
        int(N_INPUT_CHANNELS),
        tuple(MAXCHI_WINDOWS),
        float(LABEL_SIGMA),
        int(BP_WINDOW),
        int(max_events) if max_events is not None else 'all',
    )
    cache_key = hashlib.sha256(repr(payload).encode()).hexdigest()[:16]
    npz_path = CACHE_DIR / f"ds_pool_{split_name}_{cache_key}.npz"
    pkl_path = CACHE_DIR / f"ds_pool_{split_name}_{cache_key}.pkl"

    if npz_path.exists() and pkl_path.exists():
        print(f"\nCache HIT: pool/{split_name}  ({npz_path.name}, {npz_path.stat().st_size/1e6:.1f} MB)")
        data = np.load(npz_path)
        X_raw = data['X']
        X = X_raw.astype(X_DTYPE) if X_raw.dtype != X_DTYPE else X_raw
        del X_raw
        yg = data['y_g']; ybp = data['y_bp']; yrg = data['y_rg']
        with open(pkl_path, 'rb') as f:
            meta = pickle.load(f)
        print(f"  Loaded {X.shape[0]} samples from cache (X dtype={X.dtype}, {X.nbytes/1e9:.2f} GB)")
    else:
        print(f"\nCache MISS: pool/{split_name}  ({len(fa_paths)} files; will parse and cache)")
        X_l, yg_l, ybp_l, yrg_l, meta_l = [], [], [], [], []
        for fa in tqdm(fa_paths, desc=f"pool/{split_name}"):
            for t in parse_simulation(fa):
                X_l.append(t['input'].astype(X_DTYPE))
                yg_l.append(t['labels_gaussian'])
                ybp_l.append(t['labels_bp'])
                yrg_l.append(t['labels_region'])
                meta_l.append(t['meta'])
                if max_events is not None and len(X_l) >= max_events:
                    break
            if max_events is not None and len(X_l) >= max_events:
                break
        X = np.array(X_l) if X_l else np.zeros((0, MAX_SEQ_LEN, N_INPUT_CHANNELS), dtype=X_DTYPE)
        yg = np.array(yg_l,  dtype=np.float32) if yg_l  else np.zeros((0, MAX_SEQ_LEN), dtype=np.float32)
        ybp = np.array(ybp_l, dtype=np.float32) if ybp_l else np.zeros((0, MAX_SEQ_LEN), dtype=np.float32)
        yrg = np.array(yrg_l, dtype=np.float32) if yrg_l else np.zeros((0, MAX_SEQ_LEN), dtype=np.float32)
        meta = meta_l
        print(f"  Parsed {X.shape[0]} samples; saving cache ...")
        np.savez_compressed(npz_path, X=X, y_g=yg, y_bp=ybp, y_rg=yrg)
        with open(pkl_path, 'wb') as f:
            pickle.dump(meta, f)
        print(f"  Cache written: {npz_path.name} ({npz_path.stat().st_size/1e6:.1f} MB)")

    if label_mode == 'gaussian':
        y = yg
    elif label_mode == 'breakpoint':
        y = ybp
    else:
        y = yrg

    mask = (X.sum(axis=-1) > 0).astype(np.float32)
    pos_mass = float(np.sum(y))
    valid_frac = float(mask.mean()) if mask.size else 0.0
    print(f"\n{'='*60}")
    print(f"Loaded pool/{split_name}: {X.shape[0]} samples (label_mode={label_mode})")
    print(f"X shape: {X.shape}  dtype={X.dtype}  ({X.nbytes/1e9:.2f} GB)  |  y shape: {y.shape}")
    if y.size:
        print(f"Label mass (sum y): {pos_mass:.1f} / {y.size}  (mean={y.mean():.5f}, max={y.max():.3f})")
    print(f"Valid (non-padded) fraction: {valid_frac:.3f}")
    print(f"{'='*60}")
    return X, y, mask, meta
'''

# ---- new cell-12: pool-aware loading via splits/run41_*.txt ----
CELL_12_NEW = r'''# Run #41: load TRAIN/VAL from the pooled stratified split.
# Pool: XML-1..6 + UnseenTestSet, file-level 70/15/15 stratified by
# max(SimBPStart, SimBPEnd) into [0-10k, 10-20k, 20k+] buckets so train
# and test both span 0-30k. Test split is loaded separately during eval.
SPLITS_DIR = Path('splits')
TRAIN_LIST = SPLITS_DIR / 'run41_train.txt'
VAL_LIST   = SPLITS_DIR / 'run41_val.txt'

# Memory budget at MAX_SEQ_LEN=32000 and N_INPUT_CHANNELS=33 (fp16):
# per event = 32000 * 33 * 2 = 2.1 MB.
#  - Parse phase peaks at ~2x the final X array (Python list of fp16 chunks
#    + the np.array() concat). 4,000 events → ~8 GB list + 8 GB final =
#    ~16 GB peak parse. Safe under the 24 GB cgroup cap.
#  - Training phase: X_train (~8 GB) + X_val (~3 GB) + tf.data closures +
#    TF graph + model. Resident ~15 GB. Comfortable.
# Holding train slightly under the run #38 baseline (4,200) but with much
# better positional coverage thanks to the pooled split.
TRAIN_MAX_EVENTS = 4000
VAL_MAX_EVENTS   = 1500


def _read_split_list(path):
    """Returns list of full Paths under DATA_ROOT for each rel_path entry."""
    paths = []
    with open(path) as f:
        for ln in f:
            rel = ln.strip()
            if rel:
                paths.append(DATA_ROOT / rel)
    return paths


train_files = _read_split_list(TRAIN_LIST)
val_files   = _read_split_list(VAL_LIST)
print(f"Pooled split: train={len(train_files)} files, val={len(val_files)} files")

print("\n=== TRAIN: pool/run41_train ===")
X_train_all, y_train_all, mask_train_all, meta_train_all = load_filelist_dataset(
    'run41_train', train_files, max_events=TRAIN_MAX_EVENTS,
)

print("\n=== VAL: pool/run41_val ===")
X_val_all, y_val_all, mask_val_all, meta_val_all = load_filelist_dataset(
    'run41_val', val_files, max_events=VAL_MAX_EVENTS,
)

mean_y_unmasked = float((y_train_all * mask_train_all).sum() / mask_train_all.sum())
implied_pos_weight = float((1.0 - mean_y_unmasked) / mean_y_unmasked)
POS_WEIGHT = 70.0  # held constant from run #38 (the deployment baseline)
print(f"\n{'='*60}")
print(f"TRAIN: {X_train_all.shape[0]} samples; mean(y) over unmasked: {mean_y_unmasked:.5f}")
print(f"VAL:   {X_val_all.shape[0]} samples")
print(f"Implied POS_WEIGHT (from train data): {implied_pos_weight:.2f}")
print(f"POS_WEIGHT in use (hardcoded):        {POS_WEIGHT:.2f}")
print(f"{'='*60}")

import numpy as np  # noqa: F401
'''

# ---- cell-13: minor — keep the variable plumbing as-is ----
CELL_13_NEW = r'''# Train / validation split — RUN #41 POOLED STRATIFIED SPLIT.
# Files were preselected upstream in cell-12 from splits/run41_{train,val}.txt
# (stratified across XML-1..6 + UnseenTestSet by max-BP bucket). No
# additional shuffling needed here; the upstream lists are already
# bucket-balanced and randomized.
X_train = X_train_all
y_train = y_train_all
w_train = mask_train_all
meta_train = meta_train_all

X_val = X_val_all
y_val = y_val_all
w_val = mask_val_all
meta_val = meta_val_all

print(f"Pooled split (run #41): XML-1..6 + UnseenTestSet stratified by max-BP")
print(f"Training:   {X_train.shape[0]} samples  (valid positions: {w_train.mean():.3f})")
print(f"Validation: {X_val.shape[0]} samples  (valid positions: {w_val.mean():.3f})")
'''

# ---- cell-22 update: bump versioned save name ----

# Apply the changes
def update_cell(nb, cid, new_source):
    for c in nb['cells']:
        if c.get('id') == cid:
            # Convert new_source to list of lines preserving newlines (Jupyter format)
            lines = new_source.splitlines(keepends=True)
            c['source'] = lines
            return True
    return False


changed = []
if update_cell(nb, 'cell-11', CELL_11_NEW):
    changed.append('cell-11')
if update_cell(nb, 'cell-12', CELL_12_NEW):
    changed.append('cell-12')
if update_cell(nb, 'cell-13', CELL_13_NEW):
    changed.append('cell-13')

# cell-22: just bump the model save name run40 -> run41
for c in nb['cells']:
    if c.get('id') == 'cell-22':
        src = ''.join(c['source'])
        if 'cnn_breakpoint_run40_final.keras' in src:
            src = src.replace('cnn_breakpoint_run40_final.keras',
                              'cnn_breakpoint_run41_final.keras')
            c['source'] = src.splitlines(keepends=True)
            changed.append('cell-22 (save name -> run41)')
        else:
            print('WARNING: cell-22 does not contain run40 save name')

NB.write_text(json.dumps(nb, indent=1))
print(f"Updated cells: {changed}")
print(f"Notebook saved: {NB}")
