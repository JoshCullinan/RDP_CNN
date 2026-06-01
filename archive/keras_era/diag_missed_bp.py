#!/usr/bin/env python3
"""Sanity check: are missed test BPs clustered near content_end (boundary
shortcut artifact suppressed by EB=200) or in the interior (genuine OOD)?"""
import numpy as np, tensorflow as tf, glob, pickle, gc
from scipy.signal import find_peaks

MAX_SEQ_LEN = 32000
TOLERANCE = 200
MODEL = 'models_test/cnn_breakpoint_run39_final.keras'

print(f"Loading {MODEL}")
model = tf.keras.models.load_model(MODEL, compile=False)
n_ch = model.input_shape[-1]
test_npz = next(p for p in sorted(glob.glob('cache/ds_UnseenTestSet_*.npz'), reverse=True)
                if np.load(p)['X'].shape[-1] == n_ch)
test_pkl = test_npz.replace('.npz', '.pkl')
print(f"  test cache: {test_npz}")

d = np.load(test_npz)
X_test = d['X'].astype(np.float16)
with open(test_pkl, 'rb') as f:
    meta_test = pickle.load(f)

print(f"Predicting test ({len(meta_test)} events)...")
sig = tf.TensorSpec(shape=(MAX_SEQ_LEN, n_ch), dtype=tf.float16)
def gen():
    for i in range(len(X_test)):
        yield X_test[i]
ds = (tf.data.Dataset.from_generator(gen, output_signature=sig)
      .batch(2).map(lambda x: tf.cast(x, tf.float32)).prefetch(2))
y_pred = model.predict(ds, verbose=0)
print(f"  shape: {y_pred.shape}")

# Apply EB=200 suppression as in eval_run29.py
EB = 200
threshold = 0.7

# For each event, classify each true BP as: detected, missed-near-end, missed-interior
detected_count = 0
miss_near_end = 0  # missed BP within EB of content_end
miss_interior = 0  # missed BP elsewhere
miss_total = 0

# also: where are the BPs that were missed?
miss_bp_positions = []
miss_bp_positions_relative = []  # relative to content_end
detected_bp_positions = []

for i, m in enumerate(meta_test):
    content_mask = (X_test[i].astype(np.float32).sum(axis=-1) > 0)
    content_end = int(content_mask.sum())
    pred = y_pred[i].copy()
    if pred.ndim == 2:
        pred = pred[..., 0]
    if EB > 0:
        pred[:EB] = 0.0
        if content_end > EB:
            pred[content_end - EB:content_end] = 0.0
    pred[content_end:] = 0.0
    peaks, _ = find_peaks(pred, height=threshold, distance=TOLERANCE)
    true_bps = [m['bp_start'], m['bp_end']]
    for tb in true_bps:
        # Detect: any peak within tolerance?
        if len(peaks) and (np.abs(peaks - tb) <= TOLERANCE).any():
            detected_count += 1
            detected_bp_positions.append(tb)
        else:
            miss_total += 1
            miss_bp_positions.append(tb)
            miss_bp_positions_relative.append(tb - content_end)
            # Within EB of either end?
            if tb < EB or (content_end > EB and tb >= content_end - EB):
                miss_near_end += 1
            else:
                miss_interior += 1

print(f"\nDetected: {detected_count}")
print(f"Missed total: {miss_total}")
print(f"  - near content edge (within {EB} of either boundary): {miss_near_end} ({100*miss_near_end/max(1,miss_total):.1f}%)")
print(f"  - interior (suppression-safe): {miss_interior} ({100*miss_interior/max(1,miss_total):.1f}%)")

# Position distribution of missed BPs
mp = np.array(miss_bp_positions)
print(f"\nMissed BP position distribution:")
print(f"  in [0, 5k]={int(((mp >= 0) & (mp < 5000)).sum())}")
print(f"  in [5k, 10k]={int(((mp >= 5000) & (mp < 10000)).sum())}")
print(f"  in [10k, 15k]={int(((mp >= 10000) & (mp < 15000)).sum())}")
print(f"  in [15k, 20k]={int(((mp >= 15000) & (mp < 20000)).sum())}")
print(f"  in [20k, 25k]={int(((mp >= 20000) & (mp < 25000)).sum())}")
print(f"  in [25k, 32k]={int(((mp >= 25000) & (mp < 32000)).sum())}")

# Same for detected
dp = np.array(detected_bp_positions)
print(f"\nDetected BP position distribution:")
print(f"  in [0, 5k]={int(((dp >= 0) & (dp < 5000)).sum())}")
print(f"  in [5k, 10k]={int(((dp >= 5000) & (dp < 10000)).sum())}")
print(f"  in [10k, 15k]={int(((dp >= 10000) & (dp < 15000)).sum())}")
print(f"  in [15k, 20k]={int(((dp >= 15000) & (dp < 20000)).sum())}")
print(f"  in [20k, 25k]={int(((dp >= 20000) & (dp < 25000)).sum())}")
print(f"  in [25k, 32k]={int(((dp >= 25000) & (dp < 32000)).sum())}")

# Detection rate per position bucket
print(f"\nDetection rate per BP position bucket (interior, suppression-safe events only):")
all_bps = list(zip(mp, [False]*len(mp))) + list(zip(dp, [True]*len(dp)))
buckets = [(0,5000), (5000,10000), (10000,15000), (15000,20000), (20000,25000), (25000,32000)]
for lo, hi in buckets:
    in_range = [d for pos, d in all_bps if lo <= pos < hi]
    if not in_range:
        continue
    rate = sum(in_range) / len(in_range)
    print(f"  [{lo:5d}, {hi:5d}): n={len(in_range):5d}  detection rate={rate:.3f}")
