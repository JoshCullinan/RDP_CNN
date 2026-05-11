#!/usr/bin/env python3
"""Position-OOD diagnostic: detection rate as a function of breakpoint position.

If detection rate cliff-falls past the training-content-length ceiling (~19525 bp,
the XML-4 max), the failure mode is position out-of-distribution — fix is data
augmentation, not architecture.

If detection rate is flat-but-low across the whole position range, the failure
mode is multi-scale capacity — fix is U-Net or Transformer.
"""
import numpy as np
import tensorflow as tf
import glob, pickle, gc
from scipy.signal import find_peaks

MAX_SEQ_LEN = 32000
TOLERANCE = 200
N_TEST = 800  # subset for speed; ~10s on GPU


def streaming_predict(model, X):
    sig = tf.TensorSpec(shape=(MAX_SEQ_LEN, X.shape[-1]), dtype=tf.float16)
    def gen():
        for i in range(len(X)):
            yield X[i]
    with tf.device('/CPU:0'):
        ds = (tf.data.Dataset.from_generator(gen, output_signature=sig)
              .batch(2)
              .map(lambda x: tf.cast(x, tf.float32))
              .prefetch(2))
    return model.predict(ds, verbose=0)


def main():
    print("Loading model...")
    model = tf.keras.models.load_model('models_test/cnn_breakpoint_final.keras', compile=False)

    test_npz = sorted(glob.glob('cache/ds_UnseenTestSet_*.npz'))[-1]
    test_pkl = test_npz.replace('.npz', '.pkl')

    print(f"Loading test cache (first {N_TEST}) ...")
    d = np.load(test_npz)
    X_test = d['X'][:N_TEST].astype(np.float16)
    with open(test_pkl, 'rb') as f:
        meta_all = pickle.load(f)
    meta_test = meta_all[:N_TEST]
    print(f"  X_test.shape={X_test.shape}  meta n={len(meta_test)}")

    print("Predicting...")
    y_pred = streaming_predict(model, X_test)
    print(f"  y_pred.shape={y_pred.shape}")

    # Bucket BPs by position. For each true BP, mark whether a peak (height>=thr,
    # min distance=TOLERANCE) lands within +/- TOLERANCE.
    # Try at val-best threshold AND test-best threshold, since the global
    # threshold may already be position-dependent.
    buckets = [(0, 5000), (5000, 10000), (10000, 15000),
               (15000, 19525), (19525, 25000), (25000, 32000)]

    for thr_label, thr in [('val-best 0.6', 0.6), ('test-best 0.9', 0.9)]:
        print(f"\n=== Threshold = {thr} ({thr_label}) ===")
        # Per-bucket: count true BPs falling there, count detections
        bucket_total = {b: 0 for b in buckets}
        bucket_hit = {b: 0 for b in buckets}
        # Also track total peaks per sample (for FP context)
        for i in range(len(meta_test)):
            m = meta_test[i]
            true_bps = [m['bp_start'], m['bp_end']]
            peaks, _ = find_peaks(y_pred[i], height=thr, distance=TOLERANCE)
            for tb in true_bps:
                # find which bucket
                for b in buckets:
                    if b[0] <= tb < b[1]:
                        bucket_total[b] += 1
                        # detect: any peak within +/- TOLERANCE
                        if len(peaks):
                            d = np.abs(peaks - tb)
                            if d.min() <= TOLERANCE:
                                bucket_hit[b] += 1
                        break
        print(f"  {'bucket':>22s}  {'total':>6s}  {'hit':>6s}  {'rate':>6s}")
        for b in buckets:
            tot = bucket_total[b]
            hit = bucket_hit[b]
            rate = hit / tot if tot else 0.0
            mark = ' <-- past train ceiling' if b[0] >= 19525 else ''
            print(f"  [{b[0]:>5d}, {b[1]:>5d})    {tot:>6d}  {hit:>6d}  {rate:>6.3f}{mark}")

    # Also: sanity per-position hit rate as a curve
    print("\n=== Per-1000bp BP detection rate (threshold=0.6) ===")
    bucket_size = 1000
    pos_total = np.zeros(MAX_SEQ_LEN // bucket_size + 1, dtype=int)
    pos_hit = np.zeros_like(pos_total)
    for i in range(len(meta_test)):
        m = meta_test[i]
        true_bps = [m['bp_start'], m['bp_end']]
        peaks, _ = find_peaks(y_pred[i], height=0.6, distance=TOLERANCE)
        for tb in true_bps:
            b = tb // bucket_size
            pos_total[b] += 1
            if len(peaks) and np.abs(peaks - tb).min() <= TOLERANCE:
                pos_hit[b] += 1
    print(f"{'pos_kbp':>8s}  {'total':>6s}  {'hit':>6s}  {'rate':>6s}")
    for b in range(len(pos_total)):
        if pos_total[b] > 0:
            print(f"  {b}-{b+1:<3d}  {pos_total[b]:>6d}  {pos_hit[b]:>6d}  {pos_hit[b]/pos_total[b]:>6.3f}")


if __name__ == '__main__':
    main()
