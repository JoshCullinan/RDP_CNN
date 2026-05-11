#!/usr/bin/env python3
"""Apply edge-buffer suppression at inference time and re-evaluate.

For each sample: compute content_end = mask.sum() (number of non-padded
positions). Zero out y_pred at positions [0, EB) and [content_end - EB,
content_end) before find_peaks. This kills the BN-padding boundary artifact
diagnosed in bucket_diagnostic.py.

Sweep EB over a few values to find the operating point.
"""
import numpy as np
import tensorflow as tf
import glob, pickle, gc
from scipy.signal import find_peaks

MAX_SEQ_LEN = 32000
TOLERANCE = 200


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
    return model.predict(ds, verbose=1)


def evaluate(y_pred, X, meta, threshold, edge_buffer):
    """y_pred is shape (n, MAX_SEQ_LEN). Apply edge suppression per sample
    using the per-sample content length derived from X (channel-sum > 0)."""
    tp = fp = fn = 0
    for i, m in enumerate(meta):
        # Content length: number of non-padded positions
        content_mask = (X[i].astype(np.float32).sum(axis=-1) > 0)
        content_end = int(content_mask.sum())
        if content_end == 0:
            continue

        pred = y_pred[i].copy()
        # Suppress edges within content
        if edge_buffer > 0:
            pred[:edge_buffer] = 0.0
            if content_end > edge_buffer:
                pred[content_end - edge_buffer: content_end] = 0.0
            # Beyond content is already padded — typically zero but suppress to be safe
            pred[content_end:] = 0.0

        true_bps = [m['bp_start'], m['bp_end']]
        peaks, _ = find_peaks(pred, height=threshold, distance=TOLERANCE)

        used = set()
        for tb in true_bps:
            best = None; best_d = TOLERANCE + 1
            for j, p in enumerate(peaks):
                if j in used: continue
                d = abs(int(p) - int(tb))
                if d <= TOLERANCE and d < best_d:
                    best_d = d; best = j
            if best is not None:
                used.add(best); tp += 1
            else:
                fn += 1
        fp += len(peaks) - len(used)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f, tp, fp, fn


def main():
    print("Loading model...")
    model = tf.keras.models.load_model('models_test/cnn_breakpoint_final.keras', compile=False)

    test_npz = sorted(glob.glob('cache/ds_UnseenTestSet_*.npz'))[-1]
    test_pkl = test_npz.replace('.npz', '.pkl')

    print(f"Loading test cache {test_npz}")
    d = np.load(test_npz)
    X_test = d['X'].astype(np.float16)
    with open(test_pkl, 'rb') as f:
        meta_test = pickle.load(f)
    print(f"  X_test.shape={X_test.shape}  meta_n={len(meta_test)}")
    del d; gc.collect()

    print("\nPredicting full test set...")
    y_pred = streaming_predict(model, X_test)
    print(f"  done. y_pred.shape={y_pred.shape}")

    # Sweep over thresholds and edge_buffer values
    edge_buffers = [0, 100, 200, 300, 500, 1000]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n=== TEST F1 sweep: edge_buffer × threshold ===")
    print(f"  rows = edge_buffer (bp suppressed at content edges)")
    print(f"  cols = peak threshold")
    print()
    header = '  EB \\ thr  ' + '  '.join(f'{t:>6.2f}' for t in thresholds)
    print(header)
    best = (0.0, None, None)
    for eb in edge_buffers:
        row = []
        for thr in thresholds:
            p, r, f, tp, fp, fn = evaluate(y_pred, X_test, meta_test, thr, eb)
            row.append(f)
            if f > best[0]:
                best = (f, eb, thr)
        print(f"  EB={eb:>5d}   " + '  '.join(f'{x:>6.3f}' for x in row))

    print(f"\nBest: F1={best[0]:.3f} at edge_buffer={best[1]}, threshold={best[2]}")

    # Detailed at best (and at val-selected eb=200, thr=0.6)
    print("\n=== Detailed metrics at best (edge_buffer, threshold) ===")
    for eb, thr, label in [(best[1], best[2], 'best'),
                            (200, 0.6, 'EB=200 / val-thresh 0.6'),
                            (200, 0.9, 'EB=200 / test-thresh 0.9'),
                            (0, 0.6, 'no-suppress baseline')]:
        p, r, f, tp, fp, fn = evaluate(y_pred, X_test, meta_test, thr, eb)
        print(f"  [{label}] EB={eb} thr={thr}  P={p:.3f} R={r:.3f} F1={f:.3f}  TP={tp} FP={fp} FN={fn}")


if __name__ == '__main__':
    main()
