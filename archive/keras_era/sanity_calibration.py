#!/usr/bin/env python3
"""Sanity check before run #29: is the val→test logit-magnitude gap driven by
MaxChi feature stats, or by padding/position effects upstream of MaxChi?

Approach:
1. Load cnn_breakpoint_final.keras.
2. From cached val (XML-5, content ≤ 11k bp, lots of padding to 32k),
   pick samples and run predict.
3. From cached test (UnseenTestSet, content up to 30k bp, less padding),
   pick samples and run predict.
4. Compare the distribution of predicted probabilities at NON-PADDED positions
   only (mask via channel-sum > 0). If test predictions at non-padded positions
   are systematically smaller-magnitude than val at non-padded positions, the
   calibration drift is real and the next question is: is it MaxChi stats, or
   something else?

Cheap supplementary test: take ONE val sample, artificially extend its content
length by repeating the last few hundred bp (no real biology, just test-shape),
and compare predictions at the original positions. If logit magnitude shifts
meaningfully when only padding changes, the model is sensitive to padding/length
in a way that MaxChi normalization alone won't fix.
"""
import numpy as np
import tensorflow as tf
import glob, pickle, gc

MAX_SEQ_LEN = 32000


def load_one(npz_path, n_samples=20):
    d = np.load(npz_path)
    X = d['X'][:n_samples].astype(np.float16)
    return X


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


def summarize(name, X, y_pred):
    # Per-sample mask = channel-sum > 0 (non-padded positions)
    mask = (X.astype(np.float32).sum(axis=-1) > 0)  # (n, MAX_SEQ_LEN)
    # Padded predictions
    pad_preds = []
    nonpad_preds = []
    for i in range(len(X)):
        m = mask[i]
        nonpad_preds.append(y_pred[i, m])
        pad_preds.append(y_pred[i, ~m])
    nonpad = np.concatenate(nonpad_preds)
    pad = np.concatenate(pad_preds)
    print(f"\n=== {name} ===")
    print(f"  samples: {len(X)}")
    print(f"  content lengths: mean={mask.sum(axis=1).mean():.0f}  min={mask.sum(axis=1).min()}  max={mask.sum(axis=1).max()}")
    print(f"  predictions on NON-PADDED positions ({len(nonpad):,} positions):")
    print(f"    mean={nonpad.mean():.4f}  median={np.median(nonpad):.4f}  std={nonpad.std():.4f}")
    print(f"    P50={np.percentile(nonpad,50):.4f}  P90={np.percentile(nonpad,90):.4f}  P99={np.percentile(nonpad,99):.4f}  max={nonpad.max():.4f}")
    print(f"    fraction > 0.5:  {(nonpad>0.5).mean():.4f}")
    print(f"    fraction > 0.9:  {(nonpad>0.9).mean():.4f}")
    print(f"  predictions on PADDED positions ({len(pad):,} positions):")
    print(f"    mean={pad.mean():.4f}  median={np.median(pad):.4f}  std={pad.std():.4f}  max={pad.max():.4f}")
    return nonpad


def main():
    print("Loading model...")
    model = tf.keras.models.load_model('models_test/cnn_breakpoint_final.keras', compile=False)
    print(f"  ok. {model.count_params():,} params\n")

    val_npz = sorted(glob.glob('cache/ds_XML-5_*.npz'))[-1]
    test_npz = sorted(glob.glob('cache/ds_UnseenTestSet_*.npz'))[-1]

    # Use first 50 samples of each — enough to characterize the distribution
    print(f"Loading val (50 samples) from {val_npz}")
    X_val = load_one(val_npz, n_samples=50)
    print(f"Loading test (50 samples) from {test_npz}")
    X_test = load_one(test_npz, n_samples=50)

    print("\nPredicting val...")
    y_val = streaming_predict(model, X_val)
    print("Predicting test...")
    y_test = streaming_predict(model, X_test)

    val_nonpad = summarize("VAL", X_val, y_val)
    test_nonpad = summarize("TEST", X_test, y_test)

    print(f"\n=== HEADLINE COMPARISON ===")
    print(f"  val P90 prediction:  {np.percentile(val_nonpad, 90):.4f}")
    print(f"  test P90 prediction: {np.percentile(test_nonpad, 90):.4f}")
    print(f"  val mean:  {val_nonpad.mean():.4f}")
    print(f"  test mean: {test_nonpad.mean():.4f}")
    if np.percentile(test_nonpad, 90) < np.percentile(val_nonpad, 90) * 0.7:
        print("  → Test predictions ARE systematically smaller. Calibration drift confirmed.")
    else:
        print("  → Test predictions NOT meaningfully smaller. Calibration drift hypothesis weaker.")


if __name__ == '__main__':
    main()
