#!/usr/bin/env python3
"""Run all three advisor-recommended diagnostics on run #39 model:

1. Channel-utilization check: zero method-conf channels 24-32, recompute val F1.
   If F1 barely changes, the model never learned to use them.
2. Per-content-length bucket detection rate (val + test). If failure
   concentrates in long-content buckets, A5 synthetic concatenation is the fix.
3. RDP-firing coverage: stratify test detection by whether RDP5 fired (PredBPStart
   present) vs not. If accuracy is high only when RDP fired, model is parroting RDP.

Usage: python3 diag_run39.py
"""
import numpy as np
import tensorflow as tf
import glob, pickle, json, sys, time
from scipy.signal import find_peaks

MAX_SEQ_LEN = 32000
TOLERANCE = 200
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'models_test/cnn_breakpoint_run39_final.keras'

RDP_BLOCK_START = 22  # consensus channels start
RDP_BLOCK_END = 33    # 9 method scalars end
METHOD_CHANNELS_START = 24  # just the 9 method scalars (consensus 22,23 separate)
METHOD_CHANNELS_END = 33


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


def evaluate(y_pred, X, meta, threshold=0.7, edge_buffer=200):
    """Honest F1 + per-event detection: 0/1/2 BPs found."""
    if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]
    tp = fp = fn = 0
    both = one = miss = 0
    near_misses = []  # |peak_pos - true_bp| for close-but-no-cigar
    detected_per_event = []
    for i, m in enumerate(meta):
        content_mask = (X[i].astype(np.float32).sum(axis=-1) > 0)
        content_end = int(content_mask.sum())
        if content_end == 0:
            miss += 1; fn += 2; detected_per_event.append(0); continue
        pred = y_pred[i].copy()
        if edge_buffer > 0:
            pred[:edge_buffer] = 0.0
            if content_end > edge_buffer:
                pred[content_end - edge_buffer: content_end] = 0.0
        pred[content_end:] = 0.0
        true_bps = [m['bp_start'], m['bp_end']]
        peaks, _ = find_peaks(pred, height=threshold, distance=TOLERANCE)
        used = set()
        n_match = 0
        for tb in true_bps:
            best = None; best_d = TOLERANCE + 1
            for j, p in enumerate(peaks):
                if j in used: continue
                d = abs(int(p) - int(tb))
                if d <= TOLERANCE and d < best_d:
                    best_d = d; best = j
            if best is not None:
                used.add(best); tp += 1; n_match += 1
            else:
                # Find the closest peak even if outside tolerance, for near-miss stat
                if len(peaks) > 0:
                    closest = int(peaks[np.argmin(np.abs(peaks - int(tb)))])
                    near_misses.append(abs(closest - int(tb)))
                fn += 1
        fp += len(peaks) - len(used)
        if n_match == 2: both += 1
        elif n_match == 1: one += 1
        else: miss += 1
        detected_per_event.append(n_match)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        'precision': p, 'recall': r, 'f1': f,
        'both': both, 'one': one, 'miss': miss,
        'detected_per_event': detected_per_event,
        'near_misses': near_misses,
    }


def main():
    print(f"[{time.strftime('%H:%M:%S')}] loading model {MODEL}")
    model = tf.keras.models.load_model(MODEL, compile=False)
    n_ch = model.input_shape[-1]
    print(f"  ok. {model.count_params():,} params, {n_ch} input channels")

    val_npz = next(p for p in sorted(glob.glob('cache/ds_XML-5_*.npz'), reverse=True)
                   if np.load(p)['X'].shape[-1] == n_ch)
    val_pkl = val_npz.replace('.npz', '.pkl')
    test_npz = next(p for p in sorted(glob.glob('cache/ds_UnseenTestSet_*.npz'), reverse=True)
                    if np.load(p)['X'].shape[-1] == n_ch)
    test_pkl = test_npz.replace('.npz', '.pkl')
    print(f"  val cache:  {val_npz}")
    print(f"  test cache: {test_npz}")

    # --- DIAGNOSTIC 1: Channel utilization ---
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC 1: Channel utilization (zero method-conf channels)")
    print(f"{'='*60}")
    d = np.load(val_npz)
    X_val = d['X'].astype(np.float16)
    with open(val_pkl, 'rb') as f:
        meta_val = pickle.load(f)
    print(f"  X_val.shape={X_val.shape}, n_events={len(meta_val)}")

    print(f"  Predicting baseline...")
    y_pred_baseline = streaming_predict(model, X_val)
    res_baseline = evaluate(y_pred_baseline, X_val, meta_val)
    print(f"  BASELINE val F1 (EB=200, thr=0.7): {res_baseline['f1']:.3f}")

    print(f"  Predicting with channels {METHOD_CHANNELS_START}-{METHOD_CHANNELS_END} zeroed...")
    X_zero = X_val.copy()
    X_zero[:, :, METHOD_CHANNELS_START:METHOD_CHANNELS_END] = 0
    y_pred_zero = streaming_predict(model, X_zero)
    del X_zero
    res_zero = evaluate(y_pred_zero, X_val, meta_val)
    print(f"  ZERO-METHOD-CHANNELS val F1: {res_zero['f1']:.3f}")
    delta = res_baseline['f1'] - res_zero['f1']
    print(f"  Delta (baseline - zeroed): {delta:+.3f}")
    if abs(delta) < 0.02:
        print(f"  --> Method channels carry NEGLIGIBLE signal. Model ignored them.")
    elif delta > 0.02:
        print(f"  --> Method channels are DOING WORK ({delta:.3f} F1 lift). Model uses them.")
    else:
        print(f"  --> Zeroing IMPROVED F1 ({-delta:.3f}). Channels are HARMFUL — overfitting?")

    # --- DIAGNOSTIC 2: Per-bucket detection rate (test) ---
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC 2: Per-content-length bucket detection rate (TEST)")
    print(f"{'='*60}")
    d2 = np.load(test_npz)
    X_test = d2['X'].astype(np.float16)
    with open(test_pkl, 'rb') as f:
        meta_test = pickle.load(f)
    print(f"  X_test.shape={X_test.shape}, n_events={len(meta_test)}")

    print(f"  Predicting test...")
    y_pred_test = streaming_predict(model, X_test)
    res_test = evaluate(y_pred_test, X_test, meta_test)
    print(f"  Overall test F1 (EB=200, thr=0.7): {res_test['f1']:.3f}")

    # Bucket by content length
    content_lens = np.array([m['actual_len'] for m in meta_test])
    buckets = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000),
               (20000, 25000), (25000, 32000)]
    print(f"\n  Bucket   | n events | both% | one%  | miss% | mean_detected")
    print(f"  ---------|----------|-------|-------|-------|---------------")
    for lo, hi in buckets:
        in_bucket = (content_lens >= lo) & (content_lens < hi)
        n = int(in_bucket.sum())
        if n == 0:
            print(f"  {lo:5d}-{hi:5d} |        0 |       |       |       |")
            continue
        det = np.array(res_test['detected_per_event'])[in_bucket]
        both_pct = (det == 2).mean() * 100
        one_pct = (det == 1).mean() * 100
        miss_pct = (det == 0).mean() * 100
        mean_det = det.mean()
        print(f"  {lo:5d}-{hi:5d} | {n:8d} | {both_pct:4.1f}% | {one_pct:4.1f}% | {miss_pct:4.1f}% | {mean_det:.3f}")

    # --- DIAGNOSTIC 3: RDP-firing coverage ---
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC 3: RDP-firing coverage (TEST)")
    print(f"{'='*60}")
    # Channel 22 = consensus PredBPStart Gaussian. If it has any non-zero, RDP fired.
    rdp_fired = np.array([X_test[i, :, 22].max() > 0 for i in range(len(meta_test))])
    print(f"  Test events with RDP fired (PredBPStart non-null): {rdp_fired.sum()}/{len(meta_test)} ({100*rdp_fired.mean():.1f}%)")
    det = np.array(res_test['detected_per_event'])
    for label, mask in [('RDP fired', rdp_fired), ('RDP did NOT fire', ~rdp_fired)]:
        if mask.sum() == 0:
            continue
        n = int(mask.sum())
        d_sub = det[mask]
        both_pct = (d_sub == 2).mean() * 100
        one_pct = (d_sub == 1).mean() * 100
        miss_pct = (d_sub == 0).mean() * 100
        print(f"  {label} (n={n}): both={both_pct:.1f}% one={one_pct:.1f}% miss={miss_pct:.1f}% (mean_det={d_sub.mean():.3f})")

    # Near-miss distribution (peaks within 2x tolerance but outside 1x)
    print(f"\n{'='*60}")
    print(f"AUX: Near-miss distance distribution (test)")
    print(f"{'='*60}")
    nm = np.array(res_test['near_misses'])
    if len(nm) > 0:
        for cap in [200, 400, 800, 1600, 3200, 6400]:
            print(f"  |peak - true| <= {cap:5d}: {(nm <= cap).sum():5d} / {len(nm):5d}  ({100*(nm <= cap).mean():.1f}%)")
        print(f"  median: {int(np.median(nm))}, p75: {int(np.percentile(nm, 75))}, p90: {int(np.percentile(nm, 90))}")

    # Save
    summary = {
        'model': MODEL,
        'val_baseline_f1': res_baseline['f1'],
        'val_zero_method_f1': res_zero['f1'],
        'val_method_channel_lift': delta,
        'test_f1': res_test['f1'],
        'test_n_events': len(meta_test),
        'test_buckets': [
            {
                'lo': lo, 'hi': hi,
                'n': int(((content_lens >= lo) & (content_lens < hi)).sum()),
                'both_pct': float(((np.array(res_test['detected_per_event'])[
                    (content_lens >= lo) & (content_lens < hi)]) == 2).mean()) if ((content_lens >= lo) & (content_lens < hi)).sum() else 0,
            }
            for lo, hi in buckets
        ],
        'rdp_fired_n': int(rdp_fired.sum()),
        'rdp_fired_both_pct': float((det[rdp_fired] == 2).mean() * 100) if rdp_fired.sum() else 0,
        'rdp_not_fired_n': int((~rdp_fired).sum()),
        'rdp_not_fired_both_pct': float((det[~rdp_fired] == 2).mean() * 100) if (~rdp_fired).sum() else 0,
    }
    with open('diag_run39.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved diag_run39.json")


if __name__ == '__main__':
    main()
