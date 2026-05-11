#!/usr/bin/env python3
"""Eval run #29 model with edge suppression. Smaller sweep than the 6x7 grid
to save the ~10-min Python loop time. Reports val + test honest F1.

Saves a JSON summary to results_run29.json so the next iteration can read it
without re-running.
"""
import numpy as np
import tensorflow as tf
import glob, pickle, gc, json, sys, time
from scipy.signal import find_peaks

MAX_SEQ_LEN = 32000
TOLERANCE = 200
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else 'models_test/cnn_breakpoint_run29_final.keras'
RUN_TAG = sys.argv[2] if len(sys.argv) > 2 else 'run29'


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


def _squeeze(y_pred):
    """U-Net outputs (n, L, 1); dilated stack outputs (n, L). Normalise to (n, L)."""
    if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
        return y_pred[..., 0]
    return y_pred


def evaluate_with_eb(y_pred, X, meta, threshold, edge_buffer):
    y_pred = _squeeze(y_pred)
    tp = fp = fn = 0
    both = one = miss = 0
    for i, m in enumerate(meta):
        content_mask = (X[i].astype(np.float32).sum(axis=-1) > 0)
        content_end = int(content_mask.sum())
        if content_end == 0:
            miss += 1; fn += 2
            continue
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
                fn += 1
        fp += len(peaks) - len(used)
        if n_match == 2: both += 1
        elif n_match == 1: one += 1
        else: miss += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        'precision': p, 'recall': r, 'f1': f,
        'tp': tp, 'fp': fp, 'fn': fn,
        'events_both': both, 'events_one': one, 'events_missed': miss,
    }


def main():
    print(f"[{time.strftime('%H:%M:%S')}] loading model {MODEL_NAME}")
    model = tf.keras.models.load_model(MODEL_NAME, compile=False)
    print(f"  ok. {model.count_params():,} params")

    # Pick caches whose channel count matches the model's expected input.
    # Multiple cache files can coexist if MAXCHI_WINDOWS changed across runs.
    n_channels_expected = model.input_shape[-1]
    def pick_cache(pattern):
        for p in sorted(glob.glob(pattern), reverse=True):
            try:
                with np.load(p) as z:
                    if z['X'].shape[-1] == n_channels_expected:
                        return p
            except Exception:
                continue
        # fallback: latest
        return sorted(glob.glob(pattern))[-1]
    val_npz = pick_cache('cache/ds_XML-5_*.npz')
    val_pkl = val_npz.replace('.npz', '.pkl')
    test_npz = pick_cache('cache/ds_UnseenTestSet_*.npz')
    test_pkl = test_npz.replace('.npz', '.pkl')
    print(f"  using val cache:  {val_npz}")
    print(f"  using test cache: {test_npz}")

    print(f"\n[{time.strftime('%H:%M:%S')}] VAL eval (EB sweep)")
    d = np.load(val_npz)
    X_val = d['X'].astype(np.float16)
    with open(val_pkl, 'rb') as f:
        meta_val = pickle.load(f)
    del d; gc.collect()
    y_val_pred = streaming_predict(model, X_val)

    EBs = [0, 200]
    THRs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    val_results = {}
    print(f"\n  {'EB':>4s}  {'thr':>4s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  both/one/miss")
    for eb in EBs:
        for thr in THRs:
            res = evaluate_with_eb(y_val_pred, X_val, meta_val, thr, eb)
            val_results[f'eb={eb}_thr={thr}'] = res
            print(f"  {eb:>4d}  {thr:>4.2f}  {res['precision']:>6.3f}  {res['recall']:>6.3f}  {res['f1']:>6.3f}  {res['events_both']}/{res['events_one']}/{res['events_missed']}")

    # Pick val-best at each EB
    val_best = {}
    for eb in EBs:
        best = max((val_results[f'eb={eb}_thr={t}'] for t in THRs), key=lambda r: r['f1'])
        val_best[eb] = best
    print(f"\n  Val-best @ EB=0:   F1={val_best[0]['f1']:.3f}  thr=({[t for t in THRs if val_results[f'eb=0_thr={t}']['f1']==val_best[0]['f1']][0]})")
    print(f"  Val-best @ EB=200: F1={val_best[200]['f1']:.3f}  thr=({[t for t in THRs if val_results[f'eb=200_thr={t}']['f1']==val_best[200]['f1']][0]})")

    val_best_thr_eb0 = [t for t in THRs if val_results[f'eb=0_thr={t}']['f1']==val_best[0]['f1']][0]
    val_best_thr_eb200 = [t for t in THRs if val_results[f'eb=200_thr={t}']['f1']==val_best[200]['f1']][0]
    del X_val, y_val_pred; gc.collect()

    print(f"\n[{time.strftime('%H:%M:%S')}] TEST eval (EB sweep)")
    d = np.load(test_npz)
    X_test = d['X'].astype(np.float16)
    with open(test_pkl, 'rb') as f:
        meta_test = pickle.load(f)
    del d; gc.collect()
    y_test_pred = streaming_predict(model, X_test)

    test_results = {}
    print(f"\n  {'EB':>4s}  {'thr':>4s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  both/one/miss")
    for eb in EBs:
        for thr in THRs:
            res = evaluate_with_eb(y_test_pred, X_test, meta_test, thr, eb)
            test_results[f'eb={eb}_thr={thr}'] = res
            print(f"  {eb:>4d}  {thr:>4.2f}  {res['precision']:>6.3f}  {res['recall']:>6.3f}  {res['f1']:>6.3f}  {res['events_both']}/{res['events_one']}/{res['events_missed']}")

    # Apply val-selected threshold (per EB) to test
    test_at_val_thr_eb0 = test_results[f'eb=0_thr={val_best_thr_eb0}']
    test_at_val_thr_eb200 = test_results[f'eb=200_thr={val_best_thr_eb200}']

    print(f"\n=== HEADLINE COMPARISON vs RDP5 (test F1=0.367) ===")
    print(f"  RAW (EB=0)            test F1 @ val-thr {val_best_thr_eb0}:   {test_at_val_thr_eb0['f1']:.3f}  P={test_at_val_thr_eb0['precision']:.3f}  R={test_at_val_thr_eb0['recall']:.3f}")
    print(f"  HONEST (EB=200)       test F1 @ val-thr {val_best_thr_eb200}: {test_at_val_thr_eb200['f1']:.3f}  P={test_at_val_thr_eb200['precision']:.3f}  R={test_at_val_thr_eb200['recall']:.3f}")
    print(f"  HONEST best test F1 across thresholds @ EB=200: {max(test_results[f'eb=200_thr={t}']['f1'] for t in THRs):.3f}")

    summary = {
        'run_tag': RUN_TAG,
        'model': MODEL_NAME,
        'rdp5_baseline_f1': 0.367,
        'val_results': val_results,
        'test_results': test_results,
        'val_best_thr_eb0': val_best_thr_eb0,
        'val_best_thr_eb200': val_best_thr_eb200,
        'test_at_val_thr_eb0': test_at_val_thr_eb0,
        'test_at_val_thr_eb200': test_at_val_thr_eb200,
        'test_best_f1_eb200': max(test_results[f'eb=200_thr={t}']['f1'] for t in THRs),
    }
    out = f'results_{RUN_TAG}.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
