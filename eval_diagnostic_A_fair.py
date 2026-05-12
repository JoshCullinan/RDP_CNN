#!/usr/bin/env python3
"""Diagnostic (A): Fair eval on truly held-out UnseenTestSet subset.

Subset = UnseenTestSet − (run41_train ∪ run42c_train) → 11 files.

For each model (#38, #41, #42c, #42c-diag, #43):
  - Load matching cache (by input channel count)
  - Subset events to the 11 held-out files
  - Sweep threshold @ EB=0 and EB=200
  - Report honest F1 (EB=200, threshold from sweep)

Also report each model on FULL UnseenTestSet for reference.
"""
import sys, glob, pickle, json, gc, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

TOLERANCE = 200
MAX_SEQ_LEN = 32000

# Held-out subset
HELDOUT_FILE = Path('splits/honest_eval_subset_11.txt')

MODELS = [
    ('run38',       'models_test/cnn_breakpoint_run38_final.keras'),
    ('run41',       'models_test/cnn_breakpoint_run41_final.keras'),
    ('run42c',      'models_test/cnn_breakpoint_run42c_final.keras'),
    ('run42c_diag', 'models_test/cnn_breakpoint_run42c_diag_final.keras'),
    ('run43',       'models_test/cnn_breakpoint_run43_final.keras'),
]

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EBS = [0, 200]


def pick_cache_for_channels(n_channels):
    """Find a UnseenTestSet cache with matching channel count."""
    for p in sorted(glob.glob('cache/ds_UnseenTestSet_*.npz'), reverse=True):
        try:
            with np.load(p) as z:
                if z['X'].shape[-1] == n_channels:
                    return p
        except Exception:
            continue
    return None


def _squeeze(y_pred):
    if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
        return y_pred[..., 0]
    return y_pred


def streaming_predict(model, X):
    n_ch = X.shape[-1]
    sig = tf.TensorSpec(shape=(MAX_SEQ_LEN, n_ch), dtype=tf.float16)
    def gen():
        for i in range(len(X)):
            yield X[i]
    with tf.device('/CPU:0'):
        ds = (tf.data.Dataset.from_generator(gen, output_signature=sig)
              .batch(2)
              .map(lambda x: tf.cast(x, tf.float32))
              .prefetch(2))
    return model.predict(ds, verbose=0)


def evaluate(y_pred, X, meta, threshold, edge_buffer):
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
    return dict(precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn,
                events_both=both, events_one=one, events_missed=miss)


def main():
    heldout = set(Path(line.strip()).name for line in HELDOUT_FILE.read_text().splitlines() if line.strip())
    print(f"Held-out subset: N={len(heldout)} files")

    results = {}
    for tag, model_path in MODELS:
        print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] {tag}: {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        n_ch = model.input_shape[-1]
        print(f"  params={model.count_params():,}  n_channels={n_ch}")

        cache = pick_cache_for_channels(n_ch)
        if cache is None:
            print(f"  !! NO CACHE for n_channels={n_ch} — skipping")
            results[tag] = {'error': f'no cache for {n_ch}ch'}
            continue
        print(f"  cache: {cache}")

        pkl = cache.replace('.npz', '.pkl')
        d = np.load(cache)
        X_full = d['X'].astype(np.float16)
        with open(pkl, 'rb') as f:
            meta_full = pickle.load(f)
        del d; gc.collect()
        print(f"  full cache: N_events={len(meta_full)}, X.shape={X_full.shape}")

        # Subset by file name
        idx_held = [i for i, m in enumerate(meta_full) if m['file'] in heldout]
        if not idx_held:
            print(f"  !! No events match held-out subset")
            results[tag] = {'error': 'no events in subset'}
            continue
        idx_held = np.array(idx_held)
        X_sub = X_full[idx_held]
        meta_sub = [meta_full[i] for i in idx_held]
        files_sub = sorted(set(m['file'] for m in meta_sub))
        print(f"  held-out subset: N_events={len(meta_sub)}, N_files={len(files_sub)}")

        # Predict on full (we want both full and subset)
        print(f"  predicting on FULL ({len(X_full)} events)...")
        y_full = streaming_predict(model, X_full)
        print(f"  predicting on SUBSET ({len(X_sub)} events)...")
        y_sub = streaming_predict(model, X_sub)

        full_res = {}
        sub_res = {}
        for eb in EBS:
            for thr in THRESHOLDS:
                full_res[f'eb={eb}_thr={thr}'] = evaluate(y_full, X_full, meta_full, thr, eb)
                sub_res[f'eb={eb}_thr={thr}']  = evaluate(y_sub,  X_sub,  meta_sub,  thr, eb)

        # Best F1 across thresholds at EB=200 (honest)
        best_thr_full_200 = max(THRESHOLDS, key=lambda t: full_res[f'eb=200_thr={t}']['f1'])
        best_thr_sub_200  = max(THRESHOLDS, key=lambda t: sub_res[f'eb=200_thr={t}']['f1'])
        best_thr_full_0   = max(THRESHOLDS, key=lambda t: full_res[f'eb=0_thr={t}']['f1'])
        best_thr_sub_0    = max(THRESHOLDS, key=lambda t: sub_res[f'eb=0_thr={t}']['f1'])

        print(f"\n  FULL ({len(meta_full)} events):")
        print(f"    EB=0   best F1={full_res[f'eb=0_thr={best_thr_full_0}']['f1']:.4f} @ thr={best_thr_full_0}")
        print(f"    EB=200 best F1={full_res[f'eb=200_thr={best_thr_full_200}']['f1']:.4f} @ thr={best_thr_full_200}")
        print(f"  SUBSET ({len(meta_sub)} events, {len(files_sub)} files):")
        print(f"    EB=0   best F1={sub_res[f'eb=0_thr={best_thr_sub_0}']['f1']:.4f} @ thr={best_thr_sub_0}")
        print(f"    EB=200 best F1={sub_res[f'eb=200_thr={best_thr_sub_200}']['f1']:.4f} @ thr={best_thr_sub_200}")

        results[tag] = {
            'model': model_path,
            'n_channels': n_ch,
            'cache': cache,
            'full_n_events': len(meta_full),
            'sub_n_events': len(meta_sub),
            'sub_n_files': len(files_sub),
            'full_results': full_res,
            'sub_results': sub_res,
            'best_full_eb0': {'thr': best_thr_full_0, **full_res[f'eb=0_thr={best_thr_full_0}']},
            'best_full_eb200': {'thr': best_thr_full_200, **full_res[f'eb=200_thr={best_thr_full_200}']},
            'best_sub_eb0': {'thr': best_thr_sub_0, **sub_res[f'eb=0_thr={best_thr_sub_0}']},
            'best_sub_eb200': {'thr': best_thr_sub_200, **sub_res[f'eb=200_thr={best_thr_sub_200}']},
        }

        del X_full, X_sub, y_full, y_sub, meta_full, meta_sub, model
        gc.collect()

    out_path = Path('results_diagnostic_A_fair.json')
    with out_path.open('w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # Print summary table
    print(f"\n{'='*80}\nSUMMARY TABLE\n{'='*80}")
    print(f"{'model':<14s} {'N_full':>7s} {'F1_full_honest':>15s} {'N_sub':>7s} {'F1_sub_honest':>14s} {'sub_files':>10s}")
    for tag, r in results.items():
        if 'error' in r:
            print(f"  {tag:<14s} ERROR: {r['error']}")
            continue
        print(f"  {tag:<14s} {r['full_n_events']:>7d} {r['best_full_eb200']['f1']:>15.4f} "
              f"{r['sub_n_events']:>7d} {r['best_sub_eb200']['f1']:>14.4f} {r['sub_n_files']:>10d}")


if __name__ == '__main__':
    main()
