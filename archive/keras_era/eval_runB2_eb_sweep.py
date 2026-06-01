#!/usr/bin/env python3
"""EB sweep for runB2 on UnseenTestSet (full + 11-file subset).

Resolves the audit's edge-failure ambiguity: at the deployment-relevant
11-file subset, does F1 lift substantially when EB is relaxed? If yes,
the headline 0.408 is harness-pessimistic; if no, the model has a
genuine edge blindspot.

Plan:
  1. Load runB2 model (cache: ds_UnseenTestSet_c2217f407a2ba195.npz, 33ch)
  2. Predict ONCE with channels 15:22 zeroed per-sample in the generator
  3. Compute content_end then drop X
  4. Sweep EB x threshold on y_pred (cheap numpy crunching)
  5. Save JSON + print 2D table
"""
import json, pickle, time, gc, sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"WARN: memory_growth({gpu}): {e}", flush=True)

TOLERANCE = 200
MAX_SEQ_LEN = 32000
MODEL_PATH = 'models_test/cnn_breakpoint_runB2_no_rdp_channels_final.keras'
CACHE_PATH = 'cache/ds_UnseenTestSet_c2217f407a2ba195.npz'
PKL_PATH   = 'cache/ds_UnseenTestSet_c2217f407a2ba195.pkl'
HELDOUT_FILE = Path('splits/honest_eval_subset_11.txt')
OUT_JSON = Path('results_runB2_eb_sweep.json')

EBS = [0, 25, 50, 100, 150, 200, 300, 500]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ZERO_LO, ZERO_HI = 15, 22  # B2 train-time channel mask


def _squeeze(y_pred):
    if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
        return y_pred[..., 0]
    return y_pred


def streaming_predict(model, X, batch=2, zero_lo=ZERO_LO, zero_hi=ZERO_HI):
    """Predict with channels [zero_lo, zero_hi) zeroed per-sample in the
    generator (no full X.copy() — saves ~12 GB)."""
    n_ch = X.shape[-1]
    sig = tf.TensorSpec(shape=(MAX_SEQ_LEN, n_ch), dtype=tf.float16)
    def gen():
        for i in range(len(X)):
            xi = X[i].copy()
            xi[:, zero_lo:zero_hi] = 0.0
            yield xi
    with tf.device('/CPU:0'):
        ds = (tf.data.Dataset.from_generator(gen, output_signature=sig)
              .batch(batch)
              .map(lambda x: tf.cast(x, tf.float32))
              .prefetch(2))
    return model.predict(ds, verbose=0)


def evaluate_from_preds(y_pred, content_end, meta, threshold, edge_buffer):
    """Mirror of eval_one_model.evaluate_from_preds — greedy nearest-first,
    TOLERANCE=200, find_peaks with distance=TOLERANCE."""
    y_pred = _squeeze(y_pred)
    tp = fp = fn = 0
    both = one = miss = 0
    for i, m in enumerate(meta):
        ce = int(content_end[i])
        if ce == 0:
            miss += 1; fn += 2
            continue
        pred = y_pred[i].copy()
        if edge_buffer > 0:
            pred[:edge_buffer] = 0.0
            if ce > edge_buffer:
                pred[ce - edge_buffer: ce] = 0.0
        pred[ce:] = 0.0
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
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL_PATH}", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    n_ch = model.input_shape[-1]
    print(f"  params={model.count_params():,}  n_channels={n_ch}", flush=True)
    assert n_ch == 33, f"expected 33-channel runB2 model, got {n_ch}"

    print(f"  cache: {CACHE_PATH}", flush=True)
    d = np.load(CACHE_PATH)
    X_raw = d['X']
    X_full = X_raw if X_raw.dtype == np.float16 else X_raw.astype(np.float16)
    with open(PKL_PATH, 'rb') as f:
        meta_full = pickle.load(f)
    d.close()
    print(f"  full cache: N={len(meta_full)}, X.shape={X_full.shape}, dtype={X_full.dtype}, X.bytes={X_full.nbytes/1e9:.2f}GB", flush=True)

    print(f"  computing content_end...", flush=True)
    content_end = np.array(
        [int((X_full[i].astype(np.float32).sum(axis=-1) > 0).sum()) for i in range(len(X_full))],
        dtype=np.int32,
    )

    idx_held = np.array([i for i, m in enumerate(meta_full) if m['file'] in heldout])
    files_sub = sorted(set(meta_full[i]['file'] for i in idx_held))
    print(f"  subset: N={len(idx_held)}, files={len(files_sub)}", flush=True)

    print(f"  predicting on FULL with channels [{ZERO_LO},{ZERO_HI}) zeroed...", flush=True)
    t0 = time.time()
    y_full = streaming_predict(model, X_full)
    print(f"  predict took {time.time()-t0:.0f}s, y.shape={y_full.shape}, y.bytes={y_full.nbytes/1e9:.2f}GB", flush=True)

    del X_full
    gc.collect()
    print(f"  dropped X_full", flush=True)

    y_sub = y_full[idx_held]
    ce_sub = content_end[idx_held]
    meta_sub = [meta_full[i] for i in idx_held]

    full_res = {}; sub_res = {}
    print(f"\n  sweeping EB x THR ({len(EBS)} x {len(THRESHOLDS)} = {len(EBS)*len(THRESHOLDS)} configs per slice)...", flush=True)
    t0 = time.time()
    for eb in EBS:
        for thr in THRESHOLDS:
            key = f'eb={eb}_thr={thr}'
            full_res[key] = evaluate_from_preds(y_full, content_end, meta_full, thr, eb)
            sub_res[key]  = evaluate_from_preds(y_sub,  ce_sub,       meta_sub,  thr, eb)
        print(f"    eb={eb} done ({time.time()-t0:.0f}s elapsed)", flush=True)

    # Compact 2D table
    print(f"\n{'='*72}", flush=True)
    print(f"  EB SWEEP — runB2 on UnseenTestSet (TOLERANCE=200, channels 15:22 zeroed)", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'EB':>5} | {'FULL F1':>8} {'thr':>4} | {'SUB F1':>8} {'thr':>4}  (N_full=5539, N_sub={len(idx_held)})", flush=True)
    print(f"  {'-'*5}-+-{'-'*8}-{'-'*4}-+-{'-'*8}-{'-'*4}", flush=True)
    best_per_eb = {}
    for eb in EBS:
        b_full = max(THRESHOLDS, key=lambda t: full_res[f'eb={eb}_thr={t}']['f1'])
        b_sub  = max(THRESHOLDS, key=lambda t: sub_res[f'eb={eb}_thr={t}']['f1'])
        f_full = full_res[f'eb={eb}_thr={b_full}']['f1']
        f_sub  = sub_res[f'eb={eb}_thr={b_sub}']['f1']
        marker = '  <-- current "honest" EB' if eb == 200 else ''
        print(f"  {eb:>5} | {f_full:>8.4f} {b_full:>4.2f} | {f_sub:>8.4f} {b_sub:>4.2f}{marker}", flush=True)
        best_per_eb[eb] = {
            'best_full': {'thr': b_full, **full_res[f'eb={eb}_thr={b_full}']},
            'best_sub':  {'thr': b_sub,  **sub_res[f'eb={eb}_thr={b_sub}']},
        }
    print(f"{'='*72}", flush=True)

    out = {
        'model': MODEL_PATH, 'n_channels': int(n_ch), 'cache': CACHE_PATH,
        'zero_channels': [ZERO_LO, ZERO_HI],
        'full_n_events': len(meta_full),
        'sub_n_events': int(len(idx_held)),
        'sub_n_files': len(files_sub),
        'ebs': EBS, 'thresholds': THRESHOLDS,
        'full_results': full_res, 'sub_results': sub_res,
        'best_per_eb': best_per_eb,
    }
    with OUT_JSON.open('w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  saved: {OUT_JSON}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
