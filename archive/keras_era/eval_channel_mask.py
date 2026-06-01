#!/usr/bin/env python3
"""Diagnostic (B1): Inference-time channel mask on run #41.

Tests whether run #41 (33-channel model) is heavily reliant on the
hand-crafted RDP-style features (channels 15-21) or has learned the signal
in the conv stack itself.

Method:
  Load run41 model + 33-channel UnseenTestSet cache.
  Run inference with channels 15:22 zeroed (full + 11-file subset).
  Sweep threshold @ EB=200 (honest).
  Compare to run #41 baseline: 0.339 (full), 0.393 (subset).

Channel layout (CLAUDE.md "Three things easy to break"):
  0-4   recombinant one-hot (A/T/G/C/-)
  5-9   parent 1 one-hot
  10-14 parent 2 one-hot
  15-17 match_p1, match_p2, informative
  18-21 MaxChi disparity @ windows {50, 100, 200, 500}
  22-32 (run41-specific extra channels; preserved here)
"""
import sys, glob, pickle, json, gc, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

# Allow GPU memory to grow on demand instead of pre-grabbing the arena.
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"WARN: memory_growth({gpu}): {e}", flush=True)

TOLERANCE = 200
MAX_SEQ_LEN = 32000
HELDOUT_FILE = Path('splits/honest_eval_subset_11.txt')
MODEL_PATH = 'models_test/cnn_breakpoint_run41_final.keras'
MASK_CHANNELS = list(range(15, 22))   # channels 15-21 inclusive
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EBS = [0, 200]


def _peek_npz_channels(path):
    """Return last-dim of 'X' inside an .npz without loading it (~150 B)."""
    import zipfile
    from numpy.lib import format as npf
    with zipfile.ZipFile(path) as zf:
        target = 'X.npy' if 'X.npy' in zf.namelist() else next(
            (n for n in zf.namelist() if n.endswith('.npy')), None)
        if target is None:
            return None
        with zf.open(target) as f:
            version = npf.read_magic(f)
            shape, _f, _d = npf._read_array_header(f, version)
    return shape[-1]


def pick_cache_for_channels(n_channels):
    for p in sorted(glob.glob('cache/ds_UnseenTestSet_*.npz'), reverse=True):
        try:
            if _peek_npz_channels(p) == n_channels:
                return p
        except Exception as e:
            print(f"  WARN: peek {p} failed: {type(e).__name__}: {e}", flush=True)
            continue
    return None


def _squeeze(y_pred):
    if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
        return y_pred[..., 0]
    return y_pred


def streaming_predict(model, X, mask_channels=None):
    """Predict event-by-event. If mask_channels is given, zero those channels
    per-sample in the generator (no full X.copy() — saves ~12 GB on 33ch cache)."""
    n_ch = X.shape[-1]
    sig = tf.TensorSpec(shape=(MAX_SEQ_LEN, n_ch), dtype=tf.float16)
    mask_lo = mask_channels[0] if mask_channels else None
    mask_hi = mask_channels[-1] + 1 if mask_channels else None
    def gen():
        for i in range(len(X)):
            xi = X[i]
            if mask_channels is not None:
                xi = xi.copy()                  # one event (~2 MB), not 12 GB
                xi[:, mask_lo:mask_hi] = 0.0
            yield xi
    with tf.device('/CPU:0'):
        ds = (tf.data.Dataset.from_generator(gen, output_signature=sig)
              .batch(2)
              .map(lambda x: tf.cast(x, tf.float32))
              .prefetch(2))
    return model.predict(ds, verbose=0)


def evaluate_from_preds(y_pred, content_end, meta, threshold, edge_buffer):
    """Evaluate using only y_pred + per-event content_end (no X)."""
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
    print(f"Held-out subset: N={len(heldout)} files", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] Loading {MODEL_PATH}", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    n_ch = model.input_shape[-1]
    print(f"  params={model.count_params():,}  n_channels={n_ch}", flush=True)
    assert max(MASK_CHANNELS) < n_ch, f"mask channels {MASK_CHANNELS} exceed n_ch={n_ch}"

    cache = pick_cache_for_channels(n_ch)
    if cache is None:
        print(f"  !! NO CACHE for n_channels={n_ch}"); sys.exit(1)
    print(f"  cache: {cache}", flush=True)

    pkl = cache.replace('.npz', '.pkl')
    d = np.load(cache)
    X_full = d['X'].astype(np.float16)
    with open(pkl, 'rb') as f:
        meta_full = pickle.load(f)
    d.close()
    print(f"  full cache: N={len(meta_full)}, X.shape={X_full.shape}, X.bytes={X_full.nbytes/1e9:.2f}GB", flush=True)

    # Pre-compute content_end before we drop X, then drop X after both predicts.
    print(f"  computing content_end...", flush=True)
    content_end = np.array(
        [int((X_full[i].astype(np.float32).sum(axis=-1) > 0).sum()) for i in range(len(X_full))],
        dtype=np.int32,
    )

    idx_held = np.array([i for i, m in enumerate(meta_full) if m['file'] in heldout])
    meta_sub = [meta_full[i] for i in idx_held]
    files_sub = sorted(set(m['file'] for m in meta_sub))
    print(f"  subset: N={len(meta_sub)}, files={len(files_sub)}", flush=True)

    # Predict baseline (no mask). Generator yields X[i] unmodified.
    print(f"\n[{time.strftime('%H:%M:%S')}] BASELINE (no mask)", flush=True)
    t0 = time.time()
    y_full_base = streaming_predict(model, X_full, mask_channels=None)
    print(f"  baseline predict: {time.time()-t0:.0f}s", flush=True)

    # Predict masked. Generator zeros channels 15:22 per-sample (no full copy).
    print(f"\n[{time.strftime('%H:%M:%S')}] MASKED (ch {MASK_CHANNELS[0]}-{MASK_CHANNELS[-1]} zeroed per-sample)", flush=True)
    t0 = time.time()
    y_full_mask = streaming_predict(model, X_full, mask_channels=MASK_CHANNELS)
    print(f"  masked predict: {time.time()-t0:.0f}s", flush=True)

    # Drop X — eval only needs y_pred + content_end.
    del X_full
    gc.collect()
    print(f"  dropped X_full", flush=True)

    y_sub_base = y_full_base[idx_held]
    y_sub_mask = y_full_mask[idx_held]
    ce_sub = content_end[idx_held]

    results = {}
    for label, (yf, ys) in [
        ('baseline', (y_full_base, y_sub_base)),
        ('masked',   (y_full_mask, y_sub_mask)),
    ]:
        full_res = {}; sub_res = {}
        for eb in EBS:
            for thr in THRESHOLDS:
                full_res[f'eb={eb}_thr={thr}'] = evaluate_from_preds(yf, content_end, meta_full, thr, eb)
                sub_res[f'eb={eb}_thr={thr}']  = evaluate_from_preds(ys, ce_sub,       meta_sub,  thr, eb)
        b_full_0 = max(THRESHOLDS, key=lambda t: full_res[f'eb=0_thr={t}']['f1'])
        b_full_2 = max(THRESHOLDS, key=lambda t: full_res[f'eb=200_thr={t}']['f1'])
        b_sub_0  = max(THRESHOLDS, key=lambda t: sub_res[f'eb=0_thr={t}']['f1'])
        b_sub_2  = max(THRESHOLDS, key=lambda t: sub_res[f'eb=200_thr={t}']['f1'])

        print(f"\n  [{label}]")
        print(f"    FULL   EB=0   best F1={full_res[f'eb=0_thr={b_full_0}']['f1']:.4f} @ thr={b_full_0}")
        print(f"    FULL   EB=200 best F1={full_res[f'eb=200_thr={b_full_2}']['f1']:.4f} @ thr={b_full_2}")
        print(f"    SUBSET EB=0   best F1={sub_res[f'eb=0_thr={b_sub_0}']['f1']:.4f} @ thr={b_sub_0}")
        print(f"    SUBSET EB=200 best F1={sub_res[f'eb=200_thr={b_sub_2}']['f1']:.4f} @ thr={b_sub_2}", flush=True)

        results[label] = {
            'full_results': full_res, 'sub_results': sub_res,
            'best_full_eb0':  {'thr': b_full_0, **full_res[f'eb=0_thr={b_full_0}']},
            'best_full_eb200':{'thr': b_full_2, **full_res[f'eb=200_thr={b_full_2}']},
            'best_sub_eb0':   {'thr': b_sub_0,  **sub_res[f'eb=0_thr={b_sub_0}']},
            'best_sub_eb200': {'thr': b_sub_2,  **sub_res[f'eb=200_thr={b_sub_2}']},
        }

    out_path = Path('results_diagnostic_B1_channel_mask.json')
    with out_path.open('w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}", flush=True)

    # Headline table
    b_full = results['baseline']['best_full_eb200']['f1']
    m_full = results['masked']['best_full_eb200']['f1']
    b_sub  = results['baseline']['best_sub_eb200']['f1']
    m_sub  = results['masked']['best_sub_eb200']['f1']
    print(f"\n{'='*60}\nDIAGNOSTIC (B1) — channel-15-21 dependency\n{'='*60}")
    print(f"  FULL   honest F1: baseline={b_full:.4f}  masked={m_full:.4f}  drop={b_full-m_full:+.4f}")
    print(f"  SUBSET honest F1: baseline={b_sub:.4f}  masked={m_sub:.4f}  drop={b_sub-m_sub:+.4f}")
    print(f"\n  Decision rule (HANDOVER_DIAGNOSTICS.md):")
    print(f"    masked F1 ~0.05      → model heavily uses these channels")
    print(f"    masked F1 ~0.20-0.30 → partial reliance, CNN does some real learning")
    print(f"    masked F1 ~0.30+     → channels redundant, CNN learned signal itself")


if __name__ == '__main__':
    main()
