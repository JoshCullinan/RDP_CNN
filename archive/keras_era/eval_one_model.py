#!/usr/bin/env python3
"""Diagnostic (A): Eval ONE model on full UnseenTestSet + 11-file subset.

Designed to run as a subprocess (one model per python invocation) so that
when the process exits, the OS reclaims all RAM/GPU cleanly. This avoids
the WSL2 panic seen when looping 5 models in one long-lived process.

Memory plan:
  1. Load 33ch cache (peak ~11.7 GB)
  2. Predict full pass (transient ~+1 GB GPU)
  3. Pre-compute per-event content_end from X.sum(axis=-1) > 0
  4. **Drop X** — evaluate() only needs content_end after this
  5. Sweep thresholds × edge_buffers on y_pred (cheap)
  6. Save partial JSON, exit

Defensive flags:
  - tf memory growth (no pre-grab of GPU arena)
  - python heap cap via ulimit -v in the driver
"""
import argparse, glob, pickle, json, gc, time, sys
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
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EBS = [0, 200]


def _peek_npz_channels(path):
    """Return the last-dim of the 'X' array inside an .npz without loading it.

    Reads only the .npy header (~150 bytes), not the ~12 GB array — critical
    when running under a tight `ulimit -v`."""
    import zipfile
    from numpy.lib import format as npf
    with zipfile.ZipFile(path) as zf:
        # Pick X.npy (or first .npy if X missing — caches use 'X' key).
        target = 'X.npy' if 'X.npy' in zf.namelist() else next(
            (n for n in zf.namelist() if n.endswith('.npy')), None)
        if target is None:
            return None
        with zf.open(target) as f:
            version = npf.read_magic(f)
            shape, _fortran, _dtype = npf._read_array_header(f, version)
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


def streaming_predict(model, X, batch=2, zero_lo=None, zero_hi=None):
    """If zero_lo/zero_hi given, zero channels [zero_lo, zero_hi) per-sample
    in the generator (no full X.copy() — saves ~12 GB)."""
    n_ch = X.shape[-1]
    sig = tf.TensorSpec(shape=(MAX_SEQ_LEN, n_ch), dtype=tf.float16)
    mask = zero_lo is not None and zero_hi is not None
    def gen():
        for i in range(len(X)):
            xi = X[i]
            if mask:
                xi = xi.copy()
                xi[:, zero_lo:zero_hi] = 0.0
            yield xi
    with tf.device('/CPU:0'):
        ds = (tf.data.Dataset.from_generator(gen, output_signature=sig)
              .batch(batch)
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True, help='e.g. run43')
    ap.add_argument('--model', required=True, help='path to .keras checkpoint')
    ap.add_argument('--out-dir', default='results_diagnostic_A_partial')
    ap.add_argument('--zero-lo', type=int, default=None,
                    help='If set, zero channels [zero_lo, zero_hi) per-sample at inference. '
                         'Used to match a train-time mask (e.g., B2 trained with ch 15:22 zeroed).')
    ap.add_argument('--zero-hi', type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    per_json = out_dir / f'{args.tag}.json'
    if per_json.exists():
        print(f"[{time.strftime('%H:%M:%S')}] {args.tag}: cached at {per_json} — exit 0", flush=True)
        return 0

    heldout = set(Path(line.strip()).name for line in HELDOUT_FILE.read_text().splitlines() if line.strip())
    print(f"[{time.strftime('%H:%M:%S')}] {args.tag}: loading {args.model}", flush=True)
    model = tf.keras.models.load_model(args.model, compile=False)
    n_ch = model.input_shape[-1]
    print(f"  params={model.count_params():,}  n_channels={n_ch}", flush=True)

    cache = pick_cache_for_channels(n_ch)
    if cache is None:
        print(f"  !! NO CACHE for n_channels={n_ch}", flush=True)
        return 2
    print(f"  cache: {cache}", flush=True)

    pkl = cache.replace('.npz', '.pkl')
    d = np.load(cache)
    # Cache is already stored as float16; cast only if it isn't (avoid 11.7 GB
    # redundant copy that would blow ulimit -v).
    X_raw = d['X']
    X_full = X_raw if X_raw.dtype == np.float16 else X_raw.astype(np.float16)
    with open(pkl, 'rb') as f:
        meta_full = pickle.load(f)
    d.close()
    print(f"  full cache: N={len(meta_full)}, X.shape={X_full.shape}, dtype={X_full.dtype}, X.bytes={X_full.nbytes/1e9:.2f}GB", flush=True)

    # Pre-compute content_end before we drop X
    print(f"  computing content_end...", flush=True)
    content_end = np.array(
        [int((X_full[i].astype(np.float32).sum(axis=-1) > 0).sum()) for i in range(len(X_full))],
        dtype=np.int32,
    )

    idx_held = np.array([i for i, m in enumerate(meta_full) if m['file'] in heldout])
    files_sub = sorted(set(meta_full[i]['file'] for i in idx_held))
    print(f"  subset: N={len(idx_held)}, files={len(files_sub)}", flush=True)

    if args.zero_lo is not None and args.zero_hi is not None:
        print(f"  zero-mask channels [{args.zero_lo}, {args.zero_hi}) per-sample at inference", flush=True)
    print(f"  predicting on FULL ({len(X_full)} events)...", flush=True)
    t0 = time.time()
    y_full = streaming_predict(model, X_full, zero_lo=args.zero_lo, zero_hi=args.zero_hi)
    print(f"  predict took {time.time()-t0:.0f}s, y.shape={y_full.shape}, y.bytes={y_full.nbytes/1e9:.2f}GB", flush=True)

    # Drop X — we only need y_full and content_end from here on
    del X_full
    gc.collect()
    print(f"  dropped X_full", flush=True)

    # Subset predictions and metadata
    y_sub = y_full[idx_held]
    ce_sub = content_end[idx_held]
    meta_sub = [meta_full[i] for i in idx_held]

    full_res = {}; sub_res = {}
    for eb in EBS:
        for thr in THRESHOLDS:
            full_res[f'eb={eb}_thr={thr}'] = evaluate_from_preds(y_full, content_end, meta_full, thr, eb)
            sub_res[f'eb={eb}_thr={thr}']  = evaluate_from_preds(y_sub,  ce_sub,       meta_sub,  thr, eb)

    b_full_0 = max(THRESHOLDS, key=lambda t: full_res[f'eb=0_thr={t}']['f1'])
    b_full_2 = max(THRESHOLDS, key=lambda t: full_res[f'eb=200_thr={t}']['f1'])
    b_sub_0  = max(THRESHOLDS, key=lambda t: sub_res[f'eb=0_thr={t}']['f1'])
    b_sub_2  = max(THRESHOLDS, key=lambda t: sub_res[f'eb=200_thr={t}']['f1'])

    print(f"\n  FULL ({len(meta_full)} events):")
    print(f"    EB=0   best F1={full_res[f'eb=0_thr={b_full_0}']['f1']:.4f} @ thr={b_full_0}")
    print(f"    EB=200 best F1={full_res[f'eb=200_thr={b_full_2}']['f1']:.4f} @ thr={b_full_2}")
    print(f"  SUBSET ({len(meta_sub)} events, {len(files_sub)} files):")
    print(f"    EB=0   best F1={sub_res[f'eb=0_thr={b_sub_0}']['f1']:.4f} @ thr={b_sub_0}")
    print(f"    EB=200 best F1={sub_res[f'eb=200_thr={b_sub_2}']['f1']:.4f} @ thr={b_sub_2}", flush=True)

    out = {
        'model': args.model, 'n_channels': int(n_ch), 'cache': cache,
        'full_n_events': len(meta_full),
        'sub_n_events': int(len(idx_held)),
        'sub_n_files': len(files_sub),
        'full_results': full_res, 'sub_results': sub_res,
        'best_full_eb0':  {'thr': b_full_0, **full_res[f'eb=0_thr={b_full_0}']},
        'best_full_eb200':{'thr': b_full_2, **full_res[f'eb=200_thr={b_full_2}']},
        'best_sub_eb0':   {'thr': b_sub_0,  **sub_res[f'eb=0_thr={b_sub_0}']},
        'best_sub_eb200': {'thr': b_sub_2,  **sub_res[f'eb=200_thr={b_sub_2}']},
    }
    with per_json.open('w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  saved partial: {per_json}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
