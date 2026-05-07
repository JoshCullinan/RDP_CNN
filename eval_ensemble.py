#!/usr/bin/env python3
"""Ensemble evaluation. Average per-position predictions across multiple
trained models, then run threshold/edge-suppression sweep.

Models that have plateaued individually at ~0.17 honest F1 may have
orthogonal failure modes — ensembling can lift F1 by 1-5% if so.
"""
import numpy as np
import tensorflow as tf
import glob, pickle, gc, json, sys, time
from scipy.signal import find_peaks

MAX_SEQ_LEN = 32000
TOLERANCE = 200

MODELS = [
    'models_test/cnn_breakpoint_run28d_final.keras',
    'models_test/cnn_breakpoint_run29_final.keras',
    'models_test/cnn_breakpoint_run32_final.keras',
    'models_test/cnn_breakpoint_run34_final.keras',
]


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


def squeeze_pred(y):
    return y[..., 0] if (y.ndim == 3 and y.shape[-1] == 1) else y


def evaluate(y_pred, X, meta, threshold, edge_buffer):
    tp = fp = fn = both = one = miss = 0
    for i, m in enumerate(meta):
        cm = (X[i].astype(np.float32).sum(axis=-1) > 0)
        ce = int(cm.sum())
        if ce == 0:
            miss += 1; fn += 2; continue
        pred = y_pred[i].copy()
        if edge_buffer > 0:
            pred[:edge_buffer] = 0.0
            if ce > edge_buffer:
                pred[ce - edge_buffer: ce] = 0.0
        pred[ce:] = 0.0
        true = [m['bp_start'], m['bp_end']]
        peaks, _ = find_peaks(pred, height=threshold, distance=TOLERANCE)
        used = set(); n_match = 0
        for tb in true:
            best = None; bd = TOLERANCE + 1
            for j, p in enumerate(peaks):
                if j in used: continue
                d = abs(int(p) - int(tb))
                if d <= TOLERANCE and d < bd:
                    bd = d; best = j
            if best is not None:
                used.add(best); tp += 1; n_match += 1
            else:
                fn += 1
        fp += len(peaks) - len(used)
        if n_match == 2: both += 1
        elif n_match == 1: one += 1
        else: miss += 1
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    f = 2*p*r/(p+r) if (p+r) else 0.0
    return {'P':p,'R':r,'F1':f,'both':both,'one':one,'miss':miss}


def main():
    val_npz = sorted(glob.glob('cache/ds_XML-5_*.npz'))[-1]
    test_npz = sorted(glob.glob('cache/ds_UnseenTestSet_*.npz'))[-1]
    val_pkl = val_npz.replace('.npz', '.pkl')
    test_pkl = test_npz.replace('.npz', '.pkl')

    print(f"[{time.strftime('%H:%M:%S')}] loading val + test caches")
    X_val = np.load(val_npz)['X'].astype(np.float16)
    X_test = np.load(test_npz)['X'].astype(np.float16)
    with open(val_pkl, 'rb') as f: meta_val = pickle.load(f)
    with open(test_pkl, 'rb') as f: meta_test = pickle.load(f)
    print(f"  X_val: {X_val.shape}  X_test: {X_test.shape}")

    val_preds = []
    test_preds = []
    for mp in MODELS:
        print(f"\n[{time.strftime('%H:%M:%S')}] loading {mp}")
        m = tf.keras.models.load_model(mp, compile=False)
        print(f"  predicting val ...")
        yv = squeeze_pred(streaming_predict(m, X_val))
        val_preds.append(yv)
        print(f"  predicting test ...")
        yt = squeeze_pred(streaming_predict(m, X_test))
        test_preds.append(yt)
        del m; gc.collect(); tf.keras.backend.clear_session()

    print(f"\n[{time.strftime('%H:%M:%S')}] ensemble: averaging {len(val_preds)} models")
    y_val_ens = np.mean(np.stack(val_preds, axis=0), axis=0)
    y_test_ens = np.mean(np.stack(test_preds, axis=0), axis=0)
    del val_preds, test_preds; gc.collect()

    EBs = [0, 200]
    THRs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n=== ENSEMBLE VAL sweep ===")
    print(f"  {'EB':>4s} {'thr':>4s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  both/one/miss")
    val_results = {}
    for eb in EBs:
        for thr in THRs:
            r = evaluate(y_val_ens, X_val, meta_val, thr, eb)
            val_results[f'eb={eb}_thr={thr}'] = r
            print(f"  {eb:>4d} {thr:>4.2f}  {r['P']:.3f}  {r['R']:.3f}  {r['F1']:.3f}  {r['both']}/{r['one']}/{r['miss']}")
    val_best = {eb: max((val_results[f'eb={eb}_thr={t}'] for t in THRs), key=lambda x: x['F1']) for eb in EBs}
    val_best_thr = {eb: [t for t in THRs if val_results[f'eb={eb}_thr={t}']['F1']==val_best[eb]['F1']][0] for eb in EBs}

    print("\n=== ENSEMBLE TEST sweep ===")
    print(f"  {'EB':>4s} {'thr':>4s}  {'P':>6s}  {'R':>6s}  {'F1':>6s}  both/one/miss")
    test_results = {}
    for eb in EBs:
        for thr in THRs:
            r = evaluate(y_test_ens, X_test, meta_test, thr, eb)
            test_results[f'eb={eb}_thr={thr}'] = r
            print(f"  {eb:>4d} {thr:>4.2f}  {r['P']:.3f}  {r['R']:.3f}  {r['F1']:.3f}  {r['both']}/{r['one']}/{r['miss']}")

    print(f"\n=== HEADLINE vs RDP5 (test F1=0.367) ===")
    print(f"  Val-best EB=0:   F1={val_best[0]['F1']:.3f}  thr={val_best_thr[0]}")
    print(f"  Val-best EB=200: F1={val_best[200]['F1']:.3f}  thr={val_best_thr[200]}")
    print(f"  TEST RAW (EB=0)    @ val-thr {val_best_thr[0]}:   F1={test_results[f'eb=0_thr={val_best_thr[0]}']['F1']:.3f}")
    print(f"  TEST HONEST (EB=200) @ val-thr {val_best_thr[200]}: F1={test_results[f'eb=200_thr={val_best_thr[200]}']['F1']:.3f}")
    print(f"  TEST HONEST best @ EB=200 across thresholds: F1={max(test_results[f'eb=200_thr={t}']['F1'] for t in THRs):.3f}")

    out = 'results_ensemble.json'
    summary = {
        'models': MODELS,
        'val_results': val_results,
        'test_results': test_results,
        'val_best_thr_eb200': val_best_thr[200],
        'rdp5_baseline_f1': 0.367,
    }
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
