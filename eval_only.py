#!/usr/bin/env python3
"""Standalone eval pass: load saved model, predict on cached val (XML-5)
and on UnseenTestSet, print F1 numbers.

Skips the 30-min training step. Used after run #28d crashed in cell-26
post-training — final model was on disk, we just need numbers from it.

Run under a memory cap (parent script wraps this in systemd-run --scope):
    systemd-run --user --scope -p MemoryMax=18G \\
        bash -c 'source .venv/bin/activate && python3 eval_only.py'

The from_generator pipeline streams batch-by-batch — never materializes
a whole-array tf.constant on CPU or GPU.
"""
from __future__ import annotations
import sys, time, gc, pickle, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import find_peaks

# --- Inline helpers extracted from CNN.ipynb (cells 1, 3, 6, 10, 11, 16, 35) ---

DATA_ROOT = Path('dataRaw')
TEST_DIR = 'UnseenTestSet'
MAX_SEQ_LEN = 32000
NUCLEOTIDES = ['A', 'T', 'G', 'C', '-']
N_CHANNELS = len(NUCLEOTIDES)
N_MAXCHI_WINDOWS = 4
MAXCHI_WINDOWS = (50, 100, 200, 500)
N_INPUT_CHANNELS = 3 * N_CHANNELS + 3 + N_MAXCHI_WINDOWS  # 22
LABEL_SIGMA = 20
BP_WINDOW = 10
TOLERANCE = 200
BATCH_SIZE = 2

MODEL_PATH = Path('models_test/cnn_breakpoint_final.keras')


# ---------- evaluate_peaks (cell-16) ----------
def evaluate_peaks(y_pred, meta, tolerance, threshold):
    """Peak-based F1 evaluation. y_pred: (n, MAX_SEQ_LEN). meta: list of dicts
    with bp_start, bp_end. Returns (precision, recall, f1)."""
    tp = fp = fn = 0
    for i, m in enumerate(meta):
        true_bps = [m['bp_start'], m['bp_end']]
        peaks, _ = find_peaks(y_pred[i], height=threshold, distance=tolerance)
        matched = set()
        used = set()
        # greedy nearest-first match
        for tb in true_bps:
            best = None
            best_d = tolerance + 1
            for j, p in enumerate(peaks):
                if j in used: continue
                d = abs(int(p) - int(tb))
                if d <= tolerance and d < best_d:
                    best_d = d; best = j
            if best is not None:
                used.add(best); matched.add(tb)
                tp += 1
            else:
                fn += 1
        fp += len(peaks) - len(used)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ---------- build_event_df (cell-35) ----------
def build_event_df(y_pred, meta, threshold):
    rows = []
    for i, m in enumerate(meta):
        peaks, _ = find_peaks(y_pred[i], height=threshold, distance=TOLERANCE)
        true_bps = [m['bp_start'], m['bp_end']]
        used = set()
        matched_bps = []
        for tb in true_bps:
            best = None; best_d = TOLERANCE + 1
            for j, p in enumerate(peaks):
                if j in used: continue
                d = abs(int(p) - int(tb))
                if d <= TOLERANCE and d < best_d:
                    best_d = d; best = j
            if best is not None:
                used.add(best); matched_bps.append(tb)
        rows.append({
            'sample': i,
            'true_bps': true_bps,
            'n_predicted': len(peaks),
            'n_matched': len(matched_bps),
        })
    return pd.DataFrame(rows)


def print_event_summary(events_df, title):
    n_total = len(events_df)
    both = (events_df['n_matched'] == 2).sum()
    one = (events_df['n_matched'] == 1).sum()
    zero = (events_df['n_matched'] == 0).sum()
    print('=' * 60)
    print(title)
    print('=' * 60)
    print(f"Total events:    {n_total}")
    print(f"Both BPs found:  {both}  ({100*both/n_total:.1f}%)")
    print(f"One BP found:    {one}  ({100*one/n_total:.1f}%)")
    print(f"Missed (0):      {zero}  ({100*zero/n_total:.1f}%)")
    print('=' * 60)


# ---------- one_hot / encode_triplet / maxchi (cell-6) ----------
def one_hot_encode(sequence, max_length=MAX_SEQ_LEN):
    nuc_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
    encoded = np.zeros((max_length, N_CHANNELS), dtype=np.float32)
    s = sequence[:max_length].upper()
    for i, nuc in enumerate(s):
        encoded[i, nuc_idx.get(nuc, 4)] = 1.0
    return encoded


def _seq_to_index(sequence, max_length=MAX_SEQ_LEN):
    nuc_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
    idx = np.full(max_length, -1, dtype=np.int16)
    s = sequence[:max_length].upper()
    for i, nuc in enumerate(s):
        idx[i] = nuc_idx.get(nuc, 4)
    return idx


def _maxchi_features(parental_signal, windows=MAXCHI_WINDOWS):
    L = parental_signal.shape[0]
    feats = np.zeros((L, len(windows)), dtype=np.float32)
    csum = np.concatenate([[0.0], np.cumsum(parental_signal)])
    for j, w in enumerate(windows):
        for p in range(L):
            left_a = max(0, p - w); left_b = p
            right_a = p; right_b = min(L, p + w)
            left_mean = (csum[left_b] - csum[left_a]) / max(1, (left_b - left_a))
            right_mean = (csum[right_b] - csum[right_a]) / max(1, (right_b - right_a))
            feats[p, j] = right_mean - left_mean
    return feats


def encode_triplet(seq_recomb, seq_parent1, seq_parent2):
    enc_r = one_hot_encode(seq_recomb)
    enc_1 = one_hot_encode(seq_parent1)
    enc_2 = one_hot_encode(seq_parent2)
    r = _seq_to_index(seq_recomb)
    p1 = _seq_to_index(seq_parent1)
    p2 = _seq_to_index(seq_parent2)
    valid = (r >= 0) & (p1 >= 0) & (p2 >= 0)
    match_p1 = ((r == p1) & valid).astype(np.float32)[:, None]
    match_p2 = ((r == p2) & valid).astype(np.float32)[:, None]
    informative = ((p1 != p2) & valid).astype(np.float32)[:, None]
    parental_signal = (match_p1 - match_p2).reshape(-1)
    maxchi = _maxchi_features(parental_signal)
    return np.concatenate(
        [enc_r, enc_1, enc_2, match_p1, match_p2, informative, maxchi],
        axis=-1,
    ).astype(np.float32)


# ---------- parse_simulation (cell-10, with index_col=False fix) ----------
from Bio import SeqIO


def parse_simulation(fasta_path):
    fasta_path = Path(fasta_path)
    sim_csv = fasta_path.parent / f"{fasta_path.name}SimVSRealCompare.csv"
    stats_csv = fasta_path.parent / f"{fasta_path.name}RecombIdentifyStats.csv"
    if not sim_csv.exists() or not stats_csv.exists():
        return []
    try:
        seqs = {int(r.id): str(r.seq) for r in SeqIO.parse(fasta_path, 'fasta')}
    except Exception:
        return []
    try:
        sim_df = pd.read_csv(sim_csv, skipinitialspace=True, index_col=False)
        stats_df = pd.read_csv(stats_csv, skipinitialspace=True, index_col=False)
    except Exception:
        return []
    triplets = []
    for _, row in sim_df.iterrows():
        recomb_id = int(row['ActualRecomb'])
        bp_start = int(row['SimBPStart'])
        bp_end = int(row['SimBPEnd'])
        event = row.get('Event', 1)
        evt_rows = stats_df[stats_df['Event'] == event]
        if evt_rows.empty:
            continue
        iseqs_list = []
        for _, er in evt_rows.iterrows():
            iseqs = er.get('ISeqs(A)', '')
            if isinstance(iseqs, str):
                iseqs_list.extend([int(x.strip()) for x in iseqs.split(';') if x.strip().isdigit()])
        candidates = [i for i in set(iseqs_list) if i in seqs and i != recomb_id]
        if len(candidates) < 2:
            continue
        p1_id, p2_id = candidates[0], candidates[1]
        if recomb_id not in seqs:
            continue
        X = encode_triplet(seqs[recomb_id], seqs[p1_id], seqs[p2_id])
        actual_len = max(len(seqs[recomb_id]), len(seqs[p1_id]), len(seqs[p2_id]))
        triplets.append({
            'X': X,
            'meta': {
                'file': fasta_path.name,
                'event': event,
                'recomb_id': recomb_id,
                'parent1_id': p1_id,
                'parent2_id': p2_id,
                'bp_start': bp_start,
                'bp_end': bp_end,
                'actual_len': actual_len,
            },
        })
    return triplets


# ---------- streaming predict via from_generator ----------
def predict_streaming(model, X_array):
    """Run model.predict over X_array via from_generator → batched .map cast → predict.

    Avoids both CPU-side from_tensor_slices doubling and GPU eager-copy of the
    full array. X_array can be fp16 or fp32; output is np.float32.
    """
    n = len(X_array)
    n_chan = X_array.shape[-1]

    def gen():
        for i in range(n):
            yield X_array[i]

    sig = tf.TensorSpec(shape=(MAX_SEQ_LEN, n_chan), dtype=X_array.dtype)
    with tf.device('/CPU:0'):
        ds = (tf.data.Dataset.from_generator(gen, output_signature=sig)
              .batch(BATCH_SIZE)
              .map(lambda x: tf.cast(x, tf.float32),
                   num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))
    return model.predict(ds, verbose=1)


# ---------- main ----------
def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading model from {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"  ok. params={model.count_params():,}  elapsed={time.time()-t0:.1f}s")

    # ---- VAL ----
    val_npz = sorted(glob.glob('cache/ds_XML-5_*.npz'))[-1]
    val_pkl = val_npz.replace('.npz', '.pkl')
    print(f"\n[{time.strftime('%H:%M:%S')}] loading val cache {val_npz}")
    d = np.load(val_npz)
    X_val = d['X'].astype(np.float16)  # 4.12 GB
    print(f"  X_val.shape={X_val.shape} dtype={X_val.dtype} nbytes={X_val.nbytes/1e9:.2f} GB")
    with open(val_pkl, 'rb') as f:
        meta_val = pickle.load(f)
    print(f"  meta_val len={len(meta_val)}")
    del d; gc.collect()

    print(f"\n[{time.strftime('%H:%M:%S')}] predicting val (streaming)")
    t1 = time.time()
    y_val_pred = predict_streaming(model, X_val)
    print(f"  done in {time.time()-t1:.1f}s. y_val_pred.shape={y_val_pred.shape}")
    print(f"  range=[{y_val_pred.min():.4f}, {y_val_pred.max():.4f}]")
    del X_val; gc.collect()

    # ---- THRESHOLD SWEEP ON VAL ----
    print(f"\n[{time.strftime('%H:%M:%S')}] threshold sweep on val")
    print('=' * 60)
    print(f"Peak-based evaluation (+/-{TOLERANCE} bp tolerance)")
    print('=' * 60)
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    best_thr = 0.5
    best_f1 = -1.0
    rows = []
    for thr in thresholds:
        p, r, f = evaluate_peaks(y_val_pred, meta_val, TOLERANCE, thr)
        rows.append({'threshold': thr, 'precision': p, 'recall': r, 'f1': f})
        print(f"  threshold={thr:.2f}  P={p:.3f}  R={r:.3f}  F1={f:.3f}")
        if f > best_f1:
            best_f1 = f; best_thr = thr
    print(f"\nBest val threshold: {best_thr} (F1={best_f1:.3f})")

    # ---- EVENT-LEVEL ON VAL ----
    print()
    events_val = build_event_df(y_val_pred, meta_val, threshold=best_thr)
    print_event_summary(
        events_val,
        f"VAL EVENT-LEVEL DETECTION (threshold={best_thr}, tolerance=+/-{TOLERANCE} bp)",
    )

    # ---- TEST ----
    # Read from canonical cache (built by cache_test_set.py which exec's
    # the notebook's parser). Earlier in-script parser had a set()-cast
    # ordering bug that scrambled parent IDs and produced garbage triplets.
    test_caches = sorted(glob.glob('cache/ds_UnseenTestSet_*.npz'))
    if not test_caches:
        print("\nERROR: no UnseenTestSet cache file found. Run cache_test_set.py first.")
        return
    test_npz = test_caches[-1]
    test_pkl = test_npz.replace('.npz', '.pkl')
    print(f"\n[{time.strftime('%H:%M:%S')}] loading test cache {test_npz}")
    d_t = np.load(test_npz)
    X_test = d_t['X'].astype(np.float16)
    print(f"  X_test.shape={X_test.shape} dtype={X_test.dtype} nbytes={X_test.nbytes/1e9:.2f} GB")
    with open(test_pkl, 'rb') as f:
        meta_test = pickle.load(f)
    print(f"  meta_test len={len(meta_test)}")
    del d_t; gc.collect()

    print(f"\n[{time.strftime('%H:%M:%S')}] predicting test (streaming)")
    t2 = time.time()
    y_test_pred = predict_streaming(model, X_test)
    print(f"  done in {time.time()-t2:.1f}s. y_test_pred.shape={y_test_pred.shape}")

    p_t, r_t, f_t = evaluate_peaks(y_test_pred, meta_test, TOLERANCE, best_thr)

    # also sweep on test
    print('\n=== UNSEEN TEST SET (peak-based, +/-200 bp) ===')
    for thr in thresholds:
        p, r, f = evaluate_peaks(y_test_pred, meta_test, TOLERANCE, thr)
        marker = ' <-- val-best' if thr == best_thr else ''
        print(f"  threshold={thr:.2f}  P={p:.3f}  R={r:.3f}  F1={f:.3f}{marker}")
    print()

    print('=' * 60)
    print('UNSEEN TEST SET PERFORMANCE (peak-based, val-selected threshold)')
    print('=' * 60)
    print(f"Precision: {p_t:.3f}")
    print(f"Recall:    {r_t:.3f}")
    print(f"F1 Score:  {f_t:.3f}")
    print(f"Threshold: {best_thr}  |  Tolerance: +/-{TOLERANCE} bp")
    print('=' * 60)

    events_test = build_event_df(y_test_pred, meta_test, threshold=best_thr)
    print_event_summary(
        events_test,
        f"TEST EVENT-LEVEL DETECTION (threshold={best_thr}, tolerance=+/-{TOLERANCE} bp)",
    )

    print(f"\n[total] {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
