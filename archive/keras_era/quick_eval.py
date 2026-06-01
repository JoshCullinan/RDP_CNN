#!/usr/bin/env python3
"""Re-run threshold sweep + test evaluation against the cached run #7 model.
Avoids a full retrain by loading models_test/cnn_breakpoint_best.keras.
Replicates the project parsing/eval logic verbatim from the notebook.
"""
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from Bio import SeqIO
from scipy.signal import find_peaks


DATA_ROOT = Path("dataRaw")
TRAIN_DIRS = ["XML-1", "XML-2", "XML-3", "XML-4", "XML-5"]
TEST_DIR = "UnseenTestSet"
MAX_SEQ_LEN = 10000
N_CHANNELS = 5
TOLERANCE = 200
BATCH_SIZE = 16
VAL_SPLIT = 0.15
LABEL_SIGMA = 20
BP_WINDOW = 10


def one_hot_encode(sequence, max_length=MAX_SEQ_LEN):
    nuc_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
    encoded = np.zeros((max_length, N_CHANNELS), dtype=np.float32)
    for i, nuc in enumerate(sequence[:max_length].upper()):
        idx = nuc_idx.get(nuc, 4)
        encoded[i, idx] = 1.0
    return encoded


def _seq_to_index(sequence, max_length=MAX_SEQ_LEN):
    nuc_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
    idx = np.full(max_length, -1, dtype=np.int16)
    for i, nuc in enumerate(sequence[:max_length].upper()):
        idx[i] = nuc_idx.get(nuc, 4)
    return idx


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
    return np.concatenate([enc_r, enc_1, enc_2, match_p1, match_p2, informative], axis=1)


def generate_labels(bp_start, bp_end, seq_length=MAX_SEQ_LEN, mode='gaussian',
                    window=BP_WINDOW, sigma=LABEL_SIGMA):
    labels = np.zeros(seq_length, dtype=np.float32)
    if mode == 'gaussian':
        pos = np.arange(seq_length, dtype=np.float32)
        for bp in [bp_start, bp_end]:
            peak = np.exp(-0.5 * ((pos - bp) / sigma) ** 2)
            labels = np.maximum(labels, peak.astype(np.float32))
    elif mode == 'breakpoint':
        for bp in [bp_start, bp_end]:
            lo = max(0, bp - window)
            hi = min(seq_length, bp + window + 1)
            labels[lo:hi] = 1.0
    return labels


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
        sim = pd.read_csv(sim_csv, skipinitialspace=True)
        stats = pd.read_csv(stats_csv, skipinitialspace=True)
    except Exception:
        return []
    results = []
    for _, row in sim.iterrows():
        event = row['RDPEvent']
        recomb_id = int(row['ActualRecomb'])
        bp_start = int(row['SimBPStart'])
        bp_end = int(row['SimBPEnd'])
        ev_rows = stats[stats['Event'] == event]
        if len(ev_rows) != 3:
            continue
        parent_ids = []
        for _, sr in ev_rows.iterrows():
            ids = [int(s.strip()) for s in str(sr['ISeqs(A)']).split('$')
                   if s.strip().isdigit()]
            if recomb_id in ids:
                continue
            if ids:
                parent_ids.append(ids[0])
        if len(parent_ids) < 2:
            continue
        if not all(sid in seqs for sid in [recomb_id, parent_ids[0], parent_ids[1]]):
            continue
        triplet = encode_triplet(seqs[recomb_id], seqs[parent_ids[0]], seqs[parent_ids[1]])
        results.append({
            'input': triplet,
            'labels_gaussian': generate_labels(bp_start, bp_end, mode='gaussian'),
            'labels_bp': generate_labels(bp_start, bp_end, mode='breakpoint'),
            'meta': {
                'file': fasta_path.name,
                'event': event,
                'recomb_id': recomb_id,
                'bp_start': bp_start,
                'bp_end': bp_end,
                'actual_len': len(seqs[recomb_id].rstrip('-')),
            },
        })
    return results


def load_dataset(directories, label_mode='gaussian', max_files=None):
    label_keys = {'gaussian': 'labels_gaussian', 'breakpoint': 'labels_bp'}
    key = label_keys[label_mode]
    inputs, labels, meta = [], [], []
    for d in directories:
        fa_files = sorted((DATA_ROOT / d).glob("*.fa"))
        if max_files:
            fa_files = fa_files[:max_files]
        print(f"  {d}: {len(fa_files)} files", flush=True)
        for fa in fa_files:
            for t in parse_simulation(fa):
                inputs.append(t['input'])
                labels.append(t[key])
                meta.append(t['meta'])
    X = np.array(inputs, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    mask = (X.sum(axis=-1) > 0).astype(np.float32)
    return X, y, mask, meta


def detect_peaks(y_pred, threshold=0.5, min_distance=TOLERANCE):
    peaks, _ = find_peaks(y_pred, height=threshold, distance=min_distance)
    return peaks


def peak_metrics_single(true_bps, y_pred, tolerance=TOLERANCE, threshold=0.5):
    peaks = detect_peaks(y_pred, threshold=threshold)
    true_bps = list(true_bps)
    if len(peaks) == 0:
        return 0.0, 0.0, 0.0, 0, 0
    pairs = sorted(((abs(int(p) - int(t)), pi, ti)
                    for pi, p in enumerate(peaks)
                    for ti, t in enumerate(true_bps)),
                   key=lambda x: x[0])
    matched_pred, matched_true = set(), set()
    for d, pi, ti in pairs:
        if d > tolerance:
            break
        if pi in matched_pred or ti in matched_true:
            continue
        matched_pred.add(pi)
        matched_true.add(ti)
    tp = len(matched_pred)
    fp = len(peaks) - tp
    fn = len(true_bps) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2*p*r/(p+r)) if (p + r) > 0 else 0.0
    return p, r, f, len(peaks), tp


def evaluate_peaks(y_pred_batch, meta_batch, tolerance, threshold):
    metrics = []
    for yp, m in zip(y_pred_batch, meta_batch):
        true_bps = [m['bp_start'], m['bp_end']]
        p, r, f, _, _ = peak_metrics_single(true_bps, yp, tolerance, threshold)
        metrics.append((p, r, f))
    p, r, f = zip(*metrics)
    return float(np.mean(p)), float(np.mean(r)), float(np.mean(f))


def event_breakdown(y_pred_batch, meta_batch, threshold):
    n_both = n_one = n_missed = 0
    peaks_total = []
    fp_total = []
    for yp, m in zip(y_pred_batch, meta_batch):
        true_bps = [m['bp_start'], m['bp_end']]
        p, r, f, n_peaks, n_matched = peak_metrics_single(true_bps, yp, TOLERANCE, threshold)
        if n_matched == 2:
            n_both += 1
        elif n_matched == 1:
            n_one += 1
        else:
            n_missed += 1
        peaks_total.append(n_peaks)
        fp_total.append(n_peaks - n_matched)
    n_total = len(meta_batch)
    return {
        'n_total': n_total,
        'n_both': n_both,
        'n_one': n_one,
        'n_missed': n_missed,
        'pct_both': 100 * n_both / n_total,
        'pct_one': 100 * n_one / n_total,
        'pct_missed': 100 * n_missed / n_total,
        'mean_peaks': float(np.mean(peaks_total)),
        'mean_fp': float(np.mean(fp_total)),
    }


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    print(f"[{time.strftime('%H:%M:%S')}] loading training data (max_files=750)", flush=True)
    X_all, y_all, mask_all, meta_all = load_dataset(TRAIN_DIRS, label_mode='gaussian', max_files=750)
    print(f"  loaded {X_all.shape[0]} samples, X={X_all.shape}", flush=True)

    fnames = np.array([m['file'] for m in meta_all])
    unique = np.unique(fnames)
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    n_val_files = int(len(unique) * VAL_SPLIT)
    val_files = set(unique[:n_val_files].tolist())
    val_idx = np.array([i for i, f in enumerate(fnames) if f in val_files])
    X_val = X_all[val_idx]
    meta_val = [meta_all[i] for i in val_idx]
    print(f"  val: {X_val.shape[0]} samples (val files: {len(val_files)})", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading model", flush=True)
    cnn = tf.keras.models.load_model('models_test/cnn_breakpoint_best.keras', compile=False)
    print(f"  model loaded: {cnn.name}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] predicting on val", flush=True)
    y_val_pred = cnn.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    print(f"  pred shape: {y_val_pred.shape}, range [{y_val_pred.min():.4f}, {y_val_pred.max():.4f}]", flush=True)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    print(f"\n=== EXTENDED VAL THRESHOLD SWEEP (n={X_val.shape[0]}) ===", flush=True)
    print(f"{'thr':>5} {'P':>6} {'R':>6} {'F1':>6} {'Both%':>7} {'One%':>6} {'Miss%':>7} {'peaks':>7} {'fp':>6}", flush=True)
    val_results = []
    for thr in thresholds:
        p, r, f = evaluate_peaks(y_val_pred, meta_val, TOLERANCE, thr)
        eb = event_breakdown(y_val_pred, meta_val, thr)
        print(f"{thr:>5.2f} {p:>6.3f} {r:>6.3f} {f:>6.3f} {eb['pct_both']:>6.1f}% {eb['pct_one']:>5.1f}% "
              f"{eb['pct_missed']:>6.1f}% {eb['mean_peaks']:>7.2f} {eb['mean_fp']:>6.2f}", flush=True)
        val_results.append({'thr': thr, 'p': p, 'r': r, 'f1': f, **eb})

    val_df = pd.DataFrame(val_results)
    best_idx = val_df['f1'].idxmax()
    best_thr = val_df.loc[best_idx, 'thr']
    print(f"\n  Best val F1 threshold: {best_thr} (F1={val_df.loc[best_idx, 'f1']:.3f})", flush=True)
    best_both_idx = val_df['n_both'].idxmax()
    print(f"  Max val Both BPs threshold: {val_df.loc[best_both_idx, 'thr']} "
          f"(Both={val_df.loc[best_both_idx, 'n_both']}/{val_df.loc[best_both_idx, 'n_total']})", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] loading test set", flush=True)
    X_test, y_test, mask_test, meta_test = load_dataset([TEST_DIR], label_mode='breakpoint')
    print(f"  loaded {X_test.shape[0]} samples", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] predicting on test", flush=True)
    y_test_pred = cnn.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    print(f"  pred range [{y_test_pred.min():.4f}, {y_test_pred.max():.4f}]", flush=True)

    print(f"\n=== EXTENDED TEST THRESHOLD SWEEP (n={X_test.shape[0]}) ===", flush=True)
    print(f"{'thr':>5} {'P':>6} {'R':>6} {'F1':>6} {'Both%':>7} {'One%':>6} {'Miss%':>7} {'peaks':>7} {'fp':>6}", flush=True)
    for thr in thresholds:
        p, r, f = evaluate_peaks(y_test_pred, meta_test, TOLERANCE, thr)
        eb = event_breakdown(y_test_pred, meta_test, thr)
        print(f"{thr:>5.2f} {p:>6.3f} {r:>6.3f} {f:>6.3f} {eb['pct_both']:>6.1f}% {eb['pct_one']:>5.1f}% "
              f"{eb['pct_missed']:>6.1f}% {eb['mean_peaks']:>7.2f} {eb['mean_fp']:>6.2f}", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] done", flush=True)


if __name__ == '__main__':
    main()
