#!/usr/bin/env python3
"""Post-hoc top-K peak reranking. Instead of thresholding y_pred and
collecting all peaks above the threshold, find ALL local maxima (low
threshold), rank by peak height, take top K=2. This bypasses the
threshold/amplitude tradeoff entirely.

If the model finds both real BPs as the top-2 most-prominent peaks,
top-K eval should give Both BPs ≫ 14.2% even though the absolute
amplitudes are noisy.
"""
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


def one_hot_encode(sequence, max_length=MAX_SEQ_LEN):
    nuc_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
    encoded = np.zeros((max_length, N_CHANNELS), dtype=np.float32)
    for i, nuc in enumerate(sequence[:max_length].upper()):
        encoded[i, nuc_idx.get(nuc, 4)] = 1.0
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
            ids = [int(s.strip()) for s in str(sr['ISeqs(A)']).split('$') if s.strip().isdigit()]
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


def load_dataset(directories, max_files=None):
    inputs, meta = [], []
    for d in directories:
        fa_files = sorted((DATA_ROOT / d).glob("*.fa"))
        if max_files:
            fa_files = fa_files[:max_files]
        print(f"  {d}: {len(fa_files)} files", flush=True)
        for fa in fa_files:
            for t in parse_simulation(fa):
                inputs.append(t['input'])
                meta.append(t['meta'])
    X = np.array(inputs, dtype=np.float32)
    return X, meta


def topk_peaks(y_pred, actual_len, K, suppress_bp, min_distance=TOLERANCE,
               base_threshold=0.05):
    """Find all peaks above base_threshold, optionally suppress boundaries,
    then keep the top-K by height."""
    p = y_pred.copy()
    # Suppress padding region
    p[actual_len:] = 0.0
    # Optionally suppress edges of valid region
    if suppress_bp > 0:
        p[:suppress_bp] = 0.0
        p[max(0, actual_len - suppress_bp):actual_len] = 0.0

    peaks, props = find_peaks(p, height=base_threshold, distance=min_distance)
    if len(peaks) == 0:
        return np.array([])
    heights = props['peak_heights']
    # Sort by height descending, take top K
    order = np.argsort(-heights)
    return peaks[order[:K]]


def peak_metrics(true_bps, peaks, tolerance=TOLERANCE):
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


def evaluate(y_pred_batch, meta_batch, K, suppress_bp):
    n_both = n_one = n_missed = 0
    Ps, Rs, Fs = [], [], []
    n_peaks_total = []
    for yp, m in zip(y_pred_batch, meta_batch):
        true_bps = [m['bp_start'], m['bp_end']]
        peaks = topk_peaks(yp, m['actual_len'], K, suppress_bp)
        p, r, f, n_peaks, n_matched = peak_metrics(true_bps, peaks)
        Ps.append(p); Rs.append(r); Fs.append(f)
        n_peaks_total.append(n_peaks)
        if n_matched == 2: n_both += 1
        elif n_matched == 1: n_one += 1
        else: n_missed += 1
    n_total = len(meta_batch)
    return {
        'P': float(np.mean(Ps)), 'R': float(np.mean(Rs)), 'F1': float(np.mean(Fs)),
        'both': n_both, 'one': n_one, 'missed': n_missed, 'n_total': n_total,
        'pct_both': 100*n_both/n_total, 'pct_any': 100*(n_both+n_one)/n_total,
        'mean_peaks': float(np.mean(n_peaks_total)),
    }


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    print(f"[{time.strftime('%H:%M:%S')}] loading val", flush=True)
    X_all, meta_all = load_dataset(TRAIN_DIRS, max_files=750)
    fnames = np.array([m['file'] for m in meta_all])
    unique = np.unique(fnames)
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    val_files = set(unique[:int(len(unique) * VAL_SPLIT)].tolist())
    val_idx = np.array([i for i, f in enumerate(fnames) if f in val_files])
    X_val = X_all[val_idx]
    meta_val = [meta_all[i] for i in val_idx]

    print(f"[{time.strftime('%H:%M:%S')}] loading model", flush=True)
    cnn = tf.keras.models.load_model('models_test/cnn_breakpoint_best.keras', compile=False)
    print(f"[{time.strftime('%H:%M:%S')}] predicting", flush=True)
    y_val_pred = cnn.predict(X_val, batch_size=BATCH_SIZE, verbose=0)

    print(f"\n=== TOP-K peak reranking on VAL (n={X_val.shape[0]}) ===")
    print(f"{'K':>3} {'sup':>4} {'P':>5} {'R':>5} {'F1':>5} {'Both%':>6} {'One%':>6} {'peaks':>6}")
    for K in [1, 2, 3, 4, 5]:
        for sup in [0, 100, 200]:
            r = evaluate(y_val_pred, meta_val, K, sup)
            print(f"{K:>3d} {sup:>4d} {r['P']:>5.3f} {r['R']:>5.3f} {r['F1']:>5.3f} "
                  f"{r['pct_both']:>5.1f}% {100*(r['one'])/r['n_total']:>5.1f}% {r['mean_peaks']:>6.2f}")

    # Test set
    print(f"\n[{time.strftime('%H:%M:%S')}] loading test")
    X_test, meta_test = load_dataset([TEST_DIR])
    print(f"[{time.strftime('%H:%M:%S')}] predicting test", flush=True)
    y_test_pred = cnn.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

    print(f"\n=== TOP-K peak reranking on TEST (n={X_test.shape[0]}) ===")
    print(f"{'K':>3} {'sup':>4} {'P':>5} {'R':>5} {'F1':>5} {'Both%':>6} {'One%':>6} {'peaks':>6}")
    for K in [1, 2, 3, 4, 5]:
        for sup in [0, 100, 200]:
            r = evaluate(y_test_pred, meta_test, K, sup)
            print(f"{K:>3d} {sup:>4d} {r['P']:>5.3f} {r['R']:>5.3f} {r['F1']:>5.3f} "
                  f"{r['pct_both']:>5.1f}% {100*(r['one'])/r['n_total']:>5.1f}% {r['mean_peaks']:>6.2f}")

    print(f"\n[{time.strftime('%H:%M:%S')}] done")


if __name__ == '__main__':
    main()
