#!/usr/bin/env python3
"""Evaluate run #12 model with post-hoc boundary suppression: zero out
the first K bp and (actual_len - K, MAX_SEQ_LEN) bp before find_peaks.
Tests whether the boundary spike is masking interior signal during peak
detection. If yes, the threshold/F1 should improve substantially.
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
LABEL_SIGMA = 20


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


def detect_peaks_suppressed(y_pred, actual_len, suppress_bp, threshold, min_distance=TOLERANCE):
    """find_peaks on a copy of y_pred with first/last suppress_bp positions zeroed.

    Suppresses the BN+padding boundary spike that consistently fires at
    actual_len. Note: also suppresses the first suppress_bp positions
    of valid sequence — real BPs there will be missed.
    """
    p = y_pred.copy()
    p[:suppress_bp] = 0.0
    end = min(actual_len, len(p))
    p[max(0, end - suppress_bp):end] = 0.0
    p[end:] = 0.0  # all padding
    peaks, _ = find_peaks(p, height=threshold, distance=min_distance)
    return peaks


def peak_metrics(true_bps, peaks, tolerance=TOLERANCE):
    if len(peaks) == 0 and len(true_bps) == 0:
        return 1.0, 1.0, 1.0, 0, 0
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


def evaluate(y_pred_batch, meta_batch, suppress_bp, threshold):
    metrics = []
    n_both = n_one = n_missed = 0
    peaks_total = []
    fp_total = []
    for yp, m in zip(y_pred_batch, meta_batch):
        true_bps = [m['bp_start'], m['bp_end']]
        peaks = detect_peaks_suppressed(yp, m['actual_len'], suppress_bp, threshold)
        p, r, f, n_peaks, n_matched = peak_metrics(true_bps, peaks)
        metrics.append((p, r, f))
        peaks_total.append(n_peaks)
        fp_total.append(n_peaks - n_matched)
        if n_matched == 2: n_both += 1
        elif n_matched == 1: n_one += 1
        else: n_missed += 1
    p, r, f = zip(*metrics)
    n_total = len(meta_batch)
    return {
        'P': float(np.mean(p)), 'R': float(np.mean(r)), 'F1': float(np.mean(f)),
        'both': n_both, 'one': n_one, 'missed': n_missed, 'n_total': n_total,
        'pct_both': 100*n_both/n_total, 'pct_any': 100*(n_both+n_one)/n_total,
        'mean_peaks': float(np.mean(peaks_total)),
        'mean_fp': float(np.mean(fp_total)),
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

    print(f"[{time.strftime('%H:%M:%S')}] predicting val", flush=True)
    y_val_pred = cnn.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    print(f"  range [{y_val_pred.min():.4f}, {y_val_pred.max():.4f}]")

    # Sweep over suppress_bp values, then over thresholds for each
    suppress_options = [0, 100, 200, 400, 600]
    thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    best_global = (-1, 0, 0, None)
    print(f"\n=== VAL: suppress_bp × threshold sweep ===")
    print(f"{'sup':>4} {'thr':>4} {'P':>5} {'R':>5} {'F1':>5} {'Both%':>6} {'Any%':>5} {'peaks':>6} {'fp':>5}")
    for sup in suppress_options:
        for thr in thresholds:
            r = evaluate(y_val_pred, meta_val, sup, thr)
            print(f"{sup:>4d} {thr:>4.2f} {r['P']:>5.3f} {r['R']:>5.3f} {r['F1']:>5.3f} "
                  f"{r['pct_both']:>5.1f}% {r['pct_any']:>4.1f}% {r['mean_peaks']:>6.2f} {r['mean_fp']:>5.2f}")
            if r['F1'] > best_global[0]:
                best_global = (r['F1'], sup, thr, r)
        print()
    print(f"  Best val F1: {best_global[0]:.3f} at suppress_bp={best_global[1]}, threshold={best_global[2]}")
    print(f"  Both BPs at val-best: {best_global[3]['pct_both']:.1f}%")

    # Test set with the val-best (sup, thr)
    print(f"\n[{time.strftime('%H:%M:%S')}] loading test set")
    X_test, meta_test = load_dataset([TEST_DIR])

    print(f"[{time.strftime('%H:%M:%S')}] predicting test", flush=True)
    y_test_pred = cnn.predict(X_test, batch_size=BATCH_SIZE, verbose=0)

    print(f"\n=== TEST: at val-best (sup={best_global[1]}, thr={best_global[2]}) ===")
    r_test = evaluate(y_test_pred, meta_test, best_global[1], best_global[2])
    print(f"  P={r_test['P']:.3f}  R={r_test['R']:.3f}  F1={r_test['F1']:.3f}  "
          f"Both={r_test['pct_both']:.1f}%  Any={r_test['pct_any']:.1f}%  "
          f"peaks={r_test['mean_peaks']:.2f}  fp={r_test['mean_fp']:.2f}")

    # Compare: test at (sup=0, thr=0.8) -- the no-suppression baseline matching run #12
    print(f"\n=== TEST baseline (no suppression, thr=0.8 — matches run #12 cell-40) ===")
    r_baseline = evaluate(y_test_pred, meta_test, 0, 0.8)
    print(f"  P={r_baseline['P']:.3f}  R={r_baseline['R']:.3f}  F1={r_baseline['F1']:.3f}  "
          f"Both={r_baseline['pct_both']:.1f}%  Any={r_baseline['pct_any']:.1f}%  "
          f"peaks={r_baseline['mean_peaks']:.2f}  fp={r_baseline['mean_fp']:.2f}")

    print(f"\n[{time.strftime('%H:%M:%S')}] done")


if __name__ == '__main__':
    main()
