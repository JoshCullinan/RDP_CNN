#!/usr/bin/env python3
"""Plot prediction vs ground-truth for a few val samples from the cached
run #12 model. Used to discriminate boundary-fixed-interior-still-flat
vs boundary-still-spikes vs boundary-fixed-interior-better.
"""
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from Bio import SeqIO

DATA_ROOT = Path("dataRaw")
TRAIN_DIRS = ["XML-1", "XML-2", "XML-3", "XML-4", "XML-5"]
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


def generate_gaussian_labels(bp_start, bp_end, seq_length=MAX_SEQ_LEN, sigma=LABEL_SIGMA):
    labels = np.zeros(seq_length, dtype=np.float32)
    pos = np.arange(seq_length, dtype=np.float32)
    for bp in [bp_start, bp_end]:
        peak = np.exp(-0.5 * ((pos - bp) / sigma) ** 2)
        labels = np.maximum(labels, peak.astype(np.float32))
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
            'labels_gaussian': generate_gaussian_labels(bp_start, bp_end),
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
    inputs, labels, meta = [], [], []
    for d in directories:
        fa_files = sorted((DATA_ROOT / d).glob("*.fa"))
        if max_files:
            fa_files = fa_files[:max_files]
        print(f"  {d}: {len(fa_files)} files", flush=True)
        for fa in fa_files:
            for t in parse_simulation(fa):
                inputs.append(t['input'])
                labels.append(t['labels_gaussian'])
                meta.append(t['meta'])
    X = np.array(inputs, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    mask = (X.sum(axis=-1) > 0).astype(np.float32)
    return X, y, mask, meta


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    print(f"[{time.strftime('%H:%M:%S')}] loading val set", flush=True)
    X_all, y_all, mask_all, meta_all = load_dataset(TRAIN_DIRS, max_files=750)

    fnames = np.array([m['file'] for m in meta_all])
    unique = np.unique(fnames)
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    n_val_files = int(len(unique) * VAL_SPLIT)
    val_files = set(unique[:n_val_files].tolist())
    val_idx = np.array([i for i, f in enumerate(fnames) if f in val_files])
    X_val = X_all[val_idx]
    y_val = y_all[val_idx]
    meta_val = [meta_all[i] for i in val_idx]
    print(f"  val samples: {X_val.shape[0]}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading model", flush=True)
    cnn = tf.keras.models.load_model('models_test/cnn_breakpoint_best.keras', compile=False)

    print(f"[{time.strftime('%H:%M:%S')}] predicting", flush=True)
    y_val_pred = cnn.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    print(f"  pred range: [{y_val_pred.min():.4f}, {y_val_pred.max():.4f}]", flush=True)

    # Pick 4 val samples with INTERIOR breakpoints (not near edges)
    candidates = []
    for i, m in enumerate(meta_val):
        bp_s, bp_e = m['bp_start'], m['bp_end']
        actual_len = m['actual_len']
        # Want both BPs at least 300 bp from edges and from each other
        if (bp_s > 300 and bp_s < actual_len - 300
            and bp_e > 300 and bp_e < actual_len - 300
            and abs(bp_e - bp_s) > 500):
            candidates.append(i)
        if len(candidates) >= 8:
            break
    print(f"  found {len(candidates)} val samples with interior BPs", flush=True)

    if len(candidates) < 4:
        candidates = list(range(min(4, X_val.shape[0])))

    pick = candidates[:4]
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)

    for ax_idx, idx in enumerate(pick):
        m = meta_val[idx]
        bp_s, bp_e = m['bp_start'], m['bp_end']
        actual_len = m['actual_len']
        plot_len = min(actual_len + 200, MAX_SEQ_LEN)

        ax = axes[ax_idx]
        # Ground truth Gaussian
        ax.fill_between(range(plot_len), y_val[idx, :plot_len], alpha=0.3,
                        color='green', label='Ground truth')
        # Prediction
        ax.plot(range(plot_len), y_val_pred[idx, :plot_len], color='blue',
                lw=0.6, label='Prediction')
        # BP markers
        ax.axvline(bp_s, color='red', ls='--', alpha=0.5, lw=1)
        ax.axvline(bp_e, color='red', ls='--', alpha=0.5, lw=1)
        # Boundary buffer markers
        ax.axvline(100, color='orange', ls=':', alpha=0.5, lw=1, label='edge_buffer')
        ax.axvline(actual_len - 100, color='orange', ls=':', alpha=0.5, lw=1)
        ax.axvline(actual_len, color='black', ls=':', alpha=0.7, lw=1, label='actual_len')

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, plot_len)
        ax.set_title(f"Val idx {idx} | actual_len={actual_len} | BPs at {bp_s}, {bp_e}")
        ax.set_ylabel('Probability')
        if ax_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
        # stats for interior
        interior = y_val_pred[idx, 100:actual_len-100]
        ax.text(0.02, 0.95,
                f'pred_max={y_val_pred[idx, :actual_len].max():.3f}\n'
                f'edge[0:100]_max={y_val_pred[idx, :100].max():.3f}\n'
                f'edge[end-100:end]_max={y_val_pred[idx, actual_len-100:actual_len].max():.3f}\n'
                f'interior_mean={interior.mean():.3f}\n'
                f'interior_max={interior.max():.3f}',
                transform=ax.transAxes, verticalalignment='top',
                fontsize=8, family='monospace',
                bbox=dict(boxstyle='round', alpha=0.5, facecolor='wheat'))

    axes[-1].set_xlabel('Genomic position (bp)')
    plt.tight_layout()
    out = 'figures/run12_chart.png'
    Path('figures').mkdir(exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\nSaved chart: {out}", flush=True)

    # Also print summary stats across all val samples
    print(f"\n=== AGGREGATE BOUNDARY STATS (n={X_val.shape[0]}) ===")
    edge_left_max = []
    edge_right_max = []
    interior_means = []
    for i in range(X_val.shape[0]):
        actual_len = meta_val[i]['actual_len']
        edge_left_max.append(float(y_val_pred[i, :100].max()))
        edge_right_max.append(float(y_val_pred[i, actual_len-100:actual_len].max()))
        interior_means.append(float(y_val_pred[i, 100:actual_len-100].mean()))
    print(f"  edge[0:100] max:        mean={np.mean(edge_left_max):.4f}  median={np.median(edge_left_max):.4f}")
    print(f"  edge[end-100:end] max:  mean={np.mean(edge_right_max):.4f}  median={np.median(edge_right_max):.4f}")
    print(f"  interior mean:          mean={np.mean(interior_means):.4f}  median={np.median(interior_means):.4f}")


if __name__ == '__main__':
    main()
