#!/usr/bin/env python3
"""Shared training driver for Diagnostic (B2) and Diagnostic (C).

Both variants train on run41's cached pooled split (~4000 train events,
~1500 val events) with the same hyperparameters as run41, changing only
ONE thing (the variant patch). This enforces the "one-change-per-run"
discipline.

  --variant C   Extend dilation_rates from (1,2,4,8,16,32) to
                (1,2,4,8,16,32,64,128,256,512,1024) — RF 224bp → ~14kbp.
  --variant B2  Zero channels 15:22 in X_train AND X_val at load time
                (match_p1, match_p2, informative, MaxChi disparity @ 4
                windows). Model never sees those features during training.

Defensive memory plan (matches eval_one_model.py):
  - set_memory_growth on GPU
  - Cache loaded once, fp16 stays fp16 (no redundant astype copy)
  - tf.data from_generator (CPU device pin), batch=2, prefetch=AUTOTUNE
  - Per-epoch ModelCheckpoint added — survives WSL2 crash
  - Best-val-aupr checkpoint also kept (for the eval pipeline)
"""
import argparse, json, math, os, pickle, sys, time, gc
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, LayerNormalization, Activation,
                                     Dropout, Concatenate, Add, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K

# ---------------- GPU memory growth (CRITICAL for WSL2 stability) -------------
for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"WARN: memory_growth({gpu}): {e}", flush=True)

# ---------------- Constants (match cell-3 / cell-15 / cell-18 / cell-22) ------
MAX_SEQ_LEN = 32000
N_INPUT_CHANNELS = 33
N_CHANNELS = 5
N_MAXCHI_WINDOWS = 4
RDP_BLOCK_START = 3 * N_CHANNELS + 3 + N_MAXCHI_WINDOWS  # 22
RDP_BLOCK_END = N_INPUT_CHANNELS                         # 33
RDP_DROPOUT_P = 0.1
POS_WEIGHT = 70.0
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 100
EDGE_BUFFER = 200
PRIOR_POSITIVE = 0.01

# Variant-specific channels-zero band for (B2): channels 15..21 inclusive.
B2_MASK_LO, B2_MASK_HI = 15, 22
# Variant (OH) one-hot-only: zero ALL derived signal channels (15..32). Tests
# whether the CNN transfers to real-HIV from raw nucleotide one-hots alone.
OH_MASK_LO, OH_MASK_HI = 15, 33
# Variant (RC) RDP-channels only: zero the one-hot block (0..14). Inverse of
# OH — tests whether one-hots add anything beyond classical-method signals.
RC_MASK_LO, RC_MASK_HI = 0, 15

# ---------------- Cache paths (run41 pooled split) ----------------------------
CACHE_TRAIN = 'cache/ds_pool_run41_train_13767833f56c8d8d.npz'
CACHE_VAL   = 'cache/ds_pool_run41_val_e36bcb8add5ec9f0.npz'
PKL_TRAIN = CACHE_TRAIN.replace('.npz', '.pkl')
PKL_VAL   = CACHE_VAL.replace('.npz', '.pkl')


# ---------------- Loss (weighted BCE, matches cell-15) ------------------------
def weighted_bce(pos_weight=POS_WEIGHT):
    def loss(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)
        weight = pos_weight * y_true + (1.0 - y_true)
        per_pos = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        return K.mean(weight * per_pos)
    return loss


# ---------------- build_cnn (verbatim from cell-18, default dilations) --------
def build_cnn(input_shape=(MAX_SEQ_LEN, N_INPUT_CHANNELS),
              filter_sizes=(3, 7, 15, 31),
              dilation_rates=(1, 2, 4, 8, 16, 32),
              n_filters=128, dropout=0.5,
              prior_positive=PRIOR_POSITIVE):
    inputs = Input(shape=input_shape, name='triplet_input')
    x = Conv1D(64, 3, padding='same', name='conv_shared')(inputs)
    x = LayerNormalization(name='ln_shared')(x)
    x = Activation('relu', name='relu_shared')(x)

    branches = []
    for ks in filter_sizes:
        b = Conv1D(n_filters, ks, padding='same', name=f'conv_k{ks}')(x)
        b = LayerNormalization(name=f'ln_k{ks}')(b)
        b = Activation('relu', name=f'relu_k{ks}')(b)
        b = Dropout(dropout, name=f'drop_k{ks}')(b)
        branches.append(b)
    x = Concatenate(name='merge_scales')(branches)

    for i, d in enumerate(dilation_rates):
        if i == 0:
            residual = Conv1D(n_filters, 1, padding='same', name='res_proj_d1')(x)
        else:
            residual = x
        y = Conv1D(n_filters, 7, padding='same', dilation_rate=d, name=f'conv_dil_d{d}')(x)
        y = LayerNormalization(name=f'ln_dil_d{d}')(y)
        y = Activation('relu', name=f'relu_dil_d{d}')(y)
        y = Dropout(dropout, name=f'drop_dil_d{d}')(y)
        x = Add(name=f'res_add_d{d}')([y, residual])

    x = Conv1D(32, 1, name='conv_pw1')(x)
    x = Activation('relu', name='relu_pw1')(x)
    x = Dropout(dropout, name='drop_pw')(x)
    bias_init = -math.log((1 - prior_positive) / prior_positive)
    x = Conv1D(1, 1, dtype='float32',
               bias_initializer=tf.keras.initializers.Constant(bias_init),
               name='conv_out')(x)
    x = Activation('sigmoid', dtype='float32', name='sigmoid_out')(x)
    outputs = Reshape((MAX_SEQ_LEN,), dtype='float32', name='output')(x)
    return Model(inputs, outputs, name='MultiScaleCNN_PerPosition')


# ---------------- Edge-aware sample weights (from cell-22) --------------------
def edge_suppress_weights(mask, eb=EDGE_BUFFER):
    out = mask.astype(np.float32).copy()
    n = out.shape[0]
    for i in range(n):
        nz = np.flatnonzero(mask[i])
        if nz.size == 0:
            continue
        first, last = int(nz[0]), int(nz[-1])
        end = min(first + eb, last + 1)
        out[i, first:end] = 0.0
        start = max(last - eb + 1, end)
        out[i, start:last + 1] = 0.0
    return out


# ---------------- Gaussian label rebake (for --label-sigma sweep) -------------
def regenerate_gaussian_labels(meta, sigma, length=MAX_SEQ_LEN):
    """Rebake per-position Gaussian labels at a custom sigma from meta BP
    positions. Replaces y_g during a training run without rebuilding the cache."""
    n = len(meta)
    y = np.zeros((n, length), dtype=np.float32)
    pos = np.arange(length, dtype=np.float32)
    for i, m in enumerate(meta):
        for bp in [int(m['bp_start']), int(m['bp_end'])]:
            if 0 <= bp < length:
                y[i] = np.maximum(
                    y[i],
                    np.exp(-0.5 * ((pos - bp) / float(sigma)) ** 2).astype(np.float32),
                )
    return y


# ---------------- Cache loading (run41 pool, no astype copy) ------------------
def load_cache(npz_path, pkl_path):
    print(f"  loading {npz_path}", flush=True)
    d = np.load(npz_path)
    X = d['X']  # already fp16 on disk; do NOT copy via astype
    y = d['y_g']  # gaussian targets (the active label_mode in run41)
    d.close()
    with open(pkl_path, 'rb') as f:
        meta = pickle.load(f)
    assert X.dtype == np.float16, f"expected fp16 X, got {X.dtype}"
    print(f"    X.shape={X.shape}, dtype={X.dtype}, bytes={X.nbytes/1e9:.2f}GB", flush=True)
    print(f"    y.shape={y.shape}, dtype={y.dtype}", flush=True)
    return X, y, meta


# ---------------- tf.data pipeline (matches cell-22) --------------------------
def build_pipeline(X, y, w, training, dropout_rdp):
    rng = np.random.RandomState(42) if dropout_rdp else None

    def gen():
        for i in range(len(X)):
            x_i = X[i]
            if dropout_rdp and rng.rand() < RDP_DROPOUT_P:
                x_i = x_i.copy()
                x_i[:, RDP_BLOCK_START:RDP_BLOCK_END] = 0
            yield x_i, y[i], w[i]

    x_sig = tf.TensorSpec(shape=(MAX_SEQ_LEN, X.shape[-1]), dtype=tf.float16)
    y_sig = tf.TensorSpec(shape=(MAX_SEQ_LEN,), dtype=tf.float32)
    w_sig = tf.TensorSpec(shape=(MAX_SEQ_LEN,), dtype=tf.float32)

    def to_fp32_x(x, y, w):
        return tf.cast(x, tf.float32), y, w

    with tf.device('/CPU:0'):
        ds = tf.data.Dataset.from_generator(gen, output_signature=(x_sig, y_sig, w_sig))
        if training:
            ds = ds.shuffle(buffer_size=min(len(X), 256), seed=42, reshuffle_each_iteration=True)
        ds = (ds.batch(BATCH_SIZE)
              .map(to_fp32_x, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))
    return ds


# ---------------- Variant patches ---------------------------------------------
def apply_variant_x(X, variant):
    """For B2: zero channels 15..21 IN-PLACE (the cache is mmap'd-fresh into
    memory by np.load; in-place is fine). The model never sees those channels.
    For OH: zero channels 15..32 (all derived signal channels) — tests pure
    one-hot transfer. For C: no-op."""
    if variant == 'B2':
        print(f"  variant B2: zeroing X[..., {B2_MASK_LO}:{B2_MASK_HI}] in place", flush=True)
        X[:, :, B2_MASK_LO:B2_MASK_HI] = 0
    elif variant == 'OH':
        print(f"  variant OH: zeroing X[..., {OH_MASK_LO}:{OH_MASK_HI}] in place", flush=True)
        X[:, :, OH_MASK_LO:OH_MASK_HI] = 0
    elif variant == 'RC':
        print(f"  variant RC: zeroing X[..., {RC_MASK_LO}:{RC_MASK_HI}] in place", flush=True)
        X[:, :, RC_MASK_LO:RC_MASK_HI] = 0
    return X


def variant_dilations(variant):
    if variant == 'C':
        return (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    return (1, 2, 4, 8, 16, 32)


# ---------------- Main --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['B2', 'C', 'OH', 'RC'])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--out-dir', default='models_test')
    ap.add_argument('--tag', default=None,
                    help='Override the default model tag (e.g., runB2_7k). '
                         'When provided, output filenames use this tag instead of '
                         'the variant default (runB2_no_rdp_channels / runC_extended_rf).')
    ap.add_argument('--train-cache', default=CACHE_TRAIN,
                    help='Path to train .npz cache (default: run41 4000-event cache).')
    ap.add_argument('--val-cache', default=CACHE_VAL,
                    help='Path to val .npz cache (default: run41 1500-event cache).')
    ap.add_argument('--label-sigma', type=float, default=None,
                    help='If set, regenerate Gaussian labels at this sigma from '
                         'meta BP positions instead of using cached y_g (default '
                         'cache sigma = 20).')
    args = ap.parse_args()

    default_tag = {'B2': 'runB2_no_rdp_channels', 'C': 'runC_extended_rf',
                   'OH': 'runOH_one_hot_only',
                   'RC': 'runRC_rdp_channels_only'}[args.variant]
    tag = args.tag if args.tag is not None else default_tag
    final_path = Path(args.out_dir) / f'cnn_breakpoint_{tag}_final.keras'
    best_path  = Path(args.out_dir) / f'cnn_breakpoint_{tag}_best.keras'
    last_path  = Path(args.out_dir) / f'cnn_breakpoint_{tag}_last.keras'
    history_path = Path(args.out_dir) / f'history_{tag}.json'

    train_cache = args.train_cache
    val_cache   = args.val_cache
    train_pkl = train_cache.replace('.npz', '.pkl')
    val_pkl   = val_cache.replace('.npz', '.pkl')

    if final_path.exists():
        print(f"FINAL exists: {final_path} — exiting (idempotent)", flush=True)
        return 0

    print(f"[{time.strftime('%H:%M:%S')}] === Diagnostic {args.variant} (tag={tag}) ===", flush=True)
    print(f"  variant={args.variant}", flush=True)
    print(f"  dilations={variant_dilations(args.variant)}", flush=True)
    print(f"  train_cache={train_cache}", flush=True)
    print(f"  val_cache  ={val_cache}", flush=True)
    print(f"  out_final={final_path}", flush=True)
    print(f"  out_best ={best_path}", flush=True)
    print(f"  out_last ={last_path}", flush=True)

    # ---- Load data ----
    print(f"\n[{time.strftime('%H:%M:%S')}] LOAD TRAIN", flush=True)
    X_train, y_train, meta_train = load_cache(train_cache, train_pkl)
    print(f"\n[{time.strftime('%H:%M:%S')}] LOAD VAL", flush=True)
    X_val, y_val, meta_val = load_cache(val_cache, val_pkl)

    # Variant patch on X
    X_train = apply_variant_x(X_train, args.variant)
    X_val   = apply_variant_x(X_val,   args.variant)

    # Optionally rebake labels at a custom sigma
    if args.label_sigma is not None:
        print(f"\n[{time.strftime('%H:%M:%S')}] REBAKE GAUSSIAN LABELS at sigma={args.label_sigma}", flush=True)
        y_train = regenerate_gaussian_labels(meta_train, args.label_sigma)
        y_val   = regenerate_gaussian_labels(meta_val,   args.label_sigma)
        print(f"  rebaked y_train shape={y_train.shape}, sum>0={int((y_train > 0).sum())}", flush=True)
        print(f"  rebaked y_val   shape={y_val.shape}, sum>0={int((y_val > 0).sum())}", flush=True)

    # Build edge-suppressed masks
    print(f"\n[{time.strftime('%H:%M:%S')}] EDGE-SUPPRESS WEIGHTS (eb={EDGE_BUFFER})", flush=True)
    # Per-event: a full-tensor astype(float32) on 14.78GB fp16 X_train balloons
    # to ~30GB and OOM-kills the box (this crashed Wezterm on 2026-05-16).
    def padding_mask(X):
        out = np.empty(X.shape[:2], dtype=bool)
        for i in range(X.shape[0]):
            out[i] = (X[i] != 0).any(axis=-1)
        return out
    m_train = padding_mask(X_train)
    m_val   = padding_mask(X_val)
    w_train = edge_suppress_weights(m_train)
    w_val   = edge_suppress_weights(m_val)
    print(f"  train mask kept frac: {m_train.mean():.4f} -> {w_train.mean():.4f}", flush=True)
    print(f"  val   mask kept frac: {m_val.mean():.4f}   -> {w_val.mean():.4f}", flush=True)
    del m_train, m_val
    gc.collect()

    # ---- Build model ----
    print(f"\n[{time.strftime('%H:%M:%S')}] BUILD MODEL", flush=True)
    cnn = build_cnn(dilation_rates=variant_dilations(args.variant))
    cnn.compile(
        optimizer=AdamW(learning_rate=LR, weight_decay=1e-5),
        loss=weighted_bce(POS_WEIGHT),
        metrics=[tf.keras.metrics.AUC(curve='PR', name='aupr')],
    )
    print(f"  params: {cnn.count_params():,}", flush=True)
    print(f"  RF (dilations × kernel=7, +1): {1 + sum(7 - 1 for d in variant_dilations(args.variant)) * 1}", flush=True)
    # Actual RF: filter_sizes max kernel=31 + dilated stack. RF = sum_d((k-1)*d) + 1
    rf = 1 + 30 + sum(6 * d for d in variant_dilations(args.variant))  # rough; k=7 in dilated, k=31 in input branch
    print(f"  approx RF: {rf}bp (multi-scale stem: kernel 31; dilated stack: kernel 7 × dilations)", flush=True)

    # ---- Pipelines ----
    print(f"\n[{time.strftime('%H:%M:%S')}] BUILD PIPELINES", flush=True)
    train_ds = build_pipeline(X_train, y_train, w_train, training=True,  dropout_rdp=True)
    val_ds   = build_pipeline(X_val,   y_val,   w_val,   training=False, dropout_rdp=False)

    # ---- Callbacks: best + per-epoch last + early stop + LR reduce ----
    callbacks = [
        EarlyStopping(monitor='val_aupr', mode='max', patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_aupr', mode='max', factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath=str(best_path), monitor='val_aupr', mode='max',
                        save_best_only=True, verbose=1),
        # NEW: per-epoch checkpoint so a WSL2 crash mid-training keeps progress.
        # Overwrites the same file each epoch (we only need the latest).
        ModelCheckpoint(filepath=str(last_path), save_best_only=False, verbose=0),
    ]

    print(f"\n[{time.strftime('%H:%M:%S')}] FIT (epochs={args.epochs}, batch={BATCH_SIZE}, "
          f"steps/epoch ≈ {len(X_train)//BATCH_SIZE})", flush=True)
    history = cnn.fit(train_ds, validation_data=val_ds,
                      epochs=args.epochs, callbacks=callbacks, verbose=2)

    # ---- Save final + history ----
    cnn.save(str(final_path))
    print(f"\nSaved final model: {final_path}", flush=True)

    hist = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    with history_path.open('w') as f:
        json.dump(hist, f, indent=2)
    print(f"Saved history: {history_path}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
