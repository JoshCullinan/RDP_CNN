#!/usr/bin/env python3
"""Audit informative% for the 4k vs 7k runB2 training caches.

Hypothesis: runB2_7k regressed because the extra ~3000 events have lower
informative% (channel 17 = parents differ) than the 4k subset, dragging
the model toward less-discriminative training examples.

We load each cache sequentially (never both at once) to keep RAM under
control. For each event we compute:
    info_pct = mean(X[i, :content_end, 17])  where content_end is the
               number of non-padding positions.
Then compare distributions between the 4k and 7k caches, and isolate the
7k-only extras.
"""
import json
import pickle
from pathlib import Path

import numpy as np

CACHE_4K_NPZ = Path('cache/ds_pool_run41_train_13767833f56c8d8d.npz')
CACHE_4K_PKL = CACHE_4K_NPZ.with_suffix('.pkl')
CACHE_7K_NPZ = Path('cache/ds_pool_run41_train_b1c12ba31fb9e241.npz')
CACHE_7K_PKL = CACHE_7K_NPZ.with_suffix('.pkl')

INFORMATIVE_CH = 17
THRESHOLD = 0.05  # 5% — feedback_filter_weak_data.md cutoff

OUT_JSON = Path('results_audit_informative_pct_4k_vs_7k.json')


def event_key(m):
    """Best-effort unique key per event from the meta dict."""
    return (m['file'], int(m.get('event_idx', m.get('idx', -1))),
            int(m['bp_start']), int(m['bp_end']))


def compute_info_pct(npz_path, pkl_path):
    print(f"\n  Loading {npz_path}")
    d = np.load(npz_path)
    X = d['X']  # fp16
    print(f"    X.shape={X.shape}, dtype={X.dtype}, bytes={X.nbytes/1e9:.2f}GB")
    with open(pkl_path, 'rb') as f:
        meta = pickle.load(f)
    print(f"    meta n={len(meta)}")
    info_pct = np.empty(len(X), dtype=np.float32)
    content_end = np.empty(len(X), dtype=np.int32)
    # Per-event so the temp stays at ~2 MB.
    for i in range(len(X)):
        xi = X[i]
        # padding mask: any non-zero channel
        mask = (xi != 0).any(axis=-1)
        ce = int(mask.sum())
        content_end[i] = ce
        if ce == 0:
            info_pct[i] = 0.0
        else:
            info_pct[i] = float(xi[:ce, INFORMATIVE_CH].astype(np.float32).mean())
        if i % 1000 == 0:
            print(f"      .. {i}/{len(X)}", flush=True)
    d.close()
    return info_pct, content_end, meta


def summarise(name, info_pct):
    return {
        'name': name,
        'n': int(len(info_pct)),
        'mean': float(info_pct.mean()),
        'median': float(np.median(info_pct)),
        'p10': float(np.percentile(info_pct, 10)),
        'p25': float(np.percentile(info_pct, 25)),
        'p75': float(np.percentile(info_pct, 75)),
        'p90': float(np.percentile(info_pct, 90)),
        'pct_below_5pct': float((info_pct < THRESHOLD).mean()),
        'pct_below_1pct': float((info_pct < 0.01).mean()),
        'n_below_5pct': int((info_pct < THRESHOLD).sum()),
        'n_below_1pct': int((info_pct < 0.01).sum()),
    }


def main():
    print(f"INFORMATIVE% AUDIT (channel {INFORMATIVE_CH}, threshold {THRESHOLD})")
    print(f"  4k cache: {CACHE_4K_NPZ}")
    print(f"  7k cache: {CACHE_7K_NPZ}")

    info_4k, ce_4k, meta_4k = compute_info_pct(CACHE_4K_NPZ, CACHE_4K_PKL)
    info_7k, ce_7k, meta_7k = compute_info_pct(CACHE_7K_NPZ, CACHE_7K_PKL)

    s_4k = summarise('4k', info_4k)
    s_7k = summarise('7k', info_7k)

    # Identify the 7k-only extras (events in 7k but not 4k).
    keys_4k = {event_key(m): i for i, m in enumerate(meta_4k)}
    keys_7k = {event_key(m): i for i, m in enumerate(meta_7k)}
    extras_idx_7k = [i for k, i in keys_7k.items() if k not in keys_4k]
    info_extras = info_7k[extras_idx_7k]
    print(f"\n  7k-only extras: n={len(extras_idx_7k)}")
    s_extras = summarise('7k_only_extras', info_extras)

    # Also: 4k events that are NOT in 7k (sanity — should be near zero if 7k is a superset).
    only_4k_idx = [i for k, i in keys_4k.items() if k not in keys_7k]
    s_only4k = summarise('4k_only_(not_in_7k)', info_4k[only_4k_idx]) if only_4k_idx else \
        {'n': 0, 'note': 'empty — 7k is a superset of 4k'}

    out = {
        'caches': {'4k': str(CACHE_4K_NPZ), '7k': str(CACHE_7K_NPZ)},
        'channel': INFORMATIVE_CH,
        'threshold': THRESHOLD,
        'summary': {'4k': s_4k, '7k': s_7k, '7k_only_extras': s_extras,
                    '4k_only_(not_in_7k)': s_only4k},
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}\n")

    # Print compact table
    rows = ['4k', '7k', '7k_only_extras']
    print(f"{'cohort':>20} | {'n':>6} | {'mean':>7} | {'med':>7} | {'p10':>7} | {'p25':>7} | {'<5%':>7} | {'<1%':>7}")
    print('-' * 90)
    for key in rows:
        s = out['summary'][key]
        print(f"{key:>20} | {s['n']:>6} | {s['mean']:>7.4f} | {s['median']:>7.4f} | "
              f"{s['p10']:>7.4f} | {s['p25']:>7.4f} | {s['pct_below_5pct']:>7.2%} | "
              f"{s['pct_below_1pct']:>7.2%}")


if __name__ == '__main__':
    main()
