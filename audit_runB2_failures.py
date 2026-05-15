#!/usr/bin/env python3
"""Failure-mode audit on runB2 (Step 1 of post-diagnostic gameplan).

Slices runB2's UnseenTestSet event outcomes (both/one/missed) by per-event
features to find which feature(s) best predict failure. If a single feature
predicts >=70% of failures (worst quartile share), we have an actionable
filter; otherwise the failures are uniform and we push to task reframe.

Channel layout (verified from CNN.ipynb cell-3):
  0-4   recombinant one-hot
  5-9   parent 1 one-hot
  10-14 parent 2 one-hot
  15-17 match_p1, match_p2, informative   (B2 zeroes these)
  18-21 MaxChi disparity @ {50,100,200,500}   (B2 zeroes these)
  22-23 RDP consensus PredBPStart/End Gaussians  (positional)
  24-32 9 per-method scalar confidences (broadcast across positions)

B2 training masked ch 15:22 -> same mask must be applied at inference.
The RDP block (22:33) is NOT masked.
"""
import sys, glob, pickle, json, gc, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

for gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"WARN: memory_growth({gpu}): {e}", flush=True)

TOLERANCE = 200
MAX_SEQ_LEN = 32000
HELDOUT_FILE = Path('splits/honest_eval_subset_11.txt')
MODEL_PATH = 'models_test/cnn_breakpoint_runB2_no_rdp_channels_final.keras'
CACHE_PATH = 'cache/ds_UnseenTestSet_c2217f407a2ba195.npz'
MASK_LO, MASK_HI = 15, 22  # B2's training-time zero mask
THRESHOLDS = (0.6, 0.8)
EDGE_BUFFER = 200


def _squeeze(y_pred):
    if y_pred.ndim == 3 and y_pred.shape[-1] == 1:
        return y_pred[..., 0]
    return y_pred


def streaming_predict(model, X, zero_lo=MASK_LO, zero_hi=MASK_HI, batch=2):
    """Per-sample-copy generator that zeros ch [zero_lo,zero_hi) -- no full X.copy()."""
    n_ch = X.shape[-1]
    sig = tf.TensorSpec(shape=(MAX_SEQ_LEN, n_ch), dtype=tf.float16)

    def gen():
        for i in range(len(X)):
            xi = X[i]
            xi = xi.copy()
            xi[:, zero_lo:zero_hi] = 0.0
            yield xi

    with tf.device('/CPU:0'):
        ds = (tf.data.Dataset.from_generator(gen, output_signature=sig)
              .batch(batch)
              .map(lambda x: tf.cast(x, tf.float32))
              .prefetch(2))
    return model.predict(ds, verbose=0)


def classify_event(pred, ce, m, threshold, edge_buffer=EDGE_BUFFER):
    """Return n_match in {0,1,2} for one event using same logic as eval_one_model."""
    if ce == 0:
        return 0
    p = pred.copy()
    if edge_buffer > 0:
        p[:edge_buffer] = 0.0
        if ce > edge_buffer:
            p[ce - edge_buffer: ce] = 0.0
    p[ce:] = 0.0
    true_bps = [m['bp_start'], m['bp_end']]
    peaks, _ = find_peaks(p, height=threshold, distance=TOLERANCE)
    used = set()
    n_match = 0
    for tb in true_bps:
        best = None; best_d = TOLERANCE + 1
        for j, pk in enumerate(peaks):
            if j in used: continue
            d = abs(int(pk) - int(tb))
            if d <= TOLERANCE and d < best_d:
                best_d = d; best = j
        if best is not None:
            used.add(best); n_match += 1
    return n_match


def per_event_features(xi_fp32, ce, m):
    """Compute audit features for one event. xi_fp32 is (L, 33) float32.

    Returns dict of features. All identity/informative features restrict to
    content positions [0, ce) AND require recomb-called (ch 0:5 nonzero).
    """
    if ce <= 0:
        return dict(
            informative_pct=0.0, parental_identity=1.0,
            recomb_p1_identity=0.0, recomb_p2_identity=0.0,
            bp_dist=abs(int(m['bp_end']) - int(m['bp_start'])),
            edge_dist_min=0,
            rdp_methods_max=0.0, rdp_consensus_max=0.0,
            content_len=0, n_called=0,
        )

    # Slice content region
    Xc = xi_fp32[:ce]   # (ce, 33)
    recomb = Xc[:, 0:5]
    p1 = Xc[:, 5:10]
    p2 = Xc[:, 10:15]

    # Called-nucleotide masks: position has any one-hot bit set
    called_r = recomb.sum(axis=-1) > 0
    called_p1 = p1.sum(axis=-1) > 0
    called_p2 = p2.sum(axis=-1) > 0
    # Require all three called to make a meaningful comparison
    valid = called_r & called_p1 & called_p2
    n_valid = int(valid.sum())

    if n_valid == 0:
        informative_pct = 0.0
        recomb_p1_id = 0.0
        recomb_p2_id = 0.0
    else:
        # Argmax only on valid positions to avoid the 0-class default
        ar = recomb[valid].argmax(axis=-1)
        a1 = p1[valid].argmax(axis=-1)
        a2 = p2[valid].argmax(axis=-1)
        informative_pct = float((a1 != a2).mean())
        recomb_p1_id = float((ar == a1).mean())
        recomb_p2_id = float((ar == a2).mean())

    # RDP features: per-method scalars are broadcast, so [pos=0] suffices.
    # Consensus Gaussians are per-position; report max over content.
    rdp_methods_max = float(xi_fp32[0, 24:33].max())
    if ce > 0:
        rdp_consensus_max = float(xi_fp32[:ce, 22:24].max())
    else:
        rdp_consensus_max = 0.0

    bp_dist = abs(int(m['bp_end']) - int(m['bp_start']))
    edge_dist_min = int(min(int(m['bp_start']), ce - int(m['bp_end'])))

    return dict(
        informative_pct=informative_pct,
        parental_identity=1.0 - informative_pct,
        recomb_p1_identity=recomb_p1_id,
        recomb_p2_identity=recomb_p2_id,
        bp_dist=bp_dist,
        edge_dist_min=edge_dist_min,
        rdp_methods_max=rdp_methods_max,
        rdp_consensus_max=rdp_consensus_max,
        content_len=int(ce),
        n_called=n_valid,
    )


def quartile_crosstab(records, feature, outcome_key):
    """Bin events into quartiles by feature value; report both/one/missed rates."""
    vals = np.array([r[feature] for r in records])
    # Use quartile breakpoints; collapse ties
    qs = np.quantile(vals, [0.25, 0.5, 0.75])
    # Build edges; deduplicate if degenerate
    edges = [-np.inf] + list(qs) + [np.inf]
    bins = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (vals >= lo) & (vals <= hi)
        else:
            mask = (vals >= lo) & (vals < hi)
        sub = [r for r, m in zip(records, mask) if m]
        if not sub:
            bins.append(dict(
                lo=float(lo) if np.isfinite(lo) else None,
                hi=float(hi) if np.isfinite(hi) else None,
                n=0, both=0, one=0, missed=0,
                both_rate=0.0, fail_rate=0.0,
                share_of_failures=0.0,
            ))
            continue
        n = len(sub)
        both = sum(1 for r in sub if r[outcome_key] == 2)
        one = sum(1 for r in sub if r[outcome_key] == 1)
        miss = sum(1 for r in sub if r[outcome_key] == 0)
        bins.append(dict(
            lo=float(lo) if np.isfinite(lo) else None,
            hi=float(hi) if np.isfinite(hi) else None,
            n=n, both=both, one=one, missed=miss,
            both_rate=both / n,
            fail_rate=(one + miss) / n,
            # share filled in after we know total failures
            share_of_failures=0.0,
        ))
    total_fail = sum(b['one'] + b['missed'] for b in bins)
    if total_fail > 0:
        for b in bins:
            b['share_of_failures'] = (b['one'] + b['missed']) / total_fail
    return dict(quantiles=[float(q) for q in qs], bins=bins, total_fail=total_fail, total_n=len(records))


def main():
    t_start = time.time()
    heldout = set(Path(line.strip()).name for line in HELDOUT_FILE.read_text().splitlines() if line.strip())
    print(f"Held-out subset: N={len(heldout)} files", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] Loading {MODEL_PATH}", flush=True)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    n_ch = model.input_shape[-1]
    print(f"  params={model.count_params():,}  n_channels={n_ch}", flush=True)
    assert n_ch == 33, f"expected 33-channel model, got {n_ch}"

    print(f"  cache: {CACHE_PATH}", flush=True)
    pkl = CACHE_PATH.replace('.npz', '.pkl')
    d = np.load(CACHE_PATH)
    X_full = d['X']  # already fp16 on disk -- do NOT astype (would dupe 11.7 GB)
    with open(pkl, 'rb') as f:
        meta_full = pickle.load(f)
    d.close()
    print(f"  full cache: N={len(meta_full)}, X.shape={X_full.shape}, dtype={X_full.dtype}, X.bytes={X_full.nbytes/1e9:.2f}GB", flush=True)

    # Sanity check: RDP block populated?
    print(f"\n  RDP-block sanity check (first 3 events):", flush=True)
    for i in range(3):
        consensus_max = float(X_full[i, :, 22:24].astype(np.float32).max())
        methods_vec = X_full[i, 0, 24:33].astype(np.float32)
        print(f"    event {i}: consensus_max={consensus_max:.4f}, methods[24:33]={methods_vec.tolist()}", flush=True)

    # content_end per event
    print(f"\n  computing content_end...", flush=True)
    content_end = np.array(
        [int((X_full[i].astype(np.float32).sum(axis=-1) > 0).sum()) for i in range(len(X_full))],
        dtype=np.int32,
    )
    print(f"  content_end: min={content_end.min()}, median={int(np.median(content_end))}, max={content_end.max()}", flush=True)

    # Compute per-event features BEFORE predicting (and BEFORE dropping X).
    print(f"\n[{time.strftime('%H:%M:%S')}] Computing per-event features...", flush=True)
    t0 = time.time()
    records = []
    for i in range(len(X_full)):
        ce = int(content_end[i])
        # Cast to fp32 only this one event (cheap)
        xi_fp32 = X_full[i].astype(np.float32)
        feats = per_event_features(xi_fp32, ce, meta_full[i])
        feats['idx'] = i
        feats['file'] = meta_full[i]['file']
        feats['bp_start'] = int(meta_full[i]['bp_start'])
        feats['bp_end'] = int(meta_full[i]['bp_end'])
        feats['in_subset'] = meta_full[i]['file'] in heldout
        records.append(feats)
        if (i + 1) % 1000 == 0:
            print(f"    features {i+1}/{len(X_full)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  features done in {time.time()-t0:.0f}s", flush=True)

    # Predict (B2 mask applied per-sample in generator)
    print(f"\n[{time.strftime('%H:%M:%S')}] Predicting (ch 15:22 zeroed per-sample)...", flush=True)
    t0 = time.time()
    y_full = streaming_predict(model, X_full, MASK_LO, MASK_HI)
    y_full = _squeeze(y_full)
    print(f"  predict took {time.time()-t0:.0f}s, y.shape={y_full.shape}, y.bytes={y_full.nbytes/1e9:.2f}GB", flush=True)

    # Drop X
    del X_full
    gc.collect()
    print(f"  dropped X_full", flush=True)

    # Classify outcomes at both thresholds
    print(f"\n[{time.strftime('%H:%M:%S')}] Classifying outcomes at thr=0.6 and thr=0.8 (EB=200)...", flush=True)
    t0 = time.time()
    for i, r in enumerate(records):
        ce = int(content_end[i])
        m = meta_full[i]
        r['outcome_thr06'] = classify_event(y_full[i], ce, m, 0.6)
        r['outcome_thr08'] = classify_event(y_full[i], ce, m, 0.8)
    print(f"  classification done in {time.time()-t0:.0f}s", flush=True)

    # Summary headline counts
    for thr_key, label in [('outcome_thr06', 'thr=0.6 (full peak)'), ('outcome_thr08', 'thr=0.8 (subset peak)')]:
        n_both = sum(1 for r in records if r[thr_key] == 2)
        n_one = sum(1 for r in records if r[thr_key] == 1)
        n_miss = sum(1 for r in records if r[thr_key] == 0)
        sub = [r for r in records if r['in_subset']]
        s_both = sum(1 for r in sub if r[thr_key] == 2)
        s_one = sum(1 for r in sub if r[thr_key] == 1)
        s_miss = sum(1 for r in sub if r[thr_key] == 0)
        print(f"\n  {label}:")
        print(f"    FULL   (N={len(records)}): both={n_both} ({n_both/len(records):.3f})  one={n_one}  missed={n_miss}  fail_rate={(n_one+n_miss)/len(records):.3f}")
        print(f"    SUBSET (N={len(sub)}): both={s_both} ({s_both/max(1,len(sub)):.3f})  one={s_one}  missed={s_miss}  fail_rate={(s_one+s_miss)/max(1,len(sub)):.3f}")

    # Cross-tab features against outcomes
    FEATURES = ['informative_pct', 'parental_identity', 'recomb_p1_identity',
                'recomb_p2_identity', 'bp_dist', 'edge_dist_min',
                'rdp_methods_max', 'rdp_consensus_max', 'content_len']
    summary = {
        'config': dict(model=MODEL_PATH, cache=CACHE_PATH,
                       mask_lo=MASK_LO, mask_hi=MASK_HI,
                       thresholds=list(THRESHOLDS), edge_buffer=EDGE_BUFFER,
                       n_events=len(records)),
        'headline': {},
        'crosstabs': {},
        'predictive_criterion': {},
    }
    for thr_key, label in [('outcome_thr06', 'thr=0.6'), ('outcome_thr08', 'thr=0.8')]:
        n_both = sum(1 for r in records if r[thr_key] == 2)
        n_one = sum(1 for r in records if r[thr_key] == 1)
        n_miss = sum(1 for r in records if r[thr_key] == 0)
        total_fail = n_one + n_miss
        summary['headline'][label] = dict(both=n_both, one=n_one, missed=n_miss,
                                          fail_rate=total_fail / len(records),
                                          both_rate=n_both / len(records))
        summary['crosstabs'][label] = {}
        for feat in FEATURES:
            ct = quartile_crosstab(records, feat, thr_key)
            summary['crosstabs'][label][feat] = ct

    # Predictive criterion: does any single feature's worst quartile
    # (i.e., the bin with the highest failure rate among the 4 bins)
    # account for >=70% of all failures?
    print(f"\n[{time.strftime('%H:%M:%S')}] Evaluating predictive criterion: worst-bin failure share >= 70%", flush=True)
    for thr_key, label in [('outcome_thr06', 'thr=0.6'), ('outcome_thr08', 'thr=0.8')]:
        cohort = summary['crosstabs'][label]
        print(f"\n  --- {label} ---")
        crit = {}
        for feat, ct in cohort.items():
            # Bin with highest fail_rate
            bins = ct['bins']
            worst = max(bins, key=lambda b: b['fail_rate'] if b['n'] > 0 else -1)
            best = min(bins, key=lambda b: b['fail_rate'] if b['n'] > 0 else 2)
            # Separation: worst_rate - best_rate
            sep = (worst['fail_rate'] - best['fail_rate']) if (worst['n'] > 0 and best['n'] > 0) else 0.0
            crit[feat] = dict(
                worst_lo=worst['lo'], worst_hi=worst['hi'],
                worst_fail_rate=worst['fail_rate'],
                worst_share_of_failures=worst['share_of_failures'],
                best_fail_rate=best['fail_rate'],
                separation=sep,
                meets_70pct=worst['share_of_failures'] >= 0.7,
            )
            print(f"    {feat:25s}  worst-bin [{worst['lo']}, {worst['hi']}]  "
                  f"fail_rate={worst['fail_rate']:.3f}  share_of_fail={worst['share_of_failures']:.3f}  "
                  f"sep={sep:+.3f}  {'<<< MEETS 70%' if worst['share_of_failures'] >= 0.7 else ''}")
        summary['predictive_criterion'][label] = crit

    # Save outputs
    out_parquet = Path('results_audit_runB2.parquet')
    out_csv = Path('results_audit_runB2.csv')
    out_json = Path('results_audit_runB2_summary.json')

    # Try parquet first, fall back to CSV
    try:
        import pandas as pd
        df = pd.DataFrame(records)
        try:
            df.to_parquet(out_parquet, index=False)
            print(f"\n  saved {out_parquet} (N={len(df)})", flush=True)
        except Exception as e:
            print(f"  parquet failed ({type(e).__name__}: {e}); writing CSV", flush=True)
            df.to_csv(out_csv, index=False)
            print(f"  saved {out_csv} (N={len(df)})", flush=True)
    except ImportError:
        # Pure-stdlib CSV fallback
        import csv
        keys = list(records[0].keys())
        with out_csv.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in records:
                w.writerow(r)
        print(f"\n  saved {out_csv} (N={len(records)})", flush=True)

    with out_json.open('w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  saved {out_json}", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] Total wall: {(time.time()-t_start)/60:.1f} min", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
