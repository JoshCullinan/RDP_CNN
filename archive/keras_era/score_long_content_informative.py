"""Score every long_content_30k_{001,002} file by mean informative% (using
the now-regenerated event_classifier .faParents.csv).

Run AFTER regen_long_content_parents_v2.py completes. Outputs a TSV with
one row per file: dir, rel_path, n_events, mean_inf_pct, median_inf_pct,
match_avg_pct. Use this to pick a threshold and filter files for
build_pooled_split_run42b.py.

Parallelised; uses the same parse_simulation pipeline the CNN uses so
the informative% measured here is exactly what the model would see.
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import json

NB_PATH = Path('CNN.ipynb')
DATA_ROOT = Path('dataRaw')
TARGET_DIRS = ['long_content_30k_001', 'long_content_30k_002']
OUT_TSV = Path('long_content_informative_scores.tsv')


def init_namespace():
    """Build a namespace with parse_simulation et al., once per worker."""
    import json as _json
    with open(NB_PATH) as f:
        nb = _json.load(f)
    ns = {}
    needed = ['cell-1', 'cell-2', 'cell-3', 'cell-4', 'cell-5', 'cell-6',
              'cell-7', 'cell-8', 'cell-9', 'cell-10']
    for c in nb['cells']:
        if c.get('cell_type') == 'code' and c.get('id') in needed:
            src = ''.join(c['source'])
            if c.get('id') == 'cell-10':
                idx = src.find('# Smoke test')
                if idx > 0:
                    src = src[:idx]
            exec(src, ns)
    return ns


_NS = None


def score_one(fa_path_str):
    """Returns dict with file stats. Re-uses module-level namespace."""
    global _NS
    if _NS is None:
        _NS = init_namespace()
    parse_simulation = _NS['parse_simulation']

    fa = Path(fa_path_str)
    try:
        triplets = parse_simulation(fa, max_events_per_file=30)
    except Exception as e:
        return {'path': str(fa.relative_to(DATA_ROOT)), 'error': f'{type(e).__name__}: {e}'}
    if not triplets:
        return {'path': str(fa.relative_to(DATA_ROOT)), 'n_events': 0,
                'mean_inf_pct': 0.0, 'median_inf_pct': 0.0, 'match_avg_pct': 0.0}
    inf_pcts = []
    match_pcts = []
    for t in triplets:
        X = t['input']
        content = int((X.sum(-1) > 0).sum())
        if content > 100:
            inf_pcts.append(float(X[:content, 17].mean() * 100))
            match_pcts.append(float(((X[:content, 15].mean() + X[:content, 16].mean()) / 2) * 100))
    if not inf_pcts:
        return {'path': str(fa.relative_to(DATA_ROOT)), 'n_events': 0,
                'mean_inf_pct': 0.0, 'median_inf_pct': 0.0, 'match_avg_pct': 0.0}
    return {
        'path': str(fa.relative_to(DATA_ROOT)),
        'n_events': len(inf_pcts),
        'mean_inf_pct': float(np.mean(inf_pcts)),
        'median_inf_pct': float(np.median(inf_pcts)),
        'match_avg_pct': float(np.mean(match_pcts)),
    }


def main():
    fa_files = []
    for d in TARGET_DIRS:
        fa_files.extend(sorted((DATA_ROOT / d).glob('*.fa')))
    print(f"Scoring {len(fa_files)} files ...")

    rows = []
    t0 = time.time()
    n_done = 0
    n_errors = 0

    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(score_one, str(p)): p for p in fa_files}
        for fut in as_completed(futures):
            r = fut.result()
            n_done += 1
            if 'error' in r:
                n_errors += 1
                if n_errors <= 5:
                    print(f"  [err] {r['path']}: {r['error']}")
            else:
                rows.append(r)
            if n_done % 500 == 0 or n_done == len(fa_files):
                elapsed = time.time() - t0
                eta = (len(fa_files) - n_done) / max(n_done / max(elapsed, 1e-3), 1e-3)
                print(f"  [{n_done:>5}/{len(fa_files)}] errors={n_errors}  (ETA {eta:.0f}s)")

    # Write TSV
    rows.sort(key=lambda r: r['path'])
    with OUT_TSV.open('w') as f:
        f.write('path\tn_events\tmean_inf_pct\tmedian_inf_pct\tmatch_avg_pct\n')
        for r in rows:
            f.write(f"{r['path']}\t{r['n_events']}\t{r['mean_inf_pct']:.3f}"
                    f"\t{r['median_inf_pct']:.3f}\t{r['match_avg_pct']:.3f}\n")
    print(f"\nWrote {len(rows)} rows to {OUT_TSV}")

    # Quick distribution summary
    mean_infs = [r['mean_inf_pct'] for r in rows if r['n_events'] > 0]
    print(f"\nmean_inf_pct distribution across files:")
    for thr in [1, 3, 5, 7, 10, 15]:
        n_pass = sum(1 for x in mean_infs if x >= thr)
        print(f"  >= {thr:>2}%: {n_pass:>5}/{len(mean_infs)}  ({n_pass/max(len(mean_infs),1)*100:.1f}%)")
    print(f"  P25/P50/P75: {np.percentile(mean_infs, 25):.2f} / {np.median(mean_infs):.2f} / {np.percentile(mean_infs, 75):.2f}")
    print(f"  mean/max:    {np.mean(mean_infs):.2f} / {np.max(mean_infs):.2f}")


if __name__ == '__main__':
    main()
