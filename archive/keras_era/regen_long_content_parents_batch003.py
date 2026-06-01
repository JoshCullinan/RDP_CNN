"""Regenerate .faParents.csv for long_content_30k_{001,002} using the
faithful event_classifier.py port (pick_parents_rdp5ml.py).

For each .fa, computes parents via Mode B (reconstruct lineage from
.faSimVSRealCompare.csv since SANTA logs were not preserved). Validation
on XML-1 (with SANTA logs hidden, forcing Mode B) showed 6/8 exact picks
vs the rdp5ML — the 2 differences come from reduced lineage info, not
algorithm errors. For long_content data, SimVSRealCompare lists every
(event, recombinant) pair so lineage info is richer than XML-1's.

Backs up existing lineage-heuristic .faParents.csv files to
`.faParents.csv.lineage_heuristic.bak` before overwriting.

Parallelized across 6 worker processes.
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

sys.path.insert(0, str(Path(__file__).parent))
from pick_parents_rdp5ml import process_file

DATA_ROOT = Path('dataRaw')
TARGET_DIRS = ['long_content_30k_003']


def process_one(fa_path_str):
    fa = Path(fa_path_str)
    parents_csv = fa.parent / f'{fa.name}Parents.csv'
    bak = parents_csv.with_name(parents_csv.name + '.lineage_heuristic.bak')

    # Back up existing if not already backed up
    if parents_csv.exists() and not bak.exists():
        try:
            shutil.copy2(parents_csv, bak)
        except Exception as e:
            return {'error': f'backup failed: {e}'}

    try:
        res = process_file(fa)
        if 'error' in res:
            return {'error': res['error']}
        return {
            'events_processed': res.get('events_processed', 0),
            'events_written': res.get('events_written', 0),
            'source': res.get('source', '?'),
        }
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}


def main():
    fa_files = []
    for d in TARGET_DIRS:
        fa_files.extend(sorted((DATA_ROOT / d).glob('*.fa')))
    print(f"Total files: {len(fa_files)}")

    t0 = time.time()
    n_done = n_errors = n_events_total = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(process_one, str(p)): p for p in fa_files}
        for fut in as_completed(futures):
            p = futures[fut]
            res = fut.result()
            n_done += 1
            if 'error' in res:
                n_errors += 1
                if n_errors <= 10:
                    print(f"  [err] {p.name}: {res['error']}")
            else:
                n_events_total += res['events_written']
            if n_done % 250 == 0 or n_done == len(fa_files):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-3)
                eta = (len(fa_files) - n_done) / max(rate, 1e-3)
                print(f"  [{n_done:>5}/{len(fa_files)}] events={n_events_total}, errors={n_errors}  ({rate:.1f} files/s; ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. files={n_done}, errors={n_errors}, events_written={n_events_total}")


if __name__ == '__main__':
    main()
