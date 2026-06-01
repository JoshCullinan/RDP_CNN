#!/usr/bin/env python3
"""Execute CNN.ipynb cells 1..cell-22 only, exiting cleanly after training.

Bypasses cells 23-40 to avoid the post-training memory leak that OOM'd the
notebook in run #28d (cell-26 hit the 24 GB cgroup cap right after cnn.fit
returned, ~25 GB AnonPages already resident from train_ds + val_ds closures
+ matplotlib + EarlyStopping snapshot).

Use eval_only.py / eval_edge_suppressed.py for results — those are clean.

Usage:
    systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=12G \\
        bash -c 'source .venv/bin/activate && python3 -u train_only.py'
"""
from __future__ import annotations
import json, sys, time, traceback
from pathlib import Path

NB_PATH = Path('/home/joshcullinan/RDP_CNN/CNN.ipynb')
STOP_AFTER = 'cell-22'  # exec through this cell, then exit


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = []
    for c in nb['cells']:
        if c.get('cell_type') != 'code':
            continue
        cid = c.get('id', '')
        cells.append((cid, ''.join(c['source'])))
        if cid == STOP_AFTER:
            break
    print(f"[{time.strftime('%H:%M:%S')}] training plan: exec {len(cells)} code cells through {STOP_AFTER}", flush=True)

    g = {'__name__': '__main__'}
    for cid, src in cells:
        t0 = time.time()
        print(f"\n[{time.strftime('%H:%M:%S')}] --- {cid} ({len(src)} chars) ---", flush=True)
        try:
            exec(compile(src, f'CNN.ipynb#{cid}', 'exec'), g)
        except SystemExit as e:
            print(f"[{cid}] SystemExit: {e}", flush=True)
            raise
        except Exception as e:
            print(f"[{cid}] EXEC ERROR: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.exit(1)
        dt = time.time() - t0
        if dt > 1:
            print(f"[{cid}] elapsed {dt:.1f}s", flush=True)

    # Save history for downstream analysis
    hist = g.get('history')
    if hist is not None and hasattr(hist, 'history'):
        out_path = NB_PATH.parent / f'history_{time.strftime("%Y%m%d_%H%M%S")}.json'
        with open(out_path, 'w') as f:
            json.dump({k: [float(x) for x in v] for k, v in hist.history.items()}, f, indent=2)
        print(f"[history] saved {len(hist.history)} metrics to {out_path}", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] training pipeline complete. Model on disk at models_test/cnn_breakpoint_best.keras (and _final.keras if EarlyStopping fired).", flush=True)


if __name__ == '__main__':
    main()
