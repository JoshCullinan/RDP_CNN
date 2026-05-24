"""Side-process snapshotter for M1.2 training checkpoints.

The training script overwrites `models_test/backbone_mlm_v1.pt` at the end
of every stage. If a later epoch performs worse than an earlier one, we'd
lose access to the earlier checkpoint.

This script watches the ckpt file's mtime and copies it to a versioned
name under `models_test/snapshots/` every time it changes. Runs in a
separate process — no GPU contention, no code change to training.

Usage:
    python3 snapshot_ckpts.py &     # run alongside training

Stops cleanly on SIGINT/SIGTERM.
"""

from __future__ import annotations

import shutil
import signal
import sys
import time
from pathlib import Path

import torch


CKPT = Path("models_test/backbone_mlm_v1.pt")
SNAPSHOTS = Path("models_test/snapshots")
STAGE_NAMES = {0: "10k", 1: "19k", 2: "30k"}

_STOP = False


def _handle_signal(signum, _frame):
    global _STOP
    print(f"[snapshotter] received signal {signum}, exiting", flush=True)
    _STOP = True


def _read_ckpt_metadata(path: Path) -> tuple[int | None, str | None, int | None]:
    """Return (epoch, stage_name, global_step) from a ckpt without loading weights."""
    try:
        # weights_only=False because we need the python dict fields, but we
        # only access metadata keys — no need to map_location anywhere.
        ck = torch.load(path, map_location="cpu", weights_only=False)
        epoch = int(ck.get("epoch", -1))
        stage_idx = ck.get("stage_idx_done")
        stage_name = STAGE_NAMES.get(int(stage_idx)) if stage_idx is not None else "unk"
        gs = int(ck.get("global_step", -1))
        return epoch, stage_name, gs
    except Exception as exc:
        print(f"[snapshotter] failed to read metadata from {path}: {exc}", flush=True)
        return None, None, None


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    print(f"[snapshotter] watching {CKPT}, writing snapshots to {SNAPSHOTS}",
          flush=True)

    last_mtime: float | None = None
    poll_s = 20.0

    while not _STOP:
        if CKPT.exists():
            mtime = CKPT.stat().st_mtime
            if last_mtime is None or mtime > last_mtime:
                # Wait a few seconds in case the file is still being written.
                time.sleep(3)
                # Re-check size stable before reading.
                size_a = CKPT.stat().st_size
                time.sleep(1)
                size_b = CKPT.stat().st_size
                if size_a != size_b:
                    # Still being written — try again next loop
                    continue

                epoch, stage_name, gs = _read_ckpt_metadata(CKPT)
                if epoch is None:
                    # Couldn't parse — wait and retry
                    time.sleep(poll_s)
                    continue
                snap_name = f"backbone_mlm_v1_e{epoch:02d}_{stage_name}_gs{gs}.pt"
                dst = SNAPSHOTS / snap_name
                if dst.exists():
                    # Already snapshotted this exact state. Skip.
                    last_mtime = mtime
                    continue
                try:
                    shutil.copy2(CKPT, dst)
                    size_mb = dst.stat().st_size / (1024 * 1024)
                    print(f"[snapshotter] {time.strftime('%H:%M:%S')}  snapshot "
                          f"epoch={epoch} stage={stage_name} gs={gs:,} "
                          f"→ {snap_name}  ({size_mb:.1f} MB)", flush=True)
                except Exception as exc:
                    print(f"[snapshotter] copy failed: {exc}", flush=True)
                last_mtime = mtime
        # Sleep in small chunks so the signal handler can break us out promptly
        for _ in range(int(poll_s)):
            if _STOP:
                break
            time.sleep(1.0)

    print("[snapshotter] exited cleanly", flush=True)


if __name__ == "__main__":
    main()
