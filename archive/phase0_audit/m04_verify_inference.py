"""M0.4 audit — actually run runB2_sig10 and confirm 0.421 F1.

Streamed redesign after the previous version OOM-killed the box by
allocating two writable copies of the 11.7 GB X tensor. Lessons in
[[feedback-padding-mask-oom]]: never materialise the full X in RAM;
iterate per-batch.

Only one inference path is needed. m04_report.json already proved the
v2 cache's channels 0..14 are bit-identical to legacy for every
UnseenTestSet event. Running the same model on byte-identical inputs
twice is redundant — instead, run it once on the legacy cache, confirm
F1 ≈ 0.421, and the v2 contract follows by determinism.

Peak RAM: per-batch slice (B × L × C × 2 bytes) + accumulated y_pred
(N × L × 1 byte = ~178 MB for UnseenTestSet) + TF buffers. With B=32,
slice = ~64 MB. Safe on the 30 GB box.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import resource
import time
from pathlib import Path


# RSS watchdog target. We don't use RLIMIT_AS because CUDA reserves a huge
# virtual address space (20–30 GB) on startup that would trigger a tight AS
# cap before any real work happens. Instead, we check ru_maxrss after each
# batch and abort the process if we're approaching the system memory ceiling.
# 26 GB on a 30 GB box leaves ~4 GB for the OS + GUI; user explicitly chose
# this ceiling.
_RSS_CEILING_BYTES = 26 * 1024 * 1024 * 1024


def _current_rss_bytes() -> int:
    """ru_maxrss is reported in KB on Linux; convert to bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _rss_watchdog(label: str = "") -> None:
    rss = _current_rss_bytes()
    if rss > _RSS_CEILING_BYTES:
        raise MemoryError(
            f"RSS watchdog tripped {label}: process RSS "
            f"{rss / 2**30:.1f} GB exceeds ceiling "
            f"{_RSS_CEILING_BYTES / 2**30:.0f} GB. Aborting to protect "
            f"the user's desktop session."
        )

import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

# Let TF grow GPU memory on demand instead of grabbing the full pool at
# startup — leaves room for any other GPU consumer and prevents the
# "CUDA_ERROR_OUT_OF_MEMORY 5.12 GiB" reservation we saw under tight VM caps.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

TOLERANCE = 200
RDP_BLOCK_START = 15  # B2 variant zeros channels 15..21
RDP_BLOCK_END_OF_ZERO = 22
DEFAULT_BATCH = 32

# Threshold sweep matches the original runB2_sig10 SANTA eval (eval_diagnostic_A_fair.py).
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
EBS = (0, 200)

# Canonical target F1s from results_diagnostic_A_partial/runB2_sig10.json.
TARGET_FULL_EB0 = 0.3734
TARGET_SUBSET_EB0 = 0.4205
SUBSET_FILE_LIST = Path("/home/joshc/Dev/RDP_CNN/splits/honest_eval_subset_11.txt")


def evaluate_with_content_mask(y_pred: np.ndarray, X_memmap, meta: list[dict],
                               indices: np.ndarray, threshold: float,
                               edge_buffer: int = 0,
                               tolerance: int = TOLERANCE) -> tuple[float, float, float]:
    """Per-event F1 matching eval_diagnostic_A_fair.evaluate().

    For each event we:
      - read its content-end from the legacy X (where any channel is nonzero),
      - zero predictions outside the content (and the edge-buffer band),
      - run find_peaks on the masked prediction.
    """
    tp = fp = fn = 0
    for idx_in_array, m_i in enumerate(indices):
        # Per-row content mask. fp16 .any() works fine — no dtype cast needed.
        row = np.asarray(X_memmap[m_i])  # ~2 MB per row, freed each iteration
        content_mask = row.any(axis=-1)
        content_end = int(content_mask.sum())
        m = meta[m_i]
        if content_end == 0:
            fn += 2
            continue
        pred = y_pred[idx_in_array].astype(np.float32, copy=True)
        if edge_buffer > 0:
            pred[:edge_buffer] = 0.0
            if content_end > edge_buffer:
                pred[content_end - edge_buffer:content_end] = 0.0
        pred[content_end:] = 0.0
        peaks, _ = find_peaks(pred, height=threshold, distance=tolerance)
        true_bps = [int(m["bp_start"]), int(m["bp_end"])]
        used: set[int] = set()
        for tb in true_bps:
            best = None
            best_d = tolerance + 1
            for j, p in enumerate(peaks):
                if j in used:
                    continue
                d = abs(int(p) - tb)
                if d <= tolerance and d < best_d:
                    best_d = d
                    best = j
            if best is not None:
                used.add(best)
                tp += 1
            else:
                fn += 1
        fp += len(peaks) - len(used)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def best_threshold(y_pred: np.ndarray, X_memmap, meta: list[dict],
                   indices: np.ndarray, edge_buffer: int = 0
                   ) -> tuple[float, float, float, float]:
    best = (-1.0, -1.0, -1.0, -1.0)
    for thr in THRESHOLDS:
        p, r, f = evaluate_with_content_mask(
            y_pred, X_memmap, meta, indices, thr, edge_buffer=edge_buffer)
        print(f"    thr={thr:.2f}: P={p:.3f} R={r:.3f} F1={f:.3f}",
              flush=True)
        if f > best[0]:
            best = (f, p, r, thr)
    return best


def stream_inference(model, X_memmap, indices: np.ndarray,
                     batch: int) -> np.ndarray:
    """Run inference over the rows X_memmap[indices], in batches.

    Each batch slab = (batch × L × C × 2 bytes) ≈ 64 MB at batch=32. The
    accumulator y_pred = (len(indices) × L × 2 bytes) ≈ 354 MB for the
    full 5,539-event UnseenTestSet, 37 MB for the 584-event subset.
    """
    n = len(indices)
    _, L, _ = X_memmap.shape
    y_out = np.zeros((n, L), dtype=np.float16)
    t0 = time.time()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        rows = indices[start:end]
        slab = np.array(X_memmap[rows], dtype=np.float16, copy=True)
        slab[..., RDP_BLOCK_START:RDP_BLOCK_END_OF_ZERO] = 0
        pred = model.predict(slab, batch_size=batch, verbose=0)
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred[..., 0]
        y_out[start:end] = pred.astype(np.float16)
        del slab, pred
        _rss_watchdog(label=f"after batch ending at {end}")
        if (end // batch) % 10 == 0 or end == n:
            rate = end / max(time.time() - t0, 1e-3)
            eta = (n - end) / max(rate, 1e-3)
            rss = _current_rss_bytes() / 2**30
            print(f"    [{end}/{n}]  {rate:.0f} ev/s  ETA {eta:.0f}s  "
                  f"RSS {rss:.1f} GB", flush=True)
    print(f"    total inference: {time.time()-t0:.0f}s", flush=True)
    return y_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/models_test/"
                                 "cnn_breakpoint_runB2_sig10_final.keras"))
    ap.add_argument("--legacy-cache", type=Path, default=None)
    ap.add_argument("--repo-root", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN"))
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--scope", choices=("subset", "full", "both"),
                    default="subset",
                    help="subset = 584-event honest-eval-subset only "
                         "(target 0.4205); full = 5,539-event UnseenTestSet "
                         "(target 0.3734); both = run subset then full.")
    args = ap.parse_args()

    legacy_glob = sorted(
        glob.glob(str(args.repo_root / "cache/ds_UnseenTestSet_*.npz")),
        key=os.path.getmtime, reverse=True
    )
    legacy_path = args.legacy_cache or Path(legacy_glob[0])
    print(f"[{time.strftime('%H:%M:%S')}] Open legacy cache "
          f"{legacy_path.name} (memmap)", flush=True)
    npz = np.load(legacy_path, mmap_mode="r")
    X_memmap = npz["X"]
    with open(str(legacy_path).replace(".npz", ".pkl"), "rb") as f:
        meta_legacy = pickle.load(f)
    print(f"  X shape={X_memmap.shape} dtype={X_memmap.dtype}; "
          f"{len(meta_legacy)} events", flush=True)

    # Subset file list has paths like "dataRaw/UnseenTestSet/foo.fa";
    # meta_legacy has just the basename.
    subset_files = set(Path(p.strip()).name
                       for p in SUBSET_FILE_LIST.read_text().splitlines()
                       if p.strip())
    print(f"  honest-eval-subset files: {len(subset_files)}", flush=True)
    sub_indices = np.array(
        [i for i, m in enumerate(meta_legacy) if m["file"] in subset_files]
    )
    full_indices = np.arange(len(meta_legacy))
    print(f"  subset indices: {len(sub_indices)} events  /  "
          f"full indices: {len(full_indices)} events", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] Load model {args.model.name}",
          flush=True)
    model = tf.keras.models.load_model(args.model, compile=False)
    print(f"  params={model.count_params():,}", flush=True)

    results: dict[str, dict] = {}

    def run_scope(name: str, indices: np.ndarray, target: float) -> None:
        print(f"\n=== {name.upper()} ({len(indices)} events) ===", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] Streamed inference "
              f"(B2 mask per batch, batch={args.batch})", flush=True)
        y_pred = stream_inference(model, X_memmap, indices, batch=args.batch)
        print(f"\n[{time.strftime('%H:%M:%S')}] Threshold sweep @ EB=0:",
              flush=True)
        f1, p, r, thr = best_threshold(y_pred, X_memmap, meta_legacy,
                                       indices, edge_buffer=0)
        delta = abs(f1 - target)
        print(f"\n  Best @ EB=0:  thr={thr:.2f}  P={p:.3f}  R={r:.3f}  "
              f"F1={f1:.4f}", flush=True)
        print(f"  Target F1 = {target}  (±0.03)", flush=True)
        print(f"  |Δ| = {delta:.4f}  →  "
              f"{'PASS' if delta < 0.03 else 'FAIL'}", flush=True)
        results[name] = {
            "target_f1": target, "best_f1": f1, "delta": delta,
            "best_thr": thr, "precision": p, "recall": r,
            "n_events": int(len(indices)),
        }
        del y_pred

    if args.scope in ("subset", "both"):
        run_scope("subset", sub_indices, TARGET_SUBSET_EB0)
    if args.scope in ("full", "both"):
        run_scope("full", full_indices, TARGET_FULL_EB0)

    out = Path("m04_inference_audit.json")
    import json
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}", flush=True)

    all_pass = all(r["delta"] < 0.03 for r in results.values())
    print(f"\nM0.4 inference audit: {'PASS' if all_pass else 'FAIL'}",
          flush=True)


if __name__ == "__main__":
    main()
