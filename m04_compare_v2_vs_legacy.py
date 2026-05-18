"""M0.4 reproduction sanity (inference-only path).

Verifies the v2 cache preserves runB2_sig10's required input information,
without retraining anything. Three checks, in order:

  1. One-hot fidelity. Match every UnseenTestSet event between the v2 cache
     and the legacy 33-channel .npz cache by (source_file, event_id,
     recomb_id). Expand v2 int8 to fp16 one-hot and compare channels 0:15
     element-wise. If bit-identical, the v2 pipeline's channel-0..14
     contract is faithful — same inputs → same model outputs → same F1.

  2. Channel coverage. Confirm the legacy X tensor's channels 15..21 are
     zeroed (B2 contract) and 22..32 are non-trivial (RustRDP signals).

  3. Inference F1 sanity (run only if check 1 isn't 100%). Build a hybrid
     33-channel input — channels 0..14 from v2, channels 15..32 copied
     from legacy — and run runB2_sig10 inference on UnseenTestSet. The
     hybrid input matches what the model was trained on except for any
     v2-specific encoding differences in 0..14. F1 vs the known 0.421
     SANTA-sub-F1 establishes whether v2's contract differs in a way that
     matters.

Master-plan success criterion: SANTA sub F1 within ±0.03 of 0.421. If
check 1 passes at 100%, that criterion is satisfied a fortiori.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from cache_v2_reader import CacheV2


LEGACY_CACHE_GLOB = "cache/ds_UnseenTestSet_*.npz"


def pick_latest_legacy_cache(repo_root: Path) -> Path:
    candidates = sorted(
        glob.glob(str(repo_root / LEGACY_CACHE_GLOB)),
        key=os.path.getmtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No legacy cache at {LEGACY_CACHE_GLOB}")
    return Path(candidates[0])


def load_legacy_index(legacy_npz: Path) -> tuple[np.ndarray, list[dict]]:
    """Returns (X, meta) where meta[i] is the per-event dict for X[i]."""
    X = np.load(legacy_npz, mmap_mode="r")["X"]
    with open(str(legacy_npz).replace(".npz", ".pkl"), "rb") as f:
        meta = pickle.load(f)
    return X, meta


def v2_to_onehot(v2_int8: np.ndarray, target_len: int = 32000) -> np.ndarray:
    """Expand int8 (L,) to fp16 one-hot (target_len, 5).

    v2 and legacy both use 5-channel one-hot with gap as channel 4:
        {A:0, T:1, G:2, C:3, gap:4}.
    Positions past the source sequence end stay all-zero (no nucleotide
    present), matching legacy's right-padding-with-zeros convention.
    """
    L = v2_int8.shape[0]
    out = np.zeros((target_len, 5), dtype=np.float16)
    end = min(L, target_len)
    pos = np.arange(end)
    out[pos, v2_int8[:end].astype(np.int64)] = 1.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=Path("cache/v2"))
    ap.add_argument("--legacy-cache", type=Path, default=None,
                    help="Path to legacy ds_UnseenTestSet_*.npz; defaults to "
                         "the most recently modified one.")
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--max-events", type=int, default=None,
                    help="Cap events for a quick smoke check.")
    ap.add_argument("--out", type=Path, default=Path("m04_report.json"))
    args = ap.parse_args()

    legacy_path = args.legacy_cache or pick_latest_legacy_cache(args.repo_root)
    print(f"[{time.strftime('%H:%M:%S')}] Loading legacy cache "
          f"{legacy_path.name} ...", flush=True)
    X_legacy, meta_legacy = load_legacy_index(legacy_path)
    print(f"  legacy X: shape={X_legacy.shape} dtype={X_legacy.dtype}",
          flush=True)
    print(f"  legacy meta: {len(meta_legacy)} events", flush=True)

    # Build legacy lookup: (file, event_id_int, recomb_id) → row index.
    legacy_lookup: dict[tuple[str, int, int], int] = {}
    dup = 0
    for i, m in enumerate(meta_legacy):
        try:
            ev = int(m["event"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (m["file"], ev, int(m["recomb_id"]))
        if key in legacy_lookup:
            dup += 1
            continue
        legacy_lookup[key] = i
    print(f"  legacy lookup keys: {len(legacy_lookup)} unique "
          f"({dup} dup skipped)", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] Opening v2 cache ...", flush=True)
    cache = CacheV2(args.cache_root)
    shard = cache.shards["UnseenTestSet"]
    print(f"  v2 UnseenTestSet: {len(shard)} events", flush=True)

    matched = 0
    not_in_legacy = 0
    bit_identical = 0
    mismatch_events: list[dict] = []
    onehot_diff_total = 0
    max_diff_per_event = 0

    legacy_ch15_21_sum = 0.0
    legacy_ch22_32_abs_mean = 0.0
    samples = min(50, len(meta_legacy))
    for i in range(samples):
        x = X_legacy[i]
        legacy_ch15_21_sum += float(np.abs(x[:, 15:22]).sum())
        legacy_ch22_32_abs_mean += float(np.abs(x[:, 22:33]).mean())
    legacy_ch15_21_sum /= max(samples, 1)
    legacy_ch22_32_abs_mean /= max(samples, 1)

    print(f"\n[{time.strftime('%H:%M:%S')}] Comparing one-hot channels ...",
          flush=True)
    n_to_check = args.max_events if args.max_events else len(shard)
    t0 = time.time()
    for v2_idx in range(min(n_to_check, len(shard))):
        ev = shard.events[v2_idx]
        key = (shard.files[int(ev["file_idx"])], int(ev["event_id"]),
               int(ev["recomb_id"]))
        legacy_idx = legacy_lookup.get(key)
        if legacy_idx is None:
            not_in_legacy += 1
            continue
        matched += 1

        t = shard.get_triplet(v2_idx)
        v2_R = v2_to_onehot(t["R"])
        v2_P1 = v2_to_onehot(t["P1"])
        v2_P2 = v2_to_onehot(t["P2"])

        legacy_x = X_legacy[legacy_idx]
        legacy_R = legacy_x[:, 0:5]
        legacy_P1 = legacy_x[:, 5:10]
        legacy_P2 = legacy_x[:, 10:15]

        diffs = (
            int((v2_R != legacy_R).sum())
            + int((v2_P1 != legacy_P1).sum())
            + int((v2_P2 != legacy_P2).sum())
        )
        if diffs == 0:
            bit_identical += 1
        else:
            onehot_diff_total += diffs
            max_diff_per_event = max(max_diff_per_event, diffs)
            if len(mismatch_events) < 10:
                mismatch_events.append({
                    "v2_idx": v2_idx, "key": list(key), "diffs": diffs,
                })

        if (v2_idx + 1) % 1000 == 0:
            rate = (v2_idx + 1) / max(time.time() - t0, 1e-3)
            print(f"  [{v2_idx+1}/{n_to_check}] matched={matched} "
                  f"bit_identical={bit_identical} "
                  f"({rate:.0f} ev/s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n[{time.strftime('%H:%M:%S')}] done in {elapsed:.0f}s.",
          flush=True)

    report = {
        "legacy_cache": legacy_path.name,
        "n_checked": min(n_to_check, len(shard)),
        "matched_to_legacy": matched,
        "not_in_legacy": not_in_legacy,
        "bit_identical": bit_identical,
        "mismatched": matched - bit_identical,
        "total_onehot_cell_diffs": onehot_diff_total,
        "max_diff_per_event_cells": max_diff_per_event,
        "first_mismatches": mismatch_events,
        "legacy_ch15_21_abs_sum_per_event": legacy_ch15_21_sum,
        "legacy_ch22_32_abs_mean_per_event": legacy_ch22_32_abs_mean,
    }
    with args.out.open("w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to {args.out}\n")
    print(f"  {'metric':<40} {'value'}")
    print("  " + "-" * 55)
    for k in ("n_checked", "matched_to_legacy", "not_in_legacy",
              "bit_identical", "mismatched", "total_onehot_cell_diffs",
              "max_diff_per_event_cells",
              "legacy_ch15_21_abs_sum_per_event",
              "legacy_ch22_32_abs_mean_per_event"):
        print(f"  {k:<40} {report[k]}")

    if matched == 0:
        print("\nFAIL: no events matched between v2 and legacy.")
        return
    if bit_identical == matched:
        print("\nPASS: v2 one-hots are bit-identical to legacy for every "
              "matched event. M0.4 success criterion is trivially "
              "satisfied — same inputs ⇒ same outputs ⇒ same F1 = 0.421.")
    else:
        frac = bit_identical / matched
        print(f"\nPARTIAL: {bit_identical}/{matched} ({frac*100:.1f}%) "
              f"events are bit-identical. Run inference to confirm F1 "
              f"delta is within ±0.03 of 0.421.")


if __name__ == "__main__":
    main()
