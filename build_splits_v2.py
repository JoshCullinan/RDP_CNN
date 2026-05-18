"""Build the v2 held-out split definition (M0.3).

Locks the TRAIN/VAL/TEST membership at the FASTA-file level — events
within one FASTA are correlated (sibling recombinants share a lineage)
and must stay together. Whole directories go to TRAIN or VAL with one
exception: XML-6 is split 80/20 deterministically by sha256(filename).

Output: splits/v2_split.json — one canonical record per split listing
the exact FASTA filenames it contains, plus a summary of event counts.

Run as a script. Validates that:
  - every event in the cache lands in exactly one split,
  - TEST-SANTA (UnseenTestSet) and TEST-REAL (LANL CRF panel) files are
    never present in TRAIN or VAL,
  - the XML-6 split is reproducible (sorted-then-hashed assignment).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from cache_v2_reader import CacheV2


from build_cache_v2 import CACHE_ROOT as DEFAULT_CACHE
from data_loader_v2 import DATA_ROOT

REPO_ROOT = DATA_ROOT.parent  # sibling of dataRaw/
DEFAULT_SPLITS_DIR = REPO_ROOT / "splits"
DEFAULT_LANL_DIR = REPO_ROOT / "data" / "lanl_crf" / "triplets"

# Whole-dir assignments. XML-6 handled separately.
TRAIN_DIRS = ["XML-1", "XML-2", "XML-3", "XML-4",
              "long_content_30k_002", "long_content_30k_003"]
VAL_DIRS = ["XML-5", "long_content_30k_001"]
TEST_SANTA_DIRS = ["UnseenTestSet"]

XML6_DIR = "XML-6"
XML6_TRAIN_FRACTION = 0.80


def hash_to_unit(name: str, salt: str = "v2-split") -> float:
    """Deterministic [0, 1) value from a filename. Stable across sessions
    because SHA-256 + decoded-int division are not seed-dependent.
    """
    digest = hashlib.sha256(f"{salt}|{name}".encode()).digest()
    # Take first 8 bytes as a 64-bit unsigned int.
    n = int.from_bytes(digest[:8], "big")
    return n / 2**64


def xml6_split(file_names: list[str]) -> tuple[list[str], list[str]]:
    """80/20 split: file goes TRAIN if hash_to_unit(name) < 0.80.

    Sorted upfront so the function is order-invariant. The hash decides;
    sort just makes the iteration order deterministic for logging.
    """
    sorted_names = sorted(file_names)
    train, val = [], []
    for n in sorted_names:
        (train if hash_to_unit(n) < XML6_TRAIN_FRACTION else val).append(n)
    return train, val


def count_events_in_files(shard, file_subset: set[str]) -> tuple[int, int]:
    """Returns (triplet count, unique-event count) for events whose source
    FASTA is in `file_subset` for this shard.
    """
    if not len(shard.events):
        return 0, 0
    name_to_idx = {name: i for i, name in enumerate(shard.files)}
    file_idxs = {name_to_idx[n] for n in file_subset if n in name_to_idx}
    if not file_idxs:
        return 0, 0
    mask = np.isin(shard.events["file_idx"], list(file_idxs))
    selected = shard.events[mask]
    unique = {(int(r["file_idx"]), int(r["event_id"])) for r in selected}
    return int(mask.sum()), len(unique)


def build_splits(cache_root: Path, lanl_dir: Path) -> dict:
    cache = CacheV2(cache_root)

    splits: dict[str, dict] = {
        "TRAIN": {"dirs": {}, "totals": {"triplets": 0, "events": 0,
                                          "files": 0}},
        "VAL":   {"dirs": {}, "totals": {"triplets": 0, "events": 0,
                                          "files": 0}},
        "TEST_SANTA": {"dirs": {}, "totals": {"triplets": 0, "events": 0,
                                              "files": 0}},
        "TEST_REAL":  {"dirs": {}, "totals": {"triplets": 0, "events": 0,
                                              "files": 0}},
    }

    def add(split: str, dir_name: str, files: list[str]) -> None:
        shard = cache.shards.get(dir_name)
        if shard is None:
            tr, ev = 0, 0
        else:
            tr, ev = count_events_in_files(shard, set(files))
        splits[split]["dirs"][dir_name] = {
            "files": sorted(files),
            "n_files": len(files),
            "triplets": tr,
            "events": ev,
        }
        splits[split]["totals"]["triplets"] += tr
        splits[split]["totals"]["events"] += ev
        splits[split]["totals"]["files"] += len(files)

    # Whole-directory assignments.
    for d in TRAIN_DIRS:
        if d in cache.shards:
            add("TRAIN", d, cache.shards[d].files)
    for d in VAL_DIRS:
        if d in cache.shards:
            add("VAL", d, cache.shards[d].files)
    for d in TEST_SANTA_DIRS:
        if d in cache.shards:
            add("TEST_SANTA", d, cache.shards[d].files)

    # XML-6 — deterministic 80/20.
    if XML6_DIR in cache.shards:
        xml6_files = cache.shards[XML6_DIR].files
        train_x6, val_x6 = xml6_split(xml6_files)
        add("TRAIN", XML6_DIR, train_x6)
        add("VAL", XML6_DIR, val_x6)

    # TEST_REAL — LANL CRF panel (lives outside the v2 cache).
    if lanl_dir.is_dir():
        lanl_files = sorted(p.name for p in lanl_dir.glob("*.fa"))
        # Not in the cache: triplets count is the BP count from the
        # truth CSV, but for split bookkeeping we only need file names.
        splits["TEST_REAL"]["dirs"]["lanl_crf"] = {
            "files": lanl_files,
            "n_files": len(lanl_files),
            "triplets": None,  # filled out at eval time, not cache-derived
            "events": None,
            "path": str(lanl_dir.relative_to(REPO_ROOT)),
        }
        splits["TEST_REAL"]["totals"]["files"] += len(lanl_files)

    return {
        "version": "v2-split-1",
        "cache_root": str(cache_root.relative_to(REPO_ROOT))
                      if cache_root.is_relative_to(REPO_ROOT)
                      else str(cache_root),
        "lanl_dir": str(lanl_dir.relative_to(REPO_ROOT))
                    if lanl_dir.is_relative_to(REPO_ROOT)
                    else str(lanl_dir),
        "xml6_split_seed": "sha256(v2-split|<filename>) < 0.80 → TRAIN",
        "splits": splits,
    }


# ---------------- validation ----------------------------------------------

def validate_splits(split_doc: dict, cache: CacheV2) -> list[str]:
    """Returns a list of validation problems (empty list ⇒ all good)."""
    problems: list[str] = []

    # 1. Every cached event must land in exactly one split.
    event_assignment: dict[tuple[int, int], str] = {}
    # Build (shard, file_idx) → split membership
    file_to_split: dict[tuple[str, str], str] = {}
    for split_name, split in split_doc["splits"].items():
        for dir_name, info in split["dirs"].items():
            for fn in info["files"]:
                key = (dir_name, fn)
                if key in file_to_split:
                    problems.append(
                        f"file {dir_name}/{fn} appears in both "
                        f"{file_to_split[key]} and {split_name}"
                    )
                file_to_split[key] = split_name

    # 2. Walk the cache, attribute each event to its split.
    for dir_name, shard in cache.shards.items():
        if not len(shard.events):
            continue
        name_by_idx = {i: n for i, n in enumerate(shard.files)}
        unassigned = 0
        for ev in shard.events:
            fn = name_by_idx[int(ev["file_idx"])]
            key = (dir_name, fn)
            split = file_to_split.get(key)
            if split is None:
                unassigned += 1
            else:
                event_assignment[(dir_name, int(ev["file_idx"]),
                                  int(ev["event_id"]),
                                  int(ev["recomb_id"]))] = split
        if unassigned:
            problems.append(f"{dir_name}: {unassigned} events not in any split")

    # 3. Total events covered.
    expected = sum(len(s.events) for s in cache.shards.values())
    if len(event_assignment) != expected:
        problems.append(
            f"event coverage: assigned {len(event_assignment):,} but "
            f"cache holds {expected:,} events"
        )

    # 4. TEST files must not appear in TRAIN/VAL.
    test_files = {(d, fn)
                  for sname in ("TEST_SANTA", "TEST_REAL")
                  for d, info in split_doc["splits"][sname]["dirs"].items()
                  for fn in info["files"]}
    trainval_files = {(d, fn)
                      for sname in ("TRAIN", "VAL")
                      for d, info in split_doc["splits"][sname]["dirs"].items()
                      for fn in info["files"]}
    leak = test_files & trainval_files
    if leak:
        problems.append(
            f"TEST→TRAIN/VAL leak: {len(leak)} files (first 3: "
            f"{list(leak)[:3]})"
        )

    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--lanl-dir", type=Path, default=DEFAULT_LANL_DIR)
    ap.add_argument("--out", type=Path,
                    default=DEFAULT_SPLITS_DIR / "v2_split.json")
    args = ap.parse_args()

    print(f"Building splits from cache at {args.cache_root} ...", flush=True)
    doc = build_splits(args.cache_root, args.lanl_dir)

    cache = CacheV2(args.cache_root)
    problems = validate_splits(doc, cache)
    if problems:
        print("VALIDATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(doc, f, indent=2)

    print(f"\nSplit summary (saved to {args.out}):", flush=True)
    print(f"{'split':<12} {'files':>7} {'triplets':>12} {'events':>10}")
    print("-" * 45)
    for split_name, split in doc["splits"].items():
        t = split["totals"]
        print(f"{split_name:<12} {t['files']:>7} {str(t['triplets']):>12} "
              f"{str(t['events']):>10}")
    print()
    for split_name, split in doc["splits"].items():
        print(f"  {split_name}: " + ", ".join(
            f"{d}({info['n_files']})"
            for d, info in split["dirs"].items()
        ))
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
