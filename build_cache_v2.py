"""Build the v2 sharded cache from raw SANTA data (M0.2).

Layout (one shard per source directory):

  cache/v2/manifest.json
  cache/v2/<source_dir>/files.txt        # one source-FASTA name per line
  cache/v2/<source_dir>/align_idx.npy    # structured: (file_idx, n_seqs,
                                          #             seq_len, byte_offset)
  cache/v2/<source_dir>/alignments.bin   # int8 concat of all alignments
                                          #   encoded {A:0,T:1,G:2,C:3,-:4}
  cache/v2/<source_dir>/events.npy       # structured: (file_idx, event_id,
                                          #             recomb_id, p1_id, p2_id,
                                          #             bp_start, bp_end)
  cache/v2/<source_dir>/seq_flags.npy    # structured: (file_idx, fasta_id,
                                          #             is_recombinant) — used
                                          #             for MLM + neg-triplet
                                          #             sampling in M2/M3.

Sequence retrieval is a memmap slice: O(1) by file_idx into alignments.bin,
then row indexing by (fasta_id - 1) — verified at build time to be the
identity for every FASTA.

Design choices that depart from the master plan's bullet:
  - int8 (not fp16 one-hot) — one-hot expansion is cheap at training time
    and saves ~10× disk. fp16 one-hot for 3.57M triplets would be ~3 TB.
  - One alignment per FASTA, not per triplet — long-content has 17–28×
    sibling-recombinant multiplicity (one event has many recombinant rows
    in the same alignment). Storing per-triplet would waste those 10×+.
  - No pre-computed `parental_assignment_per_position` — derivable from
    (bp_start, bp_end) in two lines at training time; storing it would
    add ~30 GB for no gain.

Run as a script: builds all shards under cache/v2/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from data_loader_v2 import (
    ALL_SOURCE_DIRS,
    DATA_ROOT,
    _detect_schema,
    load_events,
)
from Bio import SeqIO


def _discover_cache_root() -> Path:
    """Place the cache as a sibling of dataRaw so it persists across worktrees.

    The 42 GB cache should live at the repo's natural data root (alongside
    dataRaw/), not inside a transient worktree under .claude/worktrees/.
    """
    return DATA_ROOT.parent / "cache" / "v2"


CACHE_ROOT = _discover_cache_root()

# Nucleotide encoding. Stored as int8.
NT_TO_INT = {ord("A"): 0, ord("T"): 1, ord("G"): 2, ord("C"): 3, ord("-"): 4}
INT_TO_NT = {0: "A", 1: "T", 2: "G", 3: "C", 4: "-"}
GAP_INT = 4

_NUC_LUT: np.ndarray | None = None


def nucleotide_lut() -> np.ndarray:
    """Lazy ASCII→int8 lookup table, treating ambiguous bases as gap."""
    global _NUC_LUT
    if _NUC_LUT is None:
        lut = np.full(256, GAP_INT, dtype=np.int8)
        for b, v in NT_TO_INT.items():
            lut[b] = v
        _NUC_LUT = lut
    return _NUC_LUT


# Structured dtypes. Kept narrow to keep the global event table small.
ALIGN_IDX_DTYPE = np.dtype([
    ("file_idx", "i4"),
    ("n_seqs", "i4"),
    ("seq_len", "i4"),
    ("byte_offset", "i8"),
])

EVENT_DTYPE = np.dtype([
    ("file_idx", "i4"),
    ("event_id", "i4"),
    ("recomb_id", "i4"),
    ("p1_id", "i4"),
    ("p2_id", "i4"),
    ("bp_start", "i4"),
    ("bp_end", "i4"),
])

SEQ_FLAG_DTYPE = np.dtype([
    ("file_idx", "i4"),
    ("fasta_id", "i4"),
    ("is_recombinant", "u1"),
])


def encode_alignment(fa_path: Path) -> tuple[np.ndarray, list[int]]:
    """Returns (alignment int8 matrix (n, L), fasta_ids list[int]).

    Raises ValueError if FASTA IDs aren't a contiguous 1..N range — the
    cache's row-by-fasta-id-minus-1 contract depends on that.
    """
    recs = list(SeqIO.parse(fa_path, "fasta"))
    if not recs:
        raise ValueError(f"{fa_path}: empty FASTA")
    ids = []
    for r in recs:
        try:
            ids.append(int(r.id))
        except ValueError as e:
            raise ValueError(f"{fa_path}: non-integer id {r.id!r}") from e
    if sorted(ids) != list(range(1, len(ids) + 1)):
        raise ValueError(f"{fa_path}: ids are not 1..N (got min/max = "
                         f"{min(ids)}/{max(ids)})")
    n = len(recs)
    L = max(len(r.seq) for r in recs)
    out = np.full((n, L), GAP_INT, dtype=np.int8)
    lut = nucleotide_lut()
    for r in recs:
        i = int(r.id) - 1
        buf = np.frombuffer(bytes(str(r.seq), "ascii"), dtype=np.uint8)
        out[i, : len(buf)] = lut[buf]
    return out, ids


def build_shard(source_dir: str,
                cache_root: Path = CACHE_ROOT,
                data_root: Path = DATA_ROOT,
                max_files: int | None = None,
                verbose: bool = True) -> dict:
    """Build one shard. Returns the per-shard summary dict."""
    src_path = data_root / source_dir
    shard_dir = cache_root / source_dir
    shard_dir.mkdir(parents=True, exist_ok=True)

    # 1. Enumerate FASTAs that have a usable schema.
    fa_files = sorted(p for p in src_path.glob("*.fa")
                      if _detect_schema(p) is not None)
    if max_files is not None:
        fa_files = fa_files[:max_files]

    if verbose:
        print(f"[{source_dir}] {len(fa_files)} usable FASTAs", flush=True)

    if not fa_files:
        # Empty shard — write empty arrays so the reader contract holds.
        (shard_dir / "files.txt").write_text("")
        np.save(shard_dir / "align_idx.npy", np.zeros(0, dtype=ALIGN_IDX_DTYPE))
        (shard_dir / "alignments.bin").write_bytes(b"")
        np.save(shard_dir / "events.npy", np.zeros(0, dtype=EVENT_DTYPE))
        np.save(shard_dir / "seq_flags.npy", np.zeros(0, dtype=SEQ_FLAG_DTYPE))
        return {"source_dir": source_dir, "n_files": 0, "n_events": 0,
                "alignments_bytes": 0}

    # 2. Encode every alignment, write the binary, build the index.
    align_idx_rows: list[tuple] = []
    seq_flag_rows: list[tuple] = []
    align_path = shard_dir / "alignments.bin"
    offset = 0

    # We need to know which fasta_ids in each file are "the recombinant of
    # some event" to populate is_recombinant — capture during the event pass
    # below. First pass: write alignments and seed seq_flag_rows with all
    # is_recombinant=0; second pass: flip the bit for recombinant ids.
    file_to_ids: dict[int, list[int]] = {}

    with align_path.open("wb") as fout:
        for file_idx, fa in enumerate(fa_files):
            try:
                align, ids = encode_alignment(fa)
            except (ValueError, OSError) as e:
                if verbose:
                    print(f"  [skip] {fa.name}: {e}", flush=True)
                continue
            n, L = align.shape
            fout.write(align.tobytes(order="C"))
            align_idx_rows.append((file_idx, n, L, offset))
            offset += n * L
            file_to_ids[file_idx] = ids
            for fid in ids:
                seq_flag_rows.append((file_idx, fid, 0))

    align_idx = np.array(align_idx_rows, dtype=ALIGN_IDX_DTYPE)
    np.save(shard_dir / "align_idx.npy", align_idx)

    # Map fa filename -> file_idx for the event pass.
    name_to_idx = {fa.name: i for i, fa in enumerate(fa_files)}

    # 3. Build the event table by re-running the loader (deterministic).
    event_rows: list[tuple] = []
    recomb_seen: set[tuple[int, int]] = set()  # (file_idx, fasta_id)
    for ev in load_events(source_dir, max_files=max_files, data_root=data_root):
        fi = name_to_idx.get(ev.source_file)
        if fi is None:
            continue
        event_rows.append((fi, ev.event_id, ev.recomb_id, ev.parent1_id,
                           ev.parent2_id, ev.bp_positions[0], ev.bp_positions[1]))
        recomb_seen.add((fi, ev.recomb_id))

    events = np.array(event_rows, dtype=EVENT_DTYPE) if event_rows else \
             np.zeros(0, dtype=EVENT_DTYPE)
    np.save(shard_dir / "events.npy", events)

    # 4. Flip is_recombinant flags now that we know which ids appear as
    # the recombinant in any event.
    seq_flags = np.array(seq_flag_rows, dtype=SEQ_FLAG_DTYPE)
    if recomb_seen:
        # Build a 2D boolean lookup (n_files, max_fasta_id) for fast flipping.
        for i in range(len(seq_flags)):
            fi = int(seq_flags[i]["file_idx"])
            fid = int(seq_flags[i]["fasta_id"])
            if (fi, fid) in recomb_seen:
                seq_flags[i]["is_recombinant"] = 1
    np.save(shard_dir / "seq_flags.npy", seq_flags)

    # 5. files.txt — index ↔ filename map.
    (shard_dir / "files.txt").write_text("\n".join(fa.name for fa in fa_files) + "\n")

    summary = {
        "source_dir": source_dir,
        "n_files": int(len(align_idx)),
        "n_events": int(len(events)),
        "n_recombinant_seqs": int(seq_flags["is_recombinant"].sum()),
        "alignments_bytes": int(offset),
    }
    if verbose:
        print(f"  [{source_dir}] done — files={summary['n_files']}, "
              f"events={summary['n_events']:,}, "
              f"recombs={summary['n_recombinant_seqs']:,}, "
              f"alignments={summary['alignments_bytes'] / 1e6:.0f} MB",
              flush=True)
    return summary


def build_all(cache_root: Path = CACHE_ROOT,
              data_root: Path = DATA_ROOT,
              dirs: list[str] | None = None,
              max_files: int | None = None) -> dict:
    cache_root.mkdir(parents=True, exist_ok=True)
    dirs_ = dirs or ALL_SOURCE_DIRS
    summaries = []
    t0 = time.time()
    for d in dirs_:
        s = build_shard(d, cache_root=cache_root, data_root=data_root,
                        max_files=max_files)
        summaries.append(s)
    elapsed = time.time() - t0

    totals = {
        "n_files": sum(s["n_files"] for s in summaries),
        "n_events": sum(s["n_events"] for s in summaries),
        "alignments_bytes": sum(s["alignments_bytes"] for s in summaries),
        "elapsed_sec": elapsed,
    }
    manifest = {
        "version": "v2",
        "encoding": {"A": 0, "T": 1, "G": 2, "C": 3, "-": 4},
        "totals": totals,
        "shards": summaries,
    }
    with (cache_root / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nTotal: files={totals['n_files']}, events={totals['n_events']:,}, "
          f"alignments={totals['alignments_bytes'] / 1e9:.1f} GB, "
          f"elapsed={elapsed:.0f}s", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", default=None,
                    help="Source directory to build (repeat for several). "
                         "Defaults to all 10.")
    ap.add_argument("--max-files", type=int, default=None,
                    help="Cap FASTAs per directory (smoke test).")
    ap.add_argument("--cache-root", type=Path, default=CACHE_ROOT,
                    help="Override cache root (default cache/v2 next to this file).")
    args = ap.parse_args()
    build_all(cache_root=args.cache_root, dirs=args.dir, max_files=args.max_files)


if __name__ == "__main__":
    main()
