"""Read-time API for the v2 sharded cache (M0.2).

A `CacheV2` instance opens all shards lazily (memmap), exposes events as
indexable records, and supports the access patterns needed by:

  - BP localization (M3): index an event → get encoded (R, P1, P2) int8
    rows + bp_positions in O(1).
  - MLM pretraining (M1.2): index any (shard, file, fasta_id) for a single
    sequence — recombinant or not.
  - Negative-triplet sampling (M2/M3): per (shard, file), pull 3 random
    non-recombinant fasta_ids.

Sequences are returned as int8 with the encoding {A:0, T:1, G:2, C:3, -:4}.
Expand to one-hot at training time via `numpy.eye(5)[arr]`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np

# Re-exported for callers that want to decode without importing the builder.
from build_cache_v2 import (
    ALIGN_IDX_DTYPE,
    EVENT_DTYPE,
    GAP_INT,
    INT_TO_NT,
    NT_TO_INT,
    SEQ_FLAG_DTYPE,
)


class CacheV2Shard:
    """One source directory's cache, memmap-backed."""

    def __init__(self, shard_dir: Path):
        self.dir = Path(shard_dir)
        files_path = self.dir / "files.txt"
        text = files_path.read_text()
        self.files: list[str] = [s for s in text.splitlines() if s]
        self.align_idx: np.ndarray = np.load(self.dir / "align_idx.npy")
        self.events: np.ndarray = np.load(self.dir / "events.npy")
        self.seq_flags: np.ndarray = np.load(self.dir / "seq_flags.npy")

        # Memmap the int8 alignments. Empty shards have a 0-byte file; we
        # still create an empty memmap for uniform API.
        align_bin = self.dir / "alignments.bin"
        if align_bin.stat().st_size > 0:
            self.alignments = np.memmap(align_bin, dtype=np.int8, mode="r")
        else:
            self.alignments = np.zeros(0, dtype=np.int8)

        # Per-file lookup tables built from the structured arrays.
        self._by_file = {int(r["file_idx"]): r for r in self.align_idx}
        # Per (file_idx) → list of fasta_ids that are NOT recombinants
        # (used by sample_non_recombinant_ids for negative triplets).
        self._non_recomb_ids: dict[int, np.ndarray] = {}
        if len(self.seq_flags):
            for fi in np.unique(self.seq_flags["file_idx"]):
                mask = (self.seq_flags["file_idx"] == fi) & \
                       (self.seq_flags["is_recombinant"] == 0)
                self._non_recomb_ids[int(fi)] = \
                    self.seq_flags["fasta_id"][mask].astype(np.int32)

    # --- introspection ----------------------------------------------------

    def __len__(self) -> int:
        return int(len(self.events))

    @property
    def n_files(self) -> int:
        return len(self.align_idx)

    # --- core access ------------------------------------------------------

    def get_alignment(self, file_idx: int) -> np.ndarray:
        """Return the int8 (n_seqs, seq_len) matrix for one source FASTA."""
        r = self._by_file[int(file_idx)]
        n, L = int(r["n_seqs"]), int(r["seq_len"])
        offset = int(r["byte_offset"])
        flat = self.alignments[offset:offset + n * L]
        return flat.reshape(n, L)

    def get_sequence(self, file_idx: int, fasta_id: int) -> np.ndarray:
        """Return one sequence as a 1-D int8 vector."""
        return self.get_alignment(file_idx)[int(fasta_id) - 1]

    def get_triplet(self, event_idx: int) -> dict:
        """Return one event's (R, P1, P2, bp_positions, source) record."""
        ev = self.events[int(event_idx)]
        align = self.get_alignment(int(ev["file_idx"]))
        return {
            "R": align[int(ev["recomb_id"]) - 1],
            "P1": align[int(ev["p1_id"]) - 1],
            "P2": align[int(ev["p2_id"]) - 1],
            "bp_start": int(ev["bp_start"]),
            "bp_end": int(ev["bp_end"]),
            "file_idx": int(ev["file_idx"]),
            "source_file": self.files[int(ev["file_idx"])],
            "event_id": int(ev["event_id"]),
            "recomb_id": int(ev["recomb_id"]),
            "parent1_id": int(ev["p1_id"]),
            "parent2_id": int(ev["p2_id"]),
            "seq_len": align.shape[1],
        }

    # --- sampling helpers used by M2/M3 -----------------------------------

    def sample_non_recombinant_ids(self, file_idx: int, n: int,
                                   rng: random.Random | None = None) -> list[int]:
        """Pick n random non-recombinant fasta_ids from one FASTA (without
        replacement). Returns [] if the file has fewer than n non-recombinants.
        """
        pool = self._non_recomb_ids.get(int(file_idx), np.zeros(0, dtype=np.int32))
        if len(pool) < n:
            return []
        rng = rng or random
        idxs = rng.sample(range(len(pool)), n)
        return [int(pool[i]) for i in idxs]

    def iter_events(self) -> Iterator[dict]:
        for i in range(len(self.events)):
            yield self.get_triplet(i)


class CacheV2:
    """Federation of all shards under one cache root."""

    def __init__(self, cache_root: Path):
        self.root = Path(cache_root)
        with (self.root / "manifest.json").open() as f:
            self.manifest = json.load(f)
        self.shards: dict[str, CacheV2Shard] = {}
        for s in self.manifest["shards"]:
            d = s["source_dir"]
            sd = self.root / d
            if sd.is_dir() and (sd / "files.txt").exists():
                self.shards[d] = CacheV2Shard(sd)

    @property
    def n_events(self) -> int:
        return sum(len(s) for s in self.shards.values())

    @property
    def n_files(self) -> int:
        return sum(s.n_files for s in self.shards.values())


# ---------- decoder used by tests + by future training data pipelines ---------

def decode_int8_to_str(arr: np.ndarray) -> str:
    """Inverse of build_cache_v2.encode_alignment for a single row.

    Returns the alignment string with gaps. ASCII bytes for speed.
    """
    table = np.frombuffer(b"".join(INT_TO_NT[i].encode() for i in range(5)),
                          dtype=np.uint8)
    return table[arr.astype(np.int8)].tobytes().decode("ascii")
