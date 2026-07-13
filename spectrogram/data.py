"""Load and unify SANTA (CacheV2) and LANL (FASTA) triplets into fixed-length
int8 rows {A:0,T:1,G:2,C:3,-:4}."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from spectrogram.config import (SEQ_LEN, GAP_INT, LANL_TRIPLET_DIR, LANL_TRIPLET_EXPANDED_DIR,
                                 SANTA_SPLIT)

FASTA_NT_TO_INT = {"A": 0, "T": 1, "G": 2, "C": 3, "-": GAP_INT}

@dataclass
class Triplet:
    rows: np.ndarray      # int8 (3, SEQ_LEN); row order is loader's, recomb at recomb_idx
    recomb_idx: int       # which row (0/1/2) is the recombinant
    source: str           # "santa" | "lanl"
    group: str            # clustering key: SANTA shard or CRF name

def fix_length(row: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    row = np.asarray(row, dtype=np.int8)
    if row.shape[0] >= seq_len:
        return row[:seq_len].copy()
    out = np.full(seq_len, GAP_INT, dtype=np.int8)
    out[: row.shape[0]] = row
    return out

def _encode_fasta_seq(seq: str) -> np.ndarray:
    up = seq.upper()
    arr = np.fromiter((FASTA_NT_TO_INT.get(c, GAP_INT) for c in up),
                      dtype=np.int8, count=len(up))
    return fix_length(arr)

def _find_recomb_idx(rec_ids: list[str], crf_name: str) -> int:
    """Identify which record is the recombinant by CRF name pattern.

    Most CRFs (02_AG, 08_BC, 12_BF) have explicit 'recomb_' prefix.
    CRF07_BC (OpenRDP source) uses just the subtype codes (B, C, 07_BC).
    """
    # Look for explicit recomb_ prefix (CRF02_AG, CRF08_BC, CRF12_BF)
    for i, rid in enumerate(rec_ids):
        if "recomb" in rid.lower():
            return i

    # CRF07_BC special case: recombinant is named "07_BC" (at the end of the list)
    if crf_name == "CRF07_BC":
        # Expected order: B, C, 07_BC → recombinant at index 2
        for i, rid in enumerate(rec_ids):
            if "07_BC" in rid or "07BC" in rid or rid == "07_BC":
                return i
        # Fallback: if not found by name, assume recombinant is last
        return len(rec_ids) - 1

    # Default: assume first record is recombinant (shouldn't reach here for valid CRFs)
    return 0

def load_lanl_triplets(triplet_dir: Path | None = None) -> list[Triplet]:
    """Each CRF FASTA is 3 records; recombinant location depends on CRF source.

    Defaults to the parent-pairing-expanded set (data/lanl_crf/triplets_expanded/,
    built by spectrogram.expand_lanl) when it exists and is non-empty, else
    falls back to the original 4-CRF triplets/ directory.
    """
    from Bio import SeqIO
    if triplet_dir is None:
        if LANL_TRIPLET_EXPANDED_DIR.exists() and any(LANL_TRIPLET_EXPANDED_DIR.glob("*.fa")):
            triplet_dir = LANL_TRIPLET_EXPANDED_DIR
        else:
            triplet_dir = LANL_TRIPLET_DIR

    out: list[Triplet] = []
    for fa in sorted(Path(triplet_dir).glob("*.fa")):
        recs = list(SeqIO.parse(str(fa), "fasta"))
        assert len(recs) == 3, f"{fa} has {len(recs)} records, expected 3"

        # Expanded files are named "<crf>__<pA_acc>__<pB_acc>"; the original
        # 4 triplets are named "<crf>" directly. Either way the CRF family
        # name is the first "__"-delimited component.
        crf_name = fa.stem.split("__")[0]
        rec_ids = [r.id for r in recs]
        recomb_idx = _find_recomb_idx(rec_ids, crf_name)

        rows = np.stack([_encode_fasta_seq(str(r.seq)) for r in recs])
        out.append(Triplet(rows=rows, recomb_idx=recomb_idx, source="lanl", group=crf_name))
    return out

def load_santa_triplets(cache, limit: int | None = None) -> list[Triplet]:
    out: list[Triplet] = []
    for shard_name, shard in cache.shards.items():
        for ev in range(len(shard)):
            t = shard.get_triplet(ev)
            rows = np.stack([fix_length(t["R"]), fix_length(t["P1"]), fix_length(t["P2"])])
            out.append(Triplet(rows=rows, recomb_idx=0, source="santa", group=shard_name))
            if limit is not None and len(out) >= limit:
                return out
    return out

def split_file_set(split_dict: dict, which: str) -> set[tuple[str, str]]:
    """Pure helper: (shard_name, filename) pairs INCLUDED in one split arm.

    `split_dict` is the parsed v2_filtered_split.json (top-level dict with a
    "splits" key). Dropped shards / files simply have empty (or absent)
    "files" lists and contribute nothing.
    """
    out: set[tuple[str, str]] = set()
    dirs = split_dict["splits"][which]["dirs"]
    for shard_name, info in dirs.items():
        for fname in info.get("files", []):
            out.add((shard_name, fname))
    return out

def load_santa_split(cache, split_path: Path | None = None, which: str = "TRAIN",
                      limit: int | None = None) -> list[Triplet]:
    """Load only SANTA events whose (shard, source_file) is kept by the
    realism-filtered split (splits/v2_filtered_split.json, design spec §3).

    The cache holds only XML-1..XML-6 (no long_content shards); split
    entries for long_content shards simply never match and are skipped.

    Round-robins across the kept shards (one triplet per shard per round)
    rather than draining `cache.shards` in dict-insertion order, so a
    `limit` below the full kept-event count still spans every shard the
    split keeps for this arm (e.g. XML-2, XML-4, XML-6 for TRAIN) instead of
    collapsing to whichever shard happens to iterate first.
    """
    if split_path is None:
        split_path = SANTA_SPLIT
    with Path(split_path).open() as f:
        split_dict = json.load(f)
    keep = split_file_set(split_dict, which)
    keep_shards = {s for s, _ in keep}

    def _shard_events(shard_name, shard):
        for ev in range(len(shard)):
            t = shard.get_triplet(ev)
            if (shard_name, t["source_file"]) not in keep:
                continue
            rows = np.stack([fix_length(t["R"]), fix_length(t["P1"]), fix_length(t["P2"])])
            yield Triplet(rows=rows, recomb_idx=0, source="santa", group=shard_name)

    active = [_shard_events(name, shard) for name, shard in cache.shards.items()
              if name in keep_shards]
    out: list[Triplet] = []
    while active and (limit is None or len(out) < limit):
        still_active = []
        for gen in active:
            try:
                out.append(next(gen))
            except StopIteration:
                continue
            still_active.append(gen)
            if limit is not None and len(out) >= limit:
                break
        active = still_active
    return out
