"""Load and unify SANTA (CacheV2) and LANL (FASTA) triplets into fixed-length
int8 rows {A:0,T:1,G:2,C:3,-:4}."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from spectrogram.config import SEQ_LEN, GAP_INT, LANL_TRIPLET_DIR, LANL_TRIPLET_EXPANDED_DIR

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
