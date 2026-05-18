"""Unified data loader for the sequence-only backbone replacement (M0.1).

Handles all three SANTA-derived FASTA schemas on disk:

  Schema A — legacy short (XML-1..5)
      *.fa
      *.faSimVSRealCompare.csv
      *.faRecombIdentifyStats.csv   (parent IDs come from ISeqs(A))

  Schema B — new short (XML-6)
      *.fa
      *.faSimVSRealCompare.csv
      (no parent CSV out-of-the-box; run pick_parents_rdp5ml.py over the
       directory to generate *.faParents.csv first — then the schema
       collapses to schema C and the loader picks the events up.)

  Schema C — long-content (long_content_30k_001..003)
      *.fa
      *.faSimVSRealCompare.csv
      *.faParents.csv               (Event,Parent1,Parent2 mapping; one row
                                     per RDPEvent shared by many recombinants)

`load_events(directory)` is a generator yielding `TripletEvent` records. Each
record carries the raw sequence strings + breakpoint positions + bookkeeping
metadata. Encoding into one-hots and on-disk caching is M0.2's job.

Run as a script to print an event-count audit across all 10 data directories.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from Bio import SeqIO

def _discover_data_root() -> Path:
    """Walk up from this file looking for a `dataRaw/` directory.

    Supports both the main checkout (where dataRaw is alongside this file)
    and worktrees under .claude/worktrees/ (where dataRaw is several levels up).
    """
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "dataRaw"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "dataRaw/ not found above " + str(here) +
        " — set data_root= explicitly when calling load_events()"
    )


DATA_ROOT = _discover_data_root()

ALL_SOURCE_DIRS = [
    "XML-1", "XML-2", "XML-3", "XML-4", "XML-5", "XML-6",
    "long_content_30k_001", "long_content_30k_002", "long_content_30k_003",
    "UnseenTestSet",
]


@dataclass
class TripletEvent:
    """One recombination event = one (R, P1, P2) training triplet.

    Sequence strings retain alignment gap characters ('-'). `original_len` is
    the longest non-padded length across the three sequences (gaps included),
    so callers can decide how to crop or pad.
    """
    R_seq: str
    P1_seq: str
    P2_seq: str
    bp_positions: tuple[int, int]
    recomb_id: int
    parent1_id: int
    parent2_id: int
    original_len: int
    source_dir: str
    source_file: str
    event_id: int


def _detect_schema(fa: Path) -> Optional[str]:
    """Pick the schema for one .fa. Returns None if the file is unusable."""
    sim = fa.with_name(fa.name + "SimVSRealCompare.csv")
    if not sim.exists():
        return None
    parents = fa.with_name(fa.name + "Parents.csv")
    if parents.exists():
        return "long"
    stats = fa.with_name(fa.name + "RecombIdentifyStats.csv")
    if stats.exists():
        return "legacy"
    # XML-6: SimVSReal exists but no parent-id source → no usable events.
    return None


def _parent_pair_legacy(stats_rows: pd.DataFrame, recomb_id: int) -> Optional[tuple[int, int]]:
    """Schema A: pull two parent IDs from the 3-row ISeqs(A) block.

    The legacy stats CSV has exactly 3 rows per event. One row corresponds to
    the recombinant itself (its ISeqs(A) contains `recomb_id`); the other two
    rows' first ISeqs(A) ID is a parent candidate.
    """
    parents: list[int] = []
    for _, sr in stats_rows.iterrows():
        raw = str(sr.get("ISeqs(A)", ""))
        ids = [int(s.strip()) for s in raw.split("$") if s.strip().isdigit()]
        if not ids or recomb_id in ids:
            continue
        parents.append(ids[0])
        if len(parents) == 2:
            return parents[0], parents[1]
    return None


def load_events(directory: str, max_files: Optional[int] = None,
                data_root: Optional[Path] = None) -> Iterator[TripletEvent]:
    """Yield one TripletEvent per usable row across all .fa files in `directory`.

    A row is usable when:
      - the SimVSReal CSV has a populated ActualRecomb / SimBPStart / SimBPEnd,
      - the schema-appropriate parent CSV identifies a valid (P1, P2) pair,
      - all three sequence IDs resolve in the FASTA, and
      - P1, P2 are distinct from each other and from the recombinant.

    Anything that fails any check is silently skipped — count_events() reports
    aggregate skips per directory so audits stay observable.
    """
    root = data_root or DATA_ROOT
    dir_path = root / directory
    if not dir_path.is_dir():
        return
    fa_files = sorted(dir_path.glob("*.fa"))
    if max_files is not None:
        fa_files = fa_files[:max_files]

    for fa in fa_files:
        schema = _detect_schema(fa)
        if schema is None:
            continue
        try:
            seqs = {int(r.id): str(r.seq) for r in SeqIO.parse(fa, "fasta")}
        except (ValueError, OSError):
            continue

        sim_path = fa.with_name(fa.name + "SimVSRealCompare.csv")
        try:
            sim = pd.read_csv(sim_path, skipinitialspace=True, index_col=False)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue

        required_cols = {"ActualRecomb", "SimBPStart", "SimBPEnd"}
        if not required_cols.issubset(sim.columns):
            continue

        # Schema C: pre-load Event -> (P1, P2) map.
        parents_by_event: dict[int, tuple[int, int]] = {}
        if schema == "long":
            try:
                pdf = pd.read_csv(fa.with_name(fa.name + "Parents.csv"),
                                  skipinitialspace=True, index_col=False)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
            for _, r in pdf.iterrows():
                try:
                    parents_by_event[int(r["Event"])] = (int(r["Parent1"]),
                                                        int(r["Parent2"]))
                except (KeyError, TypeError, ValueError):
                    continue
            stats = None
        else:
            try:
                stats = pd.read_csv(fa.with_name(fa.name + "RecombIdentifyStats.csv"),
                                    skipinitialspace=True, index_col=False)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue

        for _, row in sim.iterrows():
            try:
                event_id = int(row["RDPEvent"]) if "RDPEvent" in sim.columns else int(row.name)
                recomb_id = int(row["ActualRecomb"])
                bp_start = int(row["SimBPStart"])
                bp_end = int(row["SimBPEnd"])
            except (TypeError, ValueError):
                continue

            if schema == "long":
                pair = parents_by_event.get(event_id)
                if pair is None:
                    continue
                p1, p2 = pair
            else:
                ev_rows = stats[stats["Event"] == event_id]
                if len(ev_rows) != 3:
                    continue
                pair = _parent_pair_legacy(ev_rows, recomb_id)
                if pair is None:
                    continue
                p1, p2 = pair

            if p1 == recomb_id or p2 == recomb_id or p1 == p2:
                continue
            if not all(sid in seqs for sid in (recomb_id, p1, p2)):
                continue

            r_seq, p1_seq, p2_seq = seqs[recomb_id], seqs[p1], seqs[p2]
            yield TripletEvent(
                R_seq=r_seq,
                P1_seq=p1_seq,
                P2_seq=p2_seq,
                bp_positions=(bp_start, bp_end),
                recomb_id=recomb_id,
                parent1_id=p1,
                parent2_id=p2,
                original_len=max(len(r_seq), len(p1_seq), len(p2_seq)),
                source_dir=directory,
                source_file=fa.name,
                event_id=event_id,
            )


def count_events(directories: Optional[list[str]] = None,
                 max_files_per_dir: Optional[int] = None,
                 data_root: Optional[Path] = None) -> dict:
    """Per-directory summary of triplets and distinct RDPEvents.

    `triplets` is the row-level count (= number of TripletEvent yielded —
    in the long-content schema this can be >30× the RDPEvent count because
    sibling recombinants inherit the same parents and breakpoints).
    `events` counts distinct (source_file, event_id) pairs.
    """
    dirs = directories or ALL_SOURCE_DIRS
    per_dir: dict[str, dict[str, int]] = {}
    total_triplets = 0
    total_events = 0
    for d in dirs:
        seen_events: set[tuple[str, int]] = set()
        n_triplets = 0
        for e in load_events(d, max_files=max_files_per_dir, data_root=data_root):
            n_triplets += 1
            seen_events.add((e.source_file, e.event_id))
        per_dir[d] = {"triplets": n_triplets, "events": len(seen_events)}
        total_triplets += n_triplets
        total_events += len(seen_events)
    return {"per_dir": per_dir, "total_triplets": total_triplets,
            "total_events": total_events}


def _print_audit(max_files_per_dir: Optional[int] = None) -> None:
    print(f"{'directory':<26} {'triplets':>12} {'events':>10} {'avg/event':>10}",
          flush=True)
    print("-" * 62, flush=True)
    grand_triplets = 0
    grand_events = 0
    for d in ALL_SOURCE_DIRS:
        seen_events: set[tuple[str, int]] = set()
        n_triplets = 0
        for e in load_events(d, max_files=max_files_per_dir):
            n_triplets += 1
            seen_events.add((e.source_file, e.event_id))
        n_events = len(seen_events)
        ratio = (n_triplets / n_events) if n_events else 0.0
        print(f"{d:<26} {n_triplets:>12,} {n_events:>10,} {ratio:>10.1f}",
              flush=True)
        grand_triplets += n_triplets
        grand_events += n_events
    print("-" * 62, flush=True)
    print(f"{'TOTAL':<26} {grand_triplets:>12,} {grand_events:>10,}", flush=True)
    print()
    print("Notes:")
    print(" - 'triplets' = TripletEvent yielded; what the cache will hold.")
    print(" - 'events'   = distinct RDPEvents (deduped over sibling recombinants).")
    print(" - 'avg/event'= sibling-recombinant multiplicity (long-content >> XML).")
    print(" - XML-6 events come from pick_parents_rdp5ml.py (sim-csv mode);")
    print("   re-run that script if .faParents.csv files are missing.")


if __name__ == "__main__":
    cap = None
    if len(sys.argv) > 1:
        cap = int(sys.argv[1])
        print(f"(audit mode: capping to {cap} files per directory)\n")
    _print_audit(max_files_per_dir=cap)
