"""Round-trip + invariant tests for the v2 cache (M0.2 success criterion).

Run: python3 test_cache_v2.py --cache-root <path>

Builds against a small smoke cache by default. Tests:

  1. Round-trip: ≥10 random events decoded from the cache match the
     original FASTA byte-for-byte (under the {A,T,G,C,-} alphabet — any
     non-ATGC bases in the source map to gap and stay as gap).
  2. Event invariants: parent IDs differ from each other and from the
     recombinant; bp_start ≤ bp_end ≤ seq_len.
  3. Random-access latency: per-event get_triplet() under 50 ms.
  4. Sequence-flags coverage: every event's recomb_id is marked as
     is_recombinant=1 in seq_flags.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from build_cache_v2 import GAP_INT, NT_TO_INT, build_all
from cache_v2_reader import CacheV2, decode_int8_to_str
from data_loader_v2 import load_events


def canonicalize(s: str) -> str:
    """Map source FASTA bases through the same lookup the cache uses.

    Anything that isn't A/T/G/C/- becomes '-' (matches encode_alignment).
    """
    out = []
    for c in s:
        b = ord(c)
        if b in NT_TO_INT:
            out.append(chr(b))
        else:
            out.append("-")
    return "".join(out)


def test_round_trip(cache: CacheV2, n_samples: int = 10,
                    seed: int = 0) -> None:
    rng = random.Random(seed)
    samples_checked = 0
    for shard_name, shard in cache.shards.items():
        if not len(shard):
            continue
        if samples_checked >= n_samples:
            break
        # Build a name -> source-FASTA index mapping for this shard.
        src_dir = shard_name
        loader_iter = load_events(src_dir)
        # Build a (event_id, recomb_id) → event dict for fast lookup.
        loaded = {}
        for ev in loader_iter:
            key = (ev.source_file, ev.event_id, ev.recomb_id)
            loaded[key] = ev
        # Pick a few cached events and find their source counterparts.
        idxs = rng.sample(range(len(shard)), min(3, len(shard)))
        for ix in idxs:
            cached = shard.get_triplet(ix)
            key = (cached["source_file"], cached["event_id"],
                   cached["recomb_id"])
            if key not in loaded:
                raise AssertionError(
                    f"{shard_name}: cached event {key} not found by loader"
                )
            src = loaded[key]
            for which, c_seq, s_seq in [
                ("R", cached["R"], src.R_seq),
                ("P1", cached["P1"], src.P1_seq),
                ("P2", cached["P2"], src.P2_seq),
            ]:
                expected = canonicalize(s_seq)
                got = decode_int8_to_str(c_seq)
                # The cached sequence is padded to the alignment's max_len —
                # in SANTA the alignment is rectangular so this length matches
                # the source string. Still, guard against any mismatch.
                if len(got) != len(expected):
                    raise AssertionError(
                        f"{shard_name} event {ix} {which}: len {len(got)} "
                        f"!= source len {len(expected)}"
                    )
                if got != expected:
                    diffs = sum(a != b for a, b in zip(got, expected))
                    raise AssertionError(
                        f"{shard_name} event {ix} {which}: {diffs} char diffs "
                        f"(first 60 cached vs source):\n  {got[:60]}\n  "
                        f"{expected[:60]}"
                    )
                samples_checked += 1
                if samples_checked >= n_samples:
                    break
            print(f"  [round-trip] {shard_name} event {ix} (file="
                  f"{cached['source_file'][:50]}, rec={cached['recomb_id']}, "
                  f"p1={cached['parent1_id']}, p2={cached['parent2_id']}, "
                  f"bps=({cached['bp_start']},{cached['bp_end']})) ✓",
                  flush=True)
            if samples_checked >= n_samples:
                break
    assert samples_checked >= n_samples, (
        f"Only checked {samples_checked} samples (wanted {n_samples})"
    )
    print(f"  Round-trip: {samples_checked} sequences verified.")


def test_event_invariants(cache: CacheV2) -> None:
    checked = 0
    for shard_name, shard in cache.shards.items():
        for i in range(min(len(shard), 100)):
            t = shard.get_triplet(i)
            assert t["recomb_id"] != t["parent1_id"]
            assert t["recomb_id"] != t["parent2_id"]
            assert t["parent1_id"] != t["parent2_id"]
            assert 0 <= t["bp_start"] <= t["bp_end"] <= t["seq_len"]
            checked += 1
    print(f"  Invariants: {checked} events passed.")


def test_random_access_latency(cache: CacheV2, n: int = 200,
                               threshold_ms: float = 50.0) -> None:
    rng = random.Random(1)
    shards = [s for s in cache.shards.values() if len(s) > 0]
    if not shards:
        print("  Latency: no events, skipping.")
        return
    times = []
    for _ in range(n):
        s = rng.choice(shards)
        ix = rng.randrange(len(s))
        t0 = time.perf_counter()
        _ = s.get_triplet(ix)
        times.append((time.perf_counter() - t0) * 1000.0)
    p50 = sorted(times)[n // 2]
    p99 = sorted(times)[int(n * 0.99)]
    print(f"  Latency over {n} random reads: p50={p50:.2f} ms, p99={p99:.2f} ms")
    assert p99 < threshold_ms, (
        f"p99 {p99:.2f} ms exceeds threshold {threshold_ms} ms"
    )


def test_seq_flag_coverage(cache: CacheV2) -> None:
    """Every event's recomb_id must be marked is_recombinant=1."""
    for shard_name, shard in cache.shards.items():
        if not len(shard.events):
            continue
        recomb_set = set(
            (int(r["file_idx"]), int(r["recomb_id"]))
            for r in shard.events
        )
        flag_set = set(
            (int(r["file_idx"]), int(r["fasta_id"]))
            for r in shard.seq_flags
            if r["is_recombinant"]
        )
        missing = recomb_set - flag_set
        assert not missing, (
            f"{shard_name}: {len(missing)} recomb_ids missing from seq_flags "
            f"(first 5: {list(missing)[:5]})"
        )
    print("  Seq-flag coverage: every recomb_id flagged.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path,
                    default=Path("/tmp/cache_v2_smoke"),
                    help="Path to an existing v2 cache. If --build is set, "
                         "rebuild it at this path before testing.")
    ap.add_argument("--build", action="store_true",
                    help="Rebuild a smoke cache (XML-1 + long_content_30k_001, "
                         "5 files each) before running tests.")
    ap.add_argument("--n-samples", type=int, default=10)
    args = ap.parse_args()

    if args.build:
        print("Building smoke cache...")
        build_all(cache_root=args.cache_root,
                  dirs=["XML-1", "long_content_30k_001"],
                  max_files=5)

    print(f"\nOpening cache at {args.cache_root}")
    cache = CacheV2(args.cache_root)
    print(f"  shards: {list(cache.shards.keys())}")
    print(f"  total events: {cache.n_events:,}")
    print(f"  total files:  {cache.n_files}")
    print()

    print("Test 1: round-trip")
    test_round_trip(cache, n_samples=args.n_samples)
    print()

    print("Test 2: event invariants")
    test_event_invariants(cache)
    print()

    print("Test 3: random-access latency")
    test_random_access_latency(cache)
    print()

    print("Test 4: seq-flag coverage")
    test_seq_flag_coverage(cache)
    print()

    print("All M0.2 tests passed.")


if __name__ == "__main__":
    main()
