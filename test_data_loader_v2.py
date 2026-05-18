"""Smoke tests for data_loader_v2.

Run: python3 test_data_loader_v2.py

Checks:
  - Each schema (legacy XML, long-content) yields at least one event.
  - XML-6 yields zero events (no parent CSV on disk — known limitation).
  - Every yielded event has matching string lengths for R, P1, P2.
  - bp_positions fall within original_len.
  - recomb_id != parent1_id != parent2_id.
"""

from data_loader_v2 import load_events, ALL_SOURCE_DIRS


def check_event_invariants(evt, idx: int) -> None:
    assert evt.recomb_id != evt.parent1_id, f"event {idx}: recomb_id == parent1_id"
    assert evt.recomb_id != evt.parent2_id, f"event {idx}: recomb_id == parent2_id"
    assert evt.parent1_id != evt.parent2_id, f"event {idx}: parent1_id == parent2_id"
    bp_start, bp_end = evt.bp_positions
    assert 0 <= bp_start <= bp_end <= evt.original_len, (
        f"event {idx}: bps {evt.bp_positions} out of range for len {evt.original_len}"
    )
    assert len(evt.R_seq) == len(evt.P1_seq) == len(evt.P2_seq), (
        f"event {idx}: seq lengths mismatch "
        f"R={len(evt.R_seq)} P1={len(evt.P1_seq)} P2={len(evt.P2_seq)}"
    )


def test_schema(directory: str, expect_nonzero: bool, max_files: int = 3) -> None:
    print(f"  testing {directory} (max_files={max_files})...", end=" ", flush=True)
    events = list(load_events(directory, max_files=max_files))
    if expect_nonzero:
        assert events, f"{directory}: expected events but got 0"
    else:
        assert not events, f"{directory}: expected 0 events but got {len(events)}"
    for i, e in enumerate(events):
        check_event_invariants(e, i)
    print(f"OK ({len(events)} events)")


def main() -> None:
    print("Schema A — legacy XML (RecombIdentifyStats):")
    for d in ("XML-1", "XML-2", "XML-3", "XML-4", "XML-5"):
        test_schema(d, expect_nonzero=True)

    print("Schema B — XML-6 (no parent CSV, expected unusable):")
    test_schema("XML-6", expect_nonzero=False, max_files=5)

    print("Schema C — long-content (Parents.csv):")
    for d in ("long_content_30k_001", "long_content_30k_002", "long_content_30k_003"):
        test_schema(d, expect_nonzero=True)

    print("UnseenTestSet (held-out test, schema A):")
    test_schema("UnseenTestSet", expect_nonzero=True)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
