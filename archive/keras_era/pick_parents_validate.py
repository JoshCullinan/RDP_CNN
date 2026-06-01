"""Validate a Python parent-picking heuristic against RDP5's picks on legacy data.

For each event in an XML-1..5 file, we know RDP5's picks from .faRecombIdentifyStats.csv.
We run our heuristic and compare:
  - Do we pick the same IDs?
  - If different, is informative% similar?

The heuristic: for each event with (recomb_id, bp_start, bp_end), find two
sample sequences such that one matches recomb on the outer region and the
other matches recomb on the inner region. Maximize match scores; tie-break
by maximizing inter-parent divergence.
"""
import json
from pathlib import Path
import numpy as np
from Bio import SeqIO
import pandas as pd

DATA_ROOT = Path('dataRaw')


def pick_parents(seqs_array, recomb_idx, bp_start, bp_end, content_len):
    """seqs_array: np.array of shape (n_seqs, content_len) with int codes 0-4 (A/T/G/C/-).
    Returns (p1_idx, p2_idx) — best parent picks for outer/inner regions.
    """
    n_seqs = seqs_array.shape[0]
    recomb = seqs_array[recomb_idx]

    # Boolean mask of "informative content positions": not gap in recomb
    valid = (recomb != 4)  # 4 = gap
    outer_mask = np.zeros(content_len, dtype=bool)
    outer_mask[:bp_start] = True
    outer_mask[bp_end:] = True
    outer_mask &= valid
    inner_mask = np.zeros(content_len, dtype=bool)
    inner_mask[bp_start:bp_end] = True
    inner_mask &= valid

    n_outer = outer_mask.sum()
    n_inner = inner_mask.sum()
    if n_outer < 50 or n_inner < 50:
        return None  # event too edge-y

    # Match scores: how many positions match the recombinant in outer / inner
    outer_match = np.zeros(n_seqs, dtype=np.int32)
    inner_match = np.zeros(n_seqs, dtype=np.int32)
    for i in range(n_seqs):
        if i == recomb_idx:
            outer_match[i] = -1
            inner_match[i] = -1
            continue
        s = seqs_array[i]
        outer_match[i] = np.sum((s == recomb) & outer_mask)
        inner_match[i] = np.sum((s == recomb) & inner_mask)

    # Best outer parent: high outer_match, low inner_match (specialised)
    # Best inner parent: high inner_match, low outer_match
    outer_specialisation = outer_match / max(n_outer, 1) - inner_match / max(n_inner, 1)
    inner_specialisation = inner_match / max(n_inner, 1) - outer_match / max(n_outer, 1)

    # Score candidates by specialisation
    p1_idx = int(np.argmax(outer_specialisation))
    inner_specialisation_excl = inner_specialisation.copy()
    inner_specialisation_excl[p1_idx] = -1e9
    inner_specialisation_excl[recomb_idx] = -1e9
    p2_idx = int(np.argmax(inner_specialisation_excl))
    return p1_idx, p2_idx


def encode_seq(s, content_len=None):
    """Encode 'A T G C -' string to int array."""
    nuc_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}
    arr = np.fromiter((nuc_idx.get(c.upper(), 4) for c in s), dtype=np.int8)
    if content_len is not None and len(arr) < content_len:
        arr = np.concatenate([arr, np.full(content_len - len(arr), 4, dtype=np.int8)])
    elif content_len is not None:
        arr = arr[:content_len]
    return arr


def evaluate_pair(seqs_array, recomb_idx, p1_idx, p2_idx, bp_start, bp_end, content_len):
    """Compute informative% and match_p1/p2 stats for a given (recomb, p1, p2) triplet."""
    r = seqs_array[recomb_idx]
    p1 = seqs_array[p1_idx]
    p2 = seqs_array[p2_idx]
    valid = (r != 4) & (p1 != 4) & (p2 != 4)
    informative = (p1 != p2) & valid
    inf_pct = float(informative.sum()) / max(valid.sum(), 1) * 100

    outer = np.zeros(content_len, dtype=bool); outer[:bp_start] = True; outer[bp_end:] = True
    inner = np.zeros(content_len, dtype=bool); inner[bp_start:bp_end] = True
    outer &= valid
    inner &= valid

    outer_match_p1 = float(((r == p1) & outer).sum()) / max(outer.sum(), 1) * 100
    inner_match_p2 = float(((r == p2) & inner).sum()) / max(inner.sum(), 1) * 100
    return {'informative_pct': inf_pct,
            'outer_match_p1_pct': outer_match_p1,
            'inner_match_p2_pct': inner_match_p2}


def process_file(fa_path: Path, max_events=10):
    """Pick parents for events in this file via our heuristic, and compare
    to RDP5's picks if available."""
    sim_csv = fa_path.parent / f"{fa_path.name}SimVSRealCompare.csv"
    stats_csv = fa_path.parent / f"{fa_path.name}RecombIdentifyStats.csv"
    if not sim_csv.exists():
        return None

    # Read FASTA
    seqs_dict = {}
    for r in SeqIO.parse(fa_path, 'fasta'):
        seqs_dict[int(r.id)] = str(r.seq)
    if not seqs_dict:
        return None
    content_len = max(len(v) for v in seqs_dict.values())
    seq_ids = sorted(seqs_dict.keys())
    seqs_array = np.stack([encode_seq(seqs_dict[i], content_len) for i in seq_ids])
    id_to_idx = {sid: i for i, sid in enumerate(seq_ids)}

    sim = pd.read_csv(sim_csv, skipinitialspace=True, index_col=False)
    if stats_csv.exists():
        stats = pd.read_csv(stats_csv, skipinitialspace=True, index_col=False)
    else:
        stats = None

    rows = []
    seen = 0
    for _, row in sim.iterrows():
        try:
            event = row['RDPEvent']
            recomb_id = int(row['ActualRecomb'])
            bp_s = int(row['SimBPStart'])
            bp_e = int(row['SimBPEnd'])
        except (KeyError, TypeError, ValueError):
            continue
        if recomb_id not in id_to_idx:
            continue
        if bp_s >= bp_e or bp_s < 0 or bp_e > content_len:
            continue
        recomb_idx = id_to_idx[recomb_id]
        picked = pick_parents(seqs_array, recomb_idx, bp_s, bp_e, content_len)
        if picked is None:
            continue
        p1_idx, p2_idx = picked
        ours = evaluate_pair(seqs_array, recomb_idx, p1_idx, p2_idx, bp_s, bp_e, content_len)
        our_p1 = seq_ids[p1_idx]
        our_p2 = seq_ids[p2_idx]
        # Compare to RDP5
        rdp5 = None
        if stats is not None:
            ev_rows = stats[stats['Event'] == event]
            if len(ev_rows) == 3:
                rdp_parents = []
                for _, sr in ev_rows.iterrows():
                    ids = [int(s.strip()) for s in str(sr['ISeqs(A)']).split('$') if s.strip().isdigit()]
                    if recomb_id in ids:
                        continue
                    if ids:
                        rdp_parents.append(ids[0])
                if len(rdp_parents) >= 2:
                    rp1, rp2 = rdp_parents[0], rdp_parents[1]
                    if rp1 in id_to_idx and rp2 in id_to_idx:
                        rdp5 = evaluate_pair(seqs_array, recomb_idx,
                                             id_to_idx[rp1], id_to_idx[rp2], bp_s, bp_e, content_len)
                        rdp5['ids'] = (rp1, rp2)
        rows.append({
            'event': event, 'recomb_id': recomb_id, 'bp_s': bp_s, 'bp_e': bp_e,
            'our_picks': (our_p1, our_p2), 'our_eval': ours,
            'rdp5': rdp5,
        })
        seen += 1
        if seen >= max_events:
            break
    return rows


def main():
    # Validate against XML-1 (we have RDP5 picks here)
    print("=== VALIDATION on XML-1 (has RDP5 picks for ground-truth comparison) ===")
    xml1_files = sorted((DATA_ROOT / 'XML-1').glob('*.fa'))[:5]
    our_inf, rdp_inf = [], []
    same_picks = differ_picks = 0
    for fa in xml1_files:
        rows = process_file(fa, max_events=5)
        if not rows:
            continue
        for r in rows:
            print(f"  ev{int(r['event']):>3} bp[{r['bp_s']:>5},{r['bp_e']:>5}]: "
                  f"ours=({r['our_picks'][0]},{r['our_picks'][1]}) inf={r['our_eval']['informative_pct']:.1f}%  "
                  f"outer_m1={r['our_eval']['outer_match_p1_pct']:.0f}%  inner_m2={r['our_eval']['inner_match_p2_pct']:.0f}%", end='')
            if r['rdp5']:
                print(f"  | RDP5=({r['rdp5']['ids'][0]},{r['rdp5']['ids'][1]}) inf={r['rdp5']['informative_pct']:.1f}%")
                our_inf.append(r['our_eval']['informative_pct'])
                rdp_inf.append(r['rdp5']['informative_pct'])
                if set(r['our_picks']) == set(r['rdp5']['ids']):
                    same_picks += 1
                else:
                    differ_picks += 1
            else:
                print()
    if our_inf:
        print(f"\n  Mean informative%: ours={np.mean(our_inf):.2f}, RDP5={np.mean(rdp_inf):.2f}")
        print(f"  Pick agreement: {same_picks}/{same_picks+differ_picks} events same parents")

    # Validate on long_content_30k_002 (no RDP5 picks; just measure our quality)
    print("\n=== APPLY on long_content_30k_002 (no RDP5 ground truth) ===")
    new_files = sorted((DATA_ROOT / 'long_content_30k_002').glob('*.fa'))[:5]
    inf_ours, m1_ours, m2_ours = [], [], []
    for fa in new_files:
        rows = process_file(fa, max_events=5)
        if not rows:
            continue
        for r in rows:
            print(f"  ev{int(r['event']):>3} bp[{r['bp_s']:>5},{r['bp_e']:>5}]: "
                  f"ours=({r['our_picks'][0]},{r['our_picks'][1]}) inf={r['our_eval']['informative_pct']:.1f}%  "
                  f"outer_m1={r['our_eval']['outer_match_p1_pct']:.0f}%  inner_m2={r['our_eval']['inner_match_p2_pct']:.0f}%")
            inf_ours.append(r['our_eval']['informative_pct'])
            m1_ours.append(r['our_eval']['outer_match_p1_pct'])
            m2_ours.append(r['our_eval']['inner_match_p2_pct'])
    if inf_ours:
        print(f"\n  Mean informative% on new data with our picks: {np.mean(inf_ours):.2f}")
        print(f"  Mean outer_match_p1%: {np.mean(m1_ours):.1f}")
        print(f"  Mean inner_match_p2%: {np.mean(m2_ours):.1f}")
        print(f"  (Compare: lineage-heuristic picks gave 2.5% informative; legacy XML+RDP5 gives 7-15%)")


if __name__ == '__main__':
    main()
