"""Faithful port of event_classifier.py for picking (MinorParent, MajorParent)
per recombination event, given a SANTA simulation's outputs.

Source: RDP-ML-REDUX/event_classifier.py (commit pre-2025-05).

Algorithm preserved bit-for-bit, vectorised for ~50-200x speedup:
  - Hamming distance with gap-pair removal (vectorised via numpy bool ops)
  - 2x weight for nucleotides within +/-200 bp of breakpoints
  - Hypergeometric CI approximation for variable-length normalisation
  - Joint best-pair scoring (minor close in X & far in Y; major inverse)
  - Score gate: pair_score < 1.99 OR (sum1 < 0.75 or sum2 < 0.75)
  - deleted_nucleotides reverse-order cross-event tracking (per-sequence)
  - Generation matrix built in ASCENDING event order so higher events
    overwrite lower (matches original)

Modes:
  - santa-logs: read `recombination_events_<key>.txt` and
    `sequence_events_map_<key>.txt` directly (faithful to original).
  - sim-csv-only: reconstruct equivalent info from
    `.faSimVSRealCompare.csv` (each row's ActualRecomb+SimBPStart/End +
    grouping rows by sequence ID to recover per-seq event lineage).
    Approximation: we don't see events RDP5 failed to detect on legacy
    data; for long_content data SimVSRealCompare lists every event.

Output: writes `.faParents.csv` next to the .fa file, with columns
`Event,Parent1,Parent2` (Parent1=MinorParent, Parent2=MajorParent).
"""
from __future__ import annotations
import argparse
import csv
import ast
import re
from pathlib import Path
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from Bio import AlignIO


GAP = 4  # encoding for '-'
_NUC_LUT = None


def _nucleotide_lut():
    global _NUC_LUT
    if _NUC_LUT is None:
        lut = np.full(256, GAP, dtype=np.int8)
        for b, v in [(ord('A'), 0), (ord('T'), 1), (ord('G'), 2), (ord('C'), 3), (ord('-'), GAP)]:
            lut[b] = v
        _NUC_LUT = lut
    return _NUC_LUT


def encode_alignment(fasta_path: Path):
    """Returns (seqs_codes: (n_seqs, max_len) int8, seq_ids: list[int], max_len: int).
    seq_ids[i] is the integer FASTA id (1-based as written in SANTA output).
    """
    alignment = AlignIO.read(fasta_path, 'fasta')
    max_len = alignment.get_alignment_length()
    n = len(alignment)
    out = np.full((n, max_len), GAP, dtype=np.int8)
    ids = []
    lut = _nucleotide_lut()
    for i, r in enumerate(alignment):
        try:
            ids.append(int(r.id))
        except ValueError:
            ids.append(-1)
        s = bytes(str(r.seq), 'ascii')
        arr = lut[np.frombuffer(s, dtype=np.uint8)]
        out[i, :len(arr)] = arr[:max_len]
    return out, ids, max_len


# ---------- Vectorised hypergeometric CI + normalised distance score ----------

def hyper_ci_vec(x_arr: np.ndarray, n_arr: np.ndarray, N_arr) -> np.ndarray:
    """Vectorised hypergeometric CI normal approximation (UCL/N only).

    Matches event_classifier.hyper_ci_approximation:
      p = (x + 1) / (n + 2)
      cor = (N - n) / (N - 1)
      me = 2.575829 * sqrt(cor * p * (1-p) / (n + 4))
      ucl = p + me
      UCL = min(N, round(N * ucl))
      out = UCL / N
      if x == 0 and n < 20: out = 1.5 * UCL / N   (heuristic correction)
    """
    x = np.asarray(x_arr, dtype=np.float64)
    n = np.asarray(n_arr, dtype=np.float64)
    N = np.broadcast_to(np.asarray(N_arr, dtype=np.float64), x.shape).astype(np.float64)
    out = np.zeros_like(x)
    safe = (n > 0) & (N > 0)
    p = np.zeros_like(x)
    p[safe] = (x[safe] + 1) / (n[safe] + 2)
    cor = np.ones_like(x)
    big_N = safe & (N > 1)
    cor[big_N] = (N[big_N] - n[big_N]) / (N[big_N] - 1)
    inner = np.zeros_like(x)
    inner[safe] = cor[safe] * p[safe] * (1 - p[safe]) / (n[safe] + 4)
    np.clip(inner, 0, None, out=inner)
    me = np.zeros_like(x)
    me[safe] = 2.575829 * np.sqrt(inner[safe])
    ucl = p + me
    UCL = np.minimum(N, np.round(N * ucl))
    out[safe] = UCL[safe] / N[safe]
    # heuristic correction for x==0 and n<20
    correction = (x == 0) & (n < 20) & safe
    out[correction] = 1.5 * out[correction]
    return out


def normalised_distance_score_vec(dist: np.ndarray, n_compared: np.ndarray,
                                    fragment_length) -> np.ndarray:
    """Returns the normalised distance per candidate. inf if n_compared==0.
    fragment_length: scalar OR per-candidate array (broadcast).
    """
    dist = np.asarray(dist, dtype=np.float64)
    n_compared = np.asarray(n_compared, dtype=np.float64)
    fl = np.broadcast_to(np.asarray(fragment_length, dtype=np.float64), dist.shape).astype(np.float64)
    out = np.full_like(dist, np.inf)
    mask_full = (n_compared == fl) & (n_compared > 0)
    out[mask_full] = dist[mask_full] / n_compared[mask_full]
    mask_ci = (~mask_full) & (n_compared > 0)
    if mask_ci.any():
        out[mask_ci] = hyper_ci_vec(dist[mask_ci], n_compared[mask_ci], fl[mask_ci])
    return out


# ---------- Vectorised best-pair selection ----------

def find_best_pair_vec(minor_scores: np.ndarray, major_scores: np.ndarray,
                        valid_mask: np.ndarray):
    """Vectorised port of event_classifier.findBestParentPair.

    Score for each (i, j) pair:
      distance_X_minor = minor_scores[i] (default 1 if None/inf)
      distance_Y_minor = major_scores[i] (default 0 if None/inf)
      distance_X_major = minor_scores[j] (default 0 if None/inf)
      distance_Y_major = major_scores[j] (default 1 if None/inf)
      sum1 = distance_X_minor + (1 - distance_Y_minor)
      sum2 = distance_Y_major + (1 - distance_X_major)
      pair_score = sum1 + sum2
    Gate: pair_score < 1.99 OR (sum1 < 0.75 OR sum2 < 0.75)
    Pick the (i, j) with min pair_score among eligible.

    Returns (best_minor_idx, best_major_idx, score) or (None, None, inf).
    """
    n = minor_scores.shape[0]
    if n == 0:
        return None, None, float('inf')
    distance_X_minor = np.where(np.isfinite(minor_scores), minor_scores, 1.0)
    distance_X_major = np.where(np.isfinite(minor_scores), minor_scores, 0.0)
    distance_Y_minor = np.where(np.isfinite(major_scores), major_scores, 0.0)
    distance_Y_major = np.where(np.isfinite(major_scores), major_scores, 1.0)

    sum1_per_i = distance_X_minor + (1.0 - distance_Y_minor)
    sum2_per_j = distance_Y_major + (1.0 - distance_X_major)
    pair_score = sum1_per_i[:, None] + sum2_per_j[None, :]

    sum1_grid = np.broadcast_to(sum1_per_i[:, None], pair_score.shape)
    sum2_grid = np.broadcast_to(sum2_per_j[None, :], pair_score.shape)
    eligible = (pair_score < 1.99) | (sum1_grid < 0.75) | (sum2_grid < 0.75)

    bad = ~valid_mask
    pair_score[bad, :] = np.inf
    pair_score[:, bad] = np.inf
    eligible[bad, :] = False
    eligible[:, bad] = False
    eligible_score = np.where(eligible, pair_score, np.inf)

    flat = int(np.argmin(eligible_score))
    best_i, best_j = divmod(flat, eligible_score.shape[1])
    best_score = float(eligible_score[best_i, best_j])
    if not np.isfinite(best_score):
        return None, None, float('inf')
    return best_i, best_j, best_score


# ---------- Input adapters ----------

def load_santa_logs(rec_events_path: Path, seq_events_path: Path,
                    max_genome_length: int, ungapped_length_of_seq1: int):
    """Faithful port of event_classifier.readFiles + create_dictionaries +
    the ungapped-length end-position fix.
    """
    rec_events = pd.read_csv(rec_events_path, sep=r'*',
                              usecols=['EventNum', 'Breakpoints', 'Generation'],
                              engine='python')

    # Handle zero-event simulations gracefully: nothing to do
    if len(rec_events) == 0:
        events_dict = {}
    else:
        rec_events['Breakpoints'] = rec_events['Breakpoints'].astype(str).str.strip('[]')

        if ungapped_length_of_seq1 != max_genome_length:
            for i, bps in enumerate(rec_events['Breakpoints']):
                start_pos = bps.split(',')[0]
                end_pos = int(bps.split(',')[-1])
                if int(end_pos) == ungapped_length_of_seq1:
                    rec_events['Breakpoints'].iat[i] = start_pos + ', ' + str(max_genome_length)

        starts_ends = rec_events['Breakpoints'].str.split(',', expand=True)
        rec_events['Start'] = starts_ends[0].astype(int)
        rec_events['End'] = starts_ends[1].astype(int)

        events_dict = {int(ev): [int(s), int(e)] for ev, s, e in zip(
            rec_events['EventNum'], rec_events['Start'], rec_events['End'])}

    seq_events = pd.read_csv(seq_events_path, sep=r'*', engine='python', index_col='Sequence')
    seqmap_dict = {}
    for sid, ev_str in enumerate(seq_events['Events'].to_numpy(), 1):
        try:
            seqmap_dict[sid] = ast.literal_eval(ev_str)
        except (ValueError, SyntaxError):
            seqmap_dict[sid] = []

    inv_seqmap_dict = defaultdict(set)
    for sid, events in seqmap_dict.items():
        for ev in events:
            inv_seqmap_dict[ev].add(sid)
    return events_dict, seqmap_dict, inv_seqmap_dict


def reconstruct_from_sim_csv(sim_csv_path: Path):
    """Reconstruct (events_dict, seqmap_dict, inv_seqmap_dict) from
    `.faSimVSRealCompare.csv`. Uses SimEVentNo column when present
    (SANTA's event number) — falls back to RDPEvent when not (long_content data).
    """
    df = pd.read_csv(sim_csv_path, skipinitialspace=True, index_col=False)
    event_col = 'SimEVentNo' if 'SimEVentNo' in df.columns else 'RDPEvent'
    events_dict = {}
    inv_seqmap_dict = defaultdict(set)
    seqmap_dict = defaultdict(list)
    seen = set()
    for _, row in df.iterrows():
        try:
            ev = int(row[event_col])
            rid = int(row['ActualRecomb'])
            s = int(row['SimBPStart'])
            e = int(row['SimBPEnd'])
        except (KeyError, TypeError, ValueError):
            continue
        if s >= e:
            continue
        events_dict[ev] = [s, e]
        if (ev, rid) not in seen:
            inv_seqmap_dict[ev].add(rid)
            seqmap_dict[rid].append(ev)
            seen.add((ev, rid))
    return events_dict, dict(seqmap_dict), inv_seqmap_dict


# ---------- Main classifier ----------

class Classifier:
    """Vectorised port of event_classifier.classifier."""

    def __init__(self, fasta_path: Path,
                 rec_events_path: Path | None = None,
                 seq_events_path: Path | None = None,
                 sim_csv_path: Path | None = None):
        self.fasta_path = fasta_path

        # Encode alignment once
        self.seqs_codes, ids, self.max_genome_length = encode_alignment(fasta_path)
        if not ids:
            raise ValueError("Empty/invalid alignment")
        self.number_of_seqs = len(ids)
        # Map 1-indexed FASTA id <-> 0-indexed matrix row
        self.seq_ids_sorted = sorted(set(s for s in ids if s > 0))
        self.id_to_idx = {sid: i for i, sid in enumerate(ids)}  # original FASTA order

        # Ungapped length of sequence with id == 1 (matches original)
        idx_id1 = self.id_to_idx.get(1)
        if idx_id1 is None:
            idx_id1 = 0
        ungapped_len = int((self.seqs_codes[idx_id1] != GAP).sum())

        # Load events / lineage
        if rec_events_path and seq_events_path and rec_events_path.exists() and seq_events_path.exists():
            self.events_dict, self.seqmap_dict, self.inv_seqmap_dict = load_santa_logs(
                rec_events_path, seq_events_path, self.max_genome_length, ungapped_len)
            self.source = 'santa-logs'
        elif sim_csv_path and sim_csv_path.exists():
            self.events_dict, self.seqmap_dict, self.inv_seqmap_dict = reconstruct_from_sim_csv(sim_csv_path)
            self.source = 'sim-csv'
        else:
            raise ValueError("Need either SANTA logs or SimVSRealCompare.csv")

        # Build generation matrix (numpy int64) — events in ASCENDING order so
        # higher events overwrite lower at overlapping cells (matches original).
        self.generation_matrix = np.zeros(
            (self.number_of_seqs, self.max_genome_length), dtype=np.int64)
        for event in sorted(self.events_dict.keys()):
            start, end = self.events_dict[event]
            if start >= end or start < 0 or end > self.max_genome_length:
                continue
            seqs = self.inv_seqmap_dict.get(event, set())
            indices = [self.id_to_idx[sid] for sid in seqs if sid in self.id_to_idx]
            if not indices:
                continue
            self.generation_matrix[indices, start:end] = event

    def _find_event_positions(self):
        """Returns block_dict[event] = {seq_idx_0based: [[start, end), ...]}.
        Equivalent to event_classifier.findEventPositions."""
        block_dict = {ev: {} for ev in self.events_dict.keys()}
        # Vectorised: per row, find runs of equal nonzero values
        for seq_idx in range(self.number_of_seqs):
            row = self.generation_matrix[seq_idx]
            # transitions: where value changes
            change = np.diff(row, prepend=-1, append=-1)
            run_starts = np.flatnonzero(change != 0)
            for i in range(len(run_starts) - 1):
                s = int(run_starts[i])
                e = int(run_starts[i + 1])
                v = int(row[s])
                if v == 0:
                    continue
                if v not in block_dict:
                    continue
                block_dict[v].setdefault(seq_idx, []).append([s, e])
        return block_dict

    def calc_parents(self):
        """Returns {event_num: [(recomb_id_1idx, minor_id_1idx, major_id_1idx, score), ...]}.

        Same algorithm as event_classifier:
          - iterate events in REVERSE order (highest first)
          - for each recombinant in this event's block: compute parent scores
          - track deletion_mask per parent (positions used by higher events)
        """
        block_dict = self._find_event_positions()
        n_seqs = self.number_of_seqs
        max_len = self.max_genome_length

        # Per-sequence position deletion mask (True = position is "deleted"
        # for that parent because it was claimed by a higher-numbered event).
        # Replaces deleted_nucleotides IntervalTree with a fast bool mask.
        deletion_mask = np.zeros((n_seqs, max_len), dtype=bool)

        # Per-position precomputes that don't depend on the event
        parent_valid_all = (self.seqs_codes != GAP)  # (n_seqs, max_len)

        results = {}
        for event_number in sorted(block_dict.keys(), reverse=True):
            sequence_ranges_dict = block_dict[event_number]
            if not sequence_ranges_dict:
                continue
            bp_s, bp_e = self.events_dict[event_number]
            if bp_s >= bp_e:
                continue

            sequences_in_block = set(sequence_ranges_dict.keys())

            # Region base masks (same across all recombinants for this event)
            pos = np.arange(max_len)
            outer_base = np.ones(max_len, dtype=bool)
            outer_base[bp_s:bp_e] = False
            close_mask = np.zeros(max_len, dtype=bool)
            cs1, ce1 = max(0, bp_s - 200), min(max_len, bp_s + 200 + 1)
            cs2, ce2 = max(0, bp_e - 200), min(max_len, bp_e + 200 + 1)
            if cs1 < ce1:
                close_mask[cs1:ce1] = True
            if cs2 < ce2:
                close_mask[cs2:ce2] = True
            far_mask = ~close_mask

            event_results = []
            for sequence_idx, ranges in sequence_ranges_dict.items():
                # Recombinant's per-position state
                recomb_row = self.seqs_codes[sequence_idx]
                recomb_valid = (recomb_row != GAP)
                # eq[p, j] = (seqs[p, j] == recomb[j])  shape (n_seqs, max_len)
                eq = (self.seqs_codes == recomb_row[None, :])
                joint_valid = parent_valid_all & recomb_valid[None, :]
                # available[p, j] = joint_valid AND not deleted
                available = joint_valid & ~deletion_mask
                mismatch = (~eq) & available

                # Minor region = this recombinant's RANGES (from generation matrix)
                # Matches original: ranges_tree_minor = IntervalTree.from_tuples(ranges)
                minor_base = np.zeros(max_len, dtype=bool)
                for s, e in ranges:
                    if s < e:
                        minor_base[s:e] = True

                minor_close = minor_base & close_mask
                minor_far = minor_base & far_mask
                major_close = outer_base & close_mask
                major_far = outer_base & far_mask

                def region_stats(region_mask: np.ndarray):
                    if not region_mask.any():
                        zero = np.zeros(n_seqs, dtype=np.int64)
                        return zero, zero
                    d = mismatch[:, region_mask].sum(axis=1)
                    n = available[:, region_mask].sum(axis=1)
                    return d, n

                ic_d, ic_n = region_stats(minor_close)
                if_d, if_n = region_stats(minor_far)
                oc_d, oc_n = region_stats(major_close)
                of_d, of_n = region_stats(major_far)

                # Per-parent fragment lengths (close region 2x-weighted contributes
                # 2x to the denominator too, matching original).
                recomb_block_length = (bp_e - bp_s) + ic_n
                major_block_length = (max_len - (bp_e - bp_s)) + oc_n

                # 2x weighting on close region (matches original)
                ic_d_w = 2 * ic_d
                ic_n_w = 2 * ic_n
                oc_d_w = 2 * oc_d
                oc_n_w = 2 * oc_n

                minor_d = ic_d_w + if_d
                minor_n = ic_n_w + if_n
                major_d = oc_d_w + of_d
                major_n = oc_n_w + of_n

                minor_scores = normalised_distance_score_vec(
                    minor_d.astype(np.float64), minor_n.astype(np.float64),
                    recomb_block_length.astype(np.float64))
                major_scores = normalised_distance_score_vec(
                    major_d.astype(np.float64), major_n.astype(np.float64),
                    major_block_length.astype(np.float64))

                # Restrict candidates: original passes `sequences_not_in_block`
                # (sequences whose lineage does NOT contain this event).
                valid_mask = np.ones(n_seqs, dtype=bool)
                for s in sequences_in_block:
                    valid_mask[s] = False

                best_i, best_j, best_score = find_best_pair_vec(
                    minor_scores, major_scores, valid_mask)
                if best_i is not None:
                    # Indices best_i, best_j are 0-based matrix rows. The original
                    # used 0-based sequence_idx + 1 as output id. We have the
                    # original FASTA id directly via id reverse lookup.
                    # But event_classifier.py outputs ids as `parent + 1` where
                    # parent is 0-based — this assumes FASTA ids are 1..N
                    # contiguous. We preserve that by emitting best_i + 1 etc.
                    # (which matches `seq_ids[best_i]` only when FASTA ids are 1..N).
                    # For all SANTA-generated data here, FASTA ids ARE 1..N.
                    minor_id_1idx = best_i + 1
                    major_id_1idx = best_j + 1
                    recomb_id_1idx = sequence_idx + 1
                    event_results.append((recomb_id_1idx, minor_id_1idx, major_id_1idx, best_score))

            # AFTER processing this event, mark its ranges as deleted for the
            # involved sequences (so lower-numbered events ignore those positions).
            for seq_idx, ranges_list in sequence_ranges_dict.items():
                for s, e in ranges_list:
                    if s < e:
                        deletion_mask[seq_idx, s:e] = True

            results[event_number] = event_results
        return results


def write_faparents_csv(out_path: Path, results: dict):
    """One row per event (de-duped). Parent1=Minor, Parent2=Major."""
    seen = set()
    rows = []
    for event in sorted(results.keys()):
        for (recomb, minor, major, score) in results[event]:
            if event in seen:
                continue
            rows.append({'Event': event, 'Parent1': minor, 'Parent2': major})
            seen.add(event)
            break
    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['Event', 'Parent1', 'Parent2'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def derive_paths(fa_path: Path):
    """Locate SANTA log + companion files. Handles two naming conventions:
      - Legacy XML-1..5: alignment_<key>.fa  +  recombination_events_<key>.txt
      - New long_content_30k_003+: <base>.fa  +  <base>.recombination_events.txt
    """
    name = fa_path.name
    parent = fa_path.parent

    # Try legacy "recombination_events_<key>.txt" first
    m = re.search(r'(?<=alignment_).*', name)
    if m:
        key = m.group()
        if key.endswith('.fa'):
            key = key[:-3]
    else:
        key = name[:-3] if name.endswith('.fa') else name
    rec_legacy = parent / f'recombination_events_{key}.txt'
    seq_legacy = parent / f'sequence_events_map_{key}.txt'

    # New-style: appended to the .fa basename
    base = name[:-3] if name.endswith('.fa') else name
    rec_new = parent / f'{base}.recombination_events.txt'
    seq_new = parent / f'{base}.sequence_events_map.txt'

    rec_events = rec_legacy if rec_legacy.exists() else rec_new
    seq_events = seq_legacy if seq_legacy.exists() else seq_new

    return {
        'rec_events': rec_events,
        'seq_events': seq_events,
        'sim_csv': parent / f'{name}SimVSRealCompare.csv',
        'parents_out': parent / f'{name}Parents.csv',
        'rdp5ml_existing': parent / f'RPD_Output_{key}.rdp5ML',
    }


def process_file(fa_path: Path):
    paths = derive_paths(fa_path)
    has_logs = paths['rec_events'].exists() and paths['seq_events'].exists()
    has_sim = paths['sim_csv'].exists()
    if not (has_logs or has_sim):
        return {'error': 'no inputs'}
    cls = Classifier(
        fa_path,
        rec_events_path=paths['rec_events'] if has_logs else None,
        seq_events_path=paths['seq_events'] if has_logs else None,
        sim_csv_path=paths['sim_csv'] if has_sim else None,
    )
    results = cls.calc_parents()
    n_written = write_faparents_csv(paths['parents_out'], results)
    return {'source': cls.source, 'events_processed': len(results),
            'events_written': n_written}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('fa', type=Path)
    args = p.parse_args()
    import time
    t0 = time.time()
    print(process_file(args.fa), f'(elapsed {time.time()-t0:.2f}s)')
