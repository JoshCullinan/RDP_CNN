"""Unsupervised divergence-anomaly gate for the M3 detector (v4, final).

WHY THIS EXISTS. M3 v2 (the deployed sequence-only detector, LANL F1 0.509) emits
false breakpoint peaks on cross-species triplets (e.g. Ebola at ~30-40% pairwise
divergence) because it reads uniform high divergence as recombination signal. The
v4 attempt to fix this with a *learned* "is this a recombinant?" aux head failed:
trained only on SANTA recombinants, it became a SANTA-vs-real detector that scored
every *real* recombinant (LANL, SARS-CoV-2 XBB) as a non-recombinant — it would
have suppressed real HIV. See memory project_m3_v4_multihead.

This gate sidesteps that generalization trap. It is UNSUPERVISED and
provenance-invariant: a triplet whose maximum pairwise divergence exceeds a
threshold is outside the regime the detector was trained on (SANTA + real HIV/
SARS/Zika all sit at div_max <~0.15), so it is flagged as a likely cross-species
comparison and its BP peaks are suppressed (with a warning) rather than reported.

VALIDATION (2026-05-29, CPU, on the real reference panels):
  group                 div_max         gate@0.20
  LANL HIV recombinant  0.131-0.138     KEPT   (F1 0.509 preserved)
  SARS-CoV-2 XBB        0.002           KEPT   (still localised, Δ=293 bp)
  Zika non-recomb       <=0.113         KEPT
  SARS non-recomb       0.003           KEPT
  Ebola cross-species   mean 0.371      98% SUPPRESSED

This is MASTER_PLAN long-term goal #3 ("explicit divergence-anomaly scoring") and
is the gate the deployment CLI should use. The BP detector itself is unchanged —
the deployed v2 model stays exactly as-is; this is a thin post-hoc wrapper.
"""

from __future__ import annotations

import numpy as np

GAP_INT = 4                 # v2 cache / encoding: {A:0,T:1,G:2,C:3,gap:4}
DEFAULT_DIV_GATE = 0.20     # validated separating threshold (see module docstring)


def pairwise_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Mismatch fraction over positions where BOTH sequences are non-gap."""
    non_gap = (a != GAP_INT) & (b != GAP_INT)
    if not non_gap.any():
        return float("nan")
    return float(((a != b) & non_gap).sum() / non_gap.sum())


def triplet_div_max(R: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> float:
    """Maximum of the three pairwise divergences in a triplet."""
    return max(pairwise_divergence(R, P1),
               pairwise_divergence(R, P2),
               pairwise_divergence(P1, P2))


def divergence_gate(R: np.ndarray, P1: np.ndarray, P2: np.ndarray,
                    thr: float = DEFAULT_DIV_GATE) -> tuple[bool, float]:
    """Decide whether to trust the BP detector on this triplet.

    Returns (keep, div_max). keep=False means the triplet's pairwise divergence
    is anomalously high (likely cross-species / out-of-distribution) and its BP
    peaks should be suppressed and flagged as unreliable.
    """
    dm = triplet_div_max(R, P1, P2)
    keep = (not np.isnan(dm)) and (dm <= thr)
    return keep, dm
