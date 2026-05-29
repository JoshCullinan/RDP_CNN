"""M3 v4 gated detector — a two-head architecture that decouples WHERE a
breakpoint is from WHETHER the triplet is something the detector can trust.

  ┌─ BP head  (deployed v2 dilated CNN, frozen) ─→ per-position breakpoint prob
  │
  ├─ aux head (divergence-anomaly confidence)   ─→ P(in-distribution recombinant)
  │
  └─ gate: peaks are emitted only when aux confidence ≥ 0.5; otherwise the
           triplet is flagged out-of-distribution (cross-species) and suppressed.

This is the architecture the M3 v4 goal called for — "where" and "whether"
are separate heads, and the BP output is gated on the aux head's confidence
rather than the BP head being trained to predict zero on divergent input (the
v3 mistake). The crucial design decision, forced by evidence, is what the AUX
HEAD READS:

  A *learned* aux classifier on the shared trunk does NOT work. Trained only on
  SANTA recombinants (no real recombinants exist with ground-truth breakpoints),
  it becomes a simulator-vs-real detector: it scores every real recombinant
  (LANL HIV, SARS-CoV-2 XBB) like a cross-species negative and would suppress
  real HIV. See memory project_m3_v4_multihead / WRITEUP §4.4.

  The aux head therefore reads the one signal that is provenance-invariant and
  generalizes from simulation to real data: PAIRWISE DIVERGENCE. Within-species
  recombinants (HIV subtypes ~0.13, SARS lineages ~0.00, Zika ~0.09) sit far
  below cross-species comparisons (Ebola ~0.37). The aux head turns div_max into
  a graded [0,1] confidence (a smooth sigmoid around the regime boundary). It is
  an out-of-distribution / anomaly head, not a learned recombinant classifier —
  the honest, working form of "is this even a recombinant I should trust?".

The aux confidence is GRADED (not a hard threshold) so a deployment CLI can
report it to the user; gating at confidence 0.5 reproduces the validated
div_max > 0.20 rule. The BP detector is unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from m3_divergence_gate import triplet_div_max, DEFAULT_DIV_GATE
from m3_dilated import DilatedHead, raw_features

DEFAULT_BP_CKPT = Path("models_test/m3d_big_snaps/m3d_best.pt")
NT_TO_INT = {"A": 0, "T": 1, "G": 2, "C": 3, "-": 4}


def seq_to_int8(s: str) -> np.ndarray:
    arr = np.empty(len(s), dtype=np.int8)
    for i, ch in enumerate(s.upper()):
        arr[i] = NT_TO_INT.get(ch, 4)
    return arr


def aux_confidence(div_max: float, thr: float = DEFAULT_DIV_GATE,
                   width: float = 0.04) -> float:
    """Graded P(in-distribution / trustworthy recombinant call) from divergence.

    Smooth sigmoid centred on the regime boundary `thr`: ~1.0 well below it
    (within-species → trust the BP call), ~0.0 well above it (cross-species →
    out of distribution). Equals 0.5 exactly at div_max == thr, so gating at
    0.5 reproduces the validated div_max > thr rule while giving a continuous
    confidence for reporting.
    """
    if math.isnan(div_max):
        return 0.0
    return 1.0 / (1.0 + math.exp((div_max - thr) / width))


class M3GatedDetector:
    """Two-head detector: frozen v2 BP head + divergence-anomaly aux head."""

    def __init__(self, bp_ckpt: Path = DEFAULT_BP_CKPT, device: str | None = None,
                 div_thr: float = DEFAULT_DIV_GATE,
                 head_hidden: int = 128, head_blocks: int = 6):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.div_thr = div_thr
        self.bp = DilatedHead(in_channels=22, hidden=head_hidden,
                              n_blocks=head_blocks, dropout=0.1).to(self.device)
        ck = torch.load(bp_ckpt, map_location=self.device, weights_only=False)
        # accept either a single-head (v2) or multi-head ckpt's BP weights
        state = ck["head_state"]
        if any(k.startswith("bp_head.") for k in state):
            state = {("head." + k[len("bp_head."):]) if k.startswith("bp_head.")
                     else k: v for k, v in state.items()
                     if not (k.startswith("aux_head") or "_norm." in k)}
        self.bp.load_state_dict(state)
        self.bp.eval()
        self.bp_ckpt = str(bp_ckpt)

    @torch.no_grad()
    def bp_track(self, R, P1, P2) -> np.ndarray:
        feats = raw_features(R, P1, P2).to(self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=(self.device == "cuda")):
            logits = self.bp(feats[None]).squeeze(0)
        return torch.sigmoid(logits.float()).cpu().numpy()

    def predict(self, R, P1, P2, threshold: float = 0.50,
                edge_buffer: int = 200, peak_distance: int = 500,
                gate_conf: float = 0.50) -> dict:
        """Full two-head prediction for one triplet.

        Returns per-position BP probabilities, the aux confidence, and the gated
        peak calls. R/P1/P2 are int8 arrays ({A:0,T:1,G:2,C:3,gap:4}); use
        seq_to_int8 for strings.
        """
        # --- aux head: divergence-anomaly confidence ---
        dm = triplet_div_max(R, P1, P2)
        conf = aux_confidence(dm, self.div_thr)
        trusted = conf >= gate_conf

        # --- BP head: per-position track + raw peaks ---
        p = self.bp_track(R, P1, P2)
        L = len(p)
        p2 = p.copy()
        p2[:edge_buffer] = 0.0
        p2[L - edge_buffer:] = 0.0
        raw_peaks, _ = find_peaks(p2, height=threshold, distance=peak_distance)
        raw_peaks = [int(x) for x in raw_peaks]

        # --- gate: emit peaks only if the aux head trusts the triplet ---
        peaks = raw_peaks if trusted else []
        warning = None if trusted else (
            f"OUT-OF-DISTRIBUTION: max pairwise divergence {dm:.3f} exceeds the "
            f"within-species regime (~{self.div_thr:.2f}); this looks like a "
            f"cross-species comparison. Breakpoint calls suppressed "
            f"(aux confidence {conf:.2f}).")
        return {
            "bp_prob": p,
            "div_max": dm,
            "aux_confidence": conf,
            "trusted": trusted,
            "raw_peaks": raw_peaks,
            "peaks": peaks,
            "warning": warning,
        }


if __name__ == "__main__":
    # Quick demo: a real HIV recombinant (trusted) vs a cross-species Ebola
    # triplet (gated), showing the two-head behaviour.
    from Bio import SeqIO
    det = M3GatedDetector()
    print(f"M3GatedDetector  bp_ckpt={det.bp_ckpt}  device={det.device}  "
          f"div_thr={det.div_thr}")

    def trip(fa, ids=None):
        recs = list(SeqIO.parse(fa, "fasta"))
        if ids:
            d = {r.id: r for r in recs}
            recs = [d[i] for i in ids]
        return [seq_to_int8(str(r.seq)) for r in recs[:3]]

    crf = Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/triplets/CRF02_AG.fa")
    if crf.exists():
        r = det.predict(*trip(crf))
        print(f"\nLANL CRF02_AG (real HIV recombinant):")
        print(f"  div_max={r['div_max']:.3f}  aux_confidence={r['aux_confidence']:.3f}"
              f"  trusted={r['trusted']}  peaks={r['peaks'][:8]}")

    eb = Path("/home/joshc/Dev/RDP_CNN/data/real_recombinants/ebola/aligned.fa")
    if eb.exists():
        from itertools import combinations
        recs = {x.id: seq_to_int8(str(x.seq)) for x in SeqIO.parse(eb, "fasta")}
        # pick the most-divergent (genuinely cross-species) triplet to illustrate
        trio = max(combinations(recs, 3),
                   key=lambda t: triplet_div_max(recs[t[0]], recs[t[1]], recs[t[2]]))
        r = det.predict(recs[trio[0]], recs[trio[1]], recs[trio[2]])
        print(f"\nEbola cross-species triplet ({' / '.join(x[:14] for x in trio)}):")
        print(f"  div_max={r['div_max']:.3f}  aux_confidence={r['aux_confidence']:.3f}"
              f"  trusted={r['trusted']}  raw_peaks={len(r['raw_peaks'])} → "
              f"gated_peaks={len(r['peaks'])}")
        print(f"  warning: {r['warning']}")
