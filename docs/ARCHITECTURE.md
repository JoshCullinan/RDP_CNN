# M3 v4 architecture

How the detector works, from raw sequences to gated breakpoint calls. The design
principle of v4 is **decoupling**: *where* is the breakpoint (BP head) is a
separate question from *should I trust this input at all* (gate). The detector is
never trained to suppress itself on hard inputs — an external gate decides.

```
aligned triplet (R, P1, P2)
        │
        ▼
   encode_triplet ──► X : (L, 22)            # m3_dilated.raw_features
        │
   ┌────┴───────────────────────────┐
   ▼                                ▼
BP HEAD  (dilated CNN)        AUX HEAD (divergence gate)
"WHERE?"                      "TRUST THIS?"
   │                                │
per-position P(bp) track     aux_confidence ∈ [0,1] = f(max pairwise divergence)
   │                                │
   ▼                                │
find_peaks ───────► peaks ◄─────────┘  GATE: suppress peaks if confidence < 0.5
   │
   ▼
output: probability track + breakpoint calls + confidence / OOD warning
```

Files: `m3_dilated.py` (encoding + BP head), `m3_divergence_gate.py` (gate),
`m3_gated_detector.py` (`M3GatedDetector`, the two heads combined), `bp_detect.py`
(CLI).

---

## 1. Input encoding — `(L, 22)` per position

Each alignment column → a 22-dim vector (`raw_features`). This is **not one-hot
only**: the cross-sequence recombination signal is computed and handed to the
model explicitly (the same statistics classical MaxChi/GeneConv use).

| Channels | Meaning |
|---|---|
| 0–4 | recombinant **R** one-hot (A/T/G/C/gap) |
| 5–9 | parent **P1** one-hot |
| 10–14 | parent **P2** one-hot |
| 15 | `match_p1` = (R == P1) |
| 16 | `match_p2` = (R == P2) |
| 17 | `informative` = (P1 ≠ P2) |
| 18–21 | running parental disparity `mean(match_p1 − match_p2)` at windows {50, 100, 200, 500} bp |

A recombinant is a **mosaic**: R matches P1 on one side of a breakpoint and P2 on
the other, so `match_p1 − match_p2` flips sign at a breakpoint. Channels 18–21
pre-compute that disparity at multiple window scales, so the model sees
MaxChi-scale transitions directly.

## 2. BP head — "where?" (`DilatedHead` in `m3_dilated.py`)

A residual dilated 1-D CNN, ~300k params:

```
X (L,22) → transpose → Conv1d 22→128 (1×1 proj)
         → 6 residual blocks, dilation = {1,2,4,8,16,32}:
              h = h + [ Conv1d(128,128,k=3,dilation=d) → GELU → GroupNorm(8,128) → Dropout1d ]
         → Conv1d 128→1 (1×1) → per-position logit → sigmoid
```

- Conv-stack receptive field ≈ 130 bp; the longer-range signal (up to 500 bp)
  enters via the MaxChi input channels, so a huge receptive field isn't needed.
- Output is a per-position probability track. Breakpoints are post-hoc peaks via
  `scipy.signal.find_peaks` (height = threshold, min separation 200 bp, plus an
  **edge buffer** that suppresses boundary artifacts — `edge_buffer=200`, which
  matters: it removes edge false positives with zero true-BP loss and is what
  pushed LANL F1 from the reported 0.509 to 0.545; see WRITEUP §3.2).

This is the **entire detector**. No transformer/HyenaDNA backbone — MLM
pretraining *hurt* (it trains nucleotide *invariance*, the opposite of
breakpoint sensitivity), so it was dropped (memory `project_m13_pretraining_hurts`).

## 3. Aux head — "trust this?" (the v4 decoupling)

The BP head over-fires on cross-species triplets (Ebola at ~37% divergence reads
uniform divergence as recombination). The aux head gates it. Two versions were
built — **the failed one is load-bearing knowledge:**

- **v4a — learned head (FAILED, not deployed; `M3MultiHead`).** A head pooling
  the shared trunk (mean+max pool, LayerNorm'd) → MLP → P(recombinant). It failed
  fundamentally: with **zero real recombinants in training** (no ground-truth
  breakpoints exist for real HIV), it became a **simulator-vs-real classifier** —
  scoring every *real* recombinant (LANL, XBB) ≈ 0, like cross-species Ebola
  (SANTA-vs-real AUROC 1.000). A learned classifier cannot generalize
  "recombinant-ness" to real data when all its positives are simulated.

- **v4 — divergence gate (SHIPPED; `m3_divergence_gate.py`).** The aux confidence
  is **not learned** — it's a function of pairwise divergence, which is intrinsic
  to the triplet and carries no provenance:

  ```python
  div_max = max(mismatch(R,P1), mismatch(R,P2), mismatch(P1,P2))   # over non-gap cols
  aux_confidence = sigmoid((0.20 − div_max) / 0.04)                # graded [0,1]
  ```

  Within-species recombinants sit low (HIV subtypes ~0.13, SARS lineages ~0.00,
  Zika ~0.09); cross-species sits high (Ebola ~0.37). The 0.20 boundary separates
  them (AUROC 0.98), loses **zero** real recombinants, and **generalizes by
  construction** because it's a property of the data, not something learned from a
  biased training set. It is an **out-of-distribution detector**, not a
  recombinant classifier.

  *Caveat:* 0.20 is calibrated on the reference panels. Divergent HIV subtype pairs
  (D/G/J) can reach ~0.15–0.18, approaching the boundary — validating on those CRFs
  is an open follow-up (the gate trades recall on very-divergent recombinants for
  cross-species safety).

## 4. Gating (`M3GatedDetector.predict`)

```python
if aux_confidence < 0.5:        # i.e. div_max > 0.20
    return [] + OOD warning      # cross-species → suppress (don't report false peaks)
else:
    return find_peaks(bp_track)  # trusted → emit breakpoint calls
```

The BP head is **frozen** when the gate is attached — the gate is a thin external
wrapper that would work around any detector. This is the opposite of the v3
failure, which trained the BP head *itself* to predict zero on divergent input and
collapsed (LANL 0.51 → 0.00, because it generalized "divergent → zero" to
legitimate HIV inter-subtype recombinants).

## 5. Deployment ensemble (`m3_eval_ensemble.py`)

The deployed detector is **4 independently-seeded BP heads** (identical
architecture). Their per-position probability tracks are *averaged*, then peaks
extracted from the mean track. This halves variance (per-seed std 0.027 → 0.013
across deployment checkpoints) and is what makes "beats RDP" robust rather than a
lucky epoch. The divergence gate wraps the ensemble identically.

---

## Why this shape — the two non-obvious choices

The two components that *look* like they should be ML but deliberately aren't,
each chosen after the ML version was built and shown to fail:

1. **The backbone is a plain CNN, not a pretrained transformer.** HyenaDNA + MLM
   pretraining produced a backbone that *hurt* downstream detection (6 independent
   tests; memory `project_m13_pretraining_hurts`).
2. **The cross-species gate is a divergence rule, not a learned classifier.** The
   learned version can't recognize real recombinants (out-of-distribution positive
   class). The divergence rule generalizes because it's intrinsic to the data.

Method lesson worth keeping: on a tiny test set (LANL = 35 breakpoints), a ~1-BP
F1 difference is noise — never claim a win or read a loss from a single seed/epoch;
train ≥3 seeds, report mean ± std + an ensemble, and justify any post-processing
convention (like `edge_buffer`) independently of the test set.
