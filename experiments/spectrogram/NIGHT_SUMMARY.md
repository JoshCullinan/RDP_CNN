# Overnight experiments — genomic-spectrogram idea + follow-ons (2026-07-13)

Autonomous run following the Fable-5 advisor's ideas, per Josh's directive: keep
testing the ideas, and if one is very successful, chase that path. Judge new
representations by whether they **read the splice** (accuracy collapses under
position-scrambling) and have **headroom** — not by whether they immediately beat
the mature positional baseline A0 (which already carries the whole M3 line's tuning).

Scripts + result JSONs in this directory reproduce every number below.
Chance for the 3-way "which of the triplet" task = 0.333.

## The representation arms
- **A0** — positional control: per-position pairwise identity of the 3 sequence
  pairs, multi-scale smoothed. (≈ the M3 "which parent matches here" signal.)
- **A1** — paper's summed genomic spectrogram (per-sequence).
- **A2** — unsummed spectrogram (4 nucleotide planes/sequence).
- **A3** — difference-spectrogram: STFT of the pairwise *match track* (splice-explicit).
- **A4** — dot-plot: coarse block self-similarity maps (broken diagonal = splice).

## Results

| exp | setup | headline |
|---|---|---|
| 1 | reduced, ConvNeXt, LANL LOCO + SANTA | A0 ≈ 0.57 > A1/A2; scramble didn't hurt spectrogram → composition, not splice |
| 2 | SANTA held-out, A0/A1/A2 × {floor, ResNet-50, VGG-16}, 8k/10ep | **A0 ≈ 0.60 on every backbone**; A1 0.33–0.42; A2 ≈ chance. VGG-16 worst backbone. Negative robust to architecture |
| 3 | splice race A0/A1/A3/A4 + scramble, 2k/8ep SmallCNN | **A4 dot-plot** = only arm above chance (0.372) AND dropping on scramble (+0.069) → reads splice. A1/A3 flat |
| 5 | scale A4 vs A0, 6k/12ep, floor+ResNet-50, +scramble | A0:ResNet **0.570, drops to 0.416 on scramble (−0.154)** → A0's edge is genuinely positional. A4 only 0.372→0.398, ~0.17 below A0 → real but **weak** splice-reader, shallow headroom |
| 4 | 4-way {…,none}, A0, SmallCNN 3k, none = SCRAMBLED | none_recall 0.999, false_none 0.000, real_acc 0.414 |
| 7 | 4-way {…,none}, A0, **ResNet-50 8k/15ep**, none = SCRAMBLED | **none_recall 1.000, false_none 0.000, real_acc 0.589** — detection gate perfect + splice-proven; which-of-3 climbs with scale |
| 8 | 4-way {…,none}, A0, ResNet-50 6k+3k/15ep, none = **REAL non-recomb** | real_acc 0.436, none_recall 0.629, **false_none 0.407** — the M3-v3 confound RECURS |

## Conclusions

1. **The paper's genomic-spectrogram representation does not transfer to this task —
   a robust negative.** Across 5 experiments, 3 backbones (incl. the paper's own
   VGG-16 and the student's ResNet-50), and a fair scale-up, every image/frequency
   arm (A1/A2/A3/A4) underperformed the position-locked positional control A0. The
   splice-explicit dot-plot (A4) is the best of the family but plateaus ~0.17 below
   A0. The paper's single-genome 94% is best explained as a compositional/lineage
   fingerprint, not recombination structure (our scramble control confirms it).

2. **Position-locked comparison is the right representation, and it genuinely reads
   the splice.** A0 with ResNet-50 hits ~0.57–0.63 and loses ~half its signal under
   position-scrambling — it uses alignment, not composition.

3. **The 4-way {seq1,seq2,seq3,none} head is promising but NOT solved for real
   deployment.** On scrambled negatives it's near-perfect and provably splice-based
   (defeats the confound). But on REAL non-recombinant negatives it exhibits the
   M3-v3 collapse (41% of real recombinants falsely called "none"). The scrambled
   result was partly an artifact — scrambling is an *easy* negative.

## Recommended next steps (with Josh)
- Fold A0/dot-plot as extra channels into the M3 dilated-CNN; keep the localization head.
- Attack the real-negative problem the v3 collapse exposes: composition-matched hard
  negatives, a divergence-invariance penalty (gradient-reversed composition probe),
  or a distance/divergence curriculum — so "none" can't be won via divergence.
- Grow the real eval panel beyond 4 CRFs (LANL has 100+) before ranking models on real HIV.
- Retire the spectrogram arms to a documented ablation; they are not deployment candidates.
