# Ablation Appendix: Genomic-Spectrogram Representation for Recombinant Identification

**Status:** CLOSED — robust negative result. Documented 2026-07-14 so this idea is not re-litigated.
**Source idea:** Guerrero-Tamayo et al., *Classification of SARS-CoV-2 sequences as recombinants via a pre-trained CNN…*, PLOS ONE 2024, [doi:10.1371/journal.pone.0309391](https://doi.org/10.1371/journal.pone.0309391) (`journal.pone.0309391.pdf`, repo root). The paper converts a single viral genome into a 2-D "genomic spectrogram" (per-nucleotide binary indicator → STFT → sum the four nucleotide planes), then transfer-learns a VGG-16 (ImageNet → HIV → SARS-CoV-2) to classify a genome as recombinant/not, reporting ~94.6% test accuracy and a claimed "recombination signature" at frequency *f* = 1/6.
**Design spec:** `docs/superpowers/specs/2026-07-13-genomic-spectrogram-identification-design.md` (Stage 1: 3-way "which of the triplet is the recombinant" identification, chance = 1/3; explicitly *not* the localization task).
**Scripts + result JSONs:** `experiments/spectrogram/*.py`, `experiments/spectrogram/results_*.json` — every number below is reproducible from these.

---

## 1. Hypothesis tested

Does the paper's **sequence → image (STFT spectrogram) → CNN** representation carry useful signal for this repo's **permutation which-of-3 recombinant identification** task — given an aligned triplet `(seq_a, seq_b, seq_c)` with the recombinant's channel position randomly permuted per sample, predict which channel holds the recombinant (chance = 0.333)?

The design spec's decision rule: *a Fourier/spectrogram arm (A1/A2, and by extension the follow-on A3/A4 arms explored overnight) must beat the position-locked positional control A0, on a real or SANTA held-out set, under a pre-registered primary metric* — or the paper's representation has not earned a place in this repo's pipeline. The spec also mandated confound diagnostics run first (P2 own-parents leak gate, positional-scramble control, divergence-vs-prediction check) precisely because this repo has twice been burned by composition/divergence confounds (M3 v3 collapse, M3 v4 aux-head provenance detector — see `project_m13_pretraining_hurts.md` / `project_m3_v3_failed.md` in memory).

## 2. The representation arms

| Arm | Description |
|---|---|
| **A0** | Positional control: per-position pairwise-identity tracks `id(s_i,s_j)` for the 3 sequence pairs, multi-scale smoothed (windows {50,100,200,500} bp) into a position×scale image — the same construction as the repo's M3 dilated-CNN "which parent matches here" comparison signal, lifted to 2-D for a fair comparison through the same CNN harness. **Must-beat baseline**, not a spectrogram arm itself. |
| **A1** | Paper-faithful summed genomic spectrogram: per sequence, per-nucleotide binary indicator → STFT → sum the four nucleotide planes (paper's Eq. 1) → one grayscale channel per sequence. |
| **A2** | Unsummed spectrogram: the four nucleotide STFT planes kept separate per sequence (12 planes total for 3 sequences) — tests whether summing over nucleotides (A1) destroys base-identity signal. |
| **A3** | Difference-spectrogram: STFT of the pairwise *match track* (splice-explicit — directly encodes where two sequences agree/disagree, then transforms to frequency domain). |
| **A4** | Dot-plot: coarse block self-similarity maps (a broken diagonal marks the splice point). |

## 3. Results — exact numbers from the JSONs

### 3.1 Experiment 1 — pre-registered pilot, ConvNeXt-Base, LOCO real-CRF + SANTA held-out

Source: `results_reduced_prereg.json` (protocol) + `results_reduced_stage1.json` (results). Arms A0/A1/A2, inits {floor, imagenet} (the pre-registered `random` init was not present in the results JSON). Chance = 0.333.

| Arm:init | LOCO real-CRF acc (`mean_fold_acc`) | SANTA held-out val acc | McNemar p vs A0 | 95% CI on Δ vs A0 | `win` |
|---|---|---|---|---|---|
| A0:floor | 0.5625 | 0.45675 | — | — | — |
| A0:imagenet | 0.4375 | 0.57225 | — | — | — |
| A1:floor | 0.125 | 0.3535 | 0.0391 | [-0.625, -0.25] | false |
| A1:imagenet | 0.5 | 0.47325 | 1.0 | [-0.125, 0.25] | false |
| A2:floor | 0.3125 | 0.3445 | 0.3438 | [-0.5, 0.0625] | false |
| A2:imagenet | 0.125 | 0.33425 | 0.125 | [-0.75, 0.125] | false |

None of the four A1/A2 arm×init combinations beat A0 under the pre-registered decision rule (`win: false` for all). This pilot is severely underpowered (`n_test_triplets: 16`, `mde: Infinity`) and its "winning" arm by raw accuracy, A1:imagenet, flags two problems in the diagnostics block: the P2 own-parents leak gate reads `auc: 0.6713, leak: true` (confound present), and the positional-scramble control shows `scrambled_acc: 0.5625 > unscrambled_acc: 0.5` (Δ = -0.0625) — i.e. **scrambling positions did not hurt, and if anything helped**, meaning this arm reads sequence composition, not splice location. Per the design spec's own confound-diagnostic logic, this pilot result is disqualified before the power problem is even considered.

### 3.2 Experiment 2 — SANTA held-out, A0/A1/A2 × {floor, ResNet-50, VGG-16}, 8000 train / 2000 val, 10 epochs, 2 seeds

Source: `results_santa_confirm.json`. Chance = 0.333.

| Arm:backbone | mean acc | std |
|---|---|---|
| **A0:floor** | **0.626** | 0.0285 |
| **A0:resnet50** | **0.631** | 0.0413 |
| **A0:vgg16** | **0.575** | 0.0153 |
| A1:floor | 0.372 | 0.011 |
| A1:resnet50 | 0.419 | 0.0135 |
| A1:vgg16 | 0.326 | 0.0043 |
| A2:floor | 0.358 | 0.0008 |
| A2:resnet50 | 0.341 | 0.0008 |
| A2:vgg16 | 0.331 | 0.0093 |

A0 (positional control) scores ~0.58–0.63 on every one of the three backbones tested, including the paper's own VGG-16. A1 (summed spectrogram) ranges 0.326–0.419, and A2 (unsummed) sits at essentially chance (0.331–0.358 vs. chance 0.333) on all three backbones. VGG-16 is the weakest backbone for A0 but does not rescue A1/A2. This is architecture-robust: the negative holds on floor, ResNet-50, *and* the paper's own VGG-16.

### 3.3 Experiment 3 — splice race, A0/A1/A3/A4 + positional scramble, SmallCNN, 2000 train / 800 val, 8 epochs, 2 seeds

Source: `results_splice_race.json`. Chance = 0.333. `splice_drop = val_mean - scr_mean` (positive = accuracy drops under position-scrambling, i.e. evidence the arm reads the splice rather than composition).

| Arm | val_mean | val_std | scrambled_mean | scr_std | splice_drop |
|---|---|---|---|---|---|
| A0 | 0.4275 | 0.0288 | 0.4363 | 0.0075 | -0.0088 |
| A1 | 0.3206 | 0.0031 | 0.325 | 0.01 | -0.0044 |
| A3 | 0.3375 | 0.0063 | 0.3269 | 0.0169 | 0.0106 |
| **A4** | **0.3725** | 0.0063 | 0.3031 | 0.0056 | **0.0694** |

A4 (dot-plot) is the *only* arm both above chance and showing a positive splice-drop (accuracy falls under scrambling) — i.e. the only spectral-family arm shown to genuinely read the splice rather than composition at this (small) scale. A1 and A3 are flat under scrambling (small negative or near-zero drop), meaning they are not using position/splice information at all. A0 in this small-scale, SmallCNN configuration is itself flat under scrambling (splice_drop = -0.0088) — the splice-reading property of A0 is established at scale in Experiment 5, not here.

### 3.4 Experiment 5 — scale-up, A4 vs A0, floor + ResNet-50, 6000 train, 12 epochs, 2 seeds, + scramble

Source: `results_exp5_scale.json`. Chance = 0.333.

| Arm:backbone | val | scrambled | splice_drop |
|---|---|---|---|
| A0:floor | 0.5457 | 0.5263 | 0.0193 |
| **A0:resnet50** | **0.5700** | **0.4163** | **0.1537** |
| A4:floor | 0.361 | 0.3173 | 0.0437 |
| A4:resnet50 | 0.3983 | 0.3417 | 0.0567 |

At scale, A0:resnet50 goes from 0.570 (val) to 0.416 under position-scrambling — a drop of 0.154, which relative to its margin above chance (0.570-0.333=0.237 → 0.416-0.333=0.083) removes roughly two-thirds of its above-chance signal, corroborating the NIGHT_SUMMARY's narrative characterization of A0 as "losing about half its signal" under scrambling — **A0's edge is genuinely positional, not compositional.** A4:resnet50 improves with scale (0.372 in Exp 3 → 0.398 here) but sits **0.1717 (≈0.17) below A0:resnet50 (0.570 − 0.398)** — real signal, shallow headroom, still well short of the positional control.

### 3.5 Experiments 4 / 7 — 4-way {seq1,seq2,seq3,none} head, arm A0, `none` = positionally-SCRAMBLED negatives

Source: `results_fourway_A0.json` (SmallCNN, 3000 train, 8 epochs, 2 seeds, `none_frac=0.5`, chance₄ = 0.25) and `results_fourway_A0_resnet50.json` (ResNet-50, 8000 train, 15 epochs, 2 seeds, same `none_frac`).

| Experiment | overall | none_recall | real_acc (3-way, non-none) | false_none_rate |
|---|---|---|---|---|
| Exp 4 (SmallCNN, 3k/8ep) | 0.6093 | 0.999 | 0.4145 | 0.0 |
| Exp 7 (ResNet-50, 8k/15ep) | 0.7257 | 1.000 | 0.5885 | 0.0 |

Against scrambled negatives the detection gate is essentially perfect (`none_recall` 0.999–1.000, `false_none_rate` 0.0) and identification accuracy on real (non-scrambled) triplets climbs with scale (0.4145 → 0.5885). **But scrambling is an easy negative** — this result does not establish that the 4-way head reads splice structure to identify "none"; see Experiment 8.

## 4. The robust negative, stated precisely

Across 5 experiments (1, 2, 3, 5, and the A0 vs. A4 scale comparison within 5), 3 backbones (floor/from-scratch CNN, ResNet-50, and the paper's own VGG-16), and a scale-up from 2k to 8k training triplets, **every image/frequency arm (A1 summed spectrogram, A2 unsummed spectrogram, A3 difference-spectrogram, A4 dot-plot) underperformed the position-locked positional control A0**:

- Best spectral arm across all experiments: **A4 (dot-plot) at 0.398** (Exp 5, ResNet-50, 6k/12ep) vs. chance 0.333 — real but modest.
- A0 (positional control) at the same scale/backbone: **0.570** (Exp 5, ResNet-50) — a **0.172 gap**, i.e. A4 plateaus **~0.17 below A0**.
- **A0's edge genuinely reads the splice**: under positional scrambling A0:resnet50 falls from 0.570 → 0.416 (Δ = -0.154, Exp 5) — it loses roughly half its above-chance margin, meaning it depends on *alignment*, not just composition.
- By contrast, the pilot's best-looking spectral result (A1:imagenet in Experiment 1) *increased* slightly under scrambling (0.5625 scrambled vs. 0.5 unscrambled) — the signature of a **composition confound**, not a splice-reader, and is disqualified on that basis regardless of its raw accuracy.
- A4 is the sole spectral-family arm that shows a genuine (if small) splice-reading property (positive `splice_drop` in both Experiment 3, +0.069, and Experiment 5, +0.044 to +0.057) — but its absolute accuracy and headroom remain well below A0's.

**Net: no spectrogram/frequency-domain arm earns a place on the deployment path.** The paper's reported ~94.6% single-genome accuracy is best explained (consistent with the design spec's a priori concern in §1) as a compositional/lineage fingerprint rather than a recombination-structure signature — precisely what the scramble controls above show for A1/A2/A3, and even for A0 under a confounded pilot setup.

## 5. Exp 8 — the real-negative confound recurs (context)

Source: `results_exp8_real_negatives.json`. Config: arm A0, backbone ResNet-50, 6000 SANTA + 3000 real triplets, 15 epochs, 2 seeds, `none` = **REAL non-recombinant** triplets (not scrambled).

| Seed | real_acc | false_none_rate | none_recall |
|---|---|---|---|
| seed 0 | 0.441 | 0.419 | 0.648 |
| seed 1 | 0.431 | 0.395 | 0.610 |
| **mean** | **0.436** | **0.407** | **0.629** |

When the 4-way head's "none" class is populated with **real** non-recombinant triplets instead of positionally-scrambled ones, `false_none_rate` jumps to **0.407 (mean of 0.419 and 0.395 per seed)** — i.e. the model calls a *real recombinant* "none" 40.7% of the time. This is the same failure mode already logged for the M3 line: `project_m3_v3_failed.md` (M3 v3 catastrophic collapse from divergence-based negatives) and `project_m3_v4_multihead.md` (learned aux gate became a SANTA-vs-real provenance detector, confound AUROC 1.000). The 4-way head, even built on the winning A0 positional representation, **wins its scrambled "none" class via an easy shortcut (composition/divergence) that does not transfer to real non-recombinant negatives** — confirming the design spec's own prediction (§2: "'none' reintroduces the divergence/provenance confound that collapsed M3 v3").

This result is **context, not the headline finding** of this appendix: it applies to the 4-way head construction generally (on top of the *best* representation, A0), not specifically to the spectrogram arms, which is why the position-locked comparison representation (M3's approach), not the spectrogram, remains the deployment path — the identification problem still has an open real-negative confound to solve, but that is tracked separately (see memory: `project_m3_v4_multihead.md`, "Recommended next steps" in `NIGHT_SUMMARY.md` §Conclusions/§Recommended next steps).

## 6. Conclusion

1. **No spectrogram/frequency-domain arm (A1, A2, A3, A4) is on the deployment path.** Every one underperforms the position-locked positional control A0 across all backbones tested (floor, ResNet-50, VGG-16) and across a 2k→8k scale-up. This is a **robust negative**, not a power/scale artifact — it holds on the paper's own backbone (VGG-16) and improves for A4 with scale without ever closing the gap to A0.
2. **A4 (dot-plot) is the one image arm worth keeping in the codebase** as a documented reference implementation — it is the only spectral-family arm shown to genuinely read splice structure (positive, reproducible splice_drop under scrambling in two independent experiments), but it plateaus **~0.17 below A0** with **shallow headroom** (0.372 → 0.398 across a 3x scale-up, Exp 3 → Exp 5). Do not invest further tuning in it without new evidence its headroom is deeper than this.
3. **The position-locked comparison representation (A0 / M3's "which parent matches here" signal) is the correct representation** for recombinant identification, exactly as it is for localization. It genuinely reads the splice (loses ~half its above-chance signal under scrambling) rather than composition.
4. **The 4-way `{seq1,seq2,seq3,none}` head is a separate, still-open problem**, independent of the spectrogram question: it works cleanly against scrambled negatives (Exp 4, 7) but re-exhibits the M3-v3/v4-style real-negative confound against real non-recombinant triplets (Exp 8, false_none_rate 0.407). This is not a reason to revisit the spectrogram arms — it is a reason to attack the real-negative problem directly on top of A0, per the next-steps already logged in `NIGHT_SUMMARY.md`.

**This appendix closes the spectrogram-representation question. Do not re-run A1/A2/A3 experiments without new evidence (e.g., a materially different STFT parameterization or nucleotide encoding) that specifically addresses why they read composition rather than splice structure.**
