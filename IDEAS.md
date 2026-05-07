# Idea catalog — breaking the F1=0.17 honest interior plateau

Comprehensive list of neural-network-related approaches for the viral
recombination breakpoint detection task. Compiled 2026-05-07 after runs
#28d-#35 confirmed an honest interior F1 ceiling at ~0.17 across
{architecture, normalization, augmentation, data volume, ensemble, input
feature scale} — vs RDP5 classical baseline F1=0.367.

This file is the living catalog. As ideas are tried, mark them with
status (TRIED/IN-PROGRESS/REJECTED/UNTESTED) and link to the
Experiment Log entry.

---

## 0. Diagnostic frame — why current ideas plateau

Three facts must drive every decision:

1. **Val F1 (EB=200) ~0.58 vs test F1 (EB=200) ~0.17.** The model can
   localize when distribution matches; the gap is sim→test, not capacity.
   Architecture sweeps within the same input/output formulation will keep
   landing at 0.17.

2. **Models from {28d, 29, 32, 34} ensemble doesn't lift F1.** They make
   correlated errors — different paths converged to the same suboptimal
   solution. We need *orthogonal* signal sources.

3. **No published deep-learning baseline beats RDP/MaxChi/GeneConv on
   triplet recombination breakpoints** (literature search 2026-05-07).
   This niche is open. Our backbone is also markedly shallower than SOTA
   per-position DNA labeling networks (SpliceAI-10k = 32 residual blocks;
   ours has ~6).

**Conclusion:** change the input distribution OR the input features OR
the model class — not just the model size or normalization.

---

## 1. Tier A — highest leverage, ranked

### A1. Hybrid: feed RDP5/MaxChi posteriors as input channels  ⭐⭐ KEPT — first run beating RDP5
**Status:** TESTED as run #36 (2026-05-07). **Beats RDP5 on raw F1 (0.397 vs 0.367)**, lifts honest interior F1 from 0.175 to 0.323. Simplest variant tested: 2 Gaussian channels (σ=50) at PredBPStart and PredBPEnd from `.faSimVSRealCompare.csv`.

**A1 extension (run #39 v2, 2026-05-07)**: added 9 per-method confidence
broadcast scalars from `.fa.csv` (clip(-log10(p_method)/30, 0, 1) per
method). Val F1 jumped 0.55 → 0.69 (+0.14), but **test honest F1 = 0.309 vs
run #38 0.313 — FLAT.** Val→test gap widened 0.24 → 0.38. The
method-confidence signal is in-distribution-specific; simulator's RDP
calibration differs from UnseenTestSet's. Channel engineering hits
diminishing returns. Per-event RecombIdentifyStats (.faRecIdentifyStats.csv)
features were planned as run #40 but DEFERRED — same distribution-shift
concern likely applies. Pivoting to A2 SpliceAI-32 (representation-capacity
attack rather than input-richness attack).

**Idea:** Run OpenRDP (or MaxChi/3SEQ) over each `.fa` alignment offline.
Its per-position breakpoint posterior becomes channels 23+ in `X`. Train
the CNN to refine the classical signal.

**Why it should work:** RDP5 is already at F1=0.367 — better than any
of our pure neural attempts. Even *modest* neural lift on top of that
immediately beats our baseline. Run #13's MaxChi-feature lift collapsed
the train-val gap; this is the same idea with the actual classical
detector's output, not just its hand-crafted features.

**Why models converge to 0.17 currently:** they have to rediscover
MaxChi's combination of windowed parental disparity from scratch; we
never give them MaxChi's actual output to lean on.

**Upside:** F1 floor lifts to ~max(0.367, neural-bonus) ≈ 0.40+ if
neural can refine at all.

**Downside:** simulator-to-real transfer of RDP5's calibration is
unknown; training distribution still mostly captures classical
statistics.

**Cost:** S–M. OpenRDP is python; cache its outputs once per file.
Single change to `parse_simulation` adds N more channels.

**Refs:** [OpenRDP](https://github.com/PoonLab/OpenRDP)

---

### A2. SpliceAI-32 backbone — match SOTA depth
**Status:** UNTESTED.

**Idea:** Replace `build_cnn` with the SpliceAI-10k topology: 32
ResidualUnits (Conv→BN→ReLU twice, with dilations 1→4→10→25), skip-merges
every 4 blocks, receptive field 10kb. Adapt for 22-channel input + 1-
channel sigmoid output.

**Why it should work:** SpliceAI is the canonical "per-position labels
on long DNA" architecture. Our current ~6 dilated blocks is shallow by
comparison. Earlier ablations in the SpliceAI paper saturate above 16
blocks.

**Why our depth choice was probably wrong:** prior runs use kernel-7 +
dilations 1..32 (one round). SpliceAI does dilation rounds 1→4→10→25
which gives a much larger effective receptive field at 10kb. Our 32k-bp
inputs likely need this.

**Upside:** breaks the plateau if "the signal exists in the input but
our backbone can't extract it." F1 could go to 0.30+.

**Downside:** ~10M params, 30+ epochs at ~5min/epoch; ~3 hour training.

**Cost:** S–M. One cell rewrite. Subagent can write the code.

**Refs:** [OpenSpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI/blob/main/openspliceai/train_base/spliceai.py),
[PyTorch port](https://github.com/dohlee/spliceai-pytorch),
[OpenSpliceAI eLife 2025](https://elifesciences.org/articles/107454)

---

### A3. Use the FULL alignment, not just the triplet
**Status:** UNTESTED.

**Idea:** Each `.fa` file contains ~100 SANTA-simulated sequences. We
currently use 3 per event. Feed alignment-derived statistics (consensus
base, MSA-column entropy, parental clade indicators, derived/ancestral
state at each position) as additional channels. Or: feed the model the
entire MSA via attention.

**Why it should work:** Classical methods (RDP, GARD) achieve their
performance partly because they use full alignment context —
phylogenetic incongruence, full-MSA entropy, lineage signals. We're
throwing away ~97% of the data per file.

**Upside:** the most data-rich change available without any new
simulation.

**Downside:** Encoding 100 sequences × 32kbp efficiently is non-trivial.
Naive concatenation blows memory; need axial attention or windowed
reduction.

**Cost:** M. New parsing logic + new input shape + maybe new
architecture (e.g., MSA Transformer-lite).

---

### A4. HyenaDNA frozen feature extractor + thin head
**Status:** UNTESTED.

**Idea:** Encode each of (recomb, p1, p2) through HyenaDNA-small-32k
(single-nt resolution, 32kbp context, pretrained on real genomes).
Concatenate with our hand-crafted match/MaxChi channels. Train a thin
SpliceAI-style head on top.

**Why it should work:** HyenaDNA brings evolutionary priors learned
across multi-species genomes. Directly attacks sim→real transfer (the
val→test gap might be sim-specific feature collapse). 6.5M params, fits
in WSL2 budget if frozen.

**Upside:** unique source of orthogonal signal — pretrained knowledge
that simulator data can't provide. Could lift sim→real transfer
specifically.

**Downside:** integration complexity; HyenaDNA in HuggingFace, our stack
is TF — may need PyTorch interop or weight conversion.

**Cost:** M–L. Several hours setup. But the linked HF model is plug-and-
play.

**Refs:** [HyenaDNA repo](https://github.com/HazyResearch/hyena-dna),
[HF model](https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen-hf),
[paper](https://arxiv.org/abs/2306.15794)

---

### A5. Synthetic 30k-content training samples
**Status:** UNTESTED.

**Idea:** Concatenate two short XML-1 samples (4k content each) with a
real BP between them to manufacture 30k-content triplets matching test
distribution. Or use existing samples and randomly extend with MSA-
derived "neighbour" sequence.

**Why it should work:** Train max content is 19k; test goes to 30k.
Position-shift aug (#34) didn't help because it just moves real signal
around, doesn't expose model to long-content statistics. Synthetic
concatenation does.

**Upside:** Directly addresses the sim→test distribution shift. Cheap
data multiplication.

**Downside:** Concatenated samples are biologically weird (parent IDs
at the junction don't make sense). May introduce artifacts.

**Cost:** M. Custom data augmentation in `_make_aug_gen`.

---

### A6. Unified Focal Loss + U-Net (paired, per RDBKE 2021)
**Status:** PARTIALLY TESTED. U-Net alone failed in run #31 with
weighted_bce. The paired loss has not been tested.

**Idea:** U-Net architecture (run #31 failed) PAIRED with Unified Focal
Loss (combination of focal + Dice + Tversky). The literature shows U-Net
works on extreme-imbalance per-position breakpoint tasks specifically
when paired with this loss family — `weighted_bce` is suboptimal at ~1%
positive rate.

**Why my U-Net failed (#31):** I kept `weighted_bce(POS_WEIGHT=70)`.
Dice/Tversky losses dominate sigmoid-BCE on tiny-positive segmentation
tasks. The architecture wasn't wrong — the loss was.

**Upside:** could rescue the U-Net direction.

**Downside:** pairs two changes (arch + loss). Per CLAUDE.md "one change
per run" — sequence them: first focal/dice on dilated stack, then
U-Net+focal/dice.

**Cost:** M. Loss function implementation is small; U-Net code already
on disk from #31.

**Refs:** [Unified Focal Loss (Yeung et al. 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8785124/),
[Unified Focal Loss repo](https://github.com/mlyg/unified-focal-loss),
[RDBKE U-Net for SV breakpoints (PLOS CB 2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009186)

---

### A7. Domain adaptation (DANN) — close val→test feature distribution gap
**Status:** UNTESTED.

**Idea:** Train an adversarial discriminator that tries to distinguish
val from test inputs by the model's internal features. Backbone is
trained to fool the discriminator (DANN — Domain Adversarial Neural
Network). Test inputs are unlabeled but their features should match
val's.

**Why it should work:** The val→test gap (0.58 → 0.17) is exactly what
domain adaptation was invented for. The test labels stay unseen during
training (we only use test INPUTS for the adversarial loss).

**Upside:** specifically attacks the val→test gap structurally.

**Downside:** Adversarial training is finicky. Requires careful loss
balancing.

**Cost:** M–L. Requires modifying training loop + adding discriminator
head.

---

## 2. Tier B — worth trying, lower P(win)

### Input feature engineering
| ID | Idea | Cost | Status |
|----|------|------|--------|
| B1 | Log-spaced MaxChi windows {50, 200, 800, 3200} | S | UNTESTED |
| B2 | Per-position substitution rate (distance from MSA consensus) | S | UNTESTED |
| B3 | Codon-frame indicators (if alignments are coding) | S | UNTESTED |
| B4 | Window entropy / Shannon information per position | S | UNTESTED |
| B5 | Pairwise informative-site density (windowed) | S | UNTESTED |
| B6 | Phylogenetic position-encoding from MSA tree | M | UNTESTED |
| B7 | RC-equivariance via input duplication | S | UNTESTED |
| B8 | Compression-distance (NCD via gzip) channels | M | UNTESTED |
| B9 | Triplet→6plet by parent permutation | S | UNTESTED |
| B10 | Feature ablation study (drop each channel; measure F1) | S | UNTESTED |

### Output formulation
| ID | Idea | Cost | Status |
|----|------|------|--------|
| B11 | Windowed binary classification (P(BP in [i, i+200])) | M | UNTESTED |
| B12 | Heatmap MSE regression instead of BCE | S | UNTESTED |
| B13 | Set prediction (DETR-style) — top-K with proper edge masking | M | (top-K tried #19-21 with BN-pad bug; UNTESTED with current LN+EB) |
| B14 | Two-headed: presence + position multi-task | M | UNTESTED |
| B15 | Distance-to-nearest-BP regression (peaks = local minima) | M | UNTESTED |
| B16 | Hierarchical: coarse 100-window class → fine within-window | M | UNTESTED |

### Training procedure
| ID | Idea | Cost | Status |
|----|------|------|--------|
| B17 | SAM (Sharpness-Aware Minimization) | S | UNTESTED |
| B18 | Stochastic Weight Averaging (SWA) | S | UNTESTED |
| B19 | Lookahead optimizer + AdamW | S | UNTESTED |
| B20 | Cosine LR schedule with warmup | S | UNTESTED |
| B21 | Gradient accumulation (effective batch 16+) | S | UNTESTED |
| B22 | Mixup / sample-level interpolation | M | UNTESTED |
| B23 | Label smoothing on Gaussian targets (smaller/larger σ) | S | TESTED #9, #18 (REVERTED at smaller σ) |

### Pretraining / transfer
| ID | Idea | Cost | Status |
|----|------|------|--------|
| B24 | Self-supervised on real HIV alignments (LANL) | L | UNTESTED |
| B25 | Autoencoder on parent-comparison signals | M | UNTESTED |
| B26 | Multi-task: also predict ancestor-of-each-position | M | UNTESTED |
| B27 | Knowledge distillation from RDP5 | M | UNTESTED |

### Ensembling / post-processing
| ID | Idea | Cost | Status |
|----|------|------|--------|
| B28 | Stacking with meta-learner (CNN + RDP5 + MaxChi + GENECONV) | M | UNTESTED |
| B29 | Test-time augmentation (predict on X and RC(X), average) | S | UNTESTED |
| B30 | CRF post-processing for smoothness | M | UNTESTED |
| B31 | Adaptive thresholding (per-sequence percentile) | S | UNTESTED |
| B32 | Watershed segmentation on prediction surface | S | UNTESTED |
| B33 | Non-maximum suppression with overlap awareness | S | UNTESTED |

---

## 3. Tier C — speculative or niche

### Architecture
| ID | Idea | Notes |
|----|------|-------|
| C1 | Caduceus (BiMamba, RC-equivariant) | CUDA kernel deps tricky in WSL2 |
| C2 | Conformer (CNN+attention hybrid) | Speech-style; could capture local + global |
| C3 | Hyena / RetNet / RWKV | Efficient attention alternatives |
| C4 | Sliding-window Transformer (Longformer) | Local + global tokens |
| C5 | WaveNet causal dilated (bidirectional via fwd+rev stack) | — |
| C6 | SegFormer 1D port | Efficient segmentation Transformer |
| C7 | Multi-resolution dilated U-Net (avoids #31 bottleneck) | Multi-scale + spatial precision |

### Loss function
| ID | Idea | Notes |
|----|------|-------|
| C8 | Tversky loss | Asymmetric Dice |
| C9 | Asymmetric loss | Decoupled FP/FN weighting |
| C10 | Boundary loss | Penalty on prediction-boundary misalignment |
| C11 | Soft-IoU | Direct IoU optimization |
| C12 | Wasserstein distance on peak distributions | Earth-mover |
| C13 | Differentiable F1 surrogate | Direct metric optimization |

### Data
| ID | Idea | Notes |
|----|------|-------|
| C14 | Curriculum learning (short→long sequences) | — |
| C15 | Hard example mining | Over-sample misses |
| C16 | Active learning across multiple-model disagreement | — |
| C17 | Adversarial samples | Find decision-boundary cases |
| C18 | Synthetic samples at intermediate SANTA settings | If we can re-run simulator |
| C19 | Cross-distribution interpolation during training | — |

### Multi-task / structured prediction
| ID | Idea | Notes |
|----|------|-------|
| C20 | Joint (BP positions, parent assignments) prediction | Decode via Viterbi |
| C21 | Self-consistency loss (BPs ↔ parent assignment changes) | — |
| C22 | HMM decoder layer on top of CNN features | Neural+HMM hybrid |
| C23 | CRF as decoder | Structured smoothing |

### Domain-specific
| ID | Idea | Notes |
|----|------|-------|
| C24 | SANTA metadata (rates, divergence) as conditional input | If accessible from XML files |
| C25 | Contrastive learning across SANTA configurations | Domain-invariant features |
| C26 | Triplet → 6 ordered triplets (parent permutation aug) | RC + parent symmetry |
| C27 | Bootstrap-style alternative parent sampling | — |

### Test-time / postproc
| ID | Idea | Notes |
|----|------|-------|
| C28 | Iterative refinement (predict → mask high-conf → repredict) | — |
| C29 | Conformal prediction wrapper | Calibrated uncertainty |
| C30 | Temperature scaling per sample | — |
| C31 | Beam search over peak sets | Globalizes greedy peak find |

---

## 4. Tier D — don't bother (already-tested or theoretically poor)

| What | Why not |
|------|---------|
| Argmax / softmax-over-positions head | Runs #19-21 REVERTED with BN-pad boundary collapse. Even with edge masking, fights the per-position structure. |
| DNABERT-2 / NucleotideTransformer / GENA-LM | BPE/k-mer tokenization destroys single-nt alignment between (recomb, p1, p2). Critical for our task. |
| ReLERNN | Solves rate prediction, not per-event localization. |
| Smaller/faster architecture as compromise | Forbidden by CLAUDE.md; literature confirms SpliceAI-class is deep. |
| σ tuning paired with POS_WEIGHT | Explored in #18 (REVERTED). |
| Extra MaxChi window scales {1000, 2000, 5000} | Tested #35, no lift. |
| Position-shift augmentation alone | Tested #34, no lift. |
| More data (max_files=80) | Tested #33, REGRESSED (overfit). |
| Naive ensembling of similar runs | Tested #34b, no orthogonality. |
| BatchNormalization with 'same'-padding under per-position output | Confirmed #28d boundary shortcut; LayerNorm preferred. |

---

## 5. Recommended experiment sequence

If autonomous iteration continues, proposed order:

1. **A1 — RDP5/MaxChi posterior as input channel.** Cheapest with
   highest P(F1>0.30). Test in isolation. Immediate F1 floor at
   max(0.367, neural-bonus).

2. **A2 — SpliceAI-32 backbone.** Independent of A1. Test in isolation.
   If both win, combine.

3. **A6 — Unified Focal Loss on dilated baseline.** Loss change is
   orthogonal to A1/A2.

4. **A5 — synthetic 30k samples.** Address the data-distribution gap.

5. **A7 — DANN (domain-adversarial training).** Address the feature-
   distribution gap directly.

6. **A4 — HyenaDNA frozen.** External pretraining as orthogonal signal
   source.

7. **A3 — full-alignment input.** Most expensive but biggest data-
   richness lift.

A1 is the obvious first move because: (a) cheapest to implement,
(b) highest expected impact (immediate F1 floor at RDP5's 0.367),
(c) tests a hypothesis (CNN can refine classical) we haven't tested at
all.

---

## 6. Bookkeeping

When trying an idea:
1. Update its row to **IN-PROGRESS** with the run number.
2. After eval, update to **TESTED** with honest F1 and link to the
   Experiment Log entry in CNN.ipynb.
3. If REVERTED, note the failure mode in 1-2 sentences for future
   reference.

When adding a new idea:
- Slot into the appropriate tier (A high-leverage, B worth-trying,
  C speculative, D don't-bother).
- Include cost (S/M/L), upside, downside, and refs if any.

Last updated: 2026-05-07. Author: Claude (Opus 4.7) iteration session
covering runs #28d-#35.
