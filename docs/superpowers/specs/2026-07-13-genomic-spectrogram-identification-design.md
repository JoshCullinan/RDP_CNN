# Genomic-spectrogram image CNN for recombinant identification — Stage 1 design

**Date:** 2026-07-13
**Status:** Approved design (pre-implementation-plan)
**Author:** Josh Cullinan + Claude
**Source idea:** Guerrero-Tamayo et al., *Classification of SARS-CoV-2 sequences as recombinants via a pre-trained CNN…*, PLOS ONE 2024, [doi:10.1371/journal.pone.0309391](https://doi.org/10.1371/journal.pone.0309391) (`journal.pone.0309391.pdf` in repo root).

---

## 1. Motivation and honest framing

The source paper turns a single viral genome into a 2-D "genomic spectrogram" image (per-nucleotide binary indicator → STFT → sum the four nucleotides), then does two-stage ImageNet→HIV→SARS-CoV-2 transfer learning on a VGG-16 to classify a genome as recombinant / not, and uses Grad-CAM to localise. It reports ~94.6% test accuracy and claims a "recombination signature" at frequency *f* = 1/6 in the Spike protein.

We want to test whether this **sequence→image + transfer-learning** idea is useful on *this repo's* data (SANTA-simulated triplets + real-HIV LANL CRFs). Two facts from interrogating the paper shape everything below:

1. **The paper's task is whole-genome classification, not breakpoint localization.** Its hot zones span ~3,800 bp — roughly 10–20× too coarse for the ±200 bp this repo needs. So the technique maps onto the repo's **stated future goal** (CLAUDE.md: "take three sequences, decide *which* (if any) is the recombinant → a four-way head over `{seq1, seq2, seq3, none}`"), i.e. **recombinant *identification***, not the current breakpoint-localization work.

2. **The paper's single-sequence classifier cannot see recombination mechanistically; its "signature" is very likely a confound.** A model given one genome and no parents can only read composition. The *f* = 1/3 line is codon periodicity; *f* = 1/6 (period-6 = two codons) most plausibly reflects that the labelled recombinant lineages (XBB etc.) have distinctive Spike codon usage — i.e. a **variant-identity detector**, not a recombination detector. This repo has already been burned by exactly this class of confound:
   - the M3 v4 auxiliary "is-this-a-real-recombinant" head became a pure SANTA-vs-real *provenance* detector at CONFOUND AUROC 1.000;
   - the M3 v3 attempt to add non-recombinant negatives collapsed LANL F1 from 0.51 → 0.00 by over-learning "high divergence ⇒ not recombinant."

**Conclusion driving the design:** the paper's *claim* (a recombination signature at *f* = 1/6) is probably a confound, but the *method* (sequence→image + big-CNN + transfer) is a legitimate **third backbone family** the repo has not tried, alongside the M3 dilated CNN and the abandoned HyenaDNA. It is worth testing — **on the identification task, against the natural baseline, with the confound and the tiny real-HIV set handled by construction.** The spectrogram is a *hypothesis under test*, not an assumption.

## 2. Goal and scope

**Staged goal (user-selected):** identification first (transfer-learning-style), then a de novo localization CNN later.

**This spec fully specifies Stage 1 only:** 3-way recombinant identification ("which of an aligned triplet is the mosaic recombinant?"). Stage 2 (de novo localization) is sketched in §8 and gets its **own** spec once Stage 1 shows the image direction carries signal — keeping each spec to one implementable unit, per the repo's pattern.

**Eventual goal (documented, deferred):** extend the 3-way head to **4-way `{seq1, seq2, seq3, none}`** so the model can also decide that *none* of the three is recombinant. This is deliberately **not** in Stage 1 because "none" reintroduces the divergence/provenance confound that collapsed M3 v3; it is attempted only after Stage 1 proves the representation is reading mosaicism rather than divergence (see §5 divergence check).

**Non-goals for Stage 1:** breakpoint localization; the 4-way "none" head; beating the M3 localization F1 (0.565) or classical RDP (0.519) — those are localization numbers on a different task axis and are **not** comparable to identification accuracy.

## 3. Data

- **SANTA (train, abundant):** aligned triplets `(recombinant, parent1, parent2)` with known recombinant identity and breakpoints, via `cache_v2_reader.CacheV2` over `cache/v2/` using `splits/v2_filtered_split.json` (the realism-filtered split; do **not** use `splits/v2_split.json`). `MAX_SEQ_LEN = 10000`, gaps allowed.
- **Real HIV (train + held-out eval):** LANL CRF triplets from `build_lanl_triplets.py`. **Critical empirical fact:** this is only **4 CRF families** (CRF02_AG, CRF07_BC, CRF08_BC, CRF12_BF; CRF01_AE deliberately skipped). The recombinant is the CRF; parents are the pure subtypes. Enumerating valid parent pairings per CRF yields many triplets per family, but they are **non-independent** (same recombinant). Any further real recombinants under `data/real_recombinants/` may be folded in if they fit the identification framing.

All triplets are aligned, so their per-position features are on a shared coordinate system — the property that makes the triplet-image comparison meaningful.

## 4. Architecture — the shared harness and the representation bake-off

### 4.1 Shared identification harness

Every experiment uses the same harness so results are comparable:

- **Input:** an aligned triplet → three per-sequence 2-D feature maps, stacked as the channels of **one image**.
- **Channel permutation:** the three sequences are assigned to channels in a **randomly permuted order per sample**; the label is *which channel holds the recombinant*. Output is a **3-way softmax over channels**. (Permutation removes the trivial "answer is always channel 0" prior. It does **not**, by itself, defeat the composition confound — that is the job of §5.)
- **Fixed length:** each aligned triplet is **pre-padded with gaps to 10,000 positions before any transform**, so image width is fixed and the position axis is linear and comparable across samples. **No anisotropic resize of the position axis** (it would make the same pixel column mean different genomic positions across samples and poison both identification and any future Grad-CAM).
- **Shared transform grid:** the three channels must use an **identical** transform grid (same window, hop, pre-pad); assert equal frame counts across channels, or the R/G/B pixels do not correspond to the same position and the comparison premise silently breaks.
- **Normalization:** **symmetric joint normalization** computed across the three channels together (preserves the equality relationship that *is* the signal — e.g. recomb-magnitude == P1-magnitude in block A). **Never** per-channel z-normalization; **never** ImageNet mean/std (it bakes in channel asymmetry that fights permutation invariance).
- **Gap encoding:** explicit and **byte-identical across SANTA and LANL** (gap statistics differ between the two and would otherwise be a provenance leak). Gaps/pad map to 0 in indicator signals, consistent with the repo's zero-pad convention.

### 4.2 Backbone options

- Biggest ImageNet-pretrained backbone that trains at **batch ≥ 16 with AMP on the RTX 3070 (8 GB)** — candidates ConvNeXt-Base / EfficientNetV2-M / Swin-Base (images are small, so a large backbone is affordable). Exact pick deferred to the implementation plan. `timm` supplies the zoo (new dependency).
- Plus a **small from-scratch CNN floor** so the init A/B (§4.4) can separate "pretraining helped" from "big model helped." Images are small; upsampling a tiny spectrogram into a giant backbone is wasteful and artifact-prone, so the floor is also the more natural-capacity comparison.

### 4.3 Representation bake-off (the core experiment)

Three representation arms run through the identical harness. **The Fourier arms must beat the positional control**, or the paper's idea has not earned its place.

- **A0 — Positional control (must-beat).** Built from the three per-position **pairwise-identity tracks** `id(s_i, s_j)` (i.e. `id(s1,s2)`, `id(s1,s3)`, `id(s2,s3)`). This is the *natural* representation for a mosaic task: a recombinant R of parents A,B has `id(R,A)=1` on block 1, `id(R,B)=1` on block 2, `id(A,B)≈low` throughout — the model must find the sequence that "switches allegiance." It is a **symmetric** representation (it does not tell the model which sequence is the designated recombinant), unlike the repo's existing asymmetric `match_p1/match_p2` features.
  **2-D lift (explicit, for a fair comparison through the 2-D-CNN harness):** each pairwise-identity track is a 1-D signal, so A0 is lifted to a 2-D image by stacking **smoothed copies at a small fixed set of window scales** (rows = scales {50,100,200,500} bp as in the repo's MaxChi features, columns = position), giving a position×scale image directly analogous to the spectrograms' position×frequency. The three pairwise tracks map to the three permuted channels. (This same multi-scale positional image is what Stage 2's localizer will use, §8.)
- **A1 — Paper-faithful spectrogram.** Per sequence, per nucleotide *n* ∈ {A,C,G,T}: binary indicator signal → `scipy.signal.spectrogram(x_n, fs=1.0, nperseg=256, noverlap=<fixed, recorded>)`; sum the four: `S = S_a + S_c + S_g + S_t` (the paper's Eq. 1); log-scale. One channel per sequence. This is the user's original Fourier idea, reproduced honestly. **Departure from the paper:** we do **not** jet-colormap into RGB; each *sequence's* summed spectrogram is one grayscale channel, preserving dynamic range and enabling the triplet stacking.
- **A2 — Unsummed spectrogram.** The four nucleotide spectrograms kept **separate** (per sequence), directly testing the review's hypothesis that *summing over nucleotides destroys the base-identity signal* the identification task needs. (Channel bookkeeping: three sequences × four nucleotides = 12 planes; the permuted 3-way label still applies to the sequence-group, not the individual planes.)

### 4.4 Init / capacity A/B

Each representation arm is run under **{ImageNet-init, random-init, small-CNN floor}**. Rationale: the repo already learned pretraining can *hurt* (M1.3: MLM pretraining made the backbone invariant to local nucleotide identity), and the ImageNet→spectrogram domain gap is large with three-sequence channels that are not natural chroma. Test it; do not assume.

## 5. Confound diagnostics (blocking gates, run FIRST)

These run before trusting any bake-off result. They are the repo's "evidence-first" discipline applied to a known failure mode.

- **P2 — own-parents separability (HARD GATE).** Can a **single-sequence** classifier separate a recombinant from its **own two parents** above chance? If yes, there is a confirmed **marginal/composition leak** (the recombinant has a per-channel signature — intermediate divergence, gap pattern — that a permutation-invariant net will preferentially exploit), and it also undermines the premise that three sequences are needed. Fix the representation before trusting any triplet result.
- **Positional-scramble control.** Independently shuffle each channel's **position axis** (destroys mosaic alignment, preserves per-channel marginals). If triplet accuracy **survives** the scramble, the model is reading composition, not mosaicism → confounded. This is the single cleanest test of the anti-confound claim.
- **Divergence-vs-prediction analysis.** On the trained 3-way model, check that predictions are **not** explained by pairwise divergence (the summed-spectrogram's low-frequency energy is essentially a divergence proxy; the "none"-class collapse lived here). If the model keys on divergence, Stage-1 "success" will not survive the eventual 4-way head, and the head-swap will not fix a representational failure.

## 6. Evaluation — leave-one-CRF-out cross-validation

The real-HIV identification eval unit is the **triplet**, and there are only ~4 independent real families — so a single held-out snapshot at chance = 1/3 is statistically undecidable. Instead:

- **Leave-one-CRF-out CV.** Each fold trains on **SANTA + 3 of the 4 real CRFs** (all enumerated parent-pairings per CRF, so each contributes many triplets) and tests on the **held-out CRF family**. Rotate through all 4. The reported claim is therefore "**can it identify a recombinant family it was never trained on**" — a weaker but *decidable* claim than pure zero-shot transfer. (Trade-off accepted by the user: we give up the "train only on simulation, transfer to real biology" story in exchange for a decidable real-HIV number.)
- **Significance discipline (pre-registered).** One **primary metric** and one **decision rule**, fixed before running the backbone × init × threshold grid (the grid otherwise guarantees garden-of-forking-paths over few test points). Compare each Fourier arm to A0 with **McNemar's test on paired predictions**; report **cluster-bootstrap CIs by CRF** (block resampling — never iid CIs over non-independent parent-pairings). Run a **label-permutation null** (shuffle held-out labels; confirm chance) as a cheap leak guard.
- **Power statement (in the plan).** Compute, for the actual per-fold N, the **minimum detectable paired margin** at 80% power / α = 0.05 under McNemar. If the expected improvement is below it, say so before building.
- **Confound meter, first-class.** Report the **SANTA↔held-out-CRF accuracy gap** with CI. High absolute SANTA accuracy with a large gap is a **fail**, even if held-out > chance.

**Gate to proceed to Stage 2:** a Fourier arm (A1 or A2) beats the positional control A0 on held-out real CRFs by a margin that clears the power threshold, **and** the init/capacity question (§4.4) is settled, **and** the confound diagnostics (§5) pass.

## 7. Components / files (clean build, repo as home)

| File | Responsibility | Depends on |
|---|---|---|
| `spectrogram/encode.py` | Sequence→feature-map for all three arms (A0 pairwise-identity, A1 summed spectrogram, A2 unsummed); fixed 10 kb pre-pad; shared grid; joint normalization; explicit gap encoding. Cache images to disk once. | scipy, `cache_v2_reader`, `build_lanl_triplets` |
| `spectrogram/harness.py` | Triplet→permuted-channel image dataset; 3-way label; backbone loader (`timm` + small-CNN floor); train loop (AMP). | torch, timm |
| `spectrogram/probe.py` | §5 diagnostics: P2 own-parents gate, positional-scramble control, divergence analysis. | harness |
| `spectrogram/eval.py` | Leave-one-CRF-out CV; McNemar; cluster-bootstrap by CRF; label-permutation null; SANTA↔CRF gap; power calc. | harness |

Each unit has one clear purpose, a defined interface, and is testable independently.

## 8. Stage 2 (sketch only — own spec later)

De novo **localization** CNN: drop the STFT (too coarse for ±200 bp) and feed a **multi-scale positional triplet-image** (e.g. rows = the repo's MaxChi disparity scales {50,100,200,500}, columns = position) into a **U-Net-style dense head** emitting a per-position breakpoint track — the M3 feature set given a 2-D-CNN + transfer treatment. The fixed-length pre-padding decision (§4.1) is made now so Stage 2's dense head / Grad-CAM is not blocked by an irreversible position-axis rescale. Specified only once Stage 1 justifies the image direction.

## 9. Risks and open questions (for the implementation plan)

- **Most likely outcome, stated honestly:** the review predicts A1 (summed spectrogram) loses to A0 (positional) because summing over nucleotides + STFT deletes the base-identity/positional signal identification needs. If so, that is a **real, reportable finding** about the paper's representation, not a project failure — and A2 vs A1 quantifies whether *summing specifically* is the culprit.
- **Real-N is small even after CV;** the power statement decides whether any "win" is real.
- **ImageNet transfer may not help** (three-sequence channels ≠ natural chroma); the floor + random-init arms are there to detect this.
- **Exact backbone, `noverlap`, image resolution, primary metric, and decision rule** are pinned in the implementation plan, not here.
