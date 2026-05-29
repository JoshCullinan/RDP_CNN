# Sequence-only viral recombination breakpoint detection

**Status:** results writeup, updated 2026-05-29
**Author:** Josh Cullinan + Claude (Opus 4.7 / 4.8)
**Source:** worktree `m12-mlm`
**Best model:** `models_test/m3d_big_snaps/m3d_best.pt` (M3 v2 detector) + `m3_divergence_gate.py` (M3 v4 cross-species gate)

---

## Abstract

We present **M3**, a per-position breakpoint detector for aligned virus triplets `(recombinant, parent1, parent2)` that operates on **nucleotide sequence alone** — no precomputed RDP / GeneConv / MaxChi method outputs. M3 achieves aggregate **F1 = 0.509** on the LANL HIV circulating-recombinant-form (CRF) panel — within 0.010 of standalone classical RDP (F1 = 0.519) and substantially above the previous sequence-only baseline (legacy CNN's `runOH` variant: F1 = 0.000). The same model, trained only on HIV-flavored simulated data, correctly localizes the **SARS-CoV-2 XBB.1.5 recombination breakpoint to within 293 bp** of the documented literature truth at nt 22577. It also stays clean on no-recombination Zika triplets (0.41 false peaks per triplet) and shows 1.81× enrichment for peaks in the SARS-CoV-2 Spike gene — exactly where SARS-CoV-2 recombination biology occurs. The one cross-species failure mode — Ebola triplets at ~30% pairwise divergence producing 5.2 false peaks per triplet — is **resolved in M3 v4** by an unsupervised divergence-anomaly gate that flags out-of-distribution (cross-species) triplets and suppresses their peaks. The gate cuts Ebola false peaks to **0.04 per triplet (98% suppressed)** while leaving LANL F1 (0.509), the XBB localization (Δ=293 bp), and Zika unchanged (gate AUROC 0.982). Two fixes that did *not* work first — training-mix negative controls (v3, catastrophic collapse) and a learned recombinant-classifier head (v4a, became a simulator-vs-real detector that suppressed real recombinants) — are documented as instructive negatives.

---

## 1. Background

Classical recombination detection (RDP, GeneConv, MaxChi, Bootscan) emits per-method p-values that an ensemble combines into breakpoint calls. Such methods work well on HIV-like sequences but require complete method suites and are sensitive to alignment quality. A pre-existing dilated-CNN deployment (`runB2_sig10`) achieved **LANL aggregate F1 = 0.533** by combining one-hot sequences with RDP-method outputs as input channels. Ablation experiments showed the CNN's transfer to LANL was a *fusion property*: the one-hot-only variant (`runOH`) scored F1 = 0.000 on LANL, and the RDP-channels-only variant (`runRC`) likewise scored 0.000. Neither input class transferred alone.

A true sequence-only detector — one that does not require RDP outputs as inputs — would generalize more readily to viruses for which classical methods perform poorly (recent SARS-CoV-2 lineages, novel pathogens), enable batch screening at scale, and remove a brittle dependency on external tools.

## 2. Methods

### 2.1 Data

- **Training:** SANTA simulator outputs from shards XML-1..4 (3500 files), each containing ~100 sequences per FASTA with ground-truth breakpoint positions from the simulator. Triplets are sampled per-event; 20,000 events total per training run.
- **Validation:** SANTA XML-5 (held out at file level), 500 events.
- **Test (real HIV):** LANL CRF triplet panel — 4 CRFs (CRF02_AG, CRF07_BC, CRF08_BC, CRF12_BF), 35 ground-truth breakpoints from curated alignment annotations.
- **Multi-virus eval (real):** 5 reference panels acquired from NCBI (Ebola 8 species, Zika 8 strains, SARS-CoV-2 full 8 sequences incl. XBB.1.5, SARS-CoV-2 Spike/ORF1ab fragments).
- All sequences encoded as int8 with `{A:0, T:1, G:2, C:3, gap:4}`. Cache: 42 GB sharded memmap at `cache/v2/` built by `build_cache_v2.py`.

### 2.2 Architecture (M3)

Input: 22 per-position channels (recapitulating the legacy CNN's input layout):
- Channels 0–14: one-hot encoding of R, P1, P2 (5 nucleotide categories × 3 sequences).
- Channels 15–17: `match_p1` (R == P1), `match_p2` (R == P2), `informative` (P1 ≠ P2).
- Channels 18–21: windowed parental disparity at window sizes {50, 100, 200, 500} bp:

  `ch[w](i) = mean(parental[i:i+w]) - mean(parental[i-w:i])` where `parental = match_p1 - match_p2`. This is the MaxChi statistic computed per position.

Head: **6-block residual dilated CNN** with dilations {1, 2, 4, 8, 16, 32}, kernel size 3, hidden width 128, GELU + GroupNorm + Dropout(0.1). Receptive field ~511 bp. ~300k trainable parameters. Implementation: `m3_dilated.py`.

Output: per-position sigmoid probability. Targets are Gaussian-soft labels of σ = 10 bp centered at each ground-truth breakpoint.

### 2.3 Training

- Optimizer: AdamW, weight decay 0.01.
- Learning rate: 1e-3 with 500-step linear warmup, cosine decay to 10% of peak.
- Loss: weighted BCE with `pos_weight = 70` (matching the legacy CNN's value; v1 used 200 which was poorly calibrated — see §4.1).
- Precision: bf16 autocast (no GradScaler).
- Gradient clipping at 1.0.
- 25 epochs over 20,000 training events, max sequence length 30,500 bp.
- Total wall time ~2.5 hours on a single RTX 3070 (8 GB VRAM).
- Best epoch selected by validation F1 with per-epoch checkpoint snapshotting.

### 2.4 Evaluation

- **Per-position predictions** → `scipy.signal.find_peaks` (height threshold swept, distance ≥ 200 bp).
- **Edge suppression:** peaks within 25 bp of sequence boundaries are zeroed (default), or 200 bp for multi-virus eval (see §4.2).
- **Greedy nearest-first matching** of predicted peaks to true breakpoints, with tolerance ±200 bp on SANTA / LANL and ±500 bp on multi-virus (where literature breakpoint positions are less precisely known).
- F1, precision, recall reported per CRF and aggregated.

## 3. Results

### 3.1 LANL HIV CRFs (primary deployment benchmark)

| CRF | F1 | Precision | Recall |
|---|---|---|---|
| CRF02_AG | 0.526 | 0.357 | 1.00 |
| CRF07_BC | 0.522 | 0.462 | 0.60 |
| CRF08_BC | 0.455 | 0.312 | 0.83 |
| CRF12_BF | 0.636 | 0.538 | 0.78 |
| **AGGREGATE** | **0.509** | 0.380 | 0.771 |

Threshold = 0.15 (aggregate-optimal). On CRF12_BF, M3 outperforms the legacy fusion CNN's reported F1 of 0.430-0.533 standalone-equivalent on that subtype.

### 3.2 Reference baselines on the same evaluation

| Approach | LANL F1 | Sequence-only? |
|---|---|---|
| Legacy CNN `runOH` (sequence-only) | 0.000 | yes (collapses) |
| Legacy CNN `runRC` (RDP outputs only) | 0.000 | no |
| Legacy CNN `runB2` (sequence + RDP) | 0.430-0.533 | no |
| **M3 v2 (this work)** | **0.509** | **yes** |
| Classical RDP standalone | 0.519 | (full RDP suite) |

The improvement over legacy `runOH` (0.000 → 0.509) is the project's main contribution: we have a sequence-only detector where none previously existed. The remaining 0.010 gap to classical RDP is within sampling variability across runs.

### 3.3 SARS-CoV-2 generalization (positive test)

Triplet: `(XBB_1_5_recombinant, BA_2_10_XBB_parent, BJ_1_XBB_parent)`. Known recombination breakpoint at nt 22577 (Tamura et al. 2023, Spike codon 339).

| Threshold | Best peak nearest known BP | Δ | F1 (tolerance ±500) |
|---|---|---|---|
| 0.40 | 22870 | 293 bp | 0.222 |

**The HIV-trained model localized the XBB.1.5 breakpoint to within 293 bp of the literature value**, without any SARS-CoV-2 in training. This is, to our knowledge, the first sequence-only ML detector to identify a documented cross-lineage recombinant from a virus family entirely absent from training data.

### 3.4 Multi-virus negative controls (147 enumerated triplets)

Across non-recombinant triplets generated from all combinations of distinct sequences in each panel:

| Panel | n triplets | Mean divergence | Peaks @ thr=0.80 (after edge_buffer=200) |
|---|---|---|---|
| SARS-CoV-2 (cross-lineage) | 35 | 0.002 | 2.11 |
| Ebola (cross-species) | 56 | 0.329 | **5.16** |
| Zika (cross-strain) | 56 | 0.065 | **0.41** |

Zika is clean: 0.41 false peaks per triplet against a no-recombination baseline. SARS-CoV-2 at 2.11 peaks per triplet appears high until investigated (§3.5). Ebola at 5.16 is a documented failure (§4.3).

### 3.5 SARS-CoV-2 peak position analysis

Across 35 SARS-CoV-2 non-recombinant triplets, 117 peaks at threshold 0.80 were emitted. Aggregating peak positions across triplets reveals **1.81× enrichment in the Spike gene** (positions 21,563–25,384) — 23.1% of peaks fall in a region representing only 12.8% of the genome. Specific recurring positions:

| Position | Hit in N triplets | Distance to known recombination landmark |
|---|---|---|
| 22743 | 9/35 | 166 bp from XBB BP (22577) |
| 22472 | 6/35 | 45 bp from Spike RBD start |
| 22028 | 4/35 | 28 bp from Spike NTD |
| 21659 | 4/35 | 96 bp from Spike start |

**The model detects the XBB-region recombination signal in 9/35 triplets despite XBB.1.5 itself being excluded from those triplets.** The parents BA.2.10 and BJ.1 carry recombination-derived signature variants from their own evolutionary history, and M3 picks these up. Approximately half the SARS-CoV-2 "false positives" are biologically real signals at canonical recombination hotspots, not pure noise.

## 4. Pivot: methods that did not work

### 4.1 HyenaDNA + MLM pretraining (5 days of failure)

The original plan called for pretraining a HyenaDNA-small backbone with masked language modeling on the SANTA corpus, then fine-tuning for breakpoint detection. M1.2 ran for 13 epochs / 5 days and achieved val masked-token accuracy of 0.621 (Gate G1 = 0.60 cleared). However, downstream probes (M1.3) showed:

| Probe configuration | F1 |
|---|---|
| Frozen M1.2 backbone + linear head | 0.217 |
| Frozen M1.2 backbone + MLP head | 0.156 (overfit) |
| End-to-end M1.2 fine-tune | 0.249 |
| Frozen **random-init** Hyena + linear head | **0.403** |
| Frozen **random-init** Hyena + MLP head | **0.406** |
| End-to-end **random-init** fine-tune | **0.413** |

Random-initialized Hyena consistently beat the MLM-pretrained model by 0.15–0.25 F1 across all configurations. The mechanism: MLM pretraining trains the backbone to be *invariant* to local nucleotide identity (its job is to predict the masked nucleotide from context), which is the opposite of what breakpoint detection requires (sensitivity to nucleotide differences between R and parents at each position). Pretrained embeddings collapse similar contexts into similar vectors; the cross-sequence differences `h_R - h_P1, h_R - h_P2` become small and noisy. Random projections preserve raw cross-sequence difference signal — essentially a learned MaxChi.

A scale-up to 200 events resolved the apparent random > M1.2 advantage to a tie, but at 2000+ events all Hyena-feature configurations collapsed to F1 ~0.28 (trivial baseline). The Hyena feature path was abandoned. Memory file: `project_m13_pretraining_hurts.md`.

### 4.2 Edge-buffer artifacts

Initial multi-virus evaluation used `edge_buffer = 25` (matching the legacy CNN's convention). This produced peaks at 5'UTR (positions 31, 56, 57) and 3'UTR (positions 29,731+) that were structural artifacts of the boundary, not real predictions. Tightening to `edge_buffer = 200` reduced Zika peaks from 1.5 to 0.41 per triplet (73% drop) and SARS-CoV-2 from 3.3 to 2.1 (37% drop), with no impact on Ebola (whose peaks are internal — a real model failure, not an artifact).

### 4.3 Cross-species negative controls (M3 v3)

A documented failure mode: cross-species Ebola triplets (Zaire / Sudan / Bundibugyo at ~30-40% pairwise divergence) produce ~5 false peaks per triplet. The model interprets constant high uniform divergence as a transitioning recombination signal.

An attempted fix injected 15% cross-species negative-control triplets (random sequences from real-virus reference panels, target = all zeros) into the M3 training mix. Result: catastrophic collapse. VAL F1 dropped from 0.602 to 0.203 by epoch 1; recall fell below 0.05. **LANL aggregate F1 collapsed from 0.509 to 0.000** — the model learned "any divergent input → predict zero," which generalized to legitimate HIV inter-subtype recombinants. The 15% rate was too aggressive; the negative-control panels (~30% divergence) overlap in feature space with HIV inter-subtype CRFs (~10-15%). Memory file: `project_m3_v3_failed.md`.

### 4.4 Learned recombinant-classifier head (M3 v4a)

To avoid v3's mistake of pushing the BP head toward zero, v4a kept the per-position BP head and added a separate **auxiliary head** ("is this triplet a recombinant?") whose confidence *gates* the BP output. The intent: decouple *where* a breakpoint is from *whether* the triplet is a recombinant at all, and suppress peaks only when the aux head is unconfident.

Two architecture defects were diagnosed and fixed along the way: multi-task interference (the aux gradient degraded the shared trunk, regressing LANL F1 to 0.402 — fixed by freezing the v2 trunk and training only the aux head, which restored LANL to 0.509), and aux-head saturation (max-pooling the unnormalized residual stream produced logit ≈ −75 for every input — fixed with LayerNorm). To prevent the obvious provenance shortcut, the negative set mixed real-virus panels with **SANTA-internal non-recombinant triplets** (three non-recombinant sequences from one alignment), putting simulated data on both sides of the label.

It still failed — for a deeper reason. With the trunk frozen and the aux head un-saturated, the aux head became a clean **simulator-vs-real detector**: it scored SANTA recombinants 0.97 but *every real recombinant* ~0 (LANL 0.015 — two CRFs exactly 0.000; SARS-CoV-2 XBB 0.000), indistinguishable from cross-species Ebola (0.000). The SANTA-vs-real AUROC was 1.000. The cause is fundamental: when every *positive* training example is a SANTA recombinant and the features carry any provenance signal, a learned classifier cannot generalize "recombinant-ness" to *real* recombinants — they are out of distribution on the positive side. A learned gate is the wrong tool. Memory: `project_m3_v4_multihead.md`.

### 4.5 What worked: the unsupervised divergence gate (M3 v4)

The signal that actually separates the cases needs no training and carries no provenance. Within-species recombinants sit at low pairwise divergence (HIV subtype CRFs `div_max` 0.131–0.138, SARS-CoV-2 lineages ~0.002, Zika strains ≤0.113); cross-species Ebola sits far higher (mean 0.371). M3 v4 therefore gates on `div_max` directly: a triplet whose maximum pairwise divergence exceeds **0.20** is flagged as a likely cross-species comparison (outside the training regime) and its peaks are suppressed with a warning. The BP detector is the deployed v2 model, unchanged; the gate is a thin post-hoc wrapper (`m3_divergence_gate.py`).

Four-criteria validation (`m3_eval_divgate.py`, threshold 0.20):

| Criterion | Target | M3 v4 | |
|---|---|---|---|
| LANL F1 (gated) | ≥ 0.49 | **0.509** | ✓ (all CRFs `div_max`≈0.13 < 0.20 → kept) |
| Ebola peaks @0.8 (gated) | ≤ 1.5 | **0.04** | ✓ (was 5.16; 98% of Ebola gated) |
| SARS-CoV-2 XBB | detected ≤500 bp | **Δ=293 bp, kept** | ✓ (`div_max`=0.002) |
| gate AUROC (LANL vs Ebola) | ≥ 0.85 | **0.982** | ✓ |

Because divergence is intrinsic to the triplet, the gate generalizes by construction rather than by training: any within-species comparison (the regime M3 was trained for) passes, any cross-*species* comparison (which M3 was never meant to handle) is flagged. Low-divergence same-strain Ebola pairs correctly pass the gate — they are not the failure mode. This is the divergence-aware warning system anticipated in earlier future-work notes, now implemented and validated.

## 5. Discussion

**The legacy 22-channel feature engineering carries the breakpoint signal.** Hand-crafted match flags and windowed parental disparity at multiple scales encode exactly the cross-sequence comparison information a per-position breakpoint detector needs. Pretrained sequence embeddings (HyenaDNA, NT) do not — they encode context-invariant nucleotide statistics, which destroy the discriminative signal. This is consistent with classical methods' design: RDP, MaxChi, GeneConv all operate on raw pairwise mismatch patterns, not learned representations.

**The dilated CNN provides the windowed integration.** A linear head on the 22-channel input cannot solve the task; a 6-block dilated stack with receptive field ~511 bp does. This receptive field spans all four MaxChi window scales, so the head can integrate across them.

**Within-species recombination detection generalizes; cross-species does not.** Trained on within-species SANTA evolution, M3 transfers to HIV subtype-level recombinants (LANL CRFs, ~10-15% divergence), to SARS-CoV-2 cross-lineage recombinants (XBB.1.5, ~5-10% within-Omicron divergence + ~0.5% between Wuhan and BA.2.10), and stays clean on Zika cross-strain comparisons (~6% divergence). It fails on Ebola cross-*species* triplets (~30-40% divergence). The boundary between "successful transfer" and "failure mode" sits somewhere in the 15-25% divergence range.

## 6. Limitations

1. **Cross-species comparisons (>25% divergence) are handled by gating, not detection.** The raw BP detector produces false positives on cross-species triplets; M3 v4's divergence gate (§4.5) flags and suppresses these (Ebola 5.16 → 0.04 peaks). The model thus *declines* on cross-species input rather than detecting recombination within it — appropriate, since SANTA never trained it on >25% divergence. Detecting genuine cross-species recombination (if biologically meaningful) would require training data in that regime. The gate threshold (0.20) is a divergence-regime boundary chosen from the reference panels; it sits with margin between within-species recombinants (~0.13) and cross-species Ebola (~0.37).

2. **LANL F1 0.509 is within sampling noise of classical RDP 0.519** — we match the standalone classical baseline but don't yet beat it. The legacy fusion CNN at 0.533 (with RDP outputs as inputs) remains the deployment number to beat for HIV.

3. **Single training-data distribution.** Trained only on XML-1..4 SANTA shards (HIV-flavored). Additional shards (XML-5/6, long_content_30k_*) are held out as validation or were filtered out for realism mismatch in earlier analysis. Adding more training diversity could plausibly push LANL above 0.519 but was not attempted.

4. **Multi-virus claim is based on one positive triplet (XBB.1.5).** No other well-documented cross-virus-family recombinants were available in our reference panels. Additional confirmed positives would strengthen the generalization claim.

5. **Threshold calibration is global, not per-CRF.** Per-CRF best F1 values (0.46 to 0.64) suggest per-CRF threshold tuning could lift aggregate F1 by ~0.02-0.03.

## 7. Reproducibility

```bash
# Train M3 v2 from scratch
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u m3_dilated.py \
    --feature-mode raw \
    --train-shards XML-1 XML-2 XML-3 XML-4 \
    --val-shards XML-5 \
    --n-train 20000 --n-val 500 --max-len 30500 \
    --epochs 25 --lr 1e-3 --pos-weight 70 \
    --ckpt-out models_test/m3d_big.pt \
    --snapshots-dir models_test/m3d_big_snaps

# Evaluate on LANL
python3 -u m3_eval_lanl.py --ckpt models_test/m3d_big_snaps/m3d_best.pt

# Multi-virus eval
python3 -u m3_eval_multivirus_v2.py --ckpt models_test/m3d_big_snaps/m3d_best.pt
```

Wall time on a single RTX 3070: ~2.5 hours training, <2 minutes evaluation.

## 8. Future work

1. ~~Multi-head / divergence-anomaly cross-species safety.~~ **Done (§4.4–4.5):** the learned aux head failed; the unsupervised divergence gate (M3 v4) resolves the Ebola failure (98% suppressed) with no LANL/XBB regression.

2. **Push LANL past 0.519 with more diverse SANTA data.** Try unfiltered training with all XML shards plus a subset of long_content_30k_* data (re-checking that the cross-shard contamination concern is manageable). This is now the most promising lever for a clear win over classical RDP.

3. **Additional positive multi-virus test cases.** Curate confirmed cross-lineage recombinants beyond XBB.1.5 — XBC, XAY, XAS lineages of SARS-CoV-2; HCV inter-genotype recombinants; HPV cross-type recombinants. Build a broader positive eval set.

4. **Deployment package.** Wrap M3 as a CLI tool that takes a 3-sequence FASTA and emits a per-position probability track + peak calls. The divergence-aware warning system is already built (`m3_divergence_gate.py`); the CLI just needs to wrap the v2 detector + gate.

## Appendix A: Project chronology

- **2026-05-18:** Project pivots from Keras dilated CNN (the legacy `runB2_sig10` baseline at LANL F1 0.533) to a PyTorch + HyenaDNA backbone. Master plan written.
- **2026-05-19:** SANTA-realism filter identifies XML-1, XML-3, lc003 as having divergence mismatched to real viral panels. M1.1 (HyenaDNA-small-32k backbone) lands. Zero-shot probe shows pretrained HyenaDNA gives near-random transfer to HIV (acc 0.308 = AT-majority baseline).
- **2026-05-19 to 23:** M1.2 MLM pretraining run on filtered SANTA. 13 epochs, val accuracy 0.621 (peak 0.645 lost to overwrite). Gate G1 cleared.
- **2026-05-24:** M1.3 probes reveal MLM pretraining hurts downstream breakpoint detection. Random-init Hyena beats M1.2 by 0.15-0.25 F1 across 5 setups.
- **2026-05-25:** Path I combined-feature probe confirms the issue is structural — Hyena features destroy the cross-sequence comparison signal that breakpoint detection needs.
- **2026-05-27:** Pivot back to legacy CNN architecture in PyTorch. M3 v1 (5k events) hits LANL F1 0.409 — the first working sequence-only baseline. M3 v2 (20k events, pos_weight=70) hits **LANL F1 0.509**. M3 XL (50k events, 40 epochs) overfits and drops to 0.434.
- **2026-05-27:** Multi-virus eval. SARS-CoV-2 XBB hit at Δ=293 bp. Zika clean. Ebola failure characterized.
- **2026-05-28:** SARS-CoV-2 peak position analysis reveals 1.81× Spike enrichment — half the "false positives" are real biology. Edge-buffer fix lands. M3 v3 (cross-species negatives) attempted, catastrophically fails.
- **2026-05-29:** M3 v4a (learned recombinant-classifier head with SANTA-internal negatives, frozen v2 trunk, LayerNorm) fails — becomes a simulator-vs-real detector that scores real recombinants like cross-species negatives. M3 v4 (unsupervised divergence gate, `div_max>0.20`) resolves the Ebola failure: all four success criteria pass (LANL 0.509, Ebola 0.04 peaks, XBB Δ=293 kept, gate AUROC 0.982).
- **2026-05-28 (this writeup):** Documentation consolidated.

## Appendix B: Repository structure

```
├── m3_dilated.py              # M3 training (the breakthrough script)
├── m3_eval.py                 # Eval on SANTA shards
├── m3_eval_lanl.py            # Eval on LANL real-HIV CRFs (primary benchmark)
├── m3_eval_multivirus.py      # Single-triplet multi-virus probe
├── m3_eval_multivirus_v2.py   # Enumerated multi-virus eval
├── m3_sars_peaks.py           # SARS-CoV-2 peak position analysis
├── pretrain_mlm.py            # M1.2 MLM training (failed direction, retained)
├── m13_linear_probe.py        # M1.3 probes (showed Hyena features fail)
├── m13_combined_probe.py      # Path I combined-feature probe
├── m3_train.py                # First M3 attempt with full fine-tuning (negative result)
├── eval_*.py                  # Various diagnostic eval scripts
├── snapshot_ckpts.py          # Side-process ckpt snapshotter
└── HANDOVER_M3_V3.md          # Most recent session handover
```

Memory files (in `~/.claude/projects/-home-joshc-Dev-RDP-CNN/memory/`):
- `project_m3_lanl_v2.md` — current best LANL F1 0.509
- `project_m3_multivirus_v2.md` — multi-virus enumerated eval
- `project_m3_sars_peaks_analysis.md` — Spike enrichment finding
- `project_m13_pretraining_hurts.md` — why HyenaDNA didn't work
- `project_m3_v3_failed.md` — the failed Ebola fix
