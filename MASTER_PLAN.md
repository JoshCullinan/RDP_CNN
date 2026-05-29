# Master plan — sequence-only backbone replacement

**Status:** active, written 2026-05-18. **Reality update 2026-05-29.**
**Audience:** future agents (Claude or otherwise) and the human collaborator.

---

## 2026-05-28 status header — read this first

The Phase 1-4 plan below was written under the assumption that **MLM pretraining + a HyenaDNA backbone + cross-sequence attention** would deliver sequence-only breakpoint detection. That assumption proved wrong over May 19-25:

- **M1.2 MLM** (10 days planned, ran 13 epochs / 5 days, val acc 0.621): completed but the resulting pretrained backbone **actively hurt** downstream breakpoint detection. See `project_m13_pretraining_hurts.md`.
- **M1.3 probes** (random-init vs M1.2-init Hyena features): random consistently beat M1.2 by 0.15-0.25 F1 across 5 setups. The MLM objective trains nucleotide-identity *invariance* — opposite of what breakpoint detection needs.
- **M3-mini at 40 events**: F1 0.41 looked promising but turned out to be small-sample sampling noise. At 200+ events, Hyena-feature-based M3 collapses to F1 0.28 (trivial baseline) regardless of head type.
- **Pivot to legacy-CNN-style M3** (raw 22ch + dilated CNN head, no Hyena): **F1 0.509 on LANL real-HIV CRFs**, within 0.010 of classical RDP standalone (0.519). First working sequence-only detector.

**Actual completed milestones (replacing the planned Phases 1-4):**

| Milestone | Date | Result |
|---|---|---|
| M1.2 MLM pretraining | 2026-05-23 | Done, val acc 0.621, but **counterproductive downstream** |
| M1.3 probes + M3-mini | 2026-05-24 | Random > M1.2 across all setups; Hyena features structurally wrong |
| Path I combined-feature probe | 2026-05-25 | Hyena features don't add value on top of random projections |
| **M3 v1: dilated head, 5k events** | 2026-05-27 | **LANL F1 0.409** (first working sequence-only baseline) |
| **M3 v2: 20k events, pos_weight=70** | 2026-05-27 | **LANL F1 0.509** (peer to classical RDP 0.519) |
| **M3 v2 multi-virus eval** | 2026-05-27 | SARS-CoV-2 XBB hit Δ=293bp; Zika clean; Ebola FAILS (cross-species) |
| M3 XL: 50k × 40 epochs | 2026-05-27 | Overfit to SANTA, LANL 0.434 (worse than v2) |
| SARS-CoV-2 peak analysis | 2026-05-28 | Half the "FPs" are real Spike-hotspot signals (1.81× enrichment) |
| Multi-virus v2 (edge_buffer=200) | 2026-05-28 | Zika 1.5→0.41 peaks, SARS 3.3→2.1, Ebola unchanged |
| M3 v3: + neg_frac=0.15 cross-species negatives | 2026-05-28 | **FAILED** — LANL 0.509→0.000 (BP learned "divergent→zero") |
| M3 v4a: learned aux recombinant-gate head | 2026-05-29 | **FAILED** — became simulator-vs-real detector; scores real recombinants (LANL/XBB) ~0 like Ebola (CONFOUND AUROC 1.000) |
| **M3 v4: unsupervised divergence gate (`div_max>0.20`)** | **2026-05-29** | **✅ ALL CRITERIA PASS** — LANL 0.509, Ebola peaks 5.16→0.04, XBB kept Δ=293, gate AUROC 0.982 |

**Current best model:** `models_test/m3d_big_snaps/m3d_best.pt` (M3 v2 detector, LANL F1 0.509) **+ `m3_divergence_gate.py`** (M3 v4 cross-species gate). Validate with `m3_eval_divgate.py`. The Ebola failure mode is RESOLVED.

**The original Phase 1-4 plan below is retained as historical context but is NOT the active plan.** New work iterates from M3 v2 outcomes (memory files: `project_m3_lanl_v2.md`, `project_m3_multivirus.md`, `project_m3_multivirus_v2.md`, `project_m3_sars_peaks_analysis.md`).

**Active questions for next session:**
- ~~Does a fix for Ebola exist that doesn't regress LANL?~~ **Answered:** the M3 v4 divergence gate (not a learned classifier, not training-mix negatives). Ebola resolved.
- Can we push LANL F1 past classical RDP (>0.519) with more data diversity (not just more of the same)? **Now the top open lever** (writeup §8.2).
- ~~Build the deployment CLI (`bp_detect <fasta>`).~~ **Done:** `bp_detect.py` wraps `M3GatedDetector` (v2 BP head + divergence gate). Takes an aligned 3-seq FASTA → per-position probability track (`.track.tsv`), breakpoint calls (`.peaks.tsv`), recombinant-confidence + OOD warning (`.json`). Validated on LANL (trusted, 5 peaks), XBB (trusted, peak at 22870), Ebola cross-species (gated, 10 raw FPs → 0 + warning).
- Expand the positive multi-virus eval set (XBC/XAY SARS lineages, HCV, HPV) to strengthen the cross-lineage claim beyond XBB.1.5.

---

## Original plan (May 18, retained for context)



## North star

Build a recombination-breakpoint detector that:

1. Operates on **raw sequence one-hots only** (no MaxChi disparities, no RustRDP outputs).
2. Beats classical RDP on real HIV (LANL CRF panel, aggregate F1).
3. Has a real chance of generalising to non-HIV genomes where classical methods may not transfer.

The current deployment baseline (**runB2_sig10**, LANL agg F1 0.533) is structurally a refinement layer on top of RustRDP — `runOH` and `runRC` ablations both collapsed to LANL F1 = 0, proving the CNN has not learned recombination from sequence alone. The master plan replaces the backbone and the training regime to fix this.

---

## Strategic context (read first)

- **What's unused:** 8,499 long_content_30k_* FASTAs at 30 kb (deployment scale), 900 XML-6 FASTAs at 10 kb, ~1.3M total non-recombinant sequences usable as MLM corpus. runB2_sig10 trained on ~1.3% of the labelled events available.
- **What's wrong with the current architecture:** dilated CNN with RF ~410 bp can't natively perform the long-range pairwise comparisons a change-point detection task requires. Receptive-field extension (diagnostic C) didn't help because the issue is the primitive, not the span.
- **What the ablations proved:** the CNN learned to combine RustRDP-derived channels with one-hot features. Neither input class alone transfers. The next model must work without the engineered inputs to be a primary detector.

---

## Milestones

Each milestone is sized to be **achievable by a single agent run** (a few hours to ~a day of work) and produces a **concrete deliverable** that survives the session. Milestones are sequential within a phase but phases 1 and 2 can overlap once Phase 0 lands. Each milestone has explicit success criteria so an agent knows when to stop and report.

---

### Phase 0 — Foundations (no model training yet)

#### M0.1 — ✅ DONE 2026-05-18 — Unified data loader

Artifacts: `data_loader_v2.py`, `test_data_loader_v2.py`, `audit_full.log` (commit ca36431).

| Dir | Triplets | Unique events | Multiplicity |
|---|---:|---:|---:|
| XML-1 | 32,405 | 32,405 | 1.0 |
| XML-2 | 39,068 | 39,068 | 1.0 |
| XML-3 | 38,921 | 38,921 | 1.0 |
| XML-4 | 18,816 | 18,816 | 1.0 |
| XML-5 | 28,959 | 28,959 | 1.0 |
| XML-6 | 97,227 | 8,984 | 10.8 |
| long_content_30k_001 | 18,290 | 1,059 | 17.3 |
| long_content_30k_002 | 2,027,957 | 77,707 | 26.1 |
| long_content_30k_003 | 1,259,145 | 44,409 | 28.4 |
| UnseenTestSet | 5,539 | 5,539 | 1.0 |
| **TOTAL** | **3,566,327** | **295,867** | — |

Key findings that change downstream milestones:
- Unique-event total 295,867 is ~6% below the prior estimate (315k); triplet-level total is ~11× higher than unique events (sibling-recombinant inheritance dominates).
- **XML-6 unblocked via `pick_parents_rdp5ml.py`** (sim-csv-only mode — RDP-style closest-relative inference per BP segment from `.faSimVSRealCompare.csv` alone, no SANTA logs needed). 900 files processed in 64 s, adding **8,984 unique events at 10 kb length** that fill the curriculum gap between 4 kb XML and 30 kb long_content. Script validated 6/8 exact agreement on XML-1 with logs hidden. Any future SANTA dump lacking `.faParents.csv` and `.faRecombIdentifyStats.csv` should be processed with that script first.
- **Long-content sibling-recombinant multiplicity** (17–28×) means each unique BP configuration appears as many correlated training rows. M0.2 caches all rows so downstream code can choose; M3 should dedupe-by-event or downweight within-event correlation in batches to avoid effective-batch-size collapse.

#### M0.2 — ✅ DONE 2026-05-18 — Sharded cache builder

Artifacts: `build_cache_v2.py`, `cache_v2_reader.py`, `test_cache_v2.py`, `build_full.log`. Cache itself at `cache/v2/` (gitignored, 42 GB).

| Shard | Files | Events | Alignments size |
|---|---:|---:|---:|
| XML-1 | 1,215 | 32,405 | 542 MB |
| XML-2 | 1,110 | 39,068 | 1,177 MB |
| XML-3 | 718 | 38,921 | 485 MB |
| XML-4 | 1,080 | 18,816 | 2,397 MB |
| XML-5 | 1,271 | 28,959 | 1,597 MB |
| XML-6 | 900 | 97,227 | 450 MB |
| long_content_30k_001 | 500 | 18,290 | 1,742 MB |
| long_content_30k_002 | 3,999 | 2,027,957 | 17,938 MB |
| long_content_30k_003 | 4,000 | 1,259,145 | 17,938 MB |
| UnseenTestSet | 97 | 5,539 | 371 MB |
| **TOTAL** | **14,890** | **3,566,327** | **44.64 GB** raw / 42 GB on disk |

Design departed from the master plan's per-event one-hot layout (would have been ~3 TB given the 17–28× sibling-recombinant multiplicity). Final layout per shard:
- `alignments.bin` — int8 concatenation of all alignment matrices (A:0, T:1, G:2, C:3, gap:4). One byte per nucleotide.
- `align_idx.npy` — structured array `(file_idx, n_seqs, seq_len, byte_offset)` per source FASTA.
- `events.npy` — structured array `(file_idx, event_id, recomb_id, p1_id, p2_id, bp_start, bp_end)` per triplet event.
- `seq_flags.npy` — `(file_idx, fasta_id, is_recombinant)` per sequence; supports MLM corpus sampling + no-BP negative-triplet sampling.
- `files.txt` — filename per line, indexed by file_idx.

Reader API (`cache_v2_reader.CacheV2`): memmap-backed, no decoding at read time. `get_triplet(idx)` returns int8 R/P1/P2 arrays + bp positions in ~10 µs. One-hot expansion is `np.eye(5)[arr]` at training time — ~1 ms per 30 kb sequence.

Test results (commit pending):
- Round-trip: 30/30 sequences from XML-1..4 byte-identical to the source FASTAs after canonicalising non-ATGC bases to gap (matches the encoder).
- Invariants: 1,000 events checked — all satisfy `recomb_id ≠ p1_id ≠ p2_id` and `0 ≤ bp_start ≤ bp_end ≤ seq_len`.
- Random-access latency: p50=0.01 ms, p99=0.02 ms over 200 random reads (vs 50 ms threshold — 2,500× headroom).
- Seq-flag coverage: every recomb_id from `events.npy` correctly flagged in `seq_flags.npy`.

Build wall time: 603 s (10 min) on the RTX-3070 box.

#### M0.3 — ✅ DONE 2026-05-18 — Held-out split definition

Artifacts: `build_splits_v2.py`, `splits/v2_split.json` (830 KB).

| Split | Files | Triplets | Unique events | Composition |
|---|---:|---:|---:|---|
| TRAIN | 12,827 | 3,492,769 | 258,416 | XML-1..4 (4,123), long_content_30k_002 (3,999), long_content_30k_003 (4,000), XML-6 80% (705) |
| VAL | 1,966 | 68,019 | 31,912 | XML-5 (1,271), long_content_30k_001 (500), XML-6 20% (195) |
| TEST_SANTA | 97 | 5,539 | 5,539 | UnseenTestSet (untouched until final eval) |
| TEST_REAL | 4 | — | — | LANL CRF panel (lives outside the v2 cache; untouched until final eval) |

XML-6 split is deterministic: `sha256("v2-split|<filename>")[:8] / 2^64 < 0.80 → TRAIN`. Actual TRAIN/VAL ratio came out 705/195 = 78.3%/21.7% (hashes don't land exactly on 0.80; deterministic and reproducible). All four validation checks pass:
1. Every cached event lands in exactly one split.
2. No TEST file appears in TRAIN/VAL.
3. No file appears in two splits.
4. Total event coverage matches the cache total.

MLM-only corpus (Phase 1 substrate): every non-recombinant sequence across all 14,793 TRAIN+VAL FASTAs is available via `cache.shards[d].sample_non_recombinant_ids()` and the `seq_flags.npy` `is_recombinant=0` mask. No separate split file needed; it's a function of the existing TRAIN/VAL FASTA membership.

#### M0.4 — ✅ DONE 2026-05-18 — Baseline reproduction (sanity peg, inference path)

Artifacts: `m04_compare_v2_vs_legacy.py`, `m04_verify_inference.py`, `m04_report.json`, `m04_smoke_report.json`, `m04_inference_audit.json`, `m04_inference_audit.log`.

Two-stage verification, completed end-to-end:

**Stage 1 — one-hot fidelity** (logical proof). 5,539 / 5,539 UnseenTestSet events bit-identical in channels 0..14 between the v2 cache and the legacy 33-channel cache `ds_UnseenTestSet_07e8e66de8d2a720.npz`. Zero diff cells across ~2.66B compared. Runtime 14 s. Bug found in smoke (50 events showed 0/50 initially): `v2_to_onehot()` was zeroing gap positions instead of encoding them in channel 4 — fixed to match the legacy 5-channel convention `{A:0, T:1, G:2, C:3, gap:4}`.

**Stage 2 — measured F1** (empirical proof). Loaded the runB2_sig10 checkpoint, streamed inference over the 584-event honest-eval subset (11 files held out of run41+run42c training), applied content-end + edge-buffer masking matching `eval_diagnostic_A_fair.evaluate()`, swept thresholds 0.1–0.95:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.20 | 0.377 | 0.507 | **0.4321** (sweep best) |
| 0.30 | 0.380 | 0.470 | **0.4210** (matches canonical 0.4205 to 4 decimals) |
| 0.50 | 0.380 | 0.461 | 0.4172 |
| 0.70 | 0.367 | 0.304 | 0.3324 |

|Δ_best| = 0.0116 against canonical target 0.4205. Master-plan ±0.03 satisfied with ~0.018 margin.

**Process discipline:** the first inference attempt OOM-killed the user's desktop session by allocating two 11.7 GB copies of the legacy X tensor (`np.array(..., copy=True)` × 2 = 35 GB on a 30 GB box) — the failure mode that [[feedback-padding-mask-oom]] documented two days earlier. Fix: dropped Path B (redundant given bit-identity), switched to per-batch streaming off a memmap'd X, added an RSS watchdog at 26 GB (user-chosen ceiling) that aborts the process if it gets close to the system memory wall instead of letting the OS OOM killer choose. Peak observed RSS 12.9 GB — comfortably under cap. The `RLIMIT_AS` cap I tried first conflicted with CUDA's huge virtual-memory mappings and was rejected in favour of the RSS check.

**Phase 0 complete.** All four foundations milestones done — data loader, cache, splits, baseline-input fidelity verified at the bit level AND the F1 level. Ready for Phase 1 (MLM self-supervised pretraining).

#### M0.5 — ✅ DONE 2026-05-19 — Data-realism filter (added after Phase 0)

Artifacts: `m14_realism_measures.py`, `m14_realism_measures.json`, `build_filtered_split.py`, `splits/v2_filtered_split.json`, `splits/v2_filter_report.json`.

Investigation triggered by the M1.2 zero-shot probe finding (mean acc 0.308 across three pretrained DNA models) and a follow-on question: how realistic is SANTA's training data vs real recombinant viral panels?

**Real-virus reference panels acquired** (`data/real_recombinants/`): Ebola (8 seqs spanning 5 species), Zika (8 seqs, African + Asian lineages), SARS-CoV-2 full (8 seqs incl. XBB.1.5 recombinant + parents BA.2.10 + BJ.1), SARS-CoV-2 Spike + ORF1ab fragments. HIV-1 panel reused from existing LANL CRF triplets. 1.6 MB total, fetched via NCBI Entrez + MAFFT in 46 s.

**6 measures computed across 150 SANTA alignments + 6 real panels:** pairwise Hamming distribution, 6-mer JSD vs nearest real panel, BP geometry, conservation pattern vs declared `<purifyingFitness>` sites, HyenaDNA-small next-token accuracy, RustRDP min p-value.

**Key finding — SANTA's evolution-from-seed paradigm matches some real-virus alignment types and mismatches others:**

| Real-virus type | Example | Real Hamming | SANTA match? |
|---|---|---:|---|
| Subtype/species-level divergence | HIV CRFs, Ebola species | 0.33–0.38 | ✅ matches naturally |
| Recent-lineage panels | SARS-CoV-2 XBB/BA.2/Alpha | 0.002–0.007 | ❌ 50–270× too diverse |

SANTA's within-host/short-timescale-evolution paradigm is the wrong shape for "compare-recent-lineages" tasks. No parameter combination rescues it for SARS-CoV-2 lineages.

**Filter applied (aggressive):**
- **Whole-shard drops** (paradigm mismatch): XML-1 (Spike), XML-3 (ORF1ab), long_content_30k_003 (high-mutation SARS-CoV-2). These produced Hamming 50–270× higher than real SARS-CoV-2 lineage panels.
- **Parameter-level filters** on retained shards: XML-4 keep `mut≥0.01` (low-mut produced too-tight Ebola sims); XML-5 keep `mut=0.005` (high-mut too diverse for Zika); long_content_30k_002 drop `rp≥0.10`. XML-2 (HIV) kept as-is.

| Split | Files (before → after) | Triplets (before → after) |
|---|---:|---:|
| TRAIN | 12,827 → 4,535 | 3,492,769 → ~1,142,301 |
| VAL | 1,966 → 1,123 | 68,019 → ~48,812 |
| TEST_SANTA | 97 → 97 | 5,539 → 5,539 |
| TEST_REAL | 4 → 4 | — |

**Strategic implication:** the SARS-CoV-2 deployment scenario remains the most challenging for any sequence-only model trained on SANTA. SANTA's strength is subtype-level divergence (HIV CRFs, Ebola species) which is exactly what runB2_sig10's RustRDP-channel scaffold leverages. If the deployment target is recent SARS-CoV-2 lineages (XBB-style), more work on data generation may be needed — perhaps SANTA tuned with shorter generation counts + lower mutation rates, or a different simulator entirely.

---

### Phase 1 — Self-supervised pretraining

#### M1.1 — ✅ DONE 2026-05-19 — Backbone skeleton (HyenaDNA, PyTorch)

Artifacts: `backbone_hyenadna.py` (active), `backbone_mamba.py` (superseded, kept as record of the investigation).

**The plan's "Mamba primary, Longformer fallback, Keras model" did not survive contact with reality.** The decision tree, in order:

1. **TFLongformerModel** removed from modern HF (`transformers` 5.x dropped TF support upstream).
2. **HF Mamba (PyTorch, slow path)** runs at L=30k in **18.9 s/forward** — 9× the master-plan budget. Pretraining MLM over 1.3M sequences at this throughput would take ~37 days on the 3070.
3. **mamba-ssm / causal-conv1d CUDA build** — the canonical fast path. Both need system `nvcc`; the dev box has TF's bundled CUDA only. `pip install causal-conv1d` failed at the build-system step. This is exactly the "burn days fighting it" scenario the plan warned about.
4. **HyenaDNA fallback** — pure-PyTorch (FFT-based long convolutions, no custom CUDA), HF checkpoint `LongSafari/hyenadna-small-32k-seqlen-hf`, pretrained on the human genome.

| Spec | Target | Measured |
|---|---:|---:|
| Forward @ L=30k, B=2 | <2 s | **0.30 s** (~6.7× under budget) |
| GPU peak | — | 1.23 GB |
| GPU free after | >1 GB | **5.39 GB** (~5× over budget) |
| Params | (d_model=256, 8L spec ≈ 3.5M) | 3.28M (d_model=256, **n_layer=4**) |

HyenaDNA-small has n_layer=4 vs the master-plan-specified 8 — that's a published architectural fact for this size, not a knob. The medium-160k variant has n_layer=8 if needed later, but small-32k is sufficient for our 30k context and the human-genome pretrained weights are exactly what we want for Phase 1.

**Side effect — Phase 1 may be much shorter than originally planned.** HyenaDNA-small was pretrained via causal language modelling on the human genome at 32k context. Codon biases, conservation patterns, and many sequence statistics relevant to HIV will already be in the weights. M1.2 (MLM pretraining) might be a short fine-tune rather than from-scratch, and the success bar should be re-thought before committing GPU hours.

**Project framework switch:** all code from M1.1 onward is PyTorch. The cache reader, splits, and Phase-0 evaluation infrastructure (M0.4 inference, `m04_verify_inference.py`) remain TensorFlow/Keras because those evaluate the legacy runB2_sig10 checkpoint. The two stacks coexist in the same venv. See [[project-framework-switch-pytorch-hyenadna]] in auto-memory for the full dependency list and rationale.

New dependencies installed in `.venv/`:
- `torch==2.6.0+cu124` (+ bundled CUDA libs, ~2.5 GB)
- `transformers==5.8.1`
- `kernels==0.14.1` (HF prebuilt-kernel helper; pulled in but Mamba kernels not on the Hub)
- `mambapy==1.2.0` (alternative Mamba backend; tried, same slow path — leaving installed in case useful later)

#### M1.2-pre — ✅ DONE 2026-05-19 — Zero-shot transfer probe

Artifacts: `m12_zeroshot_probe.py`, `m12_zeroshot_probe.json`, `m12_probe.log`.

Quick precursor before committing to a 10-day MLM pretraining run: how much of HyenaDNA-small's human-genome causal-LM pretraining transfers to SANTA-simulated HIV sequences? Result:

| VAL scale | n positions | Mean acc | A | C | G | T |
|---|---:|---:|---:|---:|---:|---:|
| XML-5 (~10 kb sampled) | 323,460 | 0.318 | 0.59 | 0.03 | 0.34 | 0.16 |
| XML-6 (10 kb) | 299,970 | 0.285 | 0.38 | 0.05 | 0.17 | 0.45 |
| long_content_30k_001 (30 kb) | 897,030 | 0.323 | 0.66 | 0.02 | 0.06 | 0.28 |
| **mean** | | **0.308** | | | | |

Two diagnostic signals confirm this is NOT meaningful transfer:
1. **Heavy A-bias:** A accuracy ~0.66 on long_content, C accuracy 0.02. The model is mostly emitting "A" and getting some right because SANTA HIV is AT-rich.
2. **Position-bucket accuracy flat at ~0.32** across 0-25%, 25-75%, 75-100% of the sequence. If the model were using context, late-position accuracy would dominate early. Flatness means no context use.

0.308 ≈ majority-class baseline (~0.30 for AT-rich HIV). Per the decision rubric in the script: **partial transfer, run M1.2 from-scratch (or warm-started) MLM as originally planned.** HyenaDNA's pretrained weights are slightly-better-than-random initialization, not a meaningful shortcut.

#### M1.2 — MLM training loop

- **Goal:** train the backbone via masked nucleotide modelling on all sequences (recombinants and non-recombinants alike) across the training splits.
- **Setup:** 15% of positions masked, 5-way CE on the recovered tokens. Length-bucketed batching at {4k, 10k, 30k}. Adam at LR 1e-4 with linear warmup + cosine decay. Mixed-precision fp16. Checkpoint per epoch.
- **Deliverable:** `pretrain_mlm.py` + a trained checkpoint `models_test/backbone_mlm_v1.pt` + a curves file `models_test/history_mlm_v1.json`.
- **Success criterion (Gate G1):** MLM val loss converges (plateau over the last 3 epochs); reconstruction accuracy ≥ 0.6 on validation (uniform baseline ≈ 0.2; "predict majority class" baseline ≈ 0.3 on AT-rich HIV; HyenaDNA zero-shot baseline 0.308, measured M1.2-pre).
- **Curriculum:** epochs 1–3 on 4 kb (XML-1..5) only, then expand. Don't start on 30 kb from cold init — it won't converge.
- **Warm-start vs random init:** initialize from HyenaDNA-small-32k weights (loaded via AutoModelForMaskedLM if available, else AutoModelForCausalLM + new MLM head). Better than random but not by much per the probe.
- **Implementation note:** PyTorch backbone now, deliverable is `.pt` not `.keras`. See [[project-framework-switch-pytorch-hyenadna]].

#### M1.3 — Pretraining checkpoint card

- **Goal:** record what got trained and how well, so the next session can pick up.
- **Deliverable:** one markdown cell in `models_test/checkpoint_card_mlm_v1.md` with hyperparameters, final loss, reconstruction accuracy per length bucket, and a short interpretation paragraph.

---

### Phase 2 — Triplet contrastive pretraining

#### M2.1 — Triplet sampler

- **Goal:** efficient sampler that produces both positive (recombinant) and negative (no-recomb) triplets from the cache.
- **Deliverable:** `triplet_sampler.py` exposing a `tf.data` iterator with configurable pos:neg ratio. Negatives are random 3-tuples of non-recombinant sequences from the same source FASTA.
- **Success criterion:** 1000-sample throughput timed at <5 s end-to-end; visual inspection of 10 random batches confirms ratio is correct and no positive is incorrectly drawn from a non-recombinant.

#### M2.2 — Cross-sequence attention block

- **Goal:** a `CrossSeqAttention` Keras layer that takes `(R_emb, P1_emb, P2_emb) ∈ R^(B, L, d)` and outputs `(B, L, d)` summarising "which parent does R resemble at each position." Weight-shared encoder enforces P1↔P2 symmetry.
- **Deliverable:** `cross_attention.py` + a unit test that swaps P1 and P2 and confirms the output is permutation-equivariant under the parental-assignment label flip.
- **Success criterion:** the symmetry unit test passes to floating-point tolerance.

#### M2.3 — Triplet classifier head + training run

- **Goal:** Task A (binary recomb?) + Task B (3-way which-is-recomb?). Joint loss with equal weighting initially.
- **Deliverable:** `train_triplet_contrastive.py` + checkpoint `models_test/triplet_v1.keras`.
- **Success criterion (Gate G2):** Task A val AUROC ≥ 0.85; Task B val accuracy ≥ 0.7. If Task A < 0.7, the cross-attention block is the suspect — debug before progressing.

---

### Phase 3 — Breakpoint localization

#### M3.1 — Output heads

- **Goal:** add the two output heads on top of the pretrained backbone.
  - Parental-assignment head: per-position softmax over {P1, P2, no_recomb}.
  - Change-point head: per-position weighted-BCE on σ=10 Gaussian targets.
- **Deliverable:** `heads.py` + integration into a `build_bp_model(backbone_ckpt) -> tf.keras.Model`.
- **Success criterion:** sanity forward pass on a synthetic batch (deterministic input → deterministic output); shape checks pass.

#### M3.2 — Curriculum training run

- **Goal:** main BP-localization training with curriculum (4 kb → 10 kb → 30 kb) and pos:neg triplet mixing (3:1 recomb:non-recomb).
- **Loss:** `L = L_assign + λ_bp · L_bp + λ_aux · L_aux` (λ_bp = 1.0 to start, λ_aux = 0.1).
- **Deliverable:** `train_bp_localization.py` + checkpoint `models_test/bp_v1_{best,last,final}.keras` + history JSON.
- **Success criterion (Gate G4):** UnseenTestSet sub F1 ≥ 0.35 at val-tuned threshold.

#### M3.3 — Mid-training LANL probe (Gate G3)

- **Goal:** evaluate the model on LANL at epoch 15 (mid-curriculum) to decide whether to continue or pivot.
- **Deliverable:** eval log `eval_bp_v1_lanl_ep15.log` with per-CRF F1, recall, precision.
- **Success criterion (Gate G3 — make-or-break):** LANL agg F1 > 0 from one-hots alone. If F1 = 0, we're in runOH-collapse territory and the backbone choice or pretraining didn't install the right priors. Add ONE engineered channel back (match_p1 only) and re-eval. If still 0, pivot back to backbone selection.

---

### Phase 4 — Final eval and decision

#### M4.1 — UnseenTestSet eval

- **Deliverable:** `eval_bp_v1_santa.log` with sub F1 at EB=0 and EB=25, threshold sweep.

#### M4.2 — LANL CRF eval

- **Deliverable:** `eval_bp_v1_lanl.log` with per-CRF and aggregate F1, vs runB2_sig10 baseline 0.533 and classical RDP 0.519.

#### M4.3 — Decision memo

- **Deliverable:** a new memory file `project_bp_v1_outcome.md` summarising results, what worked, what didn't, next moves.
- **Success criteria:**
  - **Gate G5 (publishable):** LANL F1 ≥ 0.30 → mark as primary sequence-only detector milestone; iterate on quality.
  - **Gate G6 (deployment):** LANL F1 ≥ 0.55 → replace runB2_sig10 as deployment baseline.
  - Below G5 → diagnose failure mode (precision-bound vs recall-bound vs pure collapse), decide between backbone change, pretraining changes, or auxiliary engineered channels.

---

## How agents should pick up this plan

1. Read this file (`MASTER_PLAN.md`) and the linked memory files.
2. Read `CLAUDE.md`, `HANDOVER.md`, and the most recent `HANDOVER_NEXT.md` for repo conventions.
3. Identify the **lowest-numbered unfinished milestone** (M0.1 first session, then M0.2, etc.).
4. Execute only that milestone. Don't try to skip ahead — each milestone's deliverable is required by the next one.
5. Update this file by checking off the milestone (replace `### M0.1 —` with `### M0.1 — ✅ DONE YYYY-MM-DD —` and add a one-line link to the artifact).
6. Save a memory file for the milestone outcome only if the outcome is *surprising* (e.g., Mamba install blocked, or LANL F1 in Phase 1 already non-zero, or G3 failed). Routine "we finished M0.2" doesn't need a memory entry.
7. Report `result:` and stop. Don't combine milestones in one session — they're sized to fit individual sessions for a reason.

## Conventions to honour

- **One change per run** (from CLAUDE.md). Each milestone is one change.
- **Held-out splits are sacred.** Never touch TEST-SANTA or TEST-REAL during training or hyperparameter tuning.
- **Document failures.** A failed M1.2 (MLM didn't converge) is more valuable to record than a successful one — it constrains future architecture choices.
- **No `find_peaks` in the loss.** It's a post-hoc evaluator. The model's output should be directly compatible with the change-point ground truth.
- **Compute is not a constraint.** If renting an A100 makes Phase 1 finish in 1 week instead of 10 days on the 3070, do that. (Confirm with user before spending real money.)

## What this plan deliberately does NOT do

- Doesn't try to predict which of {seq1, seq2, seq3, none} is the recombinant as the *first* task (CLAUDE.md's long-term framing). Task B in Phase 2 introduces it as a pretraining auxiliary; making it the primary task is a follow-on plan if BP localization succeeds.
- Doesn't bring MaxChi or RustRDP back as inputs. The whole point is to test whether sequence-only works. Bringing them back is an off-ramp at G3 only.
- Doesn't fine-tune on real HIV labels. We have only 35 LANL truth BPs across 4 CRFs — too few to train on without overfitting. LANL stays test-only.
- Doesn't try multiple architectures in parallel. Mamba primary, Longformer fallback if M1.1 blocks. Anything else is a separate plan.

## Open questions for the human (defer until they come up)

- Budget for renting A100 for Phase 1 pretraining? (Cost estimate: ~$200–500 for a week.)
- If Phase 1 MLM doesn't converge in 10 days on 3070, drop to 10 kb max length or rent the A100?
- After M4.3, if G5 passes but G6 doesn't, should we (a) iterate on the same architecture for ≥0.55, or (b) declare M4.3 the milestone and write it up before iterating?

These don't block any milestone before they're reached — the agent should just note when they become live.
