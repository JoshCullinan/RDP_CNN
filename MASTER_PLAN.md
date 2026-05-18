# Master plan — sequence-only backbone replacement

**Status:** active, written 2026-05-18.
**Author:** strategic synthesis of `project_backbone_replacement_direction.md` and `project_backbone_replacement_training_plan.md` in auto-memory.
**Audience:** future agents (Claude or otherwise) and the human collaborator.

---

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
| XML-6 | 0 | 0 | — (no parent CSV) |
| long_content_30k_001 | 18,290 | 1,059 | 17.3 |
| long_content_30k_002 | 2,027,957 | 77,707 | 26.1 |
| long_content_30k_003 | 1,259,145 | 44,409 | 28.4 |
| UnseenTestSet | 5,539 | 5,539 | 1.0 |
| **TOTAL** | **3,469,100** | **286,883** | — |

Key findings that change downstream milestones:
- Unique-event total 286,883 is ~9% below the prior estimate (315k); triplet-level total is 11× higher.
- **XML-6 confirmed unusable for BP supervision** (no `.faParents.csv` or `.faRecombIdentifyStats.csv` on disk — SANTA-only dump, RDP5 never ran). Sequences still usable for MLM pretraining (Phase 1); cannot supply parent IDs for triplet training. The 80/20 XML-6 split in M0.3 should be reframed as "XML-6 → MLM-only corpus."
- **Long-content sibling-recombinant multiplicity** (17–28×) means each unique BP configuration appears as many correlated training rows. M0.2 caches all rows so downstream code can choose; M3 should dedupe-by-event or downweight within-event correlation in batches to avoid effective-batch-size collapse.

#### M0.2 — Sharded cache builder

- **Goal:** one-time conversion of all loadable events into on-disk shards for fast training-time access.
- **Format:** one shard per source dir, layout `(R_one_hot, P1_one_hot, P2_one_hot, parental_assignment_per_position, bp_position_list, original_len)` per event. fp8 or fp16 for one-hots (5-channel: A/T/G/C/-). Use HDF5 or `np.savez` with memmap.
- **Expected size:** ~47 GB on disk total.
- **Deliverable:** `build_cache_v2.py` + the actual shards under `cache/v2/{source_dir}/*.npz` (or `.h5`).
- **Success criterion:** cache size on disk within 2× of estimate, random-access per-event read latency < 50 ms, and a round-trip test (cache → decode → re-encode) matches the original one-hot exactly for ≥10 randomly sampled events.

#### M0.3 — Held-out split definition

- **Goal:** lock the train/val/test split before any model touches the data, to prevent leakage.
- **Splits (revised after M0.1 — XML-6 has no parent CSV, dropped from BP-supervised splits):**
  - TRAIN: XML-1..4 (~129k events), long_content_30k_002 (~78k events), long_content_30k_003 (~44k events)
  - VAL: XML-5 (~29k events), long_content_30k_001 (~1k events)
  - MLM-only corpus (Phase 1 substrate, not for BP supervision): XML-6 plus all non-recombinant sequences from every other directory's FASTAs
  - TEST-SANTA: UnseenTestSet (touched only for final eval)
  - TEST-REAL: LANL CRF panel (touched only for final eval)
- **Deliverable:** `splits/v2_split.json` listing exact file paths per split + an event count summary.
- **Success criterion:** every event in the cache is in exactly one split; XML-6 train/val files are deterministically chosen (sorted then hashed for a stable seed); TEST-SANTA and TEST-REAL files are never referenced from TRAIN or VAL.

#### M0.4 — Baseline reproduction (sanity peg)

- **Goal:** train runB2_sig10's architecture on the new cache+splits to confirm we can reproduce 0.421 SANTA / 0.533 LANL before changing anything.
- **Deliverable:** `models_test/cnn_breakpoint_reproB2_sig10_final.keras` + eval log on UnseenTestSet and LANL.
- **Success criterion:** SANTA sub F1 within ±0.03 of 0.421; LANL agg F1 within ±0.03 of 0.533. If the new infrastructure can't reproduce, fix infrastructure before touching the architecture.

---

### Phase 1 — Self-supervised pretraining

#### M1.1 — Backbone skeleton

- **Goal:** an untrained Mamba (primary) or Longformer (fallback) backbone wrapped as a Keras model with a per-position embedding output `(L, d_model)`.
- **Deliverable:** `backbone_mamba.py` with a `build_backbone(seq_len, d_model=256, n_layers=8) -> tf.keras.Model` and a 30-line sanity test that runs a random `(B, L, 5)` one-hot through it and checks the output shape.
- **Success criterion:** model builds without error on L=30000, forward pass on a batch of 2 at 30 kb completes in <2 s on the RTX 3070, memory headroom > 1 GB.
- **Notes:** if Mamba install is painful on the current env, document the blocker and fall back to Longformer immediately rather than burn days fighting it.

#### M1.2 — MLM training loop

- **Goal:** train the backbone via masked nucleotide modelling on all sequences (recombinants and non-recombinants alike) across the training splits.
- **Setup:** 15% of positions masked, 5-way CE on the recovered tokens. Length-bucketed batching at {4k, 10k, 30k}. Adam at LR 1e-4 with linear warmup + cosine decay. Mixed-precision fp16. Checkpoint per epoch.
- **Deliverable:** `pretrain_mlm.py` + a trained checkpoint `models_test/backbone_mlm_v1.keras` + a curves file `models_test/history_mlm_v1.json`.
- **Success criterion (Gate G1):** MLM val loss converges (plateau over the last 3 epochs); reconstruction accuracy ≥ 0.6 on validation (uniform baseline ≈ 0.2; "predict majority class" baseline ≈ 0.3 on AT-rich HIV).
- **Curriculum:** epochs 1–3 on 4 kb (XML-1..5) only, then expand. Don't start on 30 kb from cold init — it won't converge.

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
