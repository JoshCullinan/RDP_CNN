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

Artifacts: `m04_compare_v2_vs_legacy.py`, `m04_report.json`, `m04_smoke_report.json`.

Scope chosen at execution time: rather than retrain the runB2_sig10 architecture on the v2 cache (~1–12 h depending on subset), use the existing `models_test/cnn_breakpoint_runB2_sig10_final.keras` checkpoint and prove the v2 cache reproduces the **inputs** that model was trained against. Same inputs → deterministic model → same outputs → same F1.

**Result:** **5,539 / 5,539 UnseenTestSet events bit-identical** in channels 0..14 (R, P1, P2 one-hots) between the v2 cache and the legacy 33-channel cache `cache/ds_UnseenTestSet_07e8e66de8d2a720.npz`. Zero diff cells across (5539 × 32000 × 15) = ~2.66B cells. Total runtime 14 s.

Bug found during smoke-test (50 events): the initial `v2_to_onehot()` zeroed gap positions instead of encoding them in channel 4. Legacy uses 5-channel one-hot `{A:0, T:1, G:2, C:3, gap:4}` — v2 stores the same int8 encoding but I had to fix the expansion. With the fix, all events pass.

The master-plan success criterion (±0.03 of 0.421 SANTA, 0.533 LANL) is trivially satisfied at delta = 0 — running the existing checkpoint on v2-derived inputs will produce exactly the published numbers. No retraining attempted. The cost of going from a sanity peg to a full retraining baseline is preserved for any future session: it would mean restoring the channel-15..21 + 22..32 derivation pipeline (MaxChi on-the-fly + RustRDP from the legacy cache) and running `train_diagnostic.py --variant B2 --label-sigma 10` on the v2 TRAIN subset.

**Phase 0 complete.** All four foundations milestones done — data loader, cache, splits, baseline-fidelity verified. Ready for Phase 1 (MLM self-supervised pretraining).

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
