# Handover — RDP_CNN session, May 2026

You are picking up the multi-virus recombination breakpoint detection project mid-flight. Read this whole document before doing anything. Then read `CLAUDE.md` and `MASTER_PLAN.md`. Then read the relevant auto-memory files (paths below).

---

## TL;DR — where the project is right now

**Phase 0** (data foundations): ✅ complete, merged to `main`.
**Phase 0.5** (data realism filter, added mid-session): ✅ complete.
**M1.1** (backbone skeleton): ✅ complete — HyenaDNA-small-32k chosen.
**M1.2-pre** (zero-shot probes on 3 pretrained DNA models): ✅ complete — no pretraining shortcut.
**M1.2 actual** (MLM training loop): ⬜ **next milestone**.
**M1.3 → M3, M4**: pending.

Local `main` is 23 commits ahead of `origin/main`. Not pushed. The user is cautious about destructive remote actions and explicitly approves pushes when ready.

---

## The actual goal — re-anchoring (this matters)

**Build a sequence-only detector for viral recombination breakpoints that works on ANY recombination-capable virus, not just HIV.**

The current deployment baseline (`runB2_sig10`, LANL agg F1 0.533) is functionally a refinement layer over RustRDP outputs — its sequence-only ablation (runOH) scores F1=0.000 on real HIV. The whole point of the backbone replacement is to build something that works from raw nucleotide sequences alone, transferring to multiple virus families.

**Critical constraint:** the user does NOT want to focus exclusively on HIV. SANTA simulates 5 different viruses across the XMLs (HIV, Ebola, Zika, two SARS-CoV-2 fragments + full). The deployment scenario is multi-virus.

---

## Read these first, in this order

1. `CLAUDE.md` — project conventions, key cells (legacy notebook), iteration discipline.
2. `MASTER_PLAN.md` — the active plan with M0.1–M0.5 ✅ ticked off, M1.1 ✅, M1.2 next. Updated with the realism-filter findings.
3. Auto-memory (at `/home/joshc/.claude/projects/-home-joshc-Dev-RDP-CNN/memory/`):
   - `MEMORY.md` — index of all memories.
   - `project_framework_switch_pytorch_hyenadna.md` — PyTorch + HyenaDNA decision, dependency list, why Mamba/Longformer didn't work.
   - `project_hyenadna_zeroshot_probe.md` — probe finding 0.308 ≈ majority-class baseline.
   - `project_santa_realism_filter.md` — **read this carefully**. The paradigm-mismatch finding shapes everything downstream.
   - `project_backbone_replacement_direction.md` and `project_backbone_replacement_training_plan.md` — original strategy.
   - `feedback_padding_mask_oom.md` — **the OOM rule that nearly killed the user's desktop session twice**. Never violate it.

---

## What's already been built

### Phase 0 (data foundations, all Keras/TF)
- **M0.1** `data_loader_v2.py` — schema-unified loader for SANTA outputs (3 schemas: legacy XML-1..5, XML-6 needs `pick_parents_rdp5ml.py` first, long_content_30k_* uses `.faParents.csv`).
- **M0.2** `build_cache_v2.py` + `cache_v2_reader.py` — int8 sharded memmap cache at `cache/v2/` (42 GB, gitignored). `CacheV2.shards[name].get_alignment(file_idx)` returns int8 array. **Discovers cache root as sibling of `dataRaw/`** so it works from any worktree.
- **M0.3** `build_splits_v2.py` + `splits/v2_split.json` — held-out splits at the file level (subtype-level integrity).
- **M0.4** `m04_compare_v2_vs_legacy.py` + `m04_verify_inference.py` — verified v2 cache bit-identical to legacy 33-channel cache for UnseenTestSet (5539/5539 events). Then actually ran runB2_sig10 and measured F1 = 0.421 at thr=0.30 (canonical target 0.4205, Δ=0.0005). Master-plan ±0.03 satisfied.

### M0.5 — Data realism filter (added mid-session)
- Acquired 5 real-virus reference panels from NCBI: Ebola (8 spp), Zika (8), SARS-CoV-2 full (8 incl. XBB.1.5), SARS-CoV-2 Spike fragment, SARS-CoV-2 ORF1ab fragment. At `data/real_recombinants/`. HIV LANL CRFs already in `data/lanl_crf/triplets/`.
- Computed 6 realism measures (`m14_realism_measures.json`) across 150 SANTA + 6 real panels.
- **Finding:** SANTA's evolution-from-seed paradigm produces high within-alignment diversity (Hamming 0.15–0.45). This matches subtype/species-level real panels (HIV CRFs at 0.376, Ebola species at 0.329) but does NOT match recent-lineage panels (SARS-CoV-2 XBB-style at 0.002–0.007) — off by 50–270×.
- **Applied filter** (`build_filtered_split.py` → `splits/v2_filtered_split.json`):
  - Whole-shard drops: XML-1 (Spike), XML-3 (ORF1ab), long_content_30k_003 (mut-rate too high)
  - Per-combo trims: XML-4 keep mut≥0.01, XML-5 keep mut=0.005, long_content_30k_002 drop rp≥0.10
  - XML-2 (HIV) kept as-is — best match to real CRFs
- Final training set: TRAIN 4,535 files / ~1.14M triplets (down from 12,827 / 3.49M). VAL 1,123 files. **Future training defaults to `splits/v2_filtered_split.json`.**

### M1.1 — Backbone skeleton (PyTorch, HyenaDNA-small-32k)
- `backbone_hyenadna.py` — `build_backbone()` + `SequenceBackbone` wrapper + v2-cache→HyenaDNA vocab mapper.
- `backbone_mamba.py` — kept as record of the Mamba investigation. **Don't try to revive Mamba unless system `nvcc` becomes available** — `mamba-ssm` and `causal-conv1d` both need CUDA-toolkit at install time.
- Smoke: forward at L=30k, B=2 in 0.30s, 1.23 GB peak GPU, 5.39 GB free. Master-plan spec was <2s + >1GB free — passes by 4–6×.

### M1.2-pre — Zero-shot probes (no shortcut found)
- `m12_zeroshot_probe.py` — HyenaDNA-small probe on SANTA VAL.
- `m12_probe_hyena_medium.py` — HyenaDNA-medium probe.
- `m12_probe_nt100.py` — Nucleotide Transformer v2-100M probe (uses a side venv `/tmp/nt_venv` to avoid transformers 5.x compatibility break with NT).
- **Result:** all three models hit ~0.31 next-token-accuracy ceiling on SANTA, regardless of size, pretraining corpus (human-only vs 850 species), tokenization (char vs 6-mer), or objective (causal LM vs MLM). The ceiling IS the AT/GC composition floor of SANTA viruses.
- **Decision:** M1.2 still needs from-scratch MLM training. Warm-start from HyenaDNA helps marginally, not as a shortcut. The 0.308 floor is the measured baseline that M1.2's Gate G1 target (≥0.6) must beat.

---

## Environment and constraints

### Hardware
- Linux Ubuntu 26.04, RTX 3070 (8 GB VRAM, compute 8.6, driver 595.x)
- 30 GB RAM, 8 GB swap
- **No GPU rental authorized yet.** User explicitly anti-rental "until we have a very real chance of success" — i.e., until M3 shows non-zero LANL F1.

### Software
- Python 3.12 in `.venv/`. Activate via `source /home/joshc/Dev/RDP_CNN/.venv/bin/activate`.
- TensorFlow 2.18 (Keras) — for Phase 0 + legacy eval scripts.
- **PyTorch 2.6.0+cu124 (new for Phase 1+).** Plus `transformers==5.8.1`, `kernels==0.14.1`, `mambapy==1.2.0`. Both frameworks coexist in the same venv.
- BioPython 1.87, scipy.
- **No system `nvcc`** — TF uses bundled CUDA wheels. This is why `mamba-ssm` / `causal-conv1d` builds fail.

### The OOM rule — non-negotiable
- The user's desktop session has been OOM-killed twice in this project, both times by `np.array(X, copy=True)` or `X.astype(np.float32)` on the legacy 11.7 GB X tensor.
- **Never make in-script copies of full multi-GB tensors.** Stream per-batch.
- All new inference/training scripts MUST use the RSS-watchdog pattern from `m12_zeroshot_probe.py`:
  ```python
  import resource
  _RSS_CEILING_BYTES = 26 * 1024 * 1024 * 1024
  def _current_rss_bytes(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
  def _rss_watchdog(label=""):
      if _current_rss_bytes() > _RSS_CEILING_BYTES:
          raise MemoryError(f"RSS watchdog tripped {label}: ...")
  ```
- Do NOT use `resource.setrlimit(RLIMIT_AS, ...)` — it conflicts with CUDA's huge virtual-memory mappings. The watchdog is on RSS, not VM.

### Worktree workflow
- The session runs with `worktree.bgIsolation: worktree` — `Edit`/`Write` on tracked repo paths are blocked from main checkout. **You must enter a worktree before any code change.**
- Pattern: `EnterWorktree(name=...)` → `git merge main --no-edit -q` → edit → commit → `ExitWorktree(keep)` → from main checkout: `git merge --no-ff worktree-{name}` → `git worktree remove .claude/worktrees/{name}` → `git branch -d worktree-{name}`.
- The `cache/v2/` directory is at the repo root (alongside `dataRaw/`) and persists across worktrees — DO NOT recreate it inside a worktree.
- `Bash` git commands work without a worktree (no Edit/Write involved).

### Push policy
- Don't push to `origin/main` without explicit user approval. They will ask.
- Local main is ahead by ~23 commits at handover time.

---

## What to NOT do

1. **Don't rerun the realism investigation.** It's done, the filter is committed. If you think you have a reason to redo it, read `project_santa_realism_filter.md` first — the answer probably already addresses your concern.
2. **Don't "rescue" the dropped XML-1 / XML-3 / long_content_30k_003 shards.** The paradigm mismatch isn't a parameter-tuning issue. Filtering by Hamming threshold doesn't help because every alignment in those shards fails. If you want to add SARS-CoV-2 lineage-level training data, that requires a different simulator setup or hand-curated real data — not unfiltering.
3. **Don't try `mamba-ssm` / `causal-conv1d` install on this box.** It will fail (no system nvcc). The investigation is in `backbone_mamba.py` header.
4. **Don't switch back to Keras for the backbone.** PyTorch + HyenaDNA is the committed direction. TFLongformerModel is gone from modern HF; Mamba doesn't have a clean Keras port.
5. **Don't blindly trust subagent reports.** Earlier this session a subagent extrapolated from XML-6 (not part of main training) to conclude SANTA was unrealistic. The correction came only when the user pushed back. Read the actual XMLs at `santaSim_RDP/XMLs/{1..5}.xml` and `santaSim_RDP/Test Set XML/sarbeco_high.xml` if you need to know what SANTA is doing.
6. **Don't commit the user's thesis PDF** (`Detecting Viral Recombination with Machine Learning Thesis - Joshua Cullinan.pdf` in repo root). It's a personal artifact, not project code.
7. **Don't use TaskCreate for trivial single-step work.** The harness keeps nudging. Use it only for genuinely multi-step plans.

---

## Concrete next step — M1.2 MLM training loop

The plan is in `MASTER_PLAN.md` under M1.2. Key parameters:

**Setup:**
- Backbone: HyenaDNA-small-32k (3.3M params), warm-started from `LongSafari/hyenadna-small-32k-seqlen-hf`.
- Training data: `splits/v2_filtered_split.json` TRAIN split (4,535 files / ~1.14M triplets / ~98k unique events).
- Sequences: ALL sequences in those FASTAs (not just recombinants) for MLM.
- Mask 15% of positions, predict masked nucleotides.
- 5-way CE loss over `{A, T, G, C, gap}`.
- Length-bucketed batching at {4k, 10k, 30k}.
- Adam at LR 1e-4, linear warmup + cosine decay.
- Mixed-precision fp16.
- Checkpoint per epoch to `models_test/backbone_mlm_v1.pt`.
- Save curves to `models_test/history_mlm_v1.json`.

**Success criterion (Gate G1):** MLM val loss converges (plateau over last 3 epochs), reconstruction accuracy ≥ 0.6 on validation. Floor baseline: 0.308 (the zero-shot probe). Random baseline: 0.20.

**Curriculum:**
- Epochs 1–5: 4 kb only (XML-2 + XML-4 + XML-5 short subset)
- Epochs 6–15: add 10 kb (XML-5 full, XML-2 full)
- Epochs 16–30: add 30 kb (long_content_30k_001 + _002)
- Don't start on 30 kb from cold init — it won't converge.

**Wall-time estimate:** ~10 days on the 3070. Discuss with user before launching the full run — they may want to validate the loop end-to-end on a small subset first (1 epoch on 4 kb, ~30 min) before committing 10 days.

**Critical infra:**
- All-PyTorch from here. No Keras code in `pretrain_mlm.py`.
- Use `cache_v2_reader.CacheV2` for data access — it returns numpy arrays, framework-agnostic.
- Use `backbone_hyenadna.SequenceBackbone` for the model (read its docstring for input format — int8 v2 ints, mapped internally to HyenaDNA tokens).
- HyenaDNA's HF checkpoint has a `lm_head` that does causal LM. For MLM, you'll need a new head (or repurpose) — see the HF MambaForMaskedLM pattern, then adapt for HyenaDNA's `HyenaDNAModel`.

**Open question to flag to the user before launching:**
HyenaDNA was pretrained causally, not for MLM. The zero-shot probe used causal LM. For M1.2 you have two choices:
- (a) Stick with causal LM as the M1.2 objective — same as HyenaDNA's pretraining. The Gate G1 target of "MLM accuracy ≥ 0.6" doesn't quite apply; you'd compute causal next-token accuracy instead.
- (b) Switch to MLM — train a new MLM head from scratch on top of HyenaDNA's backbone. The probe showed the model has no MLM zero-shot ability, so this is a real from-scratch training of the masking task.

The master-plan spec was MLM. The PRACTICAL choice may be causal LM since it matches the pretraining. **Ask the user before committing.** I'd lean (a) for cheaper training but it's their call.

---

## User's working style — observed patterns

- **Ultra-think requests:** they will explicitly say "ultrathink" when they want depth. Honor it — go to first principles, lay out trade-offs explicitly.
- **Anti-friction:** they have a low tolerance for "ask 4 clarifying questions in a row." Make reasonable defaults clear, ask only for genuinely ambiguous decisions.
- **They challenge prematurely-confident framings.** If your analysis sounds too certain, they will (correctly) push back. Especially: they pushed back hard when I extrapolated from XML-6 to claim SANTA was unrealistic. Read the source artifacts.
- **They want explicit trade-offs surfaced** with concrete numbers ("would cost ~X days vs ~Y days").
- **They use AskUserQuestion well** but don't appreciate excessive scoping questions. Save it for actual fork-in-the-road decisions.
- **They explicitly authorize destructive actions individually.** Push to origin? Get a yes. Drop SANTA data? Get a yes. They appreciate the asking.
- **They appreciate honesty about errors.** When I made the XML-6 mistake, they expected (and got) a substantial correction, not a defensive reframing.

---

## Files of interest if you need to understand the project

### Phase 0 + 0.5 infrastructure (READ ME if confused)
- `MASTER_PLAN.md` — the living plan with completed milestones marked.
- `CLAUDE.md` — project conventions and historical context.
- `data_loader_v2.py`, `build_cache_v2.py`, `cache_v2_reader.py` — data infrastructure.
- `splits/v2_filtered_split.json` — **the canonical training/eval splits**. Don't use `v2_split.json` for new training.
- `build_filtered_split.py` — the filter rules.
- `m14_realism_measures.json` — the data behind the filter decision.
- `data/real_recombinants/` — real viral reference panels per virus.
- `data/lanl_crf/` — real HIV CRFs (already there from prior sessions).

### M1.x backbone work
- `backbone_hyenadna.py` — active backbone.
- `backbone_mamba.py` — superseded, kept as record.
- `m12_zeroshot_probe.py`, `m12_probe_hyena_medium.py`, `m12_probe_nt100.py` — probes.

### Legacy (don't touch unless you know why)
- `CNN.ipynb` — legacy notebook (TF/Keras). Don't open by default; the cache + scripts cover its functionality.
- `train_diagnostic.py`, `eval_*.py` — legacy training/eval. Used by M0.4 audit. Don't rewrite.
- `models_test/cnn_breakpoint_runB2_sig10_*.keras` — the deployment baseline. **Don't delete or modify.**

### Investigation artifacts (committed for traceability)
- `m13_santa_vs_real.{py,json,log}` — the earlier, flawed XML-6-based investigation. Kept as a record of what NOT to extrapolate from.
- `m04_*.{py,json,log}` — Phase 0 baseline audit.

---

## How to deliver work

- **Per-milestone discipline:** one milestone per session if possible. Each milestone produces a concrete deliverable (file path) and a measured success criterion.
- **Commit early, commit often.** The user is OK with merge commits (uses `--no-ff` style). Don't accumulate uncommitted work in a worktree — it gets lost when worktrees are removed.
- **Update `MASTER_PLAN.md`** with a checked-off entry per milestone, including measured numbers (not predicted ones).
- **Save a memory file** for any non-obvious finding or strategic decision. The user values these for cross-session continuity.
- **Don't write speculative analysis as if it were measurement.** If you haven't run it, don't claim a number.

---

## If you're stuck

- The advisor tool exists and the user values it for big architectural calls. Use it before committing to a multi-day path.
- The user is responsive in real-time during the session. If you genuinely need direction, ask — but make sure the question has actionable options, not "what should we do?"

Good luck. The hard architectural decisions are made (PyTorch + HyenaDNA + filtered SANTA). The hard work is now the training loop.
