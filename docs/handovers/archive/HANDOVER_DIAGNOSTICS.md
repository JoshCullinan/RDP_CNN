# Handover: Three-Diagnostic Plan Before Any Architecture Change

**Author:** Claude (Opus 4.7) — session 2147869b-a88e-4efd-98f9-1420b956f30f
**Date:** 2026-05-12
**Branch:** `main` (PR #3 merged; clean state)
**Prior work:** Runs #28-#43 logged in CNN.ipynb experiment log.

---

## TL;DR for the next agent

You are picking up a CNN viral-recombination-breakpoint-detection iteration that has plateaued for ~28 runs at honest F1 ≈ 0.33 (best: run #41 = 0.335 on full UnseenTestSet). The user said "this trundling needs to be improved" and asked for an ultrathink on the problem. The previous agent's instinct was to reset everything (new splits, new architecture, transformers, multi-modal). **The advisor pushed back hard against that.**

**What we're doing instead:** Three cheap diagnostics that gate any architecture decision. (A) is partially complete and is unambiguous so far — run #41's advantage over run #38 is real, not a leakage artifact. (B) and (C) are pending. **Do NOT propose architecture changes (transformers, 4-way head, multi-modal) until B and C report.** That was identified as the trundling failure mode.

The full advisor reasoning is in the session transcript at `~/.claude/projects/-home-joshcullinan-RDP-CNN/2147869b-a88e-4efd-98f9-1420b956f30f.jsonl` if you need it. The key constraints distilled below.

---

## The three diagnostics

Each diagnostic answers ONE question with ONE training run (or pure inference). All use existing infrastructure on `main`.

### Diagnostic (A) — Fair eval on truly held-out subset

**Question:** Is run #41's 0.335 vs run #38's 0.313 lift real, or a leakage artifact from the pooled split putting ~67/97 UnseenTestSet files into run #41's training?

**Subset:** `UnseenTestSet − (run41_train ∪ run42c_train) = 11 files = 584 events`. Computed by `splits/honest_eval_subset_11.txt` (already saved on disk).

**Method:** `eval_diagnostic_A_fair.py` — loads each of the 5 model checkpoints, runs inference on the cached `UnseenTestSet` events, subsets predictions to the 11 held-out files, sweeps threshold @ EB ∈ {0, 200}. Outputs JSON + summary table.

**Decision rule (from advisor):**
- Run #41 wins by ~+0.04 over run #42c family on the 11-file subset → regressions are real → proceed to (B) and (C).
- Numbers converge or invert → the "regressions" were leakage artifacts → run #42c data composition is fine, training-infra was the issue.

**Cost:** ~10 min/model × 5 = ~50 min total. No training.

### Diagnostic (B) — Channel ablation

Channels 15-21 are pre-computed RDP-style features (`match_p1`, `match_p2`, `informative`, MaxChi disparity at 4 window scales). The CNN's residual-dilated stack has ~224bp receptive field on 30000bp content — these features hand the model the recombination signal globally so the convs don't have to rediscover it. **The model may be a thin head over RDP features rather than a real deep learner.** B answers that.

**(B1) Inference-time mask** (~15 min, no training):
- Load `models_test/cnn_breakpoint_run41_final.keras`.
- At inference, zero out channels 15-21 of the 33-channel input.
- Measure honest F1 (EB=200) on UnseenTestSet.
- Outcomes:
  - F1 ~0.05 → model is heavily relying on those channels
  - F1 ~0.20-0.30 → partial reliance, CNN does some real learning
  - F1 ~0.30+ → channels are redundant, CNN learned the signal itself

**(B2) Train-time ablation** (~3-4 hr):
- Same architecture, same data, same hyperparams as run #41.
- BUT mask channels 15-21 to zero at training input (encoder stage), so model never sees them.
- Train to convergence.
- Outcomes:
  - F1 ~0.05 → architecture cannot learn the signal from raw sequences alone
  - F1 ~0.20-0.30 → CNN contributes structure on top of features
  - F1 ~0.30+ → CNN can learn from raw, features were largely redundant

The **contrast between (B1) and (B2)** is the key signal. (B1) is "is the model using these channels?" (B2) is "can it learn without them?" Tiny gap = redundancy. Huge gap = dependency.

### Diagnostic (C) — Receptive field extension

**Question:** Is the ~224bp receptive field of the current dilated stack the bottleneck on 30000bp content?

**Method** (~3-4 hr):
- Same architecture, same data, same hyperparams as run #41.
- ONE change: append 5 additional dilation layers at the end of the stack. RF goes from 224bp to ~7kbp.
- Train to convergence on run #41's data + hyperparams.

**Decision rule:**
- F1 lifts by ≥ +0.02 vs run #41 → RF is real bottleneck → next step is linear-attention transformer (Performer/Mamba/Longformer) or hybrid CNN+transformer.
- F1 doesn't lift → RF is not the bottleneck → moving to transformers would be over-engineering.

---

## Current state (where you're starting)

### Diagnostic (A) — partially complete, results unambiguous so far

`eval_diagnostic_A_fair.py` exists and works. It has been run multiple times. Each time the eval was killed by session archival before completing all 5 models. **Each restart redoes all 5 from scratch — the script doesn't cache per-model.**

Reproduced (from monitor notifications across runs, identical each time):

| Model | Full UnseenTestSet honest F1 | 11-file subset honest F1 |
|-------|-----------------------------|--------------------------|
| Run #38 | 0.316 | **0.367** |
| Run #41 | 0.339 | **0.393** (+0.026 over #38) |
| Run #42c | pending | pending |
| Run #42c_diag | pending | pending |
| Run #43 | pending | pending |

**What we know already from this:** Run #41's advantage over run #38 is preserved on truly held-out files (+0.026 on subset vs +0.023 on full). **It is not a leakage artifact.** Run #41 is genuinely better than run #38.

**What we still need to know:** Does run #42c family lag (real regression) or match/beat (regression was the leakage artifact on full UnseenTestSet eval)?

### Diagnostics (B) and (C) — not started.

No scripts written, no models trained, no results.

### Task list

You'll see Task IDs #27-#31 if running task tools. Task #27 is in_progress. Hierarchy:
- #27 (in_progress): Finish diagnostic (A)
- #28 (blocked by 27): Diagnostic (B1) — inference-time channel mask
- #29 (blocked by 28): Diagnostic (B2) — train-time channel ablation
- #30 (blocked by 28): Diagnostic (C) — RF extension
- #31 (blocked by 27,28,29,30): Synthesis + commit + memory update

### Eval (A) status

The previous attempt at finishing eval (A) was launched at PID 5530 and got through run #38 + run #41 before the session was archived. Check `eval_diagA.log` for state. If the python process is dead, restart from scratch — it's ~50 min total but only the run #42c family rows are new.

---

## How to resume (concrete next actions)

### Step 1: Finish (A)

```bash
# Check if a prior eval is still running
ps aux | grep eval_diagnostic | grep -v grep

# If not, restart
source .venv/bin/activate
nohup python -u eval_diagnostic_A_fair.py > eval_diagA.log 2>&1 &
# Then monitor:
tail -F eval_diagA.log | grep -E "best F1|Saved|run[34]"
```

Wait until `Saved results_diagnostic_A_fair.json` appears. Then read the JSON for the 5-row summary.

**Apply decision rule to results.** Use TaskUpdate to mark #27 completed.

### Step 2: (B1) inference-time channel mask

Build a new script `eval_channel_mask.py` modeled on `eval_diagnostic_A_fair.py`:

```python
# Pseudo:
model = load(run41_final.keras)
X = load(UnseenTestSet cache, 33ch)
X_masked = X.copy()
X_masked[..., 15:22] = 0.0   # zero match_p1, match_p2, informative, 4× MaxChi
y_pred = model.predict(X_masked)
F1_at_each_threshold = evaluate(y_pred, X_masked, meta, threshold, EB=200)
# Compare against run #41 baseline F1=0.339 on full, 0.393 on 11-file subset
```

Run on both full UnseenTestSet AND the 11-file subset. Report F1 drop. ~15 min.

Use TaskUpdate to mark #28 completed.

### Step 3 — fork: (B2) and (C) decision

If (A) showed run #41 advantage is real AND (B1) showed nontrivial dependency on channels 15-21: proceed with both (B2) and (C). They can run sequentially.

If (A) showed convergence (regressions were leakage): the priority shifts. The advisor said: "you have more headroom in the existing pipeline than you thought" → revisit run #42c data with proper training-infra settings. But this is unlikely given (A)'s partial results already.

### Step 4: Run (B2) — train-time channel ablation

Patch `CNN.ipynb` to mask channels 15-21 in the encoding stage (cell-6 or cell-11). Specifically:
- After `encode_triplet` builds the 33-channel tensor, set columns 15:22 to 0.
- This needs to happen at encode time, not just at predict time, so the cache is built with masked input.
- Bump CACHE_VERSION to `v9_no_rdp_channels` or similar.
- Save model as `models_test/cnn_breakpoint_runB2_no_rdp_channels.keras`.
- Use run #41's hyperparams: BATCH=2, LR=1e-4, fp32, EPOCHS=50, TRAIN_MAX_EVENTS=7000.
- Run on run #41's splits (not run42c) — these are the comparison-relevant settings.

After convergence, eval on UnseenTestSet and 11-file subset. Compare honest F1 against run #41's 0.335 (full) / 0.393 (subset).

Use TaskUpdate to mark #29 completed.

### Step 5: Run (C) — RF extension

Patch `cell-18` (`build_cnn`) to add 5 more dilation layers. The current stack has residual dilated blocks at dilation rates {1, 2, 4, 8, 16, 32}; add {64, 128, 256, 512, 1024}. RF: 7×(1+2+4+8+16+32) = 224 → 7×(1+...+1024) = 7×(1+2+4+...+1024) ≈ 14000bp. (Adjust if you want exactly 7kbp.)

Same hyperparams as run #41. Save as `models_test/cnn_breakpoint_runC_extended_rf.keras`.

Eval, compare against run #41's 0.335 / 0.393.

Use TaskUpdate to mark #30 completed.

### Step 6: Synthesis

When all three diagnostics report:

| (B1) F1 drop | (B2) F1 vs #41 | (C) F1 vs #41 | Verdict | Next direction |
|--------------|----------------|---------------|---------|----------------|
| Huge (0.05) | Catastrophic | Flat | Model = thin head on RDP features | Rebuild features more carefully OR accept hand-crafted-features paradigm |
| Small | Similar | Flat | Data ceiling reached on current task formulation | Reframe to per-position 4-way `{A, B, C, no_recomb}` — addresses detect AND identify |
| Either | Either | Lifts ≥ +0.02 | RF is the bottleneck | Linear-attention transformer (Performer, Mamba) or hybrid CNN+transformer |
| Small | Lifts | Lifts | CNN can learn raw + needs longer RF | Deeper dilated stack OR transformer; depends on cost/benefit |

Write the synthesis as a new entry in `CNN.ipynb`'s `cell-experiment-log`, update memory files, commit `eval_diagnostic_A_fair.py` and any new scripts.

Use TaskUpdate to mark #31 completed.

---

## Hard constraints (advisor + user directives — do not violate)

### Do NOT

- ❌ **Rebuild train/val/test splits from scratch.** Splits are validated; UnseenTestSet leakage analysis already accounted for. Advisor was emphatic.
- ❌ **Subsample to "make training tractable."** Training time isn't the bottleneck (runs land in 2-3 hr); signal quality is. Subsampling same-composition doesn't help.
- ❌ **Jump to transformers / multi-modal / pretrained Nucleotide Transformer before (C) reports.** Advisor explicitly flagged: "If the agent goes 'let me think about transformers' before (A), that's the trundling continuing."
- ❌ **Run more variations of the same architecture (BATCH=4, LR=2e-4, etc.).** The user said the current trundling needs to be improved — that means new information per run, not parameter sweeps.
- ❌ **Bundle multiple changes into one experiment.** Run #42c bundled 5 changes; couldn't isolate cause. One-change-per-run discipline.

### Do

- ✅ **Treat run #41 (honest F1=0.335 on full, 0.393 on 11-file subset) as the comparison baseline.** Run #38 (0.313) is the no-leakage baseline if you need a stricter floor.
- ✅ **Sharpen "informative" later.** The right metric is local informative density at ±200bp of breakpoint + parental-signal contrast, NOT population-wide informative%. But this is for after diagnostics — not now.
- ✅ **Keep memory files current.** Update `~/.claude/projects/-home-joshcullinan-RDP-CNN/memory/MEMORY.md` and project_*.md files as you learn things. There's a memory entry framework in CLAUDE.md.
- ✅ **Commit eval infrastructure.** `eval_diagnostic_A_fair.py` and `splits/honest_eval_subset_*.txt` are uncommitted as of this writing. They're the fair-comparison harness; commit after (A) is done.

---

## Files inventory

### Scripts
- `eval_diagnostic_A_fair.py` — diagnostic (A) eval script. UNCOMMITTED.
- `splits/honest_eval_subset_11.txt` — 11-file held-out list. UNCOMMITTED.
- `splits/honest_eval_subset.txt` — same content (legacy name). UNCOMMITTED.
- `eval_diagA.log` — last (A) run output. UNCOMMITTED.
- `pick_parents_rdp5ml.py` — faithful event_classifier.py port. COMMITTED on main.
- `eval_run41_on_unseentestset.py`, `eval_run42c_on_unseentestset.py`, etc. — apples-to-apples evals per run.
- `CNN.ipynb` — the notebook. Cells with stable IDs `cell-3`, `cell-6`, `cell-11`, `cell-12`, `cell-18`, etc. See CLAUDE.md cell map.

### Model checkpoints (`models_test/`)
- `cnn_breakpoint_run38_final.keras` — 24 channels (no RDP scalars), F1=0.313 honest. The clean-no-leakage baseline.
- `cnn_breakpoint_run41_final.keras` — 33 channels, F1=0.335 honest. **DEPLOYMENT BASELINE.**
- `cnn_breakpoint_run42c_final.keras` — F1=0.299, regression.
- `cnn_breakpoint_run42c_diag_final.keras` — F1=0.316 partial (OOM at E14).
- `cnn_breakpoint_run43_final.keras` — F1=0.283.

### Caches (`cache/`)
Multiple `ds_UnseenTestSet_*.npz` with different channel counts (22, 24, 25, 33). The diagnostic script picks by `n_channels` match. Each cache is ~900 MB.

### Splits (`splits/`)
- `run41_{train,val,test}.txt` — pooled stratified, ~67/14/16 UnseenTestSet leaked into train.
- `run42c_{train,val,test}.txt` — pooled stratified with long_content_30k_001/002/003 + informative% ≥5 filter, ~69/15/13 UnseenTestSet.

### Memory (`~/.claude/projects/-home-joshcullinan-RDP-CNN/memory/`)
Key entries (read before assuming things):
- `MEMORY.md` — index
- `project_run43_outcome.md` — last documented attempt's outcome
- `project_run42c_outcome.md` — same for run42c
- `project_run42c_runbook.md` — pipeline used to produce run42c data
- `project_lineage_heuristic_parents_useless.md` — why bad parents tank training
- `project_parents_are_always_approximate.md` — domain truth on parent picking
- `feedback_filter_weak_data.md` — drop informative%<5%
- `feedback_no_symlinks_for_data.md` — copy, don't symlink
- `feedback_iteration_mode.md` — autonomous looping; surface results, don't ask permission
- `feedback_architecture_freedom.md` — CNN-only constraint waived; transformers etc. OK
- `project_wsl_memory.md` — 28GB cgroup ceiling
- `project_santa_dump_10k_ceiling.md` — SANTA cfg.xml hardcodes length=10000

### CLAUDE.md
The canonical project guide. Read it. Especially:
- "Three things that are easy to break" — bias init, POS_WEIGHT, loss/metric pairing, argmax-output BN-padding interaction.
- "Iteration discipline" — one change per run, watch train PR-AUC.
- "Evaluation contract" — ±200bp tolerance, find_peaks, peak-based F1.

---

## How to test that this handover is correct

After reading, you should be able to answer:
1. **Why are we doing diagnostics before architecture changes?** → Two prior runs (#42c, #43) failed without telling us *why*; the advisor identified that we don't know whether the CNN is even using its inputs, whether RF is the bottleneck, or whether prior comparisons were leakage-confounded. Architecture changes without that information are trundling.
2. **What's the current state of (A)?** → Run #38 and run #41 reproduced (subset F1 0.367 / 0.393). Run #42c, #42c_diag, #43 pending; script kills mid-run on session archival.
3. **What's the decision rule for proceeding to architecture work?** → Use the synthesis table at the end of Step 6. Don't jump.
4. **What's the deployment baseline number?** → Run #41 = 0.335 honest F1 on full UnseenTestSet, 0.393 on 11-file truly-held-out subset.

If any answer is unclear, re-read this doc before touching code.

---

## Final note

The previous agent (me) wanted to do bigger reframes (jumble data, 4-way head, transformers) when the user asked for an ultrathink. The advisor correctly identified that as expensive speculation when cheap diagnostics would actually decide the direction. **Don't repeat that mistake.** Finish (A), run (B1) which is 15 min, and let the data tell you whether (B2) and (C) need to happen.
