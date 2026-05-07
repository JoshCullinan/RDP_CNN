# Handover: Autonomous CNN Iteration

You are taking over an ongoing ML research project. Your job is to **continuously improve the recombination-breakpoint detection model in `CNN.ipynb` without further input from the human**, and to keep a durable, structured record of every experiment you run inside the notebook itself.

> **2026-05-04 update — machine migration.** The previous agent ran 12 experiments on an Apple M4 (16 GB unified memory) over ~24 hours. Compute is now moving to a different machine. Read §11 (current state) and §12 (what's staged) before doing anything else.

---

## 1. Read these in order, then come back

1. [CLAUDE.md](CLAUDE.md) — project orientation, environment, architecture, the three things that are easy to break.
2. [TODO.md](TODO.md) — chronological log of what has been tried, what shipped, what failed and why, and the prioritised queue of next changes.
3. The "Experiment Log" section at the end of [CNN.ipynb](CNN.ipynb) — every prior run's hypothesis, configuration, headline numbers, and verdict. The most recent entry is the baseline you are improving on. (Currently 11 logged entries, runs #1–#16.)

If anything in this handover contradicts those files, **the notebook log is the most current source of truth**. Update CLAUDE.md and this file when you discover the contradiction.

## 2. Project goal

Beat MaxChi, RDP, and GeneConv on recombination-breakpoint detection, with eventual deployment on real HIV / other genomes. Quality is the only objective; compute cost is not a constraint. Do not propose smaller-or-faster architectures as compromises — bigger and more expressive is fine.

The current task setup (per-position breakpoint probabilities given a known recombinant + two known parents) is a stepping stone. The eventual deployment task is "given three sequences, decide which (if any) is the recombinant" — keep that in mind when proposing architectural changes.

## 3. The iteration loop

Repeat indefinitely until a stop condition fires (§7).

```
loop:
  1. Read TODO.md "High priority" section. Pick the top item.
  2. State a one-sentence hypothesis: "I expect change X to improve metric Y because Z."
  3. Apply EXACTLY ONE change to the notebook. (See §4 for what counts as one change.)
  4. Run the notebook end-to-end (`python3 run_notebook.py`).
  5. Record the result in the Experiment Log cell using the template in §5.
  6. Decide: keep, revert, or inconclusive. Update TODO.md to reflect.
  7. If kept: pick next item from queue.
     If reverted: pick next item, ideally one orthogonal to the failed change.
     If inconclusive: re-run with a single tightening change (more epochs, higher max_files, or fix the diagnostic that was unclear), not a different idea.
```

**There is no "and also" in step 3.** If you find yourself wanting to bundle "switch loss + add residual connections", split into two runs.

## 4. What counts as one change

Allowed in a single run:
- One change to `build_cnn` (e.g. add residual connections OR add a layer OR change kernel sizes — not all three).
- One change to the loss / its hyperparameters (e.g. `pos_weight` 70 → 30).
- One change to a single config constant in cell-3 (e.g. `LABEL_SIGMA` 20 → 10).
- One change to data augmentation (e.g. enable parent-swap).
- One change to optimiser / LR / schedule (e.g. LR 1e-4 → 5e-4).

**Always paired with the architectural change** (these are not "extra changes"):
- Re-deriving `POS_WEIGHT` from the new mean(y) if you change `LABEL_SIGMA` or `MAX_SEQ_LEN`. **Note: as of run #8 we hardcode `POS_WEIGHT=70` regardless** — see lessons below.
- Re-computing the prior-aware bias init if positive rate changes materially.
- Updating cell docstrings to match new behaviour.

If the natural unit of change touches multiple cells (e.g. a new loss requires a new metric and a new config constant), that's still one change — just keep the *idea* singular.

## 5. Experiment Log — the canonical record

The notebook contains a markdown cell titled **"Experiment Log"** near the end. After every run, append a new entry **at the top** using this exact template:

```markdown
### YYYY-MM-DD #N — One-line title

**Hypothesis:** Why I expected this to help.

**Change:** Single change applied. Cite cells touched.

**Config snapshot:**
- max_files: 750 / None
- LABEL_SIGMA: 20
- POS_WEIGHT: 70
- LR: 1e-4
- Loss: weighted_bce
- Architecture: residual dilated stack with N MaxChi windows / etc.
- MAX_SEQ_LEN: 10000
- N_INPUT_CHANNELS: 22

**Results:**
- Best epoch: N
- Train PR-AUC at best epoch: 0.XXX
- Val PR-AUC at best epoch: 0.XXX
- Val F1 (peak-based): 0.XXX
- Val "Both BPs found": XX.X%
- Test F1 (peak-based): 0.XXX
- Test "Both BPs found": XX.X%
- Test precision / recall: 0.XXX / 0.XXX
- Mean predicted peaks / event: X.XX
- Prediction range over val: [0.XXX, 0.XXX]

**Verdict:** KEPT / REVERTED / INCONCLUSIVE

**Why:** One or two sentences. Tie back to the hypothesis. If the verdict
disagrees with the hypothesis, name the surprise.

**Next:** What this run implies for the next experiment.
```

## 6. Decision criteria

When a run completes, classify it:

**KEPT** — at least one of:
- Test F1 improves by ≥0.01.
- Test "Both BPs found" improves by ≥2 percentage points.
- Test F1 is unchanged but a qualitative property improves: peaks are sharper, prediction range expands, train-val gap narrows. Note the qualitative win in **Why**.

**REVERTED** — the change clearly hurts on a headline metric, or provides no benefit and adds complexity. Revert the cell-level change *before* writing the log entry.

**INCONCLUSIVE** — results are within noise on every metric. Re-run with the cheapest tightening (more epochs, larger `max_files`, or a fix to whichever diagnostic was unclear). Two consecutive INCONCLUSIVE runs on the same change → mark as REVERTED and move on.

**Important context the previous agent learned the hard way (see lessons below):** F1 and "Both BPs found" are both headline metrics. When they conflict (F1 up but Both BPs down, or vice versa), the strategic call is what wins — do not silently apply only the favourable criterion. If the run is genuinely ambiguous, use the advisor.

## 7. Stop conditions

Stop the loop and write a final report (markdown cell at the end of the Experiment Log) when **any** of:

- Test F1 ≥ 0.85 AND test "Both BPs found" ≥ 75% — competitive with state of the art.
- Five consecutive runs land in REVERTED.
- Three consecutive runs touch the same priority queue item.
- An expected metric signature appears that suggests a deeper bug (e.g. predictions stuck at sigmoid(0)=0.5; train PR-AUC suddenly drops; loss explodes). Stop, do not paper over.
- You run out of items in the priority queue and cannot generate new ones from the run results.

## 8. Escalation — when to surface to the human

The human is opt-in for nothing during this loop, but write a `🚩 ESCALATION` block at the top of the Experiment Log if:

- A change unexpectedly **improves** test F1 by ≥0.10 or "Both BPs found" by ≥20 points. Big jumps usually mean something's wrong (e.g. data leak, evaluation bug). Verify first.
- The data on disk appears modified.
- Running the notebook starts producing NaN losses, hangs, or otherwise fails in a way that retries don't fix.
- You realise an earlier experiment's verdict was wrong.

## 9. Pitfalls — known traps

These are documented because they cost real iterations. **All of these are still active.**

- **Final-layer bias init is required, not optional.** Without `bias_initializer = -log((1-π)/π) ≈ -4.6` for π=0.01, training gets stuck at sigmoid(0)=0.5.
- **The padding mask is required**, not optional. Without `sample_weight=mask` in `fit()`, the loss is dominated by padding zeros.
- **Group-by-file train/val split is required.** As of run #15, we use a *stricter* split: TRAIN=XML-1..4, VAL=XML-5. Don't revert to mixed-XML splitting — it leaks distribution and overstates capability.
- **POS_WEIGHT is hardcoded to 70.** Don't auto-derive from `(1-p)/p` (run #7 found that over-aggressive). cell-12 still computes the data-implied value as a *diagnostic*, but `POS_WEIGHT = 70.0` in cell-3 is the operating value. If you change σ or the label encoding, POS_WEIGHT may need to scale — but treat that as a deliberate experiment (see lesson on σ/POS_WEIGHT coupling), not an automatic adjustment.
- **Notebook is too large for direct `Read`.** Edit via a Python script that loads/saves the JSON. See template below.
- **`restore_best_weights=True` returns the best epoch's weights.** Always check the train trajectory (per-epoch train PR-AUC + train loss), not just `val_aupr improved` lines.
- **PR-AUC at "base rate" looks low even for a working model.** With ~1.4% positives, a perfectly random predictor scores ≈0.014. The first working run plateaued at val_aupr ≈ 0.18 — that's 12× base rate.
- **Threshold-tuning artifacts can disguise wins as regressions.** When the prediction distribution shifts (e.g. residual+dilated, MaxChi, dropout changes), the val-best threshold moves too. Always extend the threshold sweep to 0.95 (cell-27) to find the true optimum before reading verdicts.
- **Boundary spikes are real but not a bottleneck.** The model produces ~1.0 predictions at the actual_len boundary (BN+padding artifact). Run #12 tried to mask them in the loss; it didn't help test F1. Suppressing the spike post-hoc *also* hurts F1 (some real BPs are near edges). The graded-confidence problem in the interior is the dominant issue, not the boundary.
- **Val/test gap is structural, not sample-size.** Run #14 added 34% more training data; val improved, test held flat. Run #15's strict held-out split (XML-5 only) reduced the gap a little but didn't close it. UnseenTestSet has its own distribution.

### Notebook editing template

```python
import json
nb = json.loads(open('CNN.ipynb').read())
for cell in nb['cells']:
    if cell.get('id') == 'cell-XX':
        cell['source'] = new_source.splitlines(keepends=True)
        cell['outputs'] = []
        cell['execution_count'] = None
open('CNN.ipynb', 'w').write(json.dumps(nb, indent=1))
```

Cell IDs (`cell-0`, `cell-1`, ...) are stable; preserve them.

### Running the notebook

The previous agent built a small wrapper at [run_notebook.py](run_notebook.py) that loads the notebook with nbformat and executes via nbclient. Use it via:

```bash
source .venv/bin/activate && python3 -u run_notebook.py > exec_log.txt 2>&1
```

A typical run takes ~80–100 minutes on an Apple M4. With 6 MaxChi windows (run #17 staged) it'll be slower because data prep is heavier. The cell-22 timeout is currently set to 21600s (6h). Bump if needed.

There are also helper diagnostic scripts:
- [chart_eval.py](chart_eval.py) — generates prediction-vs-truth charts on the cached `models_test/cnn_breakpoint_best.keras`. Use to diagnose where the model is failing.
- [eval_with_suppress.py](eval_with_suppress.py) — sweeps post-hoc edge suppression × threshold against the cached model. Useful to verify boundary artefact effects without retraining.
- [eval_topk.py](eval_topk.py) — evaluates the cached model under top-K peak reranking. Useful as a sanity check on whether the per-position output's top-K ordering aligns with real BPs.

## 10. Project-state-relevant lessons (newest first)

These are *load-bearing* lessons distilled from the experiment log. Read in full there for context.

- **Hand-engineered features beat the architecture's implicit synthesis.** Run #13 added 4 MaxChi-style "running parental disparity" channels and immediately collapsed the train-val gap (+0.029 → −0.003) and lifted Both BPs across the board. The residual+dilated backbone was the *foundation*; without the right input features it couldn't extract enough signal from raw match_p1/match_p2.
- **MaxChi feature representation is data-driven from `MAXCHI_WINDOWS` in cell-3.** To extend, change the tuple and bump `N_MAXCHI_WINDOWS` / `N_INPUT_CHANNELS`. cell-6's `_maxchi_features` will pick up the new windows automatically.
- **σ and POS_WEIGHT are coupled.** Run #9 tried σ=10 with hardcoded POS_WEIGHT=70 and the model under-fit (data-implied POS_WEIGHT was 166). To test σ in isolation, scale POS_WEIGHT with mean(y) — but treat that as a paired change.
- **LayerNorm is not a drop-in replacement for BatchNorm here.** Run #10 collapsed Both BPs to 0% even though test F1 nudged up. If you revisit LN, pair it with explicit LR/capacity/pos_weight changes.
- **Boundary spikes live in padding, not in the eroded valid region.** Run #12 erode-the-loss-mask change didn't suppress them (no loss at padding to begin with). The fix would be masked BatchNorm or 'valid' padding — but post-hoc suppression sweeps showed the boundary isn't the F1 limiter anyway.
- **Top-K peak reranking is a small improvement on threshold-tuning** (~+0.03 test F1) but not a breakthrough. The model has graded confidence; real-BP peaks are smaller in amplitude than spurious confident peaks. Reranking by height on top of the current model recovers some lost recall but doesn't fix the underlying confidence calibration.
- **Mixed-XML val splits hide distribution leakage.** Run #15 held out XML-5 entirely; val F1 dropped 0.356 → 0.321, val Both BPs 27.7% → 18.2%. Treat XML-5 as the iteration metric — it's both n=621 (more powerful than test n=82) and an honest cross-config measure.
- **Even XML-5 held out, val ≠ test.** UnseenTestSet F1 is roughly half of XML-5 val F1. There's a second layer of distribution shift between XML-5 and UnseenTestSet. Worth probing with explicit feature comparisons (mean(y), actual_len histograms) before assuming any single intervention will close it.
- **Train-val gap is a leading indicator of capacity vs. signal quality.** When the gap is wide (>0.05), more capacity won't help — find better features or stronger regularisation. When the gap closes, you've earned the right to scale up.

## 11. CRITICAL FINDING (end of 2026-05-05 session, after runs #17–#23)

**The val/test gap is a TRUNCATION BUG, not a generalization issue.**

After 7 consecutive REVERTED runs failed to move test F1 past run #16's 0.172, an advisor-prompted measurement of val vs test distributions revealed:

- `actual_len` mean: **val 10,788 vs test 30,009** (test is ~3× longer)
- `bp_end >= MAX_SEQ_LEN(10000)` rate: **val 32.7% vs test 74.4%**
- `bp_start >= 10000`: **val 0.6% vs test 48.8%**
- bps within 50bp of edge: **val ~30% vs test ~80%**

**Test sequences are ~3× longer than train/val.** `MAX_SEQ_LEN=10000` truncates the test set, throwing away most breakpoints. Half of test `bp_start` values and three-quarters of test `bp_end` values are positions the model literally never sees. Of the visible bps, 80% are squashed against an edge.

**Every prior REVERTED run is re-explained by this:**
- Stable test F1 ≈ 0.12–0.17 across all variants → consistent truncation rate
- Run #21 boundary suppression crashed test F1 to 0.024 → it banned the only positions truncated test bps appear at
- Bigger kernels (#22) helped val (which fits) but not test (which doesn't)
- Top-K mode-collapse to 0/9999 (#19/#20) → that "boundary attractor" was the truncated test distribution

**Fix for run #24 (one-line change in cell-3):** `MAX_SEQ_LEN = 10000 → 32000`. Max observed `bp_end` in test is 30,143; 32000 captures everything with margin.

**Required paired infra changes:**
- Cache invalidates (key includes MAX_SEQ_LEN) — cold cache run, ~10-15 min for re-parse
- VRAM: RTX 3070 has 8GB; preemptive `BATCH_SIZE 8 → 2 or 4` (per-sample input alone is ~22MB; with activations × 6 dilated blocks × 64 filters, batch 8 is at risk of OOM at 32k length)
- POS_WEIGHT may need rescaling: longer sequences → lower mean(y), implied POS_WEIGHT roughly triples (current 70 → ~178 by run-#8 calibration). Treat as paired infrastructure with the MAX_SEQ_LEN bump.

**Sanity check before run #24:** verify SANTA simulator outputs are 30k-bp sequences (not alignments with gaps making encoded length shorter). If alignments include internal gaps, the truncation arithmetic might differ.

If run #24 lifts test F1 toward val's 0.28-0.31, the project's "structural val/test gap" narrative is rewritten. Best result so far is run #16: test F1 = 0.172; under the truncation hypothesis, the achievable ceiling on UnseenTestSet should be much higher.

---

## 11-current. Current state (end of 2026-05-07 session — runs #28-#34 + ensemble)

**Status:** **Honest interior F1 ceiling at ~0.17 confirmed.** Best honest F1 across 7 architectural/data/augmentation variations + a 4-model ensemble: **F1=0.175 (run #32)**. RDP5 baseline: 0.367. **The dilated-CNN-with-per-position-sigmoid framework cannot break this ceiling.** See "Plateau finding" in the experiment log for the full table and ranked next-step directions.

**Most important learning of this session:** The reported test F1 from any run (e.g. #28d's 0.308) is largely BOUNDARY ARTIFACT — `eval_edge_suppressed.py` with EB=200 reveals the honest interior detection. **All future runs MUST report EB=200 honest F1 alongside the raw number; raw numbers without edge suppression are inflated by 40%+.**

### Earlier this session — run #28d landed (KEPT, infrastructure)

### What this session delivered

1. **Parser fix** (`pd.read_csv(..., index_col=False)` on both stats CSVs) — was silently giving the model garbage parent IDs across every prior run. Fixed in cell-10. Effect: dataset sizes jumped ~67× (UnseenTestSet went from 82 to **5,539 events**) and the comparison set is now apples-to-apples with RDP5.

2. **fp16 X storage** (cell-11) — X arrays stored as float16, cast back to fp32 inside the tf.data .map after batching. Halves host RAM. One-hot/match channels are exact in fp16; MaxChi disparity has ~3e-3 precision in [-1, 1] which is fine.

3. **`from_generator` instead of `from_tensor_slices`** (cells 22, 26, 40) — the *real* root cause of the catastrophic Vmmem deaths. `from_tensor_slices(X)` materialises a tf.constant of size(X), so X_val numpy + X_val tf.constant = 2× resident. Watched a single 8 GB Committed_AS jump at end-of-epoch-1 in monitoring CSV — exactly the X_val (4.12 GB fp16) duplicating into a tf.constant. Switched to `from_generator` which yields slices on demand. No graph copy, no doubling.

4. **cgroup memory cap** — `systemd-run --user --scope -p MemoryMax=24G` wraps the python launch. Replaces the prior catastrophic Vmmem deaths with surgical Linux OOM-kills inside the scope only — Vmmem stays alive, we get a clean Python termination + monitorable scope-failed event. Confirmed working: run #28d hit the cap during the buggy old eval cell, scope killed, WSL stayed up.

5. **Persistent monitoring** in `monitor/wsl_log.sh` (Linux side) and `C:\Users\joshc\wsl_monitor\win_log.ps1` (Windows side, run from PowerShell). Both write CSVs to `/mnt/c/Users/joshc/wsl_monitor/` every 3s. The Windows side is the only vantage point that survives a full Vmmem death. Diagnosed the 14:56:28 swap-thrash → kernel deadlock chain in run #28 attempt-1; confirmed run #28d's cgroup OOM was *not* a Vmmem death.

6. **UnseenTestSet now CACHED** — `cache_test_set.py` exec's the canonical notebook parser to populate `cache/ds_UnseenTestSet_*.npz/.pkl`. Future evals on a saved model are ~3 min from disk via `eval_only.py`, no retrain. Use this when iterating on threshold sweeps or post-hoc diagnostics.

### Run #28 results vs RDP5

| | Val (XML-5, 2925 events) | Test (UnseenTestSet, 5539 events) |
|---|---|---|
| Best F1 (raw) | 0.575 @ thr=0.6 | **0.308 @ thr=0.9** |
| Val-thresh F1 (raw) | 0.575 @ 0.6 | 0.239 @ 0.6 |
| Both BPs (raw) | 62.9% @ 0.6 | 19.5% @ 0.6 |
| **HONEST F1 (edge-suppress EB=200)** | — | **0.169–0.171** |
| RDP5 | — | F1=0.367, Both BPs 15.3% |

**CRITICAL — read this before claiming run #28d "beat" anything.** A bucket-by-position diagnostic (`bucket_diagnostic.py`) showed that **the model exploits a BN+'same'-padding boundary spike**:
- Detection rate at positions 0-1000 bp: **98.9%**
- Detection rate at positions 29000-31000 bp (content edge for 30k-test sequences): **88-100%**
- Detection rate at interior positions 5000-29000 bp: **~25-30%**

Edge-buffer suppression of just the first/last 100 bp of each sample's content during inference (`eval_edge_suppressed.py`) drops test F1 from 0.308 → 0.171. **44% of run #28d's apparent test performance is boundary artifact.**

The model learned to output peaks at content/padding boundaries because (a) some real BPs land there, and (b) BN running stats are contaminated by padded zeros, systematically inflating logits at boundaries. CLAUDE.md §"Three things that are easy to break" #5 flagged this risk under argmax-style heads but it dominates the per-position sigmoid head too — just less visibly.

**Honest interior F1 (0.171) is much further below RDP5 (0.367) than the raw 0.308 suggested.** The val→test gap (0.575 → 0.239 raw) was largely driven by val sequences having content well below the boundary spike zone, while test has many BPs near content edges that the model "detects" via the spike rather than real localization. There is no calibration drift to fix; there's a boundary shortcut to break.

**Diagnostic chain that revealed this:**
1. `sanity_calibration.py` — showed test predictions are NOT systematically smaller-magnitude (calibration drift hypothesis dead).
2. `bucket_diagnostic.py` — detection rate by 1000bp bucket; 98.9% at boundary, ~30% interior. Smoking gun.
3. `eval_edge_suppressed.py` — edge_buffer × threshold sweep; F1 collapse at EB=100 confirmed.

### Saved artifacts (do not delete)

- `models_test/cnn_breakpoint_run28d_final.keras` — boundary-shortcut model, raw F1=0.308 / honest F1=0.171. Comparison baseline.
- `models_test/cnn_breakpoint_run29_final.keras` — BN + edge-aware. Honest F1=0.174.
- `models_test/cnn_breakpoint_run30_final.keras` — LN + edge-aware. Honest F1=0.170.
- `models_test/cnn_breakpoint_run31_final.keras` — U-Net 24M params. Honest F1=0.143 (REGRESSED).
- `models_test/cnn_breakpoint_run32_final.keras` — dilated, n_filters=128, LN, edge-aware. Honest F1=0.175 — **best individual model**.
- `models_test/cnn_breakpoint_run33_final.keras` — same as #32 with max_files=80 train. Honest F1=0.151 (REGRESSED).
- `models_test/cnn_breakpoint_run34_final.keras` — same as #32 with position-shift aug. Honest F1=0.171.
- `eval_ensemble.py` — 4-model averaging eval. Honest F1=0.174. Ensemble doesn't lift over best individual — models make correlated errors.
- `cache/ds_UnseenTestSet_*.npz/.pkl` — canonical-parser test cache. Saves 10 min per eval.
- `cache/ds_XML-1..5_*.npz/.pkl` — train+val caches.
- `eval_only.py` — streaming val+test eval via from_generator (raw, no EB). **Use eval_run29.py instead — it does the honest EB=200 sweep automatically.**
- `eval_run29.py` — preferred eval. EB×threshold sweep on val+test, picks val-best, prints honest F1 vs RDP5. Outputs `results_<tag>.json`. Usage: `eval_run29.py <model.keras> <run_tag>`.
- `train_only.py` — exec's notebook cells 1-22 only, exits cleanly. Bypasses leaky cells 23-40. **Use this instead of run_notebook.py for training.**
- `eval_edge_suppressed.py` — sweeps edge_buffer × threshold to estimate honest interior F1. Run alongside any inference for a more truthful number.
- `bucket_diagnostic.py` — per-position-bucket detection rate. Use to detect boundary shortcuts in any future model.
- `sanity_calibration.py` — val vs test prediction-magnitude comparison.
- `cache_test_set.py` — exec's notebook parser to populate UnseenTestSet cache.
- `monitor/wsl_log.sh` + `C:\Users\joshc\wsl_monitor\win_log.ps1` — start before any future training run.

### Operational discipline that prevented this from getting worse

- **Don't run Bash status checks during training danger windows.** Each `bash -c ...` from the agent forks a shell from the 27 GB Python parent — page-table COW costs ~100 MB per fork. Use `tail -f` once or rely on the persistent monitors.
- **Always wrap training in `systemd-run --user --scope -p MemoryMax=24G`.** This is non-optional now. Replaces Vmmem deaths with clean Python kills.
- **Validate any new parser implementation against a cached pkl** before trusting downstream numbers. `eval_only.py` had its own `parse_simulation` that produced 2862 test events; the canonical parser produced 5539. The half that was missing came from `set()`-cast scrambling candidate ordering. The cached val pkl is ground truth — diff against it.

### Two bug fixes from this session — historical reference

1. **Parser column-shift bug (cell-10).** `pd.read_csv` was treating each `.faRecombIdentifyStats.csv` row's `Event` column as the row index because of trailing commas in data rows. Result: every column was shifted by one, the parsed `ISeqs` column was actually the float `ListCorr` column, and `parse_simulation` was building triplets with **random integers as parent IDs** — silently feeding garbage to the model on every run before #27. Fix: `index_col=False` on both `read_csv` calls. **Effect:** sim CSVs now yield ~26 events/file (vs ~0.4 before), so dataset sizes jumped ~67×. This invalidated all prior comparisons against the RDP5 classical baseline (5,671 events on UnseenTestSet); the new comparison set has the same denominator.

2. **WSL2 OOM crash from inflated dataset size (cell-11, cell-22, cell-40).** Run #27 hit a balloon-kill / kernel OOM at peak ~28-32 GB resident: X_train 8 GB fp32 + X_val 8.24 GB fp32 + tf.constant doubling under `from_tensor_slices` + `np.concatenate` doubling. WSL was raised to 28 GB but still tight. **Fix in run #28:**
   - cell-11 (`load_dataset`) casts X to **float16** on read (one-hot 0/1 channels exact in fp16, MaxChi disparity in [-1, 1] keeps ~3e-3 precision).
   - cell-22 adds `tf.data.Dataset.map(cast x → fp32)` so the model still trains in fp32 (no mixed precision).
   - cell-22 also `del X_val, y_val, w_val` after val pipeline construction (was only deleting train arrays).
   - cell-40 (test eval) drops `train_ds`, `val_ds` first, builds a fp16-numpy / fp32-cast tf.data pipeline for `cnn.predict` so X_test (5,539 events × 2.816 MB = 15.6 GB fp32) never materialises as fp32 in host RAM.
   - Existing fp32 caches still load (cast-on-read path); no re-parse needed.
   - Net: peak host RAM for run #28 should be ~12-14 GB, well under the 28 GB cap.

**WSL config:** `/mnt/c/Users/joshc/.wslconfig` is now `memory=28GB swap=12GB` (host has 32 GB). User raised it from 24 → 28 GB this session.

**TF stack:** TF 2.18.1 + CUDA on RTX 3070 (5.5 GB GPU). Working — do NOT upgrade. The OOM was a host-RAM budget issue, not a TF bug.

**Best result so far:** run #25 (test F1 = 0.218) on the **buggy parser** — model was trained with garbage parents but somehow eked out a small lift. Once #28 lands, that comparison number is no longer meaningful: the new training distribution has correct triplets, ~67× more events per file, and the test-set denominator changes too.

**RDP5 classical baseline:** F1 = 0.367 on UnseenTestSet (5,671 events from PredBPStart/PredBPEnd in faSimVSRealCompare.csv, ±200 bp tolerance). **This is the bar to beat.**

**Run #28 expectations:**
- Train events: 2,841 (XML-1..4, max_files=40 each).
- Val events: 2,925 (XML-5, max_files=200).
- Test events: 5,539 (UnseenTestSet, all files).
- If parser fix alone closes the gap to RDP5, run #29+ becomes about exceeding it. If still well below, next moves: bigger backbone, parent-swap aug (now meaningful — before, swap was no-op since parents were garbage), or a different architecture entirely (U-Net, Transformer-encoder per position). User said no CNN-only constraint.

---

## 11a. Current state (end of 2026-05-05 session — runs #17–#24)

**Status:** Run #24 KEPT (partial). 5-REVERTED stop-condition streak reset.

**Headline:** truncation hypothesis from the end-of-2026-05-05 diagnostic is **partially confirmed**. Test F1 lifted **0.172 → 0.191** (+0.019), test Both BPs 3.7% → 6.1%; val gains larger (F1 0.282 → 0.348, Both BPs 8.5% → 14.7%). Test F1 didn't crack the 0.25 line that would have rewritten the whole "0.172 ceiling" narrative — a second factor remains.

**Key learning from #24:** the data-implied POS_WEIGHT did **not** triple as the diagnostic anticipated. mean(y) over unmasked positions was 0.0125 at MAX_SEQ_LEN=32 k vs ~0.014 at 10 k — basically unchanged because the mask discards padding. Hardcoded POS_WEIGHT=178 over-weighted positives by ~2.25× vs the data-implied 79. Symptoms: best epoch=3 (very early plateau), train PR-AUC barely climbed, best threshold dropped 0.7 → 0.4 (predictions systematically inflated). **POS_WEIGHT correction is the obvious next single-axis change.**

**What's still active beyond #24:**
1. **Top-K axis closed** (runs #19–#21). Boundary-mask infra preserved per CLAUDE.md item #5. Don't revisit without upstream fix to BN+padding spike.
2. **Cross-config generalization gap narrowed but not closed.** Test F1 still trails val F1 by 0.157 (vs 0.110 in #16). Augmentation at the invariance axis (parent-swap, reverse-complement) and bigger backbone are still on the queue — but only re-attempt after POS_WEIGHT is corrected and we know what the *real* baseline at MAX_SEQ_LEN=32 k looks like.

**Pipeline infra built:**
- Disk cache for `load_dataset` (per-directory, keyed on MAX_SEQ_LEN / MAXCHI_WINDOWS / LABEL_SIGMA / BP_WINDOW) — currently warm for MAX_SEQ_LEN=32 k (~130 MB across 5 train dirs + UnseenTestSet).
- `tf.data.Dataset.from_tensor_slices` pinned to `tf.device('/CPU:0')` in cell-22 to keep bulk arrays off GPU.
- `del X_train, y_train, w_train` after dataset construction — prevents the numpy+TF doubling that OOM'd the 15 GB WSL VM at MAX_SEQ_LEN=32 k.
- WSL2 RAM cap raised 15 GB → 24 GB via `~/.wslconfig` (host has 32 GB).
- `dataRaw/` moved off the OneDrive symlink chain onto Linux ext4 (was the root cause of catastrophic vmmem-kill crashes early in this session).
- All cell IDs preserved.

**Best result is now run #24:** test F1 = 0.191, val F1 = 0.348, both improvements over #16. Models for runs #17-#24 in `models_test_backup/cnn_breakpoint_best.run{17..24}.keras`.

**Recommended #25:** drop POS_WEIGHT 178 → 70. Single-axis change. Same MAX_SEQ_LEN=32k baseline. If POS_WEIGHT was the over-weighting culprit: best epoch should land 5-10 (not 3), train PR-AUC should climb, and val/test F1 should rebalance toward higher precision. If test F1 jumps materially with the corrected POS_WEIGHT, we have a much cleaner read on what's left of the cross-distribution gap.

---

## 11b. Current state (end of previous session)

**Best configuration so far** (run #16):
- MAX_SEQ_LEN = 10000
- N_INPUT_CHANNELS = 22 (15 triplet one-hot + 3 comparison + 4 MaxChi)
- MAXCHI_WINDOWS = (50, 100, 200, 500)
- Residual dilated stack: kernel=7, dilations (1,2,4,8,16,32), residual connections via 1×1 projection on first block, post-activation, n_filters=64, dropout=0.5.
- Loss: weighted_bce(pos_weight=70.0).
- Optimiser: AdamW(lr=1e-4, weight_decay=1e-5), bias_initializer=Constant(-log(99)) on the final sigmoid.
- Train: XML-1..4, max_files=None.
- Val: XML-5 (held out, n=621 events).

**Headline metrics under run #16:**
| metric | val (XML-5) | test (UnseenTestSet) |
|---|---|---|
| F1 (peak-based) | 0.282 | **0.172** |
| Both BPs found | 8.5% | 3.7% |
| Mean peaks / event | 1.85 | 2.34 |
| Best threshold | 0.7 | (val-tuned) |

**The prior best on Both BPs is run #15** (val 18.2%, test 4.9%) at dropout=0.3. Run #16 traded Both BPs for higher per-peak precision and a +0.021 test F1 lift. The right move on the new machine may be to back off dropout (0.5 → 0.4) and see if you can recover Both BPs without losing the test F1 gain — or to attack the test set's distribution shift directly via top-K reformulation (see §12).

**Compute used in previous session:** 12 successful runs (~80–100 min each on an M4) plus a few killed/failed; ~24 hours total. Several runs hit the 7200s default cell-22 timeout when training under stronger regularisation (LayerNorm, run #10) — the timeout was bumped to 21600s in run_notebook.py.

## 12. What's staged for run #17 (RUN THIS FIRST)

Before the migration, cell-3 was edited to extend `MAXCHI_WINDOWS` from `(50, 100, 200, 500)` to `(50, 100, 200, 500, 1000, 2000)` and `N_MAXCHI_WINDOWS` from 4 to 6. The run was started but **killed before any epoch completed** (data prep is slower with 6 windows; the run hadn't even logged a val_aupr after 48 minutes).

**The notebook on disk has cell-3's source updated for #17 but cell-22's outputs are still from run #16 (4 MaxChi windows).** The cached `models_test/cnn_breakpoint_best.keras` is run #16's model.

When you start on the new machine:
1. Verify cell-3 still has the 6-window MAXCHI_WINDOWS staged.
2. Run the notebook (`python3 run_notebook.py`). This is run #17.
3. Append a log entry per §5. Use the template; the prior #16 entry is the comparison baseline.
4. Decide KEPT / REVERTED / INCONCLUSIVE per §6. Update TODO.md.

If #17 works (val Both BPs and/or test F1 improve), the next experiment is:
- **σ tuning paired with POS_WEIGHT scaling** (σ=10 with POS_WEIGHT≈140) — finally close the σ/POS_WEIGHT coupling story from run #9.
- Or **top-K coordinate regression head** if you want to take a bigger swing at the val/test distribution gap.

If #17 doesn't help (Both BPs flat or down), don't keep stacking MaxChi variants. Move to:
- **Top-K coordinate regression** — replace per-position binary classification with a structured K=2 (position, confidence) output. Hungarian matching or sorted-pair L1 loss. The previous agent debated this several times but kept choosing smaller swings; with the per-position approach plateaued at test F1 ~0.15–0.17, this is now the right move. Major refactor but cleanly motivated.
- **Masked BatchNorm** — compute BN stats only over valid (non-padded) positions. Custom layer needed. Targets the BN+padding boundary artefact directly.

## 13. Project queue (synced with TODO.md High Priority section)

1. **(STAGED)** Run #17: extended MaxChi windows {50, 100, 200, 500, 1000, 2000}.
2. **σ tuning paired with POS_WEIGHT** — re-attempt run #9 with paired POS_WEIGHT scaling.
3. **Top-K coordinate regression head** — strategic backstop if input-channel feature engineering plateaus.
4. **Masked BatchNorm** — addresses the BN+padding boundary artefact. Custom layer.
5. **Bigger multi-scale kernels** — `(7, 31, 63, 127)` at the input layer.

Keep the queue alive — when you exhaust it, generate new items from the **Why** column of recent KEPT/REVERTED entries (what they ruled in / out) and from the saliency/length-stratified diagnostics in cells 38, 45, 47.

## 14. Concrete first action

1. Activate the venv: `source .venv/bin/activate`. Make sure `tensorflow-metal` is installed if running on Apple Silicon — without it training falls back to CPU silently.
2. Verify the staged cell-3 (6 MaxChi windows). Decide whether to run as-is (recommended) or back off.
3. `python3 -u run_notebook.py > exec_log.txt 2>&1` — log run #17.
4. Read the result, update Experiment Log + TODO.md, decide.
5. Pick the next item from the queue. Apply one change. Loop.

Good luck.
