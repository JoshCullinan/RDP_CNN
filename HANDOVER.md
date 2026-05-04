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

## 11. Current state (end of previous session)

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
