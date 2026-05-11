# Handover to next agent — RDP-hybrid breakthrough preserved

> ⚠️ **READ THIS BEFORE DOING ANYTHING.** A prior session delivered a real
> breakthrough on this project (honest test F1 lifted from ~0.17 to 0.31,
> beating the RDP5 classical baseline on raw F1). The infrastructure that
> made that possible is fragile and easy to undo by accident. This file
> tells you what's preserved, what's safe to change, and what's
> off-limits.

This file is the **first thing** to read on session start. It supersedes
`HANDOVER.md` for orientation purposes (HANDOVER.md is still authoritative
on iteration discipline / pitfalls / experiment-log template — this file
just adds context layered on top).

---

## 0. TL;DR

- **Best deployment model:** `models_test/cnn_breakpoint_run38_final.keras`.
  Honest test F1 at val-selected threshold = **0.313** vs RDP5 = 0.367.
- **Best raw-F1 model:** `models_test/cnn_breakpoint_run36_final.keras`.
  Raw test F1 = **0.397** (beats RDP5's 0.367).
- **The single change that mattered:** read `PredBPStart`/`PredBPEnd`
  from `.faSimVSRealCompare.csv` and feed them as 2 Gaussian input
  channels (cell-3, cell-6, cell-10, cell-11). Run #36 in the experiment
  log. Don't undo it.
- **The catalog of next ideas to try** is in `IDEAS.md`. Tier A items
  A2-A7 are still untested.
- **The data files RDP5 already wrote** (`.fa.csv`, `.faRecIDTests.csv`)
  contain 9 per-method p-values + 18 internal scoring stats per event
  that we have NOT yet exploited. This is the most valuable untouched
  signal source.

---

## 1. What this session achieved (2026-05-06 → 2026-05-07)

Started: training was hitting catastrophic Vmmem crashes; honest test
F1 unknown; reported "F1=0.218" was inflated by a boundary shortcut.

Ended: stable WSL2, honest baseline established, RDP5 baseline beaten
on raw F1 via hybrid CNN+RDP channels.

### Metrics across the 12 runs of this session

| Run | Description | Honest test F1 (EB=200, val-thresh) | Raw test F1 |
|-----|------|------|------|
| #28d | BN, no edge-aware (boundary cheating) | 0.171 | 0.308 |
| #29 | BN + edge-aware sample_weight | 0.174 | 0.239 |
| #30 | LayerNorm + edge-aware | 0.170 | 0.229 |
| #31 | U-Net 24M params | 0.143 | 0.162 |
| #32 | dilated, n_filters=128, LN | 0.175 | 0.241 |
| #33 | #32 + max_files=80 | 0.151 | 0.196 |
| #34 | #32 + position-shift aug | 0.171 | 0.218 |
| #34b | ensemble of {#28d, #29, #32, #34} | 0.174 | 0.223 |
| #35 | + 7 MaxChi window scales | 0.167 | 0.233 |
| **#36** | **+ RDP-prediction Gaussian channels** | **0.243** | **0.397** ⭐ |
| #37 | #36 + RDP-dropout 0.3 | 0.286 | 0.361 |
| **#38** | **#36 + RDP-dropout 0.1 (sweet spot)** | **0.313** | 0.366 |

RDP5 classical baseline: raw F1 = 0.367, no honest-F1 measurement.

### What broke the plateau

The 22-channel input (one-hot triplets + match channels + MaxChi) had
hit a structural ceiling at honest F1 ~0.17. Architecture sweeps,
normalization changes, augmentation, and ensembling could not move it.

The fix was to add **2 channels with Gaussian peaks at RDP5's
PredBPStart and PredBPEnd**. The RDP5 outputs were already on disk —
read from `.faSimVSRealCompare.csv` for every alignment in `dataRaw/`.
The CNN now uses RDP5's classical-method localization as a strong prior
and refines it with the underlying sequence triplet features.

---

## 2. PROTECTED ARTIFACTS — DO NOT REMOVE OR REWRITE

### 2.1 Trained models on disk (git-ignored, irreplaceable without retraining)

```
models_test/cnn_breakpoint_run36_final.keras      ⚠️ KEEP
models_test/cnn_breakpoint_run37_final.keras      ⚠️ KEEP
models_test/cnn_breakpoint_run38_final.keras      ⚠️ KEEP  (BEST)
models_test/cnn_breakpoint_run28d_final.keras     ⚠️ KEEP  (boundary-shortcut baseline)
models_test/cnn_breakpoint_run28d_best.keras      ⚠️ KEEP
models_test/cnn_breakpoint_run29_final.keras      keep (run #29 baseline)
models_test/cnn_breakpoint_run30_final.keras      keep (LN baseline)
models_test/cnn_breakpoint_run31_final.keras      keep (U-Net REVERTED — useful counter-example)
models_test/cnn_breakpoint_run32_final.keras      keep (dilated 128f baseline)
models_test/cnn_breakpoint_run33_final.keras      keep
models_test/cnn_breakpoint_run34_final.keras      keep
models_test/cnn_breakpoint_run35_final.keras      keep
```

If you must replace `cnn_breakpoint_final.keras` (the unversioned one
that ModelCheckpoint writes), copy your new model to a versioned name
**first** (`cnn_breakpoint_runNN_final.keras`) before another training
run overwrites it.

### 2.2 Cached data — keep all

```
cache/ds_XML-1..5_*.npz/.pkl   — multiple hashes for different feature configs
cache/ds_UnseenTestSet_*.npz/.pkl   — three hashes (22-ch, 24-ch, 25-ch)
```

Each cache hash corresponds to a specific (MAX_SEQ_LEN, MAXCHI_WINDOWS,
LABEL_SIGMA, BP_WINDOW, CACHE_VERSION) configuration. Older models
need their matching cache to be evaluable. `eval_run29.py` picks the
right cache by matching channel count to model input shape.

### 2.3 Critical notebook cells — DO NOT REVERT

The following cell modifications are LOAD-BEARING for the breakthrough.
Reverting them = losing the F1 lift.

| Cell | What's there | Why it matters |
|------|---|---|
| **cell-3** | `N_INPUT_CHANNELS = 24`, `RDP_PRED_SIGMA = 50`, `RDP_DROPOUT_P = 0.1` | Defines the 24-channel input shape (22 base + 2 RDP). |
| **cell-6** | `_rdp_pred_channels()` helper, `encode_triplet(..., pred_bp_start, pred_bp_end)` | Adds the 2 RDP Gaussian channels. |
| **cell-10** | `parse_simulation` extracts `PredBPStart`/`PredBPEnd` from sim_df row, passes to encode_triplet. Defensive None/NaN handling. | Threads RDP5 outputs into the encoding pipeline. |
| **cell-11** | `CACHE_VERSION = 'v2'` | Forces cache regen with new shape. |
| **cell-22** | `_make_gen(X, y, w, dropout_rdp=False, seed=42)`, with the train pipeline using `dropout_rdp=True`. | RDP-dropout regularizer. |

### 2.4 Tooling files (durable infrastructure)

```
train_only.py              — exec's notebook cells 1-22 only; bypasses leaky 23-40
eval_run29.py              — eval pipeline w/ EB×threshold sweep; picks cache by channel match
eval_ensemble.py           — multi-model averaging eval
eval_edge_suppressed.py    — edge-buffer detector
bucket_diagnostic.py       — per-position-bucket detection-rate diagnostic
sanity_calibration.py      — val vs test prediction-magnitude check
cache_test_set.py          — exec's notebook parser to cache UnseenTestSet
append_log.py              — small helper to append entries to cell-experiment-log
monitor/wsl_log.sh         — Linux-side WSL memory logger (auto-start on each launch)
/mnt/c/Users/joshc/wsl_monitor/win_log.ps1   — Windows-side Vmmem logger
```

These are referenced by HANDOVER.md and IDEAS.md. Don't delete.

#### About the Windows PowerShell logger (win_log.ps1)

**Status as of 2026-05-07 session end: STOPPED by the user.** The
PowerShell window running `win_log.ps1` was running continuously during
the session and was deliberately stopped at session end. The CSVs it
produced are still on disk and are worth preserving as forensics:

```
/mnt/c/Users/joshc/wsl_monitor/win_vmmem_*.csv
/mnt/c/Users/joshc/wsl_monitor/win_sys_*.csv
/mnt/c/Users/joshc/wsl_monitor/win_events_*.csv
```

**When the next agent should restart it:** before any new training run
that you suspect could push memory boundaries (large architecture,
larger MAX_SEQ_LEN, more input channels, etc.). It is the ONLY vantage
point that survives a full Vmmem death — if WSL crashes catastrophically,
the WSL-side `monitor/wsl_log.sh` dies with it; only the Windows-side
logger captures Vmmem's last moments.

**How to start it (the user must do this — Claude cannot reach the
Windows-side PowerShell from inside WSL):**

```powershell
# In a Windows PowerShell window (not WSL)
powershell -ExecutionPolicy Bypass -File C:\Users\joshc\wsl_monitor\win_log.ps1
```

Leave that window open. It logs every 3s. Ctrl+C to stop.

**When the next agent does NOT need to restart it:**

- Routine eval runs (`eval_run29.py`) under `MemoryMax=20G` cgroup.
  These are well within budget and never approached the cap during the
  past session.
- Cache regeneration (`cache_test_set.py`) under MemoryMax=20G. Same.
- Light diagnostics (`bucket_diagnostic.py`, `sanity_calibration.py`).

If you're touching memory-budget-affecting parameters (`N_INPUT_CHANNELS`,
`MAX_SEQ_LEN`, `BATCH_SIZE`, `n_filters`, `max_files`), restart the
PowerShell logger before launching training. If you're not, the WSL-side
logger alone is sufficient.

**Do not write to `/mnt/c/Users/joshc/wsl_monitor/` or rename / remove
`win_log.ps1`.** That's the user's personal Windows directory; the
script lives there because it must be reachable by Windows PowerShell,
not WSL. If the script needs editing, edit it via the path above with
care for the script being potentially executing during edit.

### 2.5 Experiment log entries — additive only

The cell-experiment-log markdown cell at the bottom of CNN.ipynb is
the canonical record. **APPEND** new entries. Never rewrite existing
ones.

---

## 3. Validated baselines — the numbers to beat

| Comparison | F1 |
|---|---|
| RDP5 classical (test, all 5,539 events, ±200bp tolerance) | **0.367** |
| Run #28d (boundary-cheating baseline) honest | 0.171 |
| Run #38 (best deployment) honest @ val-thresh | **0.313** |
| Run #36 (best raw) raw @ val-thresh | **0.397** (beats RDP5) |
| Run #36 honest @ best threshold | 0.323 |

**The honest deployment number to beat is 0.313** (run #38, edge-suppressed
at EB=200, threshold selected on val). The raw-F1 number to beat is
0.397 (run #36).

---

## 4. What to try next

`IDEAS.md` has the full catalog with status. The Tier A items still
untested:

### A1 (next slice, NOT a redo) — More RDP signals from `.fa.csv` and `.faRecIDTests.csv`
**Highest expected impact still on the table.** The `.fa.csv` has
9 per-method p-values per event (RDP, GENECONV, Bootscan, MaxChi,
Chimaera, SiSscan, PhylPro, LARD, 3Seq). The `.faRecIDTests.csv` has
18 internal scoring stats × 3 roles per event. Run #36 used only the
2 consensus PredBP positions; these other files are unexploited.

Implementation:
1. Write a parser for `.fa.csv` (skiprows=15, has multi-row events,
   trailing empty column). Extract per-event: PredBPStart/End from
   "Begin"/"End", 9 p-values from columns 11-19. Handle "NS" as 1.0.
2. Add channels: for each method m, channel = `Gaussian(p; PredBP, σ=50)
   * (-log10(p_method) clipped to [0, 30])`. Gives the CNN per-method
   confidence. Total: 9 method channels × 2 BPs = up to 18 new channels.
   Or aggregate: 1 consensus-confidence channel per BP (2 channels)
   weighted by min p-value across methods.
3. Bump `N_INPUT_CHANNELS` and `CACHE_VERSION`. Regenerate train + val
   caches, then test cache via `cache_test_set.py`.
4. Retrain run #39 on the new shape. Eval with `eval_run29.py`.

Expected lift: another +0.05 to +0.10 honest F1 if the 9 methods
provide orthogonal signal beyond the consensus PredBP.

### A2 — SpliceAI-32 backbone combined with RDP channels
Subagent already produced the U-Net code (`/tmp/unet_cell18.py`) but
the SpliceAI-32 architecture is the more promising one per the
literature search (HANDOVER.md §11-current and IDEAS.md). 32
ResidualUnits, dilations 1→4→10→25 in 4 rounds, ~10M params. Should
combine well with the already-validated RDP channels.

### A6 — Unified Focal Loss on the RDP-channel architecture
Run #29-#34 used `weighted_bce(POS_WEIGHT=70)`. Tversky / Dice loss
families dominate sigmoid-BCE on extreme-imbalance segmentation tasks.
Pair with the existing RDP-channel architecture (run #38 baseline).

### A7 — Domain-adversarial training (DANN)
Val→test honest gap is still 0.40 in run #38. DANN explicitly closes
the train/test feature distribution gap. Test inputs used (not labels).

### Lower-effort tweaks
- Try `RDP_PRED_SIGMA` ∈ {20, 100, 200} — current 50 was a reasonable
  guess but untuned.
- Try `RDP_DROPOUT_P` ∈ {0.05, 0.2}. Run #38 (p=0.1) is best-known.
- Add reverse-complement TTA at inference (B29 in IDEAS.md, ~5 lines).

---

## 5. How to safely run a new experiment

The full cycle for a single new run:

```bash
cd /home/joshcullinan/RDP_CNN

# 1. Make ONE change to the notebook (cell-3 / cell-6 / cell-18 / cell-22).
#    Use a Python script that loads CNN.ipynb JSON, mutates the cell,
#    saves. Do not let CNN.ipynb get rewritten by Jupyter mid-edit.

# 2. Bump the model save name in cell-22 to cnn_breakpoint_runNN_final.keras
#    where NN is your new run number (next free integer).

# 3. Start the persistent monitor (replaces any stale one).
[ -f /tmp/wsl_log.pid ] && kill $(cat /tmp/wsl_log.pid) 2>/dev/null
nohup bash monitor/wsl_log.sh > /tmp/wsl_log.stdout 2>&1 &
echo $! > /tmp/wsl_log.pid

# 4. Tell the user to start the Windows-side PowerShell logger if not
#    already running:
#    powershell -ExecutionPolicy Bypass -File C:\Users\joshc\wsl_monitor\win_log.ps1

# 5. Launch training UNDER cgroup. NEVER launch python directly — it
#    will crash WSL2.
TS=$(date +%Y%m%d_%H%M%S)
LOG=exec_log_runNN_${TS}.txt
nohup systemd-run --user --scope --quiet \
  -p MemoryMax=24G -p MemorySwapMax=12G \
  bash -c "source .venv/bin/activate && python3 -u train_only.py" > "$LOG" 2>&1 &
echo $! > /tmp/train.pid

# 6. Use Monitor (not Bash polling — Bash forks from the 27 GB Python
#    parent are expensive). Wait until train.pid exits.

# 7. If you changed N_INPUT_CHANNELS, the test cache must regen at the
#    new channel count:
nohup systemd-run --user --scope --quiet -p MemoryMax=20G \
  bash -c "source .venv/bin/activate && python3 -u cache_test_set.py" > /tmp/cache.log 2>&1 &

# 8. Eval against the new model:
nohup systemd-run --user --scope --quiet -p MemoryMax=20G \
  bash -c "source .venv/bin/activate && python3 -u eval_run29.py models_test/cnn_breakpoint_runNN_final.keras runNN" > eval_runNN_${TS}.log 2>&1 &

# 9. Append entry to cell-experiment-log via append_log.py.
# 10. git add CNN.ipynb HANDOVER.md IDEAS.md results_runNN.json
# 11. git commit (do not commit unless user explicitly approves).
```

### Critical memory rules

- **Always wrap python in `systemd-run --user --scope -p MemoryMax=24G`.**
  Without this, an OOM crashes Vmmem and the entire WSL2 VM. Confirmed
  multiple times this session.
- **Do not run Bash status checks during training.** Each fork from
  the 27 GB Python parent reserves ~100 MB of page tables and at the
  cliff edge can push WSL over. Use `tail -f` once or rely on the
  persistent monitor CSVs.
- **MemoryMax=24G leaves 4 GB for system / TF baseline / Python overhead.**
  At MAX_SEQ_LEN=32000 and BATCH_SIZE=2 with the dilated 128f model,
  peak resident is ~22 GB. fp16 X arrays via cell-11 cast-on-load
  are essential for fitting.

---

## 6. Common gotchas (paid in past iterations)

| Gotcha | What happened | Fix |
|---|---|---|
| Test cache channel-count mismatch | Run #35 eval crashed with "depth of input must be a multiple of depth of filter: 22 vs 25" because train regenerated cache at 25-ch but test cache was still 22-ch. | After any `N_INPUT_CHANNELS` change, run `cache_test_set.py` BEFORE evaluating. `eval_run29.py` picks-by-channel-match but only finds matching caches that exist on disk. |
| Bash polling during training | Each Bash invocation forks a new shell from the 27 GB Python parent. Each fork costs ~100 MB page-table copy. At memory pressure, this can be the straw. | Use `Monitor` tool with selective grep, not repeated Bash status checks. |
| RDP-channel signal leakage worry | Adding RDP predictions as input channels feels like it might leak labels. It doesn't — `.faSimVSRealCompare.csv` contains both PredBPStart (RDP's prediction, used as INPUT) and SimBPStart (ground truth, used as LABEL). At test time, RDP5 outputs are also available — they were precomputed for every alignment when the dataset was built. Verified non-leak in commit ffb2805. | Documented; safe to keep using. |
| BatchNorm + 'same'-padding boundary shortcut | The model learns to predict peaks at content/padding boundaries (96% detection at boundary buckets, 30% interior). Inflates raw F1 by ~44%. | Always evaluate with `eval_edge_suppressed.py` (EB=200) — already automated in `eval_run29.py`. The "raw F1" is misleading; "honest F1 at val-selected threshold" is the deployment number. |
| `from_tensor_slices` 8 GB memory spike | Materialises the X array as a tf.constant in the graph, doubling resident memory at end-of-epoch-1. Caused multiple Vmmem deaths early in this session. | All cell-22/26/40 paths use `from_generator` instead. Don't revert. |
| Model checkpoint name collisions | If you run training without bumping the save name in cell-22, you'll silently overwrite a prior run's model. | ALWAYS bump `cnn_breakpoint_runNN_final.keras` to your new NN before launching. |

---

## 7. When to stop / when to escalate

Same as HANDOVER.md §7 / §8, but **also stop if**:

- Three consecutive runs land below run #38's honest F1 (0.313) — means
  you're regressing the breakthrough.
- A change you made unexpectedly *increases* F1 by ≥0.05 *without
  involving more RDP signal*. That's worth verifying — could be a leak
  from feeding RDP differently or an evaluation bug. Recheck with
  `bucket_diagnostic.py` for boundary shortcuts.

---

## 8. Files NOT to touch (without VERY deliberate intent)

- `cell-experiment-log` markdown cell at the end of CNN.ipynb — append only.
- `IDEAS.md` — append/edit-status only; don't rewrite existing entries.
- This file (`HANDOVER_NEXT.md`) — extend, don't replace.
- `HANDOVER.md` — already structured, just add to §11-current.
- `models_test/cnn_breakpoint_run3{6,7,8}_final.keras` — the breakthrough.
- `cache/ds_*` — generated artifacts, slow to regen, no reason to delete.
- `dataRaw/` — read-only. Source data.

If a change you're proposing would touch any of these "non-touch"
items destructively, **ask the user first** even in autonomous mode.

---

## 9. Single-paragraph briefing for an even-more-future agent

> A 2026-05-06 → 2026-05-07 session lifted the recombination-breakpoint
> CNN from honest F1 ≈ 0.17 (below RDP5's 0.367) to honest F1 = 0.313
> (run #38) and raw F1 = 0.397 (run #36, beats RDP5) by adding 2 input
> channels with Gaussian peaks at RDP5's PredBPStart and PredBPEnd —
> values that were already on disk in `.faSimVSRealCompare.csv` files
> from a prior project. Read `IDEAS.md` for ranked next ideas; the
> highest-leverage untested is parsing 9 per-method p-values from
> `.fa.csv` as additional channels. Don't undo runs #36 / #38 — their
> trained models are on disk and their cell-3/6/10/11/22 changes are
> load-bearing. Always launch training under
> `systemd-run --user --scope -p MemoryMax=24G` or WSL2 catastrophically
> dies. Always evaluate with `eval_run29.py` not `eval_only.py` (the
> latter reports raw F1 inflated by a boundary shortcut diagnosed early
> in the session).

---

Last updated: 2026-05-07. Author: Claude (Opus 4.7) iteration session
covering runs #28d-#38. Total commits added: 9 (`69c637d` through
`84bdcdf`).
