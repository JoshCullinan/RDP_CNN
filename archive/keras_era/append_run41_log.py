#!/usr/bin/env python3
"""Append run #41 entry to cell-experiment-log in CNN.ipynb."""
import json
from pathlib import Path

NB = Path('CNN.ipynb')
nb = json.loads(NB.read_text())

ENTRY = """

## Run #41 — Pooled stratified split (XML-1..5 + UnseenTestSet), run #38 arch

**Hypothesis (user-directed):** Run #39 diagnostics (commit eb8d116) showed
the val→test plateau is a positive-label position-distribution shift —
train BPs concentrated 0-10k, test BPs spread 0-30k with 37% past 20k.
Mixing the OOD long-content events (UnseenTestSet, 25-32k content) into
the training pool with a fresh stratified split should narrow the gap.

**Config:**
- Architecture: reverted cell-18 to run #38's residual dilated stack with
  LayerNorm, n_filters=128 (was SpliceAI-32 from run #40).
- Input channels: 33 (kept run #39's 9 method-confidence scalars; data
  asymmetry where UnseenTestSet has them but new XML-6 doesn't was
  absorbed by run #38's RDP-dropout=0.1 regularizer).
- Data: imported 900-file SANTA dump to `dataRaw/XML-6/` (XML-6), built
  pooled stratified split across {XML-1..6 + UnseenTestSet} at file level,
  70/15/15 with stratification by max-BP bucket [0-10k, 10-20k, 20k+].
- XML-6 SETBACK: `.faRecombIdentifyStats.csv` files are an RDP5 output and
  the dump never had RDP5 run. `parse_simulation` (cell-10) requires that
  file to get parent triplet IDs from `ISeqs(A)`, so every XML-6 file
  silently yielded 0 events. Run #41 effectively trains on
  {XML-1..5 + UnseenTestSet} pooled — no XML-6 scale-up.
- Cells changed: cell-11 (added `load_filelist_dataset`, bumped
  CACHE_VERSION='v5_pool'), cell-12 (read splits/run41_{train,val}.txt
  with max_events caps), cell-13 (renames only), cell-18 (reverted to
  run #38 arch), cell-22 (save name → run41_final).
- TRAIN_MAX_EVENTS=4000, VAL_MAX_EVENTS=1500, TEST_MAX_EVENTS=2500.
- POS_WEIGHT held at 70 despite implied=298 on the new split (longer
  content → more negatives per positive). Kept fixed for clean A/B vs #38.

**Training:** 27 epochs (EarlyStopping fired at epoch 27, restored best
epoch 12). Best val_aupr=0.390 on the pooled val set. Note: val_aupr is
NOT directly comparable to prior runs — pooled val has long-content
events that run #38's XML-5-only val didn't.

**Eval — apples-to-apples on UnseenTestSet (5,539 events, same as #38):**

| Metric | Run #38 | Run #41 | Δ |
|---|---|---|---|
| Raw F1 (EB=0) @ val-thr | 0.366 | 0.376 | +0.010 |
| **Honest F1 (EB=200) @ val-thr** | **0.313** | **0.335** | **+0.022** |
| Honest precision | ~0.32 | 0.294 | trade |
| Honest recall | ~0.31 | 0.389 | **+0.08** |
| Honest best across thresholds | — | 0.339 | — |

**Eval on pooled test split (961 files, mostly 0-20k content):**
- Honest F1 = 0.618 — INFLATED, do not cite as headline. Most of the
  pooled test is in-distribution; the apples-to-apples comparison is
  UnseenTestSet above.

**Verdict:** KEPT, modest but real lift. +0.022 honest F1 confirms the
position-shift diagnosis (eb8d116) — exposing the model to long-content
training events shifts recall from 0.31→0.39 with a small precision cost.

**Caveats / why the gain is small:**
- Only ~16 of 97 UnseenTestSet files (~14% of long-position events) made
  it into training via 70/15/15 split. The rest stayed in test.
- XML-6's 900 files (would have ~2-3× the existing training volume)
  contributed nothing due to missing `.faRecombIdentifyStats.csv`.
- POS_WEIGHT not retuned for the new data distribution (implied jumped
  to 298 vs held value 70) — under-weighting positives.

**Files:**
- Model: `models_test/cnn_breakpoint_run41_final.keras` (19 MB)
- Results: `results_run41_unseentestset.json` (apples-to-apples),
           `results_run41.json` (pooled test).
- Split lists: `splits/run41_{train,val,test}.txt`, `splits/run41_index.tsv`
- Caches: `cache/ds_pool_run41_{train,val,test}_*.npz/.pkl`
- Backup of pre-change notebook: `CNN.ipynb.bak_pre_run41`

**Next-step candidates ranked by expected lift:**
1. Recover XML-6 parent triplets — either user regenerates XML-6 with
   `RecombIdentifyStats` from SANTA (cheapest), or modify `parse_simulation`
   to fall back to synthetic parent selection. Unlocks ~3× training scale.
2. Oversample UnseenTestSet long-content in training (without
   reducing it in test) — augment the 67 long-content files with copies
   or interpolated variants. Quick to test.
3. Retune POS_WEIGHT for the new data distribution — 70 → 200 range.
   One-axis tweak, fast.
4. Bigger TRAIN_MAX_EVENTS — current 4,000 is held to run #38's parse-peak
   memory limit. With cgroup at 24G we could probably reach 5,500-6,000
   safely if parse is split into chunks.
"""

for c in nb['cells']:
    if c.get('id') == 'cell-experiment-log':
        existing = ''.join(c['source'])
        new_src = existing + ENTRY
        c['source'] = new_src.splitlines(keepends=True)
        break

NB.write_text(json.dumps(nb, indent=1))
print(f"Appended {len(ENTRY)} chars to cell-experiment-log")
