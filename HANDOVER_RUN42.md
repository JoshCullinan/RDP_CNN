# Handover to next agent — long-content data is incoming

> ⚠️ **Read this BEFORE acting.** The user is currently running fresh
> SANTA simulations at `<length>30000` on the Windows side to deliver
> long-content training data that the model has never seen. This file
> supersedes `HANDOVER_NEXT.md` for orientation (HANDOVER_NEXT.md is now
> stale on several points — see §6 below). Your job is to land the new
> data when it arrives, retrain, and lift honest test F1 past run #41's
> 0.335. If the data hasn't arrived yet, stop and ask — do not start
> speculative work.

---

## 0. TL;DR

- **Latest deployed model:** `models_test/cnn_breakpoint_run41_final.keras`.
  Apples-to-apples honest test F1 on UnseenTestSet = **0.335** (vs run #38
  0.313, vs RDP5 0.367 raw).
- **The single change that drove run #41's lift:** pooled stratified split
  across XML-1..5 + UnseenTestSet, so train and test both span 0–30k
  breakpoint content. Diagnosed bottleneck (commit eb8d116) is positive-
  label position-distribution shift; mixing a slice of long-content events
  into training narrows the gap. Only 14% of long-content files made it
  into training — the lift is therefore proportional and modest.
- **The user is generating MORE long-content data right now** at length
  30000 bp with proper parent-ID metadata. See `HANDOVER_RUN42_brief.txt`
  (the brief sent to the data-generation agent — paste copy below).
- **Your job:** when the new data lands, integrate it, retrain run #42,
  and measure how much further the test F1 climbs.

---

## 1. What's already in this branch / repo

### 1.1 Trained models on disk (gitignored; irreplaceable without retraining)

```
models_test/cnn_breakpoint_run41_final.keras   ⚠️ NEW BEST DEPLOYMENT (0.335)
models_test/cnn_breakpoint_run38_final.keras   ⚠️ KEEP (prior best, 0.313)
models_test/cnn_breakpoint_run36_final.keras   ⚠️ KEEP (best raw F1)
models_test/cnn_breakpoint_run40_final.keras   keep (SpliceAI-32 — regression)
models_test/cnn_breakpoint_run39_final.keras   keep (33-ch baseline)
models_test/cnn_breakpoint_run37_final.keras   keep
```

Do not overwrite the unversioned `cnn_breakpoint_final.keras` without
first copying it to a versioned name. ModelCheckpoint will silently
clobber it.

### 1.2 Cached datasets (gitignored)

Active caches relevant for run #42:
```
cache/ds_pool_run41_train_*.npz   # 4,000 events, 33 ch — POOLED, current
cache/ds_pool_run41_val_*.npz     # 1,500 events
cache/ds_pool_run41_test_*.npz    # 2,500 events (pooled — inflated metric)
cache/ds_UnseenTestSet_*.npz      # 5,539 events @ 33 ch (apples-to-apples test)
```

When you bring new data online, **bump `CACHE_VERSION` in cell-11** to
invalidate these and force regeneration.

### 1.3 Split lists (in git)

```
splits/run41_index.tsv     # per-file: rel_path, dir, n_events, max_bp
splits/run41_train.txt     # 4,469 files
splits/run41_val.txt       # 957 files
splits/run41_test.txt      # 961 files
```

Built by `build_pooled_split.py` — 70/15/15 stratified by max-BP bucket
[0-10k, 10-20k, 20k+]. Will need to be rebuilt to include the new long-
content data (and any other directory you add).

### 1.4 Critical notebook cells — current state

| Cell | What's there | Don't break |
|------|---|---|
| **cell-3** | `N_INPUT_CHANNELS=33`, `RDP_PRED_SIGMA=50`, `RDP_DROPOUT_P=0.1`, `RDP_BLOCK_START=22`, `RDP_BLOCK_END=33` | 24 base + 2 RDP-pred + 9 method-confidence. |
| **cell-6** | `encode_triplet` with `pred_bp_start`, `pred_bp_end`, `method_pvalues` args; all gracefully accept `None` | Don't remove None-handling. |
| **cell-10** | `parse_simulation` requires `<base>.faSimVSRealCompare.csv` AND `<base>.faRecombIdentifyStats.csv`. **Returns `[]` silently if either is missing.** | This is the bottleneck for the new data — see §3 below. |
| **cell-11** | `load_dataset` (legacy per-dir) + `load_filelist_dataset` (new pooled-split path). `CACHE_VERSION='v5_pool'`. | Bump version when adding new data. |
| **cell-12** | Reads `splits/run41_{train,val}.txt`, caps at `TRAIN_MAX_EVENTS=4000`, `VAL_MAX_EVENTS=1500`. | Update to read run #42 splits when ready. |
| **cell-13** | Variable plumbing (X_train, w_train, etc.). | No real changes needed. |
| **cell-18** | Run #38 architecture (residual dilated stack, LN, 128 filters). Reverted from run #40's SpliceAI-32. | Keep this baseline. |
| **cell-22** | Save name → `cnn_breakpoint_run41_final.keras`. RDP-dropout zeroes channels [22, 33) jointly per sample at p=0.1. | Bump save name to `run42` before launching. |
| **cell-experiment-log** | Append-only run notes. Most recent entry: run #41. | Append #42 after eval, don't rewrite history. |

---

## 2. What the user is generating right now

Per the brief I wrote on 2026-05-11 (full text in §7 below), the user
has SANTA installed on the Windows side at
`C:\Users\joshc\OneDrive - University of Cape Town\University\Masters\RDP-ML-REDUX\santaSim\`
and is generating:

- **≥300 simulations** (target 500–1000)
- **`<length>30000`** (vs the previous batch's hardcoded 10000)
- Same recombination/mutation rates as XML-1..5 (the user is matching them)
- Each simulation produces **three files**: `<base>.fa`,
  `<base>.faSimVSRealCompare.csv`, and **`<base>.faParents.csv`**
  (a new 3-column CSV `Event,Parent1,Parent2` emitted directly from
  SANTA's simulation log — this is the key innovation that lets us
  use SANTA-generated data without RDP5 running on it).

Output destination is a new directory (likely `long_content_30k_001/` or
similar) under `C:\Users\joshc\Dev\Student Sims\` on the Windows side.
Confirm with the user where they actually dropped it before copying.

### 2.1 BP-position requirement

Validate that the new data actually has long-position breakpoints
before training. From `splits/run41_index.tsv` we know XML-1..5 max BP
is 19,025 and UnseenTestSet is up to 30,238. The new data should have
**≥30% of breakpoints past position 20,000** to materially help. Quick
check after copy:

```bash
python3 -c "
import csv, glob
n=0; long20k=0; long25k=0
for csvp in glob.glob('dataRaw/<new_dir>/*.faSimVSRealCompare.csv'):
    with open(csvp) as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            try:
                bp = max(int(float(row['SimBPStart'])), int(float(row['SimBPEnd'])))
            except (TypeError, ValueError, KeyError):
                continue
            n += 1
            if bp >= 20000: long20k += 1
            if bp >= 25000: long25k += 1
print(f'{n} events, {long20k}/{n} past 20k ({long20k/n:.1%}), {long25k}/{n} past 25k')
"
```

If the fraction past 20k is <10%, **stop and tell the user** — the
generation params didn't take effect. Don't waste a training run on
mis-shaped data.

---

## 3. Required notebook change before run #42 can train

`parse_simulation` (cell-10) currently requires
`<base>.faRecombIdentifyStats.csv` (an RDP5 output). The new data won't
have it — instead it will have `<base>.faParents.csv` with columns
`Event,Parent1,Parent2`. **You need to teach `parse_simulation` to
prefer `faParents.csv` when it exists, falling back to
`faRecombIdentifyStats.csv` for the legacy XML-1..5 data.**

Sketch of the change (cell-10):

```python
def parse_simulation(fasta_path):
    fasta_path = Path(fasta_path)
    sim_csv = fasta_path.parent / f"{fasta_path.name}SimVSRealCompare.csv"
    stats_csv = fasta_path.parent / f"{fasta_path.name}RecombIdentifyStats.csv"
    parents_csv = fasta_path.parent / f"{fasta_path.name}Parents.csv"  # NEW
    fa_csv = fasta_path.parent / f"{fasta_path.name}.csv"

    if not sim_csv.exists():
        return []
    if not stats_csv.exists() and not parents_csv.exists():
        return []  # need at least one source of parent IDs

    # ... existing FASTA + sim parse ...

    parents_by_event = {}  # event -> (parent1_id, parent2_id) or None
    if parents_csv.exists():
        # NEW path: read Event,Parent1,Parent2 directly
        pdf = pd.read_csv(parents_csv, skipinitialspace=True)
        for _, r in pdf.iterrows():
            try:
                ev = int(r['Event']); p1 = int(r['Parent1']); p2 = int(r['Parent2'])
                parents_by_event[ev] = (p1, p2)
            except (TypeError, ValueError, KeyError):
                continue
    elif stats_csv.exists():
        # LEGACY path: existing 3-hypothesis logic, unchanged
        stats = pd.read_csv(stats_csv, skipinitialspace=True, index_col=False)
        # ... existing code that derives parent_ids from ISeqs(A) ...

    # In the per-event loop, replace:
    #   if len(parent_ids) < 2: continue
    # with:
    #   if event in parents_by_event:
    #       p1, p2 = parents_by_event[event]
    #       parent_ids = [p1, p2]
    #   else:
    #       # legacy 3-hypothesis path
    #       ...
```

After this change, the legacy XML-1..5 data still works (no `faParents.csv`
present → falls through to existing logic) AND the new long-content data
works via the new path. **Bump `CACHE_VERSION` since you changed parsing
semantics.**

You should also test the parser change on ONE legacy file and ONE new
file before bulk-running to confirm both paths yield identical event
counts to what existed before.

---

## 4. Workflow for run #42 (long-content data integrated)

The full cycle:

1. **Confirm new data has landed.** Ask the user, or check
   `C:\Users\joshc\Dev\Student Sims\<new_dir>\`. Copy into a fresh
   `dataRaw/XML-7/` (or whatever name you/user prefer). Run the BP
   position-sanity check from §2.1.

2. **Patch `parse_simulation` (cell-10)** to handle `faParents.csv`
   (see §3 sketch). Test on one legacy file + one new file.

3. **Bump `CACHE_VERSION` in cell-11** (e.g., 'v5_pool' → 'v6_long30k').
   This invalidates the run #41 caches.

4. **Rebuild the pooled split** with `build_pooled_split.py`. Edit the
   `DIRS` list to include the new directory. Re-run. Output new split
   files at `splits/run42_*.txt` (don't overwrite run #41's lists).

5. **Update cell-12** to read `splits/run42_{train,val}.txt`. Decide
   whether to bump `TRAIN_MAX_EVENTS` from 4,000 (the run #38 baseline
   memory budget) — if you can fit 5,500-6,500 events under the 24 GB
   cgroup cap, do it. Memory budget is in cell-12 comments.

6. **Bump model save name in cell-22** → `cnn_breakpoint_run42_final.keras`.

7. **Train under cgroup** (mandatory — WSL2 crashes without it):
   ```bash
   TS=$(date +%Y%m%d_%H%M%S)
   LOG=exec_log_run42_${TS}.txt
   nohup systemd-run --user --scope --quiet \
     -p MemoryMax=24G -p MemorySwapMax=12G \
     bash -c "source .venv/bin/activate && python3 -u train_only.py" > "$LOG" 2>&1 &
   echo $! > /tmp/train.pid
   ```
   Use the `Monitor` tool (not Bash polling — fork cost from the ~24 GB
   Python parent burns memory).

8. **Cache the new pooled test set.** Adapt `cache_pooled_test.py` to
   point at `splits/run42_test.txt` and run under cgroup.

9. **Eval against the new test split AND apples-to-apples on UnseenTestSet.**
   `eval_run41.py` and `eval_run41_on_unseentestset.py` are the
   templates — duplicate, point at run #42 caches/model, run both. The
   apples-to-apples-on-UnseenTestSet number is the headline.

10. **Append run #42 entry to cell-experiment-log.** Use run #41's
    entry as the template. Commit. Open PR.

---

## 5. Expected outcome — what to look for

If the data generation worked and the parser change is correct:

- **Honest test F1 on UnseenTestSet should rise above 0.335.** The
  exact ceiling depends on how many long-content events end up in
  training. Rough Fermi: run #41 trained on ~16 long-content files
  (out of 67 in train), lift = +0.022. If run #42 has 300+ new long-
  content files entering training, the lift should be 5–10× larger.
  Best case: honest F1 in the 0.40–0.50 range, comfortably beating
  RDP5's 0.367.
- **Recall should jump more than precision.** Run #41 saw recall
  0.31 → 0.39 with precision 0.32 → 0.29. Same pattern expected.
- **POS_WEIGHT may need retuning.** Hardcoded at 70; implied on the
  run #41 pooled split was 298. The longer content + more events
  shifts the class balance further. Worth a one-axis sweep (70, 150,
  220, 300) after the data-only run lands.

If the test F1 doesn't move or regresses, suspect:
1. Parser bug — the new data isn't actually yielding events. Print
   `len(parse_simulation(<one new file>))` to verify.
2. Data shape — BPs aren't spread to 30k. Re-run the §2.1 check.
3. POS_WEIGHT under-weighting positives.

---

## 6. What's stale in HANDOVER_NEXT.md

Don't trust these claims in `HANDOVER_NEXT.md` (written after run #38):

- "Highest expected impact still on the table: A1 — 9 method p-value
  channels." → **REVERTED.** Run #39 added these and it was flat on
  test (commit 1deba04). Skip A1.
- "SpliceAI-32 backbone (A2) is the more promising one per the
  literature search." → **REVERTED.** Run #40 tested SpliceAI-32 and
  regressed (-0.033 honest F1). Cell-18 has been put back to run #38's
  dilated stack.
- "Run #38 is the best deployment model with honest F1 = 0.313." →
  **Outdated.** Run #41 superseded it at 0.335. New deployment baseline
  is `cnn_breakpoint_run41_final.keras`.
- "Stop-condition: 3 runs below run #38." → run #41 is above, so reset
  the counter.

The rest of HANDOVER_NEXT.md (memory rules, cgroup launcher, parser
fix, cache invalidation rules, `eval_run29.py`-style harness) is still
correct and load-bearing.

---

## 7. The data-generation brief (copy for context)

The brief I sent to the Windows-side data-generation agent on
2026-05-11. Reproduced here so you know exactly what's coming:

> Generate ≥300 SANTA simulations with `<length>30000` and matching
> XML-1..5 recombination/mutation rates. Each simulation must produce
> `.fa + .faSimVSRealCompare.csv + .faParents.csv` (the last one is a
> 3-column `Event,Parent1,Parent2` CSV emitted directly from SANTA's
> simulation log; this replaces the RDP5-derived
> `.faRecombIdentifyStats.csv` the prior batch lacked). Drop output in
> a fresh `Student Sims\long_content_30k_001\` directory. Validate:
> BPs distributed across 0–30k with ≥30% past 20k, triplet completeness,
> file count ≥300. No RDP5 outputs required.

If the new files use a different column-naming convention than
`Event,Parent1,Parent2`, adapt §3's parser sketch accordingly. The
guarantee is "some per-event mapping from recombinant ID to its two
parents, in a parseable file alongside the `.fa`."

---

## 8. Stop conditions

Stop and ask the user if:

- The new data hasn't landed yet — don't speculate or try to start
  ahead of it.
- The new data's BP-position distribution looks wrong (§2.1 check
  fails). Don't burn a training run on bad data.
- Your parser change for `faParents.csv` causes ANY legacy file to
  yield a different event count than it did before. The legacy
  XML-1..5 data must continue to parse identically.
- The new data uses a file naming convention or schema you didn't
  expect. Confirm before adapting blindly.

---

## 9. Files to NOT touch destructively

- `cell-experiment-log` — append only.
- `models_test/cnn_breakpoint_run4[01]_final.keras` — the breakthroughs.
- `models_test/cnn_breakpoint_run3[6789]_final.keras` — historical baselines.
- `cache/ds_pool_run41_*` and `cache/ds_UnseenTestSet_*` — slow to regen.
- `splits/run41_*.txt` — preserved for reproducibility.
- `dataRaw/XML-1..5/` and `dataRaw/UnseenTestSet/` — source data.
- `dataRaw/XML-6/` — gitignored but on disk; the failed 10k-bounded
  dump from 2026-05-09. Won't contribute training events without
  parent metadata. The user has indicated XML-6 is not the priority
  to revisit; the new 30k batch (XML-7-or-similar) is.

---

Last updated: 2026-05-11. Author: Claude (Opus 4.7 1M context).
Predecessor: `HANDOVER_NEXT.md` (run #38 era, now partially stale).
Branch / PR: `cnn-iteration-runs-28-41` →
https://github.com/JoshCullinan/RDP_CNN/pull/new/cnn-iteration-runs-28-41
