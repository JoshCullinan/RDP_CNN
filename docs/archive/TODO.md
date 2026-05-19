# TODO

Tracking remaining improvements for `CNN.ipynb`. Items are roughly ordered
by expected impact / cost ratio. Tick `[x]` when shipped.

## Project goal (north star)

Beat MaxChi, RDP, and GeneConv on recombination-breakpoint detection,
and ultimately deploy on real HIV / other genomes. Compute cost is not
a constraint — favour expressive architectures over fast ones.

Two reframings are still pending and shape what's worth trying:

- **Benchmark vs MaxChi / RDP / GeneConv** on the same simulated test
  set (`UnseenTestSet/`). Until we have those numbers, "we're at F1=0.2"
  has no comparison anchor — RDP may score similarly, in which case
  the CNN is already useful as a complementary signal; or it may score
  much higher, in which case we have a clear ceiling to chase.
- **Long-term: recombinant identification, not just breakpoint location.**
  The current task assumes the recombinant is already identified
  (`ActualRecomb` from SANTA). Real deployment will need a model that
  takes 3 sequences and decides which (if any) is the recombinant —
  a four-way head `{seq1, seq2, seq3, none}` is one candidate
  framing. The current per-position output is a stepping stone toward
  that, not the final form.

## Already implemented

- [x] **Group-by-file train/val split** — events from the same SANTA run no
  longer leak across the split boundary.
- [x] **Explicit parent-comparison channels** (`match_p1`, `match_p2`,
  `informative`) — exposes the recombination signal directly instead of
  forcing the conv stack to rediscover it.
- [x] **Peak-based evaluation** with `scipy.signal.find_peaks` — counts true
  breakpoints, not above-threshold positions.
- [x] **Conv → BN → ReLU ordering** in `build_cnn`.
- [x] **PR-AUC instead of binary accuracy** as the live training metric.
- [x] Removed the unused `pos_weight` from the split cell.
- [x] **Soft Gaussian targets** (`LABEL_MODE='gaussian'`, σ=20) — replaces
  the hard 21-bp window. Each breakpoint becomes a smooth Gaussian peak
  in [0, 1], matching how `find_peaks` evaluates and giving the model a
  smooth gradient toward the true position. Was rolled out as the fix for
  "model finds zero peaks" on the 100-file subset.
- [x] **Padding mask threaded through fit()** — `load_dataset` now
  returns `mask = (X.sum(-1) > 0)` and the training cell passes it as
  `sample_weight`. The loss is no longer diluted by thousands of empty
  positions per sample.
- [x] **Checkpoint / early-stop on `val_aupr`** — `ModelCheckpoint`,
  `EarlyStopping`, and `ReduceLROnPlateau` all monitor `val_aupr` (mode
  `'max'`) instead of `val_loss`.
- [x] **Output bias init** on final 1x1 sigmoid conv —
  `bias_initializer = -log((1-π)/π) ≈ -4.6` for prior π=0.01. Without
  this, predictions sat in [0.41, 0.51] (sigmoid stuck near midpoint)
  and `val_aupr` was pinned at the random-init lottery score across
  three runs. With it: prediction range opened to [0.02, 0.35],
  `val_aupr` climbed monotonically from 0.16 to 0.18 over ~20 epochs,
  test F1 jumped from 0.05 to 0.17. RetinaNet paper, eq. 1, §3.3.
- [x] **Loss switched from focal to weighted BCE** — focal loss with
  continuous Gaussian targets created a loss-metric mismatch (loss
  kept descending while `val_aupr` plateaued at ~0.177). Plain
  weighted BCE with `pos_weight=70` (the inverse of the ~1.4%
  positive rate) pairs cleanly with soft targets: `weight(y) =
  pos_weight * y + (1 - y)` interpolates linearly, so peak centres get
  full pull, background gets weight 1. `focal_loss` is kept in
  cell-15 for A/B reference.
- [x] **Dilated integration block (k=7, d=1..32)** — receptive field
  expanded ~70 bp → ~410 bp. Logged as run #5, INCONCLUSIVE on its own
  (train PR-AUC stayed flat at ~0.16, prediction range compressed,
  best epoch=1). Falsified the receptive-field-alone hypothesis but
  the failure mode pointed at gradient flow rather than capacity.
  Architecture retained as the base for #6.
- [x] **Residual connections in the dilated stack (WaveNet-style)** —
  each dilated layer wraps as `out = relu(BN(Conv1D(64,7,d=d)(x))) + residual`,
  with a 1×1 channel projection on the first block (256→64). Logged as
  run #6, KEPT. Train PR-AUC unstuck for the first time (0.157 → 0.220),
  best epoch 1 → 17, prediction range expanded to [0.0001, 0.9995], and
  val "Both BPs found" jumped 0% → 24.8% — the persistent ceiling broken.
  Test F1 0.195 → 0.122 looks like a regression but is a threshold-tuning
  artifact (val-best threshold moved from 0.3 to 0.7 because predictions
  span the full sigmoid range now); val Both BPs improved at every
  threshold. Test Both BPs 0.0% → 1.2% — small lift, val/test gap
  attributable to test set skewing longer than 4000 bp (val 3-4kb gets
  57.3% Both BPs; ≥4kb gets 16.8%).
- [x] **`MAX_SEQ_LEN` 4000 → 10000 with auto-derived POS_WEIGHT** —
  cell-3 and cell-12 updated; mean_y_unmasked dropped 0.0139 → 0.0120,
  POS_WEIGHT auto-bumped 70 → 82.62. Logged as run #7, KEPT under §6
  qualitative criterion. Val F1 jumped 0.238 → 0.338 (+0.100) once
  cell-27's threshold sweep was extended to 0.95. Test F1 essentially
  flat at val-tuned threshold (0.122 → 0.113). Truncation hypothesis
  partially confirmed: at thr=0.7 val ≥4kb Both BPs went 16.8% → 88.3%,
  but at thr=0.7 the model fires ~21 peaks/genome and most are false
  positives — peak-density inflation. At val-best thr=0.95, val Both
  BPs is 26.0% (not 89.6%) and test Both BPs is 2.4%. The "Both BPs"
  jump was mostly artifact; the val F1 gain is real.
- [x] **Extended threshold sweep (cell-27, 0.2-0.99)** — added in #7
  after the original 0.2-0.7 sweep clipped the optimum; revealed val-best
  threshold is now 0.95 (not 0.7). One-line cell change, no model effect.
- [x] **Revert POS_WEIGHT to 70 (override auto-derivation)** — run #8,
  KEPT. Test F1 0.113 → 0.149 (+0.036), precision 0.085 → 0.135 (+0.050),
  recall basically flat. Auto-derived 82.62 was over-aggressive given
  the residual stack already finds signal. Standard going forward:
  diagnose data-implied pos_weight in cell-12 but use a hardcoded value.
- [x] **σ=10 sharper targets** — run #9, REVERTED. Test F1 dropped
  0.149 → 0.097 because hardcoded POS_WEIGHT=70 underweighted positives
  when mean(y) halved (data-implied was 166). σ tuning is coupled to
  POS_WEIGHT — needs to be re-attempted as a paired change after other
  axes are settled. cell-3 reverted to LABEL_SIGMA=20.
- [x] **BatchNorm → LayerNorm** — run #10, REVERTED. Test F1 +0.024
  on precision alone (recall −0.037), but val and test "Both BPs found"
  both collapsed to 0.0% (capability regression on hard-won metric).
  Train PR-AUC dropped 0.224 → 0.16. cell-18 reverted to BatchNorm.
  LayerNorm could be revisited with paired LR/capacity/pos_weight
  changes but is not a drop-in replacement here.
- [x] **100bp edge-buffer mask in loss** — run #12, INCONCLUSIVE/
  effectively REVERTED. Diagnostic chart showed boundary spikes still
  present after training (the spike lives in padding, where there was
  no loss anyway). Post-hoc suppression sweep showed suppressing the
  boundary HURTS F1 at every threshold. Boundary isn't the limiter;
  interior signal sharpness is. cell-12 reverted to #8 baseline.
- [x] **MaxChi-style "parental switch disparity" channels** —
  run #13, KEPT. Added 4 new input channels (windows 50/100/200/500 bp).
  Test Both BPs 2.4% → 4.9% (+2.5pp, clears KEPT bar). Val Both BPs
  15.7% → 22.2% (+6.5pp). Train-val PR-AUC gap collapsed
  +0.029 → −0.003 (overfitting eliminated). The architectural fix
  (residual+dilated, run #6) wasn't enough on its own; the model also
  needed the right input features.
- [x] **`max_files = 750 → None` (full training set)** — run #14, KEPT.
  Val F1 0.341 → 0.356 (+0.015), Val Both BPs 22.2% → 27.7% (+5.5pp),
  Val 3-4kb Both BPs 28.0% → 38.8% (+10.8pp). Test F1 +0.006 (within
  noise on n=82), Test Both BPs flat at 4.9%. Val improvements real;
  test stagnation tells us the val/test gap is *structural* (cross-
  configuration distribution shift), not sample-size.
- [x] **Stricter held-out split (TRAIN=XML-1..4, VAL=XML-5)** — run #15,
  KEPT (diagnostic). Val F1 0.356 → 0.321, Val Both BPs 27.7% → 18.2%
  on the honest signal. Test F1 0.151 (similar to before, threshold
  retuned to 0.5). Train-val gap reappeared at +0.072 — overfitting
  to XML-1..4 specifically. Val/test gap remains: even held out, val
  F1 ≫ test F1, so UnseenTestSet has a *second* layer of distribution
  shift beyond XML-5.
- [x] **Dropout 0.3 → 0.5** — run #16, KEPT (under criterion 1).
  Test F1 0.151 → 0.172 (+0.021, biggest test gain in many runs).
  Cost: Val Both BPs 18.2% → 8.5%, Test Both BPs 4.9% → 3.7% — the
  model is now in a higher-precision lower-recall regime. Train-val
  gap narrowed +0.072 → +0.037. Mixed result; the test F1 gain is the
  reason this is KEPT, but watch for further capability regression.

---

## High priority (likely to move the needle)

### 1. ~~Run #17: extended MaxChi windows {50, 100, 200, 500, 1000, 2000}~~ — REVERTED

Ran on 2026-05-04. Val improved trivially (+0.004 F1, +2.3pp Both BPs),
but Test F1 dropped 0.172 → 0.161 and Test Both BPs stayed at 3.7%.
Train-val gap widened slightly (+0.037 → +0.043). cell-3 reverted to
4 windows. See Experiment Log #17. The val/test structural gap is the
real bottleneck — adding more parental-disparity channels does not
attack it.

**Hardware note:** Migration to Linux/RTX 3070 (8 GB VRAM) was completed
in run #17. `BATCH_SIZE` dropped 16 → 8 to fit VRAM; runtime ~19 min
end-to-end (vs 80–100 min on M4).

### 2. ~~σ tuning, paired with POS_WEIGHT scaling~~ — REVERTED (run #18)

Tested σ=10 + POS_WEIGHT=140 (data-implied was 155.77) on 2026-05-04.
Test F1 0.172 → 0.160, Both BPs flat at 3.7%, train PR-AUC dropped
0.176 → 0.117 (under-fitting). Closes the σ/POS_WEIGHT coupling
story — neither σ=10 alone (run #9) nor σ=10 paired with POS_WEIGHT
helps. cell-3 reverted to LABEL_SIGMA=20 and cell-12 to POS_WEIGHT=70.
**Don't revisit σ tuning unless the underlying loss formulation changes.**

### 3. ~~Top-K coordinate regression head~~ — REVERTED (runs #19, #20)

Tested on 2026-05-04 with hard one-hot (#19) and soft Gaussian σ=5 (#20)
targets — both stalled at val_topk_match_rate=0.32965 (best epoch=1,
identical to 5 decimal places because of deterministic seeds + same
gradient direction in epoch 1). Test F1: #19=0.159, #20=0.152; Both BPs:
0% on both. The optimization landscape of softmax over 10000 positions
with this backbone has very weak signal beyond an early-epoch attractor.
**Target sharpness is not the bottleneck; the head/backbone combination
is.** Top-K axis is exhausted with the current backbone. See Experiment
Log entries #19 and #20.

### 3b. ~~Top-K boundary-attractor fix~~ — REVERTED (run #21)

Three-piece fix (target clamp, edge-buffer logit suppression). Mode
collapse broken (training ran 41 epochs, best epoch=26), but test F1
crashed to 0.024 — the previous 0.16 was almost entirely the boundary
attractor by chance matching test bps. With boundaries banned, the
backbone can't localize. **Top-K axis CLOSED with this backbone.**

### 4. ~~Run #24: MAX_SEQ_LEN 10000 → 32000~~ — KEPT (partial)

Ran on 2026-05-05. Test F1 0.172 → **0.191** (+0.019, in [0.18, 0.25]
band → KEPT but partial). Val F1 0.282 → 0.348 (+0.066), Val Both BPs
8.5% → 14.7% (+6.2pp), Test Both BPs 3.7% → 6.1% (+2.4pp).

**Truncation hypothesis is real but not the whole story.** Test
sequences are now fully visible to the model (the 74.4% of `bp_end`
values that didn't exist in the encoded input now do), and gains
materialised — but cross-config generalization gap is *narrowed*, not
*closed*. Val/test F1 spread is still 0.157 (was 0.110 in #16).

**Discovery during the run:** the diagnostic's anticipated POS_WEIGHT
rescale (70 → 178, predicting mean(y) would drop ~3×) was wrong.
mean(y) over UNMASKED positions stayed at 0.0125 (data-implied
POS_WEIGHT = 79, basically unchanged). The mask discards padding so
the per-non-padded-position positive rate is invariant to MAX_SEQ_LEN.
Hardcoded POS_WEIGHT=178 over-weighted positives by ~2.25×, and
showed it: best epoch=3 (very early plateau), train PR-AUC barely
climbed +0.019 over 18 epochs, best threshold dropped 0.7 → 0.4.

**Forced infra changes that DID work** (kept for future MAX_SEQ_LEN
=32k runs):
- BATCH_SIZE 8 → 2 (RTX 3070 8 GB VRAM at 3.2× length).
- max_files=500 cap on each train dir (15 GB WSL2 VM panicked on
  full-set `np.concatenate` at 32 k length).
- cell-22 `from_tensor_slices` pinned to `tf.device('/CPU:0')`
  (otherwise OOMs the 5.5 GB free VRAM).
- cell-22 `del X_train, y_train, w_train` after dataset construction
  (`from_tensor_slices` materialises a tf.constant copy alongside the
  resident numpy; without del, peak ~14 GB CPU pushed the 15 GB WSL2
  VM into vmmem balloon territory and caused catastrophic VM-kill on
  three earlier #24 attempts).
- `~/.wslconfig` → WSL2 RAM cap 15 GB → 24 GB (host has 32 GB).
- `dataRaw/` moved off OneDrive `/mnt/c/...` symlink chain onto Linux
  ext4 (was the root cause of the catastrophic vmmem-kill crashes).

See Experiment Log #24 for full results and lessons.

### 5. (DONE) RUN #28 LANDED 2026-05-06: parser + fp16 + from_generator + cgroup

**Outcome:** Test F1 = 0.239 (val-thresh) / 0.308 (test-best). Val F1 = 0.575. Above prior CNN best (~0.218) but below RDP5 (0.367). All 6 infrastructure fixes KEPT — see HANDOVER.md §11-current and Experiment Log #28 for the full list.

**Key learning that reshapes the queue below:** the val→test gap is now **calibration drift**, not capacity. Same model gets 62.9% Both-BPs on val and 19.5% on test at the same threshold; test wants threshold 0.9 vs val 0.6. Model logits are systematically smaller-magnitude on test — driven by longer sequences and different MaxChi disparity scale. Prior items in the queue were chosen against the buggy parser; many are now stale.

### 5b. NEXT (run #29) — Per-sample feature normalization on MaxChi channels

**Hypothesis:** The MaxChi disparity channels (18-21) have different value ranges between train (XML-1..4) and test (UnseenTestSet) because the simulator settings produce sequences of different lengths and substitution rates. The model learns logit calibration against train-statistics; at test time the same parental-disparity signal lands in a different channel-value range and pushes through the network differently. Per-sample standardization (subtract sample mean, divide by sample std on each MaxChi window channel, masked to non-padded positions) makes the disparity statistic comparable across train/test.

**Change:** cell-6 `_maxchi_features` returns standardized output. Single function modification, no other cells touched.

**Expected:**
- Cache invalidates (MAXCHI windows are part of cache key only by tuple value, not statistic; we'd want to bump CACHE_VERSION to force regen).
- If hypothesis correct: val→test threshold gap narrows (0.6 vs 0.9 → both within 0.1 of each other), test F1 at val-thresh climbs.
- Risk: we lose information about the *absolute* disparity magnitude. If that matters, val numbers regress.

**Verdict criteria:**
- Test F1 ≥ 0.367 (above RDP5) → KEPT, primary goal achieved.
- Test F1 ∈ [0.30, 0.367] → KEPT, partial — calibration was a real factor but not the whole story.
- Test F1 < 0.27 → REVERTED, calibration wasn't the limiter; pivot to U-Net or Transformer.

**Wall time:** ~30 min training + 3 min eval (test cache exists).

### 5c. AFTER #29: U-Net or Transformer encoder

If #29 doesn't get us to 0.30 test F1, the architecture has to change. Two candidates ranked by expected impact:

- **U-Net** (encoder-decoder with skip connections). Naturally per-position output. Skip connections preserve fine-grained location info that the dilated stack loses. Estimated 1-2M params, single-day implementation.
- **Transformer encoder** over positions (e.g. 6-8 layers, 8 heads, 256 model dim). Attention is intrinsically calibrated to relative position, less sensitive to absolute length. Would be 3-5M params. More implementation work; warrants a clean phase plan.

User has waived the CNN-only constraint explicitly. Pick whichever attacks the failure mode best. If #29 reveals "calibration was 80% of the gap," go U-Net (better local feature extraction). If "calibration was 20% of the gap," go Transformer (better long-range and length-invariant).

---

### 5-prev. (DONE) run #25: POS_WEIGHT 178 → 70 — correct the over-weight

The over-weighting was a paired-infra mistake from #24, not a research
choice. Drop POS_WEIGHT back to ~70 (matching the data-implied 79 ×
0.847 factor from run #8 — same rule, applied to the actual data-
implied at MAX_SEQ_LEN=32 k, not the predicted one).

**One-line change:** cell-12 `POS_WEIGHT = 178.0` → `70.0` (and matching
doc comment in cell-3).

**Expected if POS_WEIGHT was the over-weight culprit:**
- Best epoch later (5–10, not 3).
- Train PR-AUC climbs further (was capped at 0.166).
- Val/test F1 rebalance: precision recovers, recall comes down some,
  best threshold rises back toward 0.5–0.7.
- If test F1 jumps materially (≥ 0.22, say), the cross-distribution
  gap is much smaller than #24 suggested.
- If test F1 stays flat (~0.19), then POS_WEIGHT wasn't the limiter
  and the residual gap is a true generalization issue → next axes
  are augmentation (parent-swap, reverse-complement) or backbone
  capacity (n_filters 64 → 128).

Cache stays warm (no MAX_SEQ_LEN/MAXCHI/SIGMA/BP_WINDOW change).
Wall time ~15 min total like #24.

### 5. Masked BatchNorm under per-position (deferred)

Direct empirical motivation from runs #19-#21: BN+'same'-padding
boundary spike is now confirmed to dominate any argmax-style prediction.
A custom Keras layer that computes BN stats only over valid (non-padded)
positions would eliminate the artifact at the architectural level. Apply
under the per-position #16 baseline pipeline. Single change. Custom
layer work is the only complexity. Advisor consultation for design first.

### 4. Masked BatchNorm

Custom layer that computes BN stats only over valid (non-padded)
positions. Targets the BN+padding boundary artefact directly. More
invasive than dropping in LayerNorm (run #10 REVERTED) but more
conservative — it preserves BN's batch-statistics regularisation
where it's appropriate.

### 5. Bigger multi-scale kernels

E.g. `(7, 31, 63, 127)` — increases receptive field at the input layer
rather than at the integration layer. Sub-axis: try doubling
`n_filters` across the dilated stack.

### 5. Masked BatchNorm

Conservative alternative to #10. Compute BN mean/var over valid
positions only, ignoring padding. Tensorflow doesn't ship this — would
need a custom layer. Worth doing if the BN+padding artifact hypothesis
turns out to be the bottleneck.

---

## Medium priority

### 5. Parent-swap augmentation

Channel order for parent 1 vs parent 2 is arbitrary — the task is
symmetric in the parents. Cheap regularisation: at training time, swap
`X[..., 5:10]` with `X[..., 10:15]` (and `match_p1` with `match_p2`)
with probability 0.5.

### 6. Stronger split: hold out an entire `XML-N` directory

Group-by-file already kills the most obvious leak, but each `XML-N`
likely uses one SANTA configuration. For a stricter test of cross-config
generalisation:

```python
TRAIN_DIRS = ["XML-1", "XML-2", "XML-3", "XML-4"]
VAL_DIRS   = ["XML-5"]
```

Worth doing as a *second* held-out evaluation alongside the current
file-level split.

### 7. Fix saliency target

`tf.reduce_mean(pred)` averages over 4000 outputs and produces a smeared
attribution map. Target the predicted peaks instead:

```python
peaks = detect_peaks(y_pred[idx], threshold=best_thr)
with tf.GradientTape() as tape:
    tape.watch(x)
    pred = model(x)
    target = tf.reduce_sum(tf.gather(pred[0], peaks))
```

### 8. Tune `min_distance` separately from `tolerance`

`evaluate_peaks` currently sets `min_distance = tolerance`. These are
conceptually different: `tolerance` is "how close does a peak need to be
to count?", `min_distance` is "how close can two peaks be before we
collapse them?". Sweep `min_distance ∈ {50, 100, 200, 400}` independently.

### 9. `actual_len` semantics

`len(seq.rstrip('-'))` counts internal alignment gaps. If the goal is
"how long is the real sequence?", use `len(seq.replace('-', ''))`. If the
goal is "alignment span" (which matters for breakpoint coordinates),
keep current behaviour but rename to `aligned_span`.

---

## Low priority / exploratory

### 10. ~~Increase `MAX_SEQ_LEN` for full HIV genomes~~ — promoted to High Priority #4.

### 11. Cross-validation across files

5-fold grouped CV (by file) would give an honest mean ± std for each
configuration during ablation runs.

### 12. Hyperparameter sweep

Reasonable axes to sweep once the architecture is stable: `n_filters`,
`dropout`, `LR`, `BATCH_SIZE`, focal `(alpha, gamma)`, label `sigma`
(once #4 lands).

### 13. Autoencoder redesign

Out of scope for current work, but listed for completeness:

- Replace `GlobalAveragePooling1D` + `Dense` bottleneck with a U-Net
  (downsample / upsample with skip connections). Without skips the AE
  cannot localise reconstruction error.
- Train on real triplets of *non-recombinant* sequences `[s1, s2, s3]`
  instead of `[parent, parent, parent]` so the inference distribution
  matches.
- Replace `sigmoid + MSE` with three softmax-over-5 categorical cross
  entropies (one per sequence in the triplet); reconstruction targets
  are one-hot, so a categorical loss is the right shape.

### 14. Mixed-precision training

`tf.keras.mixed_precision.set_global_policy('mixed_float16')` typically
gives a 1.5–2× speedup on modern GPUs at no accuracy cost. Verify the
focal loss numerics still behave (it has a `clip` already, which helps).

### 15. Class-weighted dataset balancing at the file level

If some `XML-N` configurations dominate the event count, the model
will overfit to their characteristics. Worth checking with a histogram
of events per directory before deciding whether to weight or subsample.
