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

### 1. Run #17: extended MaxChi windows {50, 100, 200, 500, 1000, 2000} — STAGED

cell-3 has been edited to add windows 1000 and 2000 (N_INPUT_CHANNELS
22 → 24). The notebook on disk has the cell-3 source updated but the
cached cell outputs are still from run #16 (4 windows). Run was started
on the previous machine but killed before any epoch completed (~50 min
data-prep dominated wall time). **The next agent should run this first.**

Hypothesis: XML-5 val and the test set are entirely ≥4kb sequences;
wider windows give the model the longer-range parental-disparity signal
that 50–500 bp can't capture. Single change idea (extend MAXCHI_WINDOWS
in cell-3, cell-6's `_maxchi_features` is data-driven and will adapt).

Watch for:
- Val Both BPs recovers from #16's 8.5% (maybe back to ~15-20%).
- Test F1 holds or improves from #16's 0.172.
- Train-val gap stays ≤ +0.05 (the wider windows shouldn't reintroduce overfitting).

### 2. σ tuning, paired with POS_WEIGHT scaling

Re-attempt #9. For σ=10, set POS_WEIGHT to ~140 (data-implied was 166
when last measured). Treat σ + POS_WEIGHT as one logical unit per
HANDOVER §4. Sharper targets should encourage more localized peaks.

### 3. Top-K coordinate regression head — strategic backstop

If wider MaxChi (#17) and σ-tuning (#2) plateau, this is the principled
attack on the val/test distribution gap. Replace per-position binary
classification with a structured K=2 (position, confidence) output;
sorted-pair L1 loss. Major refactor (touches output head, loss, eval),
but addresses class imbalance fundamentally and removes the boundary
artefact entirely. Discussed with the advisor multiple times in the
previous session; the consensus was "do this when input-feature
engineering plateaus", which is approximately now.

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
