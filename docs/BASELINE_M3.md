# BASELINE_M3 — frozen spec for the M3 sequence-only breakpoint detector

Status: **FROZEN reference baseline**, recorded 2026-07-14. Numbers below (except
the median/rank-mean diagnostic in §5) are authoritative and must not be
silently recomputed/overwritten by future work — if you re-run and get
different numbers, that is itself a finding (see §6) and should be recorded
alongside these, not in place of them.

## 1. Training config

- Entry point: `m3_dilated.py --feature-mode raw --head-mode single`
  (produces a `DilatedHead` detector — **not** the v4 multi-head/aux-gate
  variant).
- Data: train shards `XML-1..4`, val shard `XML-5` (**unfiltered** — i.e. the
  raw SANTA shards, not `splits/v2_filtered_split.json`'s realism-filtered
  subset).
- Sizes: `n_train=20000`, `n_val=500`, `max_len=30500`.
- Optimization: 25 epochs, `lr=1e-3`, AdamW `wd=0.01`, `pos_weight=70`,
  `LABEL_SIGMA=10`, bf16 autocast.
- Seeds: 0, 1, 2, 3 (one deployment checkpoint per seed, model-selected by
  best SANTA val F1 during training).
- Environment: torch 2.13.0+cu130, NVIDIA driver 595.x, single RTX 3070 (8 GB).

## 2. Checkpoints (frozen)

| seed | checkpoint |
|---|---|
| 0 | `models_test/m3d_big_snaps/m3d_best.pt` |
| 1 | `models_test/m3d_seed1_snaps/m3d_best.pt` |
| 2 | `models_test/m3d_seed2_snaps/m3d_best.pt` |
| 3 | `models_test/m3d_seed3_snaps/m3d_best.pt` |

## 3. Eval protocol

- Benchmark: 4 real-HIV LANL CRF triplets — `CRF02_AG`, `CRF07_BC`,
  `CRF08_BC`, `CRF12_BF`.
- Scoring: micro/pooled tp+fp+fn across all 4 CRFs at a **single global
  threshold** (swept, best-F1 reported), `find_peaks`-based peak extraction,
  `TOLERANCE=200` bp match window, greedy nearest-first true/pred pairing.
- Two edge_buffer (peak-suppression-near-content-ends) conventions are
  reported side by side: `eb=200` (0-TP-loss convention, no true LANL BP lies
  within 300 bp of a content end) and the legacy `eb=25`.
- Tooling: `m3_eval_lanl.py` (single-model eval), `m3_eval_ensemble.py`
  (multi-seed variance + mean-of-tracks ensemble).

## 4. Per-seed LANL F1 (authoritative, do not recompute in place)

| seed | eb=200 | eb=25 |
|---|---|---|
| 0 | 0.489 | 0.451 |
| 1 | 0.559 | 0.531 |
| 2 | 0.593 | 0.571 |
| 3 | 0.490 | 0.459 |
| **mean ± std** | **0.533 ± 0.045** | **0.503 ± 0.050** |

SANTA val F1 (best snapshot, model-selection metric during training):

| seed | SANTA val F1 | epoch |
|---|---|---|
| 0 | 0.578 | 18 |
| 1 | 0.574 | 21 |
| 2 | 0.601 | 24 |
| 3 | 0.590 | 20 |

4-seed mean-of-tracks **ENSEMBLE** (`m3_eval_ensemble.py`, plain mean of the 4
per-seed per-position BP-probability tracks, then peak-extract):

| eb | F1 | tp/fp/fn |
|---|---|---|
| 200 | **0.512** | 21/26/14 |
| 25 | **0.477** | 21/32/14 |

This is **below** the per-seed mean (0.533 @ eb=200) — seed0 and seed3 are
comparatively low-precision/FP-heavy, and averaging their noisy tracks in
with seed1/seed2's dilutes otherwise-clean consensus peaks.

## 5. Diagnostic: does robust aggregation recover the ensemble shortfall?

Added `m3_agg_diagnostic.py` (new, additive script; imports `load_crfs`,
`tracks_for`, `best_f1` from `m3_eval_ensemble.py` unmodified — does not alter
the mean-of-tracks path or its outputs) to test whether replacing the mean
with a robust aggregator recovers F1 toward the per-seed mean (0.533) by
suppressing seed0/seed3's spurious peaks.

Tested MEDIAN-of-tracks (element-wise median across the 4 seed tracks/position)
and RANK-MEAN (average of per-seed percentile-rank-transformed tracks, via
`scipy.stats.rankdata`, so each seed contributes equally regardless of its
absolute probability calibration):

| eb | aggregator | F1 | tp/fp/fn |
|---|---|---|---|
| 200 | mean (baseline, frozen) | 0.512 | 21/26/14 |
| 200 | **median** | **0.512** | **21/26/14** (same pooled count as mean) |
| 200 | rank-mean | 0.466 | 31/67/4 |
| 25 | mean (baseline, frozen) | 0.477 | 21/32/14 |
| 25 | **median** | **0.477** | **21/32/14** (same pooled count as mean) |
| 25 | rank-mean | 0.443 | 31/74/4 |

**This is a DIAGNOSTIC of aggregation fragility, not a new headline** — the
pre-registered ensemble number for deployment/comparison purposes remains
mean-of-tracks, 0.512 (eb=200) / 0.477 (eb=25), exactly as computed by
`m3_eval_ensemble.py`.

**Finding:** median produces the *exact same pooled tp/fp/fn* as mean at both
edge buffers, but this is **not** because the two aggregators produce the
same peaks. Verified directly: the mean and median tracks differ
substantially (up to 0.11-0.12 absolute probability at the max, ~99.9% of
positions non-identical), and the extracted peak *sets* are correspondingly
different per CRF (e.g. CRF02_AG: mean gives 18 peaks at one set of
positions, median gives 19 peaks at a shifted/different set; CRF08_BC: mean 8
peaks, median 7). What's identical is only the **pooled count of
truth-matches within the 200 bp tolerance** — both aggregators' differently-
located peaks happen to net to the same tp/fp/fn once scored against ground
truth. So "median == mean" here is a real but coincidental tie at the
scoring resolution this benchmark uses (200 bp tolerance, 35 true BPs), not a
sign the two aggregation formulas behave identically, and not a bug.

Why median can't reject the noisy seeds: checked the per-position order
statistics directly. With 4 seeds and 2 of them noisy (seed0, seed3), the
median is the average of the 2 *middle*-ranked values at each position. If
noisy/clean seeds were randomly distributed across ranks, both middle slots
would be noisy seeds only ~1/6 (16.7%) of positions by chance. Instead, **both
middle-ranked values are the noisy seeds (0,3) at 46.3% of positions**, and
**at least one middle value is a noisy seed at 92.9% of positions** — i.e.
seed0/seed3 are not producing a small number of extreme (top/bottom
order-statistic) outlier values that a median would naturally exclude; they
sit in the *middle* of the per-position distribution most of the time. A
median has no leverage against contamination that lives in the middle of the
order statistics rather than the tails — this is the textbook failure mode
of median-of-4 with 50% contamination (2-of-4 noisy is the worst case for a
median: the two "good" values are not guaranteed to be the ones selected).

Rank-mean is *worse* (recall jumps to 31/35 but precision collapses to
roughly half, ~2x the FPs of mean/median), because equalizing every seed's
contribution by percentile rank gives seed0/seed3's many weak spurious peaks
the same voting weight as seed1/seed2's few strong true peaks — it actively
removes the (partial) protection that raw-probability averaging gives when a
noisy seed's spurious peak is merely weak rather than absent. (Caveat:
rank-mean's swept-threshold grid caps at 0.80; since rank-mean values are
percentiles in [0,1], its true optimum could sit above that cap, so its
reported F1 may be mildly understated. This doesn't change the conclusion —
it is still clearly worse than mean/median by a wide margin — but is noted
for completeness since this is the "if easy" secondary aggregator.)

**Interpretation:** robust aggregation does **not** recover the ensemble
toward ~0.55+. The "2 noisy seeds dilute the mean" framing, in the simple
sense of outlier suppression via order statistics, is **not the mechanism**
that a swap to median or rank-mean can fix — median is specifically robust to
a *minority of extreme-valued* tracks, but seed0/seed3's noise is not
extreme-valued, it's middle-of-distribution, which is exactly the regime
(2-of-4 contamination) where median has no leverage. Recovering to ~0.55+
likely requires either seed selection (drop seed0/seed3, ensemble only
seed1+seed2) or more (better) seeds so a minority-outlier assumption actually
holds, not a smarter aggregation formula applied to the same 4 tracks.

Script: `m3_agg_diagnostic.py` (repo root). Report JSON:
`models_test/m3_agg_diagnostic.json`.

## 6. Comparison to classical RDP + overall conclusion

- Classical RDP standalone: F1 **0.519**.
- 4-seed mean-of-tracks ensemble: F1 0.512 (eb=200), i.e. **-0.007** vs RDP —
  statistically indistinguishable given per-seed std 0.045 on a 35-true-BP
  test set.
- Per-seed mean 0.533 is nominally above RDP (+0.014), but the *deployed*
  aggregation (the ensemble, which is what would actually ship) sits at 0.512,
  below RDP.

**Conclusion: in the current environment, M3 does NOT robustly beat RDP** —
it is statistically indistinguishable from it. This is a **regression** from
the previously-documented result (memory: `project_m3_beats_rdp.md`,
2026-05-30) of a 4-seed ensemble LANL F1 0.565 (eb=200), per-seed
0.554 ± 0.013, every seed > RDP 0.519. Those original weights are
**unrecoverable** (checkpoints from that run no longer exist / were
overwritten); this document's 4 checkpoints are a **fresh retrain** under the
frozen config in §1, not the original May run.

Causes investigated and **ruled out** (confirmed identical to the May run):
- eval convention (edge_buffer, tolerance, single-global-threshold scoring)
- code drift in the training/eval scripts
- training/eval data (same shards, same split, same LANL CRF panel)

By elimination, the gap is attributed to **training-side environment-induced
variance** (e.g. torch/CUDA version drift since May, nondeterministic
kernels/driver behavior) rather than a methodology or data regression. This
is a hypothesis by elimination, not directly confirmed — no further
diagnostic was run to pin down which environment factor specifically.

## 7. KEY FINDING — SANTA val F1 does not predict LANL F1 (record prominently)

SANTA (in-domain, simulated) val F1 and real-HIV LANL F1 are **uncorrelated**
across the 4 seeds:

| seed | SANTA val F1 (rank) | LANL F1 @ eb=200 (rank) |
|---|---|---|
| 0 | 0.578 (3rd of 4) | 0.489 (**worst**) |
| 1 | 0.574 (worst) | 0.559 (2nd best) |
| 2 | 0.601 (**best**) | 0.593 (**best**) |
| 3 | 0.590 (2nd best) | 0.490 (2nd worst) |

Seed0 is 3rd-best on the in-domain val metric but **worst** on the real
deployment benchmark; seed3 is 2nd-best on val but 2nd-worst on LANL. Only
seed2 is consistent (best on both).

**Implication:** there is no a-priori in-domain signal available at training
time to select which seed to deploy — SANTA val F1 cannot be used as a proxy
for real-world (LANL) performance. **Picking a seed by its LANL score would
be leakage** (LANL is the held-out real-world benchmark, not a validation
set) — the correct practice is to commit to an aggregation/selection rule
*before* looking at LANL, which is exactly why this document freezes the
mean-of-tracks ensemble as the pre-registered number rather than
post-hoc-picking the best single seed (seed2, 0.593) or the best aggregator.

## 8. Files

- Training: `m3_dilated.py`
- Single-model eval: `m3_eval_lanl.py`
- Multi-seed variance + mean ensemble (frozen, unmodified): `m3_eval_ensemble.py`
  → `models_test/m3_ensemble_lanl.json`
- Robust-aggregation diagnostic (new, additive): `m3_agg_diagnostic.py`
  → `models_test/m3_agg_diagnostic.json`
