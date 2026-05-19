# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current direction (May 2026) — read this first

The project pivoted **away from the Keras dilated-CNN** to a PyTorch + HyenaDNA
sequence-only backbone, aimed at multi-virus transfer (not HIV-only).

- **Active plan:** [`MASTER_PLAN.md`](MASTER_PLAN.md). M0.1–M0.5, M1.1, M1.2-pre complete; **M1.2 (MLM training loop) is next.**
- **Active handover:** [`HANDOVER_NEXT_AGENT.md`](HANDOVER_NEXT_AGENT.md).
- **Active backbone:** `backbone_hyenadna.py` (HyenaDNA-small-32k, PyTorch 2.6 + transformers 5.8). `backbone_mamba.py` is a record of an abandoned attempt — don't revive without system `nvcc`.
- **Active data:** `splits/v2_filtered_split.json` (the realism-filtered split; whole-shard drops of XML-1, XML-3, long_content_30k_003 + per-combo trims). Cache is `cache/v2/` via `cache_v2_reader.CacheV2`. **Do not train on `splits/v2_split.json` going forward.**
- **Active deployment baseline (legacy):** `models_test/cnn_breakpoint_runB2_sig10_*.keras` — LANL agg F1 0.533. The HyenaDNA work must beat this.
- **OOM rule (non-negotiable):** never `np.array(X, copy=True)` or `X.astype(...)` on multi-GB cached tensors. Use the RSS-watchdog pattern from `m12_zeroshot_probe.py`. See `feedback_padding_mask_oom.md` in memory.

Pre-pivot context (the original task framing, cell IDs, iteration discipline, and the legacy Keras CNN pipeline) is preserved below for reference. Older one-shot handovers live in [`docs/handovers/archive/`](docs/handovers/archive/); pre-pivot idea/TODO queues in [`docs/archive/`](docs/archive/) — those are historical, **do not act on their recommendations.**

---

## Legacy pipeline (reference)

## What this is

ML research project for detecting **viral recombination breakpoints** in aligned nucleotide sequences. Inputs are triplets `(recombinant, parent1, parent2)` produced by the SANTA simulator; outputs are per-position breakpoint probabilities. The work was originally delivered as Jupyter notebooks; the active backbone work is now in PyTorch scripts (`backbone_hyenadna.py`, `m12_*.py`). Training and evaluation for the legacy CNN live in `CNN.ipynb`.

## Project goal

The aim is to **beat the current recombination-detection state of the art** — MaxChi, RDP, GeneConv — and ultimately deploy on real HIV (and other) genomes. Quality is the only metric that matters; compute cost is not a constraint, so favour proposing larger / deeper / more expressive architectures over faster ones.

Key implications of the goal that shape what's worth trying:

- **Training data is simulated; deployment data is real.** We cannot train on real HIV alignments because there is no ground truth. The model has to transfer from SANTA simulations to real biology — be wary of features that exploit simulation-specific artifacts.
- **Triplet input is the current simplification, not the long-term contract.** The current model is given the recombinant's identity (via `ActualRecomb` in the SANTA CSVs) and only has to localise breakpoints. The eventual goal is a model that takes three sequences and decides *which* (if any) is the recombinant — a possible future framing is a four-way head over `{seq1, seq2, seq3, none}`. Treat the current "breakpoints in a known recombinant" task as a stepping stone.
- **`RDPML.ipynb` and `RDPML_PSNN.ipynb` are referenced in `Autoencoder.ipynb` but live elsewhere.** They solve a different problem (classifiers trained on signals derived from existing detection methods); they are not baselines for what's being built here. Do not conflate them.

## Read these first (legacy CNN context only)

- The **Experiment Log** markdown cell at the end of [CNN.ipynb](CNN.ipynb) — every prior run's hypothesis, configuration, headline numbers, and verdict. The most recent entry is the legacy CNN baseline (runB2_sig10).
- [`docs/handovers/archive/HANDOVER.md`](docs/handovers/archive/HANDOVER.md) — original autonomous-iteration handover, kept for historical context.
- [`docs/archive/TODO.md`](docs/archive/TODO.md) — pre-pivot architectural decisions log.

**For current work, read [`HANDOVER_NEXT_AGENT.md`](HANDOVER_NEXT_AGENT.md) and [`MASTER_PLAN.md`](MASTER_PLAN.md) instead.**

## Environment

- Python 3.12 in `.venv/`
- TensorFlow 2.18.1 with bundled CUDA 12.x wheels (`tensorflow[and-cuda]`) on Linux + NVIDIA. Current dev box: bare-metal Ubuntu 26.04, NVIDIA RTX 3070 (8 GB, compute 8.6), driver 595.x. On other platforms TF falls back to CPU silently.
- BioPython 1.87 for FASTA parsing, scipy for `find_peaks`-based evaluation.

```bash
source .venv/bin/activate
jupyter notebook CNN.ipynb     # primary work
```

## Data layout

`dataRaw/` (gitignored) contains SANTA simulation outputs. Each simulation produces three correlated files keyed off the same `.fa` filename:

| Suffix | Contents |
|---|---|
| `.fa` | FASTA alignment (~100 sequences per file) |
| `.faSimVSRealCompare.csv` | Ground truth: `ActualRecomb`, `SimBPStart`, `SimBPEnd` |
| `.faRecombIdentifyStats.csv` | Three rows per event with `ISeqs(A)` listing triplet members |

`XML-1` … `XML-5` are training directories (~750–1300 files each). `UnseenTestSet/` is held out for final evaluation. `dataRaw.zip` is the 8GB source archive.

## Architecture in one diagram

```
(recomb, p1, p2)  ──▶  encode_triplet  ──▶  X (n, 10000, 22)        ──┐
                                                                       ├──▶  build_cnn  ──▶  y_pred (n, 10000)
                       generate_labels       y (n, 10000)  Gaussian ──┘                            │
                                             mask (n, 10000)  sample_weight                         ▼
                                                                                              find_peaks
                                                                                                    │
                                                                                                    ▼
                                                                                              evaluate_peaks
                                                                                              (+/- 200 bp tolerance)
```

**Per-position channels (22 total as of run #13):**
- 0–4: recombinant one-hot (A/T/G/C/-)
- 5–9: parent 1 one-hot
- 10–14: parent 2 one-hot
- 15–17: comparison channels — `match_p1`, `match_p2`, `informative` (parents differ). These hand the model the recombination signal directly so the convs don't have to rediscover cross-channel comparison from scratch.
- 18–21: MaxChi-style "running parental disparity" at windows {50, 100, 200, 500} bp. Each is `mean(parental_signal[p:p+w]) - mean(parental_signal[p-w:p])` where `parental_signal = match_p1 - match_p2`. These are the same statistics classical methods (MaxChi, GeneConv) compute. Run #13 added them and immediately collapsed the train-val gap and lifted Both BPs across the board.

**Padding past every sequence's end is all-zero across the 22 channels.** `load_dataset` returns a per-position `mask = (X.sum(-1) > 0)` and `cnn.fit(..., sample_weight=mask)` zeros out those positions in the loss — without this, the loss is dominated by training the model to predict 0 on thousands of empty positions per sample.

**Labels are soft Gaussian peaks** (`LABEL_MODE='gaussian'`, σ=20) at each true breakpoint, not hard +/-window binary masks. This matches how `find_peaks` evaluates and gives the model a smooth gradient.

## Three things that are easy to break

1. **Train/val split is held-out by directory.** As of run #15, train = XML-1..4 and val = XML-5 (entire directory held out). Earlier runs used file-level shuffling within all 5 XMLs and silently leaked distribution between train/val — val numbers overstated capability by ~10pp Both BPs. Don't revert to mixed-XML splitting unless you understand what you're giving up.

2. **Final layer needs a prior-aware bias init.** With ~1% positives and zero bias, `sigmoid(0)=0.5` is a strong attractor that the negative-class gradient cannot escape — predictions stay clustered in [~0.41, ~0.51] and `val_aupr` is pinned at the random-init lottery score. The fix is `bias_initializer = Constant(-log((1-π)/π)) ≈ -4.6` for π=0.01. This was discovered after three failed runs; it is required, not optional.

3. **POS_WEIGHT is hardcoded.** As of run #8 (#7 tested auto-derivation and was over-aggressive), `POS_WEIGHT = 70.0` in cell-3. cell-12 still computes the data-implied value as a *diagnostic*, but the model uses the hardcoded value. If you change `LABEL_SIGMA` or the label encoding, POS_WEIGHT may need to scale — but that's a deliberate paired experiment (run #9 didn't pair them and was REVERTED), not an automatic adjustment.

4. **Loss/metric pairing matters.** The earlier focal loss with continuous Gaussian targets produced a loss that kept descending while `val_aupr` plateaued. The active loss is `weighted_bce(POS_WEIGHT=70)` (inverse class ratio) defined alongside `focal_loss` in cell-15. `focal_loss` is kept available for A/B reference.

5. **Argmax-style outputs (top-K, etc.) need explicit boundary masking.** Runs #19-#21 explored a K=2 softmax-over-positions head (one-hot targets, σ=5 Gaussian targets, with edge-buffer suppression). All three REVERTED — under argmax-style output the BN+'same'-padding interaction makes positions 0 and L−1 systematically extreme, so the model collapses to predicting the literal sequence boundaries (head-0 → 0 in 99.8% of samples, head-1 → 9999 in 86.2% in run #19). The boundary-mask trick added in run #21 (logits at `[0, EDGE_BUFFER) ∪ [L−EDGE_BUFFER, L)` set to `-1e9`, *and* targets at the same positions zeroed out so cross-entropy contributes 0 loss/gradient there) is **necessary infrastructure** for any future argmax-style head. The cells are reverted to per-position from run #22 onward, but `K_TOPK`, `TOPK_TARGET_SIGMA`, `EDGE_BUFFER` constants in cell-3 and the legacy `build_cnn_topk` / `topk_xent_loss` / `make_topk_targets` functions are preserved. The deeper takeaway: *don't put any argmax-style head on this backbone without masking the BN-padding boundary first*. The artifact existed at #16 too but was bounded in [0,1] under sigmoid+threshold and didn't dominate; argmax amplifies small logit differences into huge probability differences.

## Evaluation contract

A predicted breakpoint is correct if `find_peaks` returns a peak within +/-`TOLERANCE` (200 bp) of a true breakpoint, with `min_distance=TOLERANCE` so broad ridges collapse to single peaks. Each true breakpoint can match at most one predicted peak (greedy nearest-first). Per-position metrics like ROC/PR-AUC are reported but **peak-based F1 and the event-level "Both BPs found / One / Missed" breakdown are the headline numbers**. With two true BPs per event, recall caps at 0.5 if the model only emits one peak per genome.

## Notebook cell IDs

Cells have stable IDs (`cell-0`, `cell-1`, …) that are referenced from TODO.md and across sessions. Key cells:

- `cell-3` — configuration constants (`MAX_SEQ_LEN`, `MAXCHI_WINDOWS`, `LABEL_SIGMA`, `POS_WEIGHT`, `TOLERANCE`, etc.)
- `cell-6` — `one_hot_encode`, `encode_triplet`, `_maxchi_features` (the MaxChi feature extractor; data-driven from `MAXCHI_WINDOWS`)
- `cell-11` — `load_dataset` (returns `X, y, mask, meta`)
- `cell-12` — loads train (XML-1..4) and val (XML-5) datasets; hardcodes `POS_WEIGHT=70`
- `cell-13` — directory-level train/val split (no file-shuffling needed)
- `cell-15` — `focal_loss` and `weighted_bce`
- `cell-18` — `build_cnn` and compile (residual dilated stack)
- `cell-21` — callbacks (all monitor `val_aupr`, `mode='max'`)
- `cell-22` — `cnn.fit(...)` with `sample_weight=w_train`
- `cell-27` — threshold sweep (extends through 0.95 for residual+dilated regime)
- `cell-36` — event-level detection summary
- `cell-experiment-log` — the canonical Experiment Log markdown cell

When editing, prefer rewriting via a Python script that loads/saves the notebook JSON — the file is too large for `Read`, and the in-place editor preserves cell IDs that downstream tooling and TODO.md references depend on.

## Tooling

- `run_notebook.py` — wrapper that executes the notebook end-to-end via `nbclient`. Use `python3 -u run_notebook.py > exec_log.txt 2>&1`.
- `chart_eval.py` — generates prediction-vs-truth charts on the cached model. Useful when diagnosing where the model is failing.
- `eval_with_suppress.py` — sweeps post-hoc edge suppression × threshold on the cached model.
- `eval_topk.py` — evaluates the cached model under top-K peak reranking.

## Iteration discipline

- **One change per run.** Bundling architecture + loss + bias init makes failures unattributable; this has bitten us twice.
- **Watch train PR-AUC, not just val.** When train loss drops 2.5× while train PR-AUC stays flat, the architecture has saturated its ranking ability — loss is no longer the bottleneck.
- For quick iteration, set `max_files` to a small number in cell-12; for headline numbers, set it to `None` (or 750+). The full dataset is the comparison-relevant configuration since the goal is to beat existing tools, not to optimise epoch time.
- **Don't propose smaller / faster architectures as compromises.** Compute is not a constraint here. If something larger is more likely to work, that's the right recommendation.
