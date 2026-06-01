# Handover — RDP_CNN, June 2026

You are picking up a working sequence-only viral recombination breakpoint
detector. **Read `README.md` first** (master index + repo map), then this file
(current state + what to do next), then `docs/ARCHITECTURE.md` and
`docs/WRITEUP_M3.md`. Cross-session memory is at
`/home/joshc/.claude/projects/-home-joshc-Dev-RDP-CNN/memory/` — skim `MEMORY.md`,
deep-read entries dated 2026-05-27 onward.

This supersedes all earlier handovers (now in `docs/handovers/archive/`).

---

## TL;DR — where we are

The project's deliverable exists and the headline goal is met: **M3, a
sequence-only detector, beats classical RDP on real HIV** and **handles
cross-species inputs safely**.

- **Detector:** dilated CNN on a 22-channel parental-comparison encoding. LANL
  real-HIV F1 — 4-seed ensemble **0.565**, per-seed **0.554 ± 0.013**, every seed
  > classical RDP standalone (0.519). (The old "0.509" was a single low-end seed +
  a conservative edge-buffer convention, not a capability gap.)
- **Cross-species safety (M3 v4):** an unsupervised **divergence gate** flags
  out-of-distribution triplets (`div_max > 0.20`) and suppresses their peaks —
  Ebola false peaks 5.16 → 0.04 per triplet, with **zero** loss on real
  recombinants (LANL 0.509 kept, SARS-CoV-2 XBB still localized Δ=293 bp). Gate
  AUROC 0.982. All four v4 success criteria pass (`m3_eval_divgate.py`).
- **Deployment:** `bp_detect.py` — aligned 3-seq FASTA → probability track +
  breakpoint calls + recombinant-confidence + OOD warning.

**The active branch is `worktree-m12-mlm`** (this worktree). It is **not pushed**
and is many commits ahead of `main`. The user authorizes pushes explicitly — do
not push without checking.

**Do not delete or overwrite** `models_test/m3d_big_snaps/m3d_best.pt` (seed-0
detector) or `models_test/m3d_seed{1,2,3}_snaps/` (the ensemble).

---

## What is working vs what is dead

**Working / active (this is the whole product):**
- `m3_dilated.py` — training + model (`DilatedHead` = the BP detector).
- `m3_divergence_gate.py` + `m3_gated_detector.py` — the v4 two-head gated detector.
- `bp_detect.py` — CLI.
- `m3_eval_lanl.py`, `m3_eval_ensemble.py`, `m3_eval_divgate.py`,
  `m3_eval_multivirus{,_v2}.py`, `m3_sars_peaks.py` — eval harnesses.
- Data infra: `cache_v2_reader.py`, `build_cache_v2.py`, `data_loader_v2.py`,
  `splits/v2_filtered_split.json`, `cache/v2/` (gitignored, repo root).

**Dead — do NOT revive (all in `archive/`, all documented in memory + WRITEUP §4):**
- HyenaDNA + MLM pretraining (M1.x): the pretrained backbone *hurt* downstream
  detection. `backbone_hyenadna.py` / `pretrain_mlm.py` remain at root ONLY because
  `m3_dilated.py` imports them at module level for the (unused) hyena feature mode.
- Learned aux recombinant-classifier head (v4a): became a simulator-vs-real
  detector that suppressed real recombinants. Ckpt `models_test/m3d_v4froz.pt`
  kept as the record — do NOT deploy its gate. (The class `M3MultiHead` survives in
  `m3_dilated.py` for the record.)
- Training-mix cross-species negatives (v3): collapsed LANL F1 0.51 → 0.00.
- The pre-pivot Keras CNN + run29–43 era (`archive/keras_era/`).

---

## Concrete ways forward (pick one; none are started)

1. **Broaden the multi-virus positive set (component 2, ~40% → higher).** Curate
   confirmed cross-lineage recombinants beyond SARS-CoV-2 XBB.1.5 — XBC/XAY/XAS
   SARS lineages, HCV inter-genotype, HPV cross-type. Build them into the
   multi-virus eval (`m3_eval_multivirus.py` pattern). Strengthens the central
   "multi-virus" claim, which currently rests on one positive (XBB). **No training
   needed** — eval-only curation + aligning new triplets.

2. **Validate the gate threshold (0.20) on divergent CRFs.** The gate is calibrated
   between within-species (~0.13) and cross-species (~0.37) divergence, but
   divergent HIV subtype pairs (D/G/J-containing) run ~0.15–0.18, close to the
   boundary. Get D/G CRF triplets, confirm the gate keeps them. Eval-only.

3. **Push LANL further / solidify the win.** The margin over RDP is real but modest
   (n=35 breakpoints). The lever is *diversity*, not volume — M3 XL (50k events ×
   40 epochs of the *same* shards) overfit and regressed to 0.434. Candidate:
   train on more diverse SANTA shards (XML-6 has 97k unused HIV events; see the
   per-shard divergence table the prior session computed — XML-1..3 are ~0.27–0.34
   while LANL is ~0.13). **Caution (advisor-flagged):** don't narrow training toward
   LANL's divergence band — it would regress the low-divergence XBB/Zika wins. And
   characterize variance across ≥3 seeds before claiming any improvement.

4. **Polish the preprint (`docs/WRITEUP_M3.md`).** It's a structured draft. Add
   figures (divergence-vs-FP-rate, per-CRF F1 bars, SARS Spike-enrichment scatter),
   and the honest caveats are already in the text (n=35, test-optimized threshold,
   gate-threshold calibration).

---

## How to run things

See `README.md` "Quickstart". The essentials:
```bash
source /home/joshc/Dev/RDP_CNN/.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 bp_detect.py triplet.fa --out-prefix result          # deploy on a triplet
python3 m3_eval_ensemble.py --seed-ckpts <4 m3d_best.pt>      # headline benchmark
python3 m3_eval_divgate.py                                    # v4 gate scorecard
```

## Hard constraints (non-negotiable — full list in `CLAUDE.md`)
1. **OOM rule:** never `np.array(X, copy=True)` / `X.astype(...)` on multi-GB
   cached tensors — stream per-event (the desktop has been OOM-killed twice).
2. **GPU contention:** one CUDA training process at a time; use
   `CUDA_VISIBLE_DEVICES=""` for concurrent CPU-only eval.
3. **Per-epoch checkpointing from epoch 1** (wired into `m3_dilated.py` already).
4. Held-out splits are sacred: XML-5 = val, UnseenTestSet/LANL = test.
5. **No pushing to `origin` without explicit user approval.**
6. No `mamba-ssm` / `causal-conv1d` installs (no system nvcc).

---

## User's working style (observed across many sessions)
- **"ultrathink" requests deserve depth** — use the `advisor` tool, lay out
  trade-offs with concrete numbers.
- **Challenges premature framings** — bring data, not intuition. (The advisor
  caught two important framing errors this project: the learned-aux-head premise
  and the "beat 0.519" gap being noise. Heed it.)
- **Anti-friction on clarifying questions** — make reasonable defaults clear, ask
  only at genuine forks.
- **Authorizes destructive actions individually** — pushing, deleting, dropping
  data: get an explicit yes.
- **Values honest reporting of unexpected/negative results.** Several of this
  project's best contributions are well-characterized negative results.
