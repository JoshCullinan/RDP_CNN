# Handover — M3 v3 session, May 2026

You are picking up after M3 v2 → multi-virus exploration → M3 v3 launch.

**TL;DR:** M3 v2 (LANL F1 0.509, peer to classical RDP) is our deployed model. Multi-virus eval revealed:
- **SARS-CoV-2 generalization works** (XBB.1.5 hit at Δ=293 bp from literature truth).
- **Zika clean** after fixing edge artifacts.
- **Ebola cross-species FAILS** (~5 peaks/triplet) — model can't distinguish constant high divergence from real recombination.

A retrain with **15% cross-species negative-control triplets** (M3 v3) is currently running as background job `bosbvhrf3` to address the Ebola failure. Expected to finish ~3 hours from launch.

**Active branch:** `worktree-m12-mlm`, multiple commits ahead of main, NOT pushed.

---

## Read these first

1. `MASTER_PLAN.md` — has a 2026-05-28 status header at the top documenting actual completed milestones (the original Phase 1-4 plan is wrong).
2. `MEMORY.md` index — most relevant entries (newest first):
   - `project_m3_sars_peaks_analysis.md` — SARS "FPs" are partially real biology (1.81× Spike enrichment)
   - `project_m3_multivirus_v2.md` — 147-triplet enumerated eval
   - `project_m3_multivirus.md` — initial multi-virus single-triplet test
   - `project_m3_lanl_v2.md` — **the current best model** (LANL F1 0.509)
   - `project_m3_lanl_breakthrough.md` — M3 v1 (LANL F1 0.409, superseded)
   - `project_m13_pretraining_hurts.md` — why HyenaDNA + MLM didn't work
3. `HANDOVER_M3_V2.md` — previous session's handover (still mostly relevant).

---

## Current best models

| Ckpt | Architecture | Training | LANL F1 |
|---|---|---|---|
| `models_test/m3d_big_snaps/m3d_best.pt` | Dilated CNN on raw 22ch | 20k events × 25 ep, XML-1..4 | **0.509** |
| `models_test/m3d_xl_snaps/m3d_best.pt` | Same, 50k × 40 ep | Same data | 0.434 (overfit) |
| `models_test/m3d_neg_snaps/m3d_best.pt` | Same + 15% negative-controls | 20k events × 25 ep + ~3k cross-species negatives | **PENDING (job bosbvhrf3)** |

---

## What needs doing in the next session

### 1. Evaluate M3 v3 (when it finishes)

The neg-mix retrain should be complete or close to it when you pick up. To evaluate:

```bash
source /home/joshc/Dev/RDP_CNN/.venv/bin/activate
cd .claude/worktrees/m12-mlm

# Check if still running
ps aux | grep "m3_dilated.py.*neg-frac" | grep -v grep

# When done, eval on LANL (compare to v2's 0.509)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u m3_eval_lanl.py \
    --ckpt models_test/m3d_neg_snaps/m3d_best.pt \
    --out models_test/m3_lanl_v3.json

# Eval on multi-virus (compare to v2 with edge_buffer=200)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u m3_eval_multivirus_v2.py \
    --ckpt models_test/m3d_neg_snaps/m3d_best.pt \
    --out models_test/m3_mv_v3.json
```

Success criteria:
- **LANL F1 stays ≥ 0.50** (don't regress while fixing Ebola)
- **Ebola peaks@0.8 drops below ~2.0** (currently 5.16 at edge_buffer=200)
- **SARS-CoV-2 stays ≤ 2.5** (currently 2.11 at edge_buffer=200)

### 2. Multi-virus production pipeline

If M3 v3 works, the natural next milestone is a single-script eval pipeline that takes any 3-sequence FASTA and reports breakpoint probability + peak positions + a "divergence-anomaly" warning. Useful for biologists. ~2 hours.

### 3. Push past classical RDP (LANL F1 > 0.519)

Current v2 is 0.010 below classical RDP. Plausible avenues:
- Add more shards: filtered set + long_content_30k_001 to widen distribution
- Ensemble multiple checkpoints (v2 + early v2 + later v2 epochs)
- Two-pass training: train v3, fine-tune the head on LANL-like (within-species, mixed-subtype) augmented data
- Tune threshold per CRF rather than aggregate

### 4. Multi-virus deployment paper

Currently we have:
- LANL F1 0.509 (HIV)
- SARS-CoV-2 XBB hit Δ=293 bp from literature truth
- Spike-region peak enrichment 1.81× on cross-strain SARS-CoV-2 triplets
- Zika clean negative controls

This is genuinely a publishable result. Could write up as "first sequence-only ML breakpoint detector matching classical methods across multiple virus families."

---

## What NOT to do

1. **Do NOT touch the M3 v2 ckpt.** It's the current best. Even if v3 fails, v2 should remain accessible.
2. **Do NOT resurrect HyenaDNA / MLM.** Conclusively dead end (5 days of failure + 6 independent tests).
3. **Do NOT add more data diversity by adding XML-5 or long_content shards to TRAIN.** Those are val/test. Using them in train would invalidate the LANL benchmark. The cross-species negatives I added come from `data/real_recombinants/` (different lineage, not the SANTA test sets).
4. **Do NOT push to origin without explicit user approval.**

---

## File map

### M3 active code
- `m3_dilated.py` — main training script (now with `--neg-frac` for v3)
- `m3_eval.py` — eval on SANTA shards (UnseenTestSet)
- `m3_eval_lanl.py` — LANL real-HIV CRF eval (deployment benchmark)
- `m3_eval_multivirus.py` — single-triplet multi-virus probe (v1)
- `m3_eval_multivirus_v2.py` — enumerated multi-virus (now with edge_buffer=200)
- `m3_sars_peaks.py` — peak position analysis (revealed Spike enrichment)
- `m3_train.py` — earlier full-fine-tune attempt (negative result, kept for posterity)

### Older infrastructure
- `pretrain_mlm.py`, `m12_*.py`, `m13_*.py`, `snapshot_ckpts.py` — M1.2/M1.3 era

### Real virus panels
- `data/lanl_crf/` — 4 LANL CRF triplets + truth BPs (HIV)
- `data/real_recombinants/{ebola, zika, sarscov2_full, sarscov2_orf1ab, sarscov2_spike}/` — reference panels for multi-virus eval

---

## Background job state

When you pick up, check:
```bash
# Is M3 v3 training still running?
ls -la /home/joshc/.claude/jobs/60855c79/m3d_neg.log
tail -20 /home/joshc/.claude/jobs/60855c79/m3d_neg.log

# Process state
ps aux | grep m3_dilated | grep -v grep
nvidia-smi
```

If still running → wait for completion.
If finished cleanly → run the M3 v3 evals.
If failed/crashed → check log, decide whether to relaunch with `--resume` or different hyperparameters.

---

## Reward

Two days ago: 5 days of failed HyenaDNA pretraining and an unclear path forward.
Now: A sequence-only HIV breakpoint detector that matches classical RDP, generalizes to SARS-CoV-2 XBB.1.5, and has a fixable known failure mode on cross-species comparisons.

This is publishable work. Keep iterating.
