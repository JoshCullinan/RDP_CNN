# Handover — RDP_CNN project, May 2026

You are picking up after a 10-day arc of work on a sequence-only viral recombination breakpoint detector. Read this whole document before doing anything. Then read `WRITEUP_M3.md` for the full story.

This file supersedes the May-19 handover by the same name; that older version is preserved in git history (commit `fc315e3` and earlier) if you need it.

---

## TL;DR — where we are right now

The project's deliverable exists. **M3 v2** is a sequence-only breakpoint detector hitting **LANL real-HIV F1 0.509** (within 0.010 of classical RDP's 0.519 standalone). It also correctly localizes the SARS-CoV-2 XBB.1.5 recombination breakpoint to within 293 bp of the documented literature truth. This is the first sequence-only ML detector in the project's history that actually works on real virus data.

There's one known failure mode (cross-species Ebola triplets at ~30-40% divergence produce ~5 false peaks per triplet) and one failed fix attempt (v3, abandoned). The whole journey including the failed HyenaDNA pretraining detour and the v3 collapse is documented in `WRITEUP_M3.md`.

**The active branch is `worktree-m12-mlm`.** It's ~13 commits ahead of `main` and NOT pushed. The user explicitly authorizes pushes only when they say so — do not push without checking.

**The deployed model:** `models_test/m3d_big_snaps/m3d_best.pt` (M3 v2). 3.6 MB. Do not delete or overwrite it.

---

## Your specific goal for this session

**Build M3 v4: a multi-head architecture that fixes the Ebola cross-species failure without regressing LANL F1.**

### Why this and not something else

The Ebola failure is the largest documented limitation in our results writeup. Fixing it converts "first sequence-only HIV+SARS-CoV-2 detector with a known limitation" into "first sequence-only multi-virus detector that handles cross-species comparisons safely." That's substantively more publishable.

The v3 attempt failed because it used the wrong mechanism: injecting cross-species negative-control triplets at 15% rate produced a backbone-level "predict zero on divergent input" prior that destroyed LANL transfer (F1 0.509 → 0.000). See `project_m3_v3_failed.md` for the post-mortem.

### The architecture you should build

A two-head model that decouples "where" (BP head) from "whether" (recombinant classifier):

```python
class M3MultiHead(nn.Module):
    def __init__(self, in_channels=22, hidden=128, n_blocks=6, dropout=0.1):
        super().__init__()
        # Shared trunk (same dilated CNN as M3 v2)
        self.trunk = DilatedTrunk(in_channels, hidden, n_blocks, dropout)
        # BP head: per-position breakpoint probability (same as M3 v2)
        self.bp_head = nn.Conv1d(hidden, 1, kernel_size=1)
        # Auxiliary head: single output per triplet — "is this a recombinant?"
        # Uses global average pooling over the trunk features.
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: (B, L, C_in)
        feats = self.trunk(x.transpose(1, 2))     # (B, hidden, L)
        bp_logits = self.bp_head(feats).squeeze(1)             # (B, L)
        aux_logit = self.aux_head(feats).squeeze(-1)           # (B,)
        return bp_logits, aux_logit
```

### The training regime

For each training step, randomly sample either a SANTA-positive event OR a real-virus negative-control triplet (use `--neg-frac` from the existing code, but **set it to 0.05, not 0.15** — gentler dose).

Loss = `L_bp + λ_aux * L_aux` where:

- `L_bp` = weighted BCE per position, **only computed on positives** (skip for negatives — don't push BP predictions to zero across negatives, that's what killed v3)
- `L_aux` = single-class BCE: `BCELoss(sigmoid(aux_logit), is_recombinant)` where `is_recombinant` is 1 for SANTA positives and 0 for negative-controls
- `λ_aux` starts at 0.5; tune if needed

### At inference time

Compute `aux_prob = sigmoid(aux_logit)`. Use it as a confidence gate:

```python
if aux_prob < 0.5:
    return "no recombination signal in this triplet (aux_prob={aux_prob:.3f})"
else:
    peaks = extract_peaks(bp_probs, threshold)
    return peaks
```

The expected behavior:
- HIV LANL CRFs: `aux_prob` should be high (true recombinants), BP predictions flow normally → preserve F1 0.509.
- SARS-CoV-2 XBB.1.5: `aux_prob` high, BP localization at ~22577 preserved.
- Zika cross-strain: `aux_prob` low, no peaks emitted.
- Ebola cross-species: `aux_prob` low, no peaks emitted ← **this is the fix.**

### Success criteria for M3 v4

1. **LANL aggregate F1 ≥ 0.49** (within 0.02 of v2's 0.509 — don't regress)
2. **Ebola peaks @ thr=0.80 ≤ 1.5** (down from 5.16 with edge_buffer=200)
3. **SARS-CoV-2 XBB.1.5 still detected** within Δ=500 bp of position 22577
4. **Auxiliary classifier AUROC ≥ 0.85** on a held-out positive/negative mix

If you hit all four, commit, eval, write up. If you miss any, debug and iterate before declaring done.

### Concrete first steps

1. Read `m3_dilated.py` to understand the current single-head architecture and the `--neg-frac` plumbing (already there from the v3 attempt — you only need to add the auxiliary head and the joint loss).
2. Add a new flag `--head-mode {single, multi}` to support the architecture switch.
3. Implement `M3MultiHead` as above. Reuse the existing `DilatedHead` body — just refactor to expose the per-position features before the final `Conv1d` head, then add the aux head.
4. Implement the joint-loss training step. The current loop builds `feats` from the backbone (None for raw), runs `head(feats)`, computes BCE. You need to:
   - Track whether the current event is a positive (SANTA) or negative (sampled from real panels) — already implied by `ev["bps"]` being empty or not.
   - Compute `L_bp` only on positives (use `loss = bce(bp_logits, y, pos_weight=pw) if len(ev["bps"]) > 0 else 0`).
   - Compute `L_aux` always: `target = 1.0 if positive else 0.0`.
   - Sum and backward.
5. Add the auxiliary head's prediction to the eval reports (multi-virus + LANL).
6. Train: 20k events × 25 epochs, neg_frac=0.05, lr=1e-3, pos_weight=70 (matching v2).
7. Eval on LANL, multi-virus, and a positive-vs-negative confusion matrix for the aux head.

ETA: 6-10 hours wall time (mostly the 2.5h training run twice if you need to iterate).

---

## What to do if M3 v4 hits success criteria

Update `MASTER_PLAN.md` with a v4 entry. Save a memory file `project_m3_v4_multihead.md`. Update the writeup section that currently says "An attempted fix... is not deployed" to reflect the new working version. Then move to one of the longer-term goals below.

## What to do if M3 v4 fails

Document the failure mode (memory: `project_m3_v4_failed.md`) and consider alternatives:

- **λ_aux too aggressive?** Try 0.1 instead of 0.5.
- **Negative-control divergence still too close to LANL?** Use only very-high-divergence (>25%) negatives: Ebola Zaire vs Reston, Ebola vs Bombali, etc.
- **Multi-task interference?** Try sequential training: train the aux head only first (with the trunk frozen), then unfreeze and add the BP loss.
- **Architecture too small for two tasks?** Bump trunk hidden from 128 to 256 and re-test.

After 1-2 iterations, if nothing works, fall back to the gentler-neg-frac approach with a single head (v3 with neg_frac=0.05 — currently untried) and report.

---

## The long-term goal

**Production-grade multi-virus sequence-only recombination breakpoint detector matching or exceeding classical-method performance, suitable for both research deployment and publication.**

Concretely, this means:

1. **LANL F1 ≥ 0.55** on real HIV (clearly beats classical RDP at 0.519, beats the legacy CNN runB2_sig10 fusion baseline at ~0.43-0.53).
2. **Multi-virus positive detection** confirmed on at least 3 cross-lineage recombinants beyond HIV (SARS-CoV-2 XBB.1.5 is the existing one; XBC, XAY, HCV inter-genotype, HPV cross-type are candidates).
3. **Safe failure mode handling** — the model declines to predict (or warns) on inputs outside its training distribution. The multi-head architecture above is one path; others include explicit divergence-anomaly scoring.
4. **Deployment CLI** — wrap M3 as `bp_detect <fasta>` that takes any 3-sequence FASTA and emits a probability track + peak calls + a confidence rating. ~3-4 hours of focused work once the model is settled.
5. **Publication / preprint** — `WRITEUP_M3.md` is already structured as a draft. Polish, add figures (a divergence-vs-FP-rate plot, a per-CRF F1 bar chart, a peak-position scatter on SARS-CoV-2 showing Spike enrichment), submit to bioRxiv / Bioinformatics / similar.

Hitting all 5 is the project's terminal state. Right now (May 28) we have ~80% of (1), 30% of (2), failed first attempt at (3), 0% of (4), 50% of (5).

---

## What is "active" vs "dead"

**Active:**
- M3 architecture (dilated CNN on raw 22 channels) — this is the right backbone
- Memory files dated 2026-05-27 onward (`project_m3_lanl_v2.md`, `project_m3_multivirus_*.md`, `project_m3_sars_peaks_analysis.md`)
- `m3_dilated.py`, `m3_eval_lanl.py`, `m3_eval_multivirus_v2.py`, `m3_sars_peaks.py`
- `models_test/m3d_big_snaps/m3d_best.pt` — the v2 ckpt

**Dead — do not resurrect:**
- HyenaDNA backbone. Five days of MLM pretraining (M1.2) produced a backbone that ACTIVELY HURT downstream breakpoint detection. Memory `project_m13_pretraining_hurts.md` documents 6 independent tests confirming this. Random-init Hyena even loses to random raw 22ch.
- M1.2 pretrained checkpoint (`models_test/snapshots/backbone_mlm_v1_e13_30k_gs292630.pt`). Keep it for reference, do not load it.
- M3 v3 (cross-species negatives at 15%). Catastrophic collapse documented in `project_m3_v3_failed.md`. Do not retry with neg_frac ≥ 0.10 on a single head.
- The original MASTER_PLAN.md Phase 1-4 milestones. The 2026-05-28 status header at the top supersedes them.

---

## Hard constraints (non-negotiable)

1. **OOM rule:** never `np.array(X, copy=True)` or `X.astype(np.float32)` on multi-GB cached tensors. The user's desktop has been OOM-killed twice in this project. Use the RSS-watchdog pattern from `pretrain_mlm.py` (26 GB cap).
2. **GPU contention:** if a training run is active, do not launch a second CUDA process. Use `CUDA_VISIBLE_DEVICES=""` for any concurrent CPU-only eval.
3. **Per-stage checkpointing from epoch 1.** The M1.2 run lost its peak ckpt because the snapshotter was added late. Don't repeat that mistake — wire snapshot saving into every training run from the start.
4. **No `find_peaks` in the loss.** It's a post-hoc evaluator. Model outputs are per-position sigmoids.
5. **Held-out splits are sacred.** XML-5 is val, UnseenTestSet is test, LANL is test. Don't train on them.
6. **No pushing to origin without explicit user approval.** Local main is ~13 commits ahead.
7. **No `mamba-ssm` / `causal-conv1d` install attempts.** Will fail (no system nvcc). Investigation is in `backbone_mamba.py` header.

---

## Hardware & environment

- Linux Ubuntu 26.04, RTX 3070 (8 GB VRAM, compute 8.6, driver 595.71.05 with CUDA 13.2 runtime)
- 30 GB RAM, 8 GB swap
- Python 3.12 in `.venv/`. Activate: `source /home/joshc/Dev/RDP_CNN/.venv/bin/activate`
- PyTorch 2.6.0+cu124 — the active stack
- TensorFlow 2.18 — only for legacy CNN eval (don't touch for M3+ work)
- bf16 is supported and is the default for M3 training (no GradScaler needed)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — always set for training to reduce VRAM fragmentation

---

## Read these in order

1. **This file.**
2. `WRITEUP_M3.md` — full results writeup, ~12 pages, the canonical narrative.
3. `MEMORY.md` index at `/home/joshc/.claude/projects/-home-joshc-Dev-RDP-CNN/memory/MEMORY.md`. Skim all entries; deep-read the ones dated 2026-05-27 or 28.
4. `MASTER_PLAN.md` — read the 2026-05-28 status header at top, skim the rest.
5. `CLAUDE.md` — repo conventions.
6. `m3_dilated.py` — the architecture you'll be extending. ~500 lines, the core training loop is in `train()`.
7. `m3_eval_lanl.py` and `m3_eval_multivirus_v2.py` — the eval harnesses you'll use to validate v4.

---

## Quick reference: key files

```
# M3 era (active)
m3_dilated.py              # training script (with --neg-frac support)
m3_eval.py                 # SANTA eval
m3_eval_lanl.py            # LANL real-HIV eval (deployment benchmark)
m3_eval_multivirus.py      # single-triplet probe (v1)
m3_eval_multivirus_v2.py   # enumerated multi-virus (now with edge_buffer=200)
m3_sars_peaks.py           # SARS-CoV-2 peak position analysis
m3_train.py                # earlier full fine-tune attempt (negative result)

# Pre-M3 era (legacy, retained)
pretrain_mlm.py            # M1.2 MLM training (dead direction)
m13_linear_probe.py        # M1.3 probes (showed Hyena features fail)
m13_combined_probe.py      # Path I (confirmed Hyena adds nothing)
m12_*.py                   # M1.2-pre zero-shot probes
eval_*.py                  # M1.2 diagnostics
snapshot_ckpts.py          # ckpt snapshotter — reuse this in v4

# Data
data/lanl_crf/             # LANL HIV CRFs + truth_bps.csv (test only)
data/real_recombinants/    # 5 reference panels (eval, also negative-control source)
cache/v2/                  # SANTA cache (42 GB, gitignored, at repo root)
splits/v2_filtered_split.json  # The M0.5 realism-filtered split

# Models
models_test/m3d_big_snaps/m3d_best.pt    # M3 v2 — current deployment
models_test/m3d_xl_snaps/                # M3 XL (overfit, kept for ensembling)
models_test/m3d_neg_snaps/m3d_best.pt    # M3 v3 (failed, do not load)
models_test/snapshots/                   # M1.2 ckpts (dead direction)

# Documentation
WRITEUP_M3.md              # The canonical results writeup
HANDOVER_M3_V3.md          # Previous session handover
HANDOVER_NEXT_AGENT.md     # This file
MASTER_PLAN.md             # Now has a 2026-05-28 status header at top
CLAUDE.md                  # Repo conventions
```

---

## User's working style (observed across 10+ sessions)

- **Ultra-think requests** ("ultrathink") deserve depth. Use the advisor tool. Lay out trade-offs explicitly with concrete numbers.
- **Anti-friction** on clarifying questions — make reasonable defaults clear, ask only for genuinely ambiguous decisions.
- **Challenges premature framings.** Bring data, not intuition.
- **Authorizes destructive actions individually.** Pushing, deleting, dropping data — get explicit approval.
- **Values honest reporting of unexpected results.** The M1.2 → M1.3 → "MLM hurts" finding was a difficult report; the user appreciated the framing of "5 days of compute = 1 valuable negative result."
- **Has a real thesis on this work.** The original repo includes `Detecting Viral Recombination with Machine Learning Thesis - Joshua Cullinan.pdf`. M3's LANL F1 0.509 is a genuine contribution to that thesis line.

---

## Reward

This project has produced a result that didn't exist 10 days ago: a sequence-only ML detector that matches classical RDP on real HIV. The HyenaDNA detour and the v3 failure are both teachable moments captured in memory. The writeup is structured for publication.

Your job is to push it further — fix Ebola, beat 0.519, ship a CLI, get the preprint out.

Good luck.
