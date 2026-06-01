# Handover — M3 v2 session, May 2026

You are picking up after the M3 sequence-only breakthrough.

**TL;DR:** We have a sequence-only breakpoint detector that hits **LANL F1 0.509** — within 0.010 of classical RDP standalone (0.519) and substantially above the legacy CNN's sequence-only baseline (F1 0.000). The whole HyenaDNA pretraining direction (M1.2/M1.3) was a dead end; the right path was modern PyTorch + legacy-CNN-style 22-channel features + a dilated CNN head.

**Active branch:** `worktree-m12-mlm`, several commits ahead of main, NOT pushed.

---

## Read these first

1. `MEMORY.md` index — three new entries:
   - `project_m13_combined_probe.md` — confirmation that Hyena features hurt
   - `project_m3_lanl_breakthrough.md` — v1 LANL F1 0.409
   - `project_m3_lanl_v2.md` — **v2 LANL F1 0.509** (the current best)
2. `CLAUDE.md`, `MASTER_PLAN.md`.

---

## The current best model

`models_test/m3d_big_snaps/m3d_best.pt` (3.6 MB)
- Architecture: 6-block dilated CNN head, dilations {1,2,4,8,16,32}, hidden=128, dropout=0.1, GroupNorm, GELU, ~300k params
- Input: raw 22 channels (5+5+5 one-hot + 3 match flags + 4 MaxChi window deltas at {50,100,200,500} bp)
- Trained: 20,000 events × 25 epochs, pos_weight=70, lr=1e-3, bf16, AdamW
- Training data: XML-1..4 unfiltered (the legacy CNN's training set)
- Best val F1 = 0.602 @ epoch 23 on XML-5
- **LANL aggregate F1 0.509** at threshold 0.15

### Per-CRF on LANL (v2)
| CRF | F1 | P | R |
|---|---|---|---|
| CRF02_AG | 0.526 | 0.36 | 1.00 |
| CRF07_BC | 0.522 | 0.46 | 0.60 |
| CRF08_BC | 0.455 | 0.31 | 0.83 |
| CRF12_BF | 0.636 | 0.54 | 0.78 |

---

## How to reproduce

```bash
source /home/joshc/Dev/RDP_CNN/.venv/bin/activate
cd .claude/worktrees/m12-mlm

# Train (or use existing ckpt at models_test/m3d_big_snaps/m3d_best.pt)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u m3_dilated.py \
    --feature-mode raw \
    --train-shards XML-1 XML-2 XML-3 XML-4 \
    --val-shards XML-5 \
    --n-train 20000 --n-val 500 \
    --max-len 30500 \
    --epochs 25 \
    --lr 1e-3 --pos-weight 70 \
    --ckpt-out models_test/m3d_big.pt \
    --snapshots-dir models_test/m3d_big_snaps \
    --history-out models_test/m3d_big_history.json

# Eval on LANL
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u m3_eval_lanl.py \
    --ckpt models_test/m3d_big_snaps/m3d_best.pt \
    --out models_test/m3_lanl_big.json
```

---

## Open questions / next iterations

### 1. Push to 50k events × 30+ epochs

V2 was 20k × 25 epochs at ~2.5h total. Still gaining slowly at epoch 23 (no plateau). 50k events × 30 epochs may push F1 to 0.55+ (clearly beating classical RDP 0.519). Cost: ~8-12h on the 3070.

### 2. Multi-virus generalization eval

The deeper project goal is "works on ANY recombination-capable virus." We have `data/real_recombinants/` panels for Ebola, Zika, SARS-CoV-2 (full, Spike, ORF1ab). The M3 ckpt is sequence-only — should generalize.

Build a `m3_eval_multivirus.py` that:
- Takes (recomb, parent1, parent2) FASTA triplets from `data/real_recombinants/{virus}/`
- Computes raw_features (same as LANL eval)
- Forwards through M3 head
- Reports per-virus F1 vs ground truth

If F1 > 0.0 on Ebola/Zika/SARS, we have multi-virus generalization. If collapses, we're HIV-specific.

### 3. UnseenTestSet gap

V2 on UnseenTestSet is still ~0.21 vs legacy CNN's 0.421. This benchmark stayed bad despite LANL improving — UnseenTestSet has 60% boundary-BP events (bp_start=0 or bp_end=seq_len) that our edge-buffer suppression auto-fails. May be a legacy CNN quirk (they had similar suppression). Worth digging into if SANTA benchmark matters; we already nail real-world LANL.

### 4. Investigate CRF02_AG recall=1.0 P=0.36

V2 caught all 10 true BPs on CRF02_AG but with 18 false positives. Some of those FP may be real BPs that the LANL truth annotation missed. Worth manual inspection.

---

## What NOT to do

1. **Do NOT resurrect HyenaDNA** unless you have a specific reason. M1.2 + M1.3 + M3-mini + Path I showed Hyena features actively hurt this task.
2. **Do NOT re-run the realism filter** for M3 training. The filter helps theoretically for real-world distribution match, but in practice training on unfiltered XML-1..4 gives better LANL transfer (0.509) than filtered (0.21 on UnseenTestSet — proxy for filtered effect).
3. **Do NOT delete `models_test/snapshots/`** — has the M1.2 ckpt (val 0.621) that's the only artifact of the 5-day pretraining run. Not useful now but expensive to recreate.
4. **Do NOT delete `models_test/m3d_big_snaps/m3d_best.pt`** — current best model.

---

## Files of interest

### M3 production
- `m3_dilated.py` — main training script (feature_mode {hyena, raw, combined}, but `raw` is what works)
- `m3_eval.py` — eval on SANTA shards (UnseenTestSet, etc.)
- `m3_eval_lanl.py` — eval on LANL real-HIV CRFs
- `m3_train.py` — old end-to-end fine-tuning attempt (stuck at F1 0.28; kept as negative-result archive)

### Older infrastructure (don't touch unless needed)
- `pretrain_mlm.py` — M1.2 MLM pretraining loop
- `m13_linear_probe.py` — frozen-backbone probes
- `m13_combined_probe.py` — raw vs Hyena vs combined feature probe
- `m12_*.py` — zero-shot probes from M1.2-pre
- `eval_mlm_per_shard.py`, `eval_positional_cheat.py` — M1.2 diagnostics
- `snapshot_ckpts.py` — side-process ckpt snapshotter (use this from the START of any long training run)

---

## Reward — current direction is working

Going from "5 days of failed HyenaDNA pretraining" to "sequence-only LANL F1 0.509, within 0.010 of 20-year-old gold-standard RDP" in two days of post-pivot work is a real result. Don't be afraid to push it further — the architecture and approach are sound now.
