# RDP_CNN — sequence-only viral recombination breakpoint detection

Detect recombination **breakpoints** in aligned virus triplets
`(recombinant, parent1, parent2)` from **nucleotide sequence alone** — no
precomputed RDP / GeneConv / MaxChi method outputs. The goal is to match or beat
the classical detection suite (RDP, MaxChi, GeneConv) and deploy across multiple
virus families.

> **New here? Read in this order:** this README → `HANDOVER_NEXT_AGENT.md` (current
> state + ways forward) → `docs/ARCHITECTURE.md` (how the model works) →
> `docs/WRITEUP_M3.md` (full results, paper draft) → `CLAUDE.md` (repo conventions).
> Cross-session memory lives at
> `/home/joshc/.claude/projects/-home-joshc-Dev-RDP-CNN/memory/`.

---

## Status (2026-06-01)

The deliverable exists and works. **M3** is a sequence-only detector; **M3 v4**
adds a cross-species safety gate. Long-term goal = 5 components:

| # | Component | Status | Evidence |
|---|---|---|---|
| 1 | Detector ≥ classical methods | ✅ **done** | LANL real-HIV: 4-seed ensemble **F1 0.565** (per-seed 0.554 ± 0.013), every seed beats classical RDP 0.519 |
| 2 | Multi-virus | 🟡 ~40% | HIV ✅ + SARS-CoV-2 XBB.1.5 ✅ (Δ=293 bp); Zika/Ebola handled safely |
| 3 | Safe failure handling | ✅ ~80% | divergence gate suppresses cross-species Ebola (5.16 → 0.04 false peaks/triplet) |
| 4 | Deployment CLI | ✅ **done** | `bp_detect.py` — FASTA in, track + calls + confidence/warning out |
| 5 | Preprint | 🟡 ~65% | `docs/WRITEUP_M3.md` is a structured draft |

**What is working, in one paragraph:** a residual dilated CNN reads a 22-channel
parental-comparison encoding to produce a per-position breakpoint-probability
track (the *detector*, F1 ≈ 0.55 on real HIV, beating classical RDP across 4
seeds). An unsupervised **divergence gate** decides whether a triplet is within
the trained regime; cross-species comparisons (divergence > 0.20) are flagged
out-of-distribution and their peaks suppressed. The whole thing wraps a 4-seed
ensemble for variance control and is exposed via the `bp_detect` CLI.

**Dead ends (do not revive — see `archive/` and memory):** HyenaDNA + MLM
pretraining (the pretrained backbone *hurt* downstream detection); a learned
"is-this-a-recombinant?" aux head (became a simulator-vs-real detector that
suppressed real recombinants); training-mix cross-species negatives (collapsed
LANL F1 0.51 → 0.00).

---

## Repository map

```
RDP_CNN/
├── README.md                 ← you are here (master index)
├── HANDOVER_NEXT_AGENT.md    ← current state + concrete ways forward
├── CLAUDE.md                 ← conventions, OOM rule, hard constraints
├── requirements.txt
│
│  ── ACTIVE M3 PIPELINE (flat at root; sibling imports — keep together) ──
├── m3_dilated.py             ← training + model defs (DilatedHead BP head, M3MultiHead)
├── m3_divergence_gate.py     ← the unsupervised cross-species gate (no training)
├── m3_gated_detector.py      ← M3GatedDetector: two-head (BP + divergence gate)
├── bp_detect.py              ← deployment CLI
├── m3_eval_lanl.py           ← LANL real-HIV benchmark (the headline metric)
├── m3_eval_ensemble.py       ← multi-seed variance + ensemble (beats-RDP claim)
├── m3_eval_divgate.py        ← 4-criteria scorecard for the v4 gate
├── m3_eval_multivirus.py / _v2.py  ← SARS/Ebola/Zika positives + negatives
├── m3_eval_aux.py            ← aux-head AUROC harness (from the failed v4a)
├── m3_sars_peaks.py          ← SARS-CoV-2 Spike-enrichment analysis
├── cache_v2_reader.py, build_cache_v2.py, data_loader_v2.py  ← data infra
├── build_splits_v2.py, build_filtered_split.py, build_lanl_triplets.py  ← splits
├── pick_parents_rdp5ml.py, parse_lanl_breakpoints.py, fetch_real_recombinants.py
├── backbone_hyenadna.py, pretrain_mlm.py  ← DEAD HyenaDNA dir, retained ONLY
│                                            because m3_dilated imports them at
│                                            module level (raw path never uses them)
│
├── docs/
│   ├── ARCHITECTURE.md       ← how the model works, end to end
│   ├── WRITEUP_M3.md         ← full results / paper draft (canonical narrative)
│   ├── MASTER_PLAN.md        ← milestone log + status header
│   ├── handovers/archive/    ← superseded handovers (historical)
│   └── archive/              ← pre-pivot idea/TODO logs (historical)
│
├── tests/                    ← test_cache_v2.py, test_data_loader_v2.py
├── data/                     ← real panels (lanl_crf/, real_recombinants/) + truth
├── splits/                   ← v2_filtered_split.json (the canonical training split)
├── santaSim_RDP/             ← the SANTA simulator XMLs (training-data generator)
├── models_test/              ← checkpoints (GITIGNORED; see below)
├── cache/                    ← int8 SANTA cache (GITIGNORED, ~42 GB, at repo root)
└── archive/                  ← all superseded code + outputs (see archive/README.md)
    ├── keras_era/            ← pre-pivot Keras CNN + run29–43 one-offs
    ├── hyenadna_era/         ← dead M1.x HyenaDNA/MLM/realism scripts
    ├── superseded_m3/        ← early M3 attempts (negative results)
    ├── phase0_audit/         ← M0.4 baseline verification
    └── old_outputs/          ← stale result JSONs, run logs, diagnostics
```

**Key checkpoints** (in `models_test/`, gitignored — not in version control):
- `m3d_big_snaps/m3d_best.pt` — seed-0 deployed BP detector (M3 v2).
- `m3d_seed{1,2,3}_snaps/m3d_best.pt` — the 3 extra seeds for the ensemble.
- `m3d_v4froz.pt` — the **failed** learned-aux v4a (kept as record; do NOT deploy).

---

## Quickstart

```bash
source /home/joshc/Dev/RDP_CNN/.venv/bin/activate   # Python 3.12, torch 2.6+cu124
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1) Run the detector on an aligned 3-sequence FASTA (recombinant first)
python3 bp_detect.py triplet.fa --out-prefix result
#    → result.track.tsv (per-position prob), result.peaks.tsv, result.json
#    cross-species inputs are auto-flagged OOD and their peaks suppressed.

# 2) Reproduce the headline benchmark (beats classical RDP)
python3 m3_eval_ensemble.py --seed-ckpts \
    models_test/m3d_big_snaps/m3d_best.pt models_test/m3d_seed1_snaps/m3d_best.pt \
    models_test/m3d_seed2_snaps/m3d_best.pt models_test/m3d_seed3_snaps/m3d_best.pt
#    → per-seed F1 0.554±0.013, ensemble 0.565  (RDP standalone = 0.519)

# 3) Validate the v4 cross-species gate (4-criteria scorecard)
python3 m3_eval_divgate.py        # LANL 0.509, Ebola 0.04, XBB Δ=293, gate AUROC 0.982

# 4) Train a detector from scratch (~70 min on an RTX 3070)
python3 m3_dilated.py --feature-mode raw --head-mode single \
    --train-shards XML-1 XML-2 XML-3 XML-4 --val-shards XML-5 \
    --n-train 20000 --n-val 500 --max-len 30500 --epochs 25 --lr 1e-3 \
    --pos-weight 70 --seed 0 \
    --ckpt-out models_test/m3d.pt --snapshots-dir models_test/m3d_snaps
```

---

## Ways forward (see `HANDOVER_NEXT_AGENT.md` for detail)

1. **Push LANL higher / broaden the win** — the margin over RDP is real but modest
   (n=35 breakpoints). More diverse SANTA shards *might* help (M3 XL showed more of
   the *same* data overfits — diversity, not volume).
2. **Broaden multi-virus positives** (component 2) — curate confirmed cross-lineage
   recombinants beyond XBB.1.5 (XBC/XAY SARS, HCV, HPV).
3. **Validate the gate threshold (0.20)** on more-divergent HIV CRFs (D/G/J subtypes,
   divergence ~0.15–0.18) to confirm it doesn't suppress genuine recombinants.
4. **Polish the preprint** (`docs/WRITEUP_M3.md`) — figures, the threshold caveat.

## Hard constraints (non-negotiable — full list in `CLAUDE.md`)
- **OOM rule:** never copy/`astype` multi-GB cached tensors; stream per-event.
- Held-out splits are sacred: XML-5 = val, UnseenTestSet/LANL = test.
- No pushing to `origin` without explicit user approval.
- No HyenaDNA/Mamba revival; no `mamba-ssm`/`causal-conv1d` installs (no system nvcc).
