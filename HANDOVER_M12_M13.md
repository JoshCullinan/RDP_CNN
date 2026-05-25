# Handover — end of M1.2 + M1.3 session, May 2026

You are picking up after the M1.2 MLM pretraining run + M1.3 linear-probe
ablations. **Read this before doing anything.** Then re-read `CLAUDE.md`
and `MASTER_PLAN.md`. The auto-memory files at
`/home/joshc/.claude/projects/-home-joshc-Dev-RDP-CNN/memory/` are also
up to date — start with `MEMORY.md`, especially the two new entries:

- `project_m12_outcome.md` — M1.2 result summary
- `project_m13_pretraining_hurts.md` — the consequential finding

---

## TL;DR — where the project is

**M1.2 (MLM pretraining): COMPLETE** — 13/30 epochs in ~5 days. Gate G1
cleared (val acc 0.605–0.645 on 30 kb), then stopped early. Best preserved
checkpoint at `models_test/snapshots/backbone_mlm_v1_e13_30k_gs292630.pt`
(val 0.621). The 0.645 peak ckpt was lost (snapshotter added too late).

**M1.3 (probing): DONE — but the result is the opposite of what we
expected.** Linear probe, MLP probe, and small fine-tune ALL show
random-init HyenaDNA beating the M1.2 ckpt by 0.15-0.25 F1 on
breakpoint detection. MLM pretraining is *counterproductive* for the
downstream task. See `project_m13_pretraining_hurts.md` for numbers.

**M3 (the real breakpoint detector): NOT STARTED.** Direction needs to
be decided before launching — see "Open decisions" below.

The branch with all this work is `worktree-m12-mlm` in
`.claude/worktrees/m12-mlm/`. Several commits ahead of main and NOT
pushed. The user explicitly authorizes pushes only when they say so.

---

## The unexpected finding (most important context)

We spent 5 days pretraining HyenaDNA with bidirectional MLM on filtered
SANTA. The pretraining itself succeeded — MLM val acc hit 0.62-0.64 on
30 kb sequences, well above the 0.31 random baseline and the 0.31
zero-shot floor. Gate G1 was cleanly cleared.

But when we tested whether the pretraining transferred to breakpoint
detection (M1.3, linear probe), we found:

| Probe setup | M1.2 ckpt F1 | Random init F1 |
|---|---|---|
| Linear probe (frozen)    | 0.217 | **0.403** |
| MLP probe (frozen)       | 0.156 | **0.406** |
| End-to-end fine-tune     | 0.249 | **0.413** |

Random consistently wins by 0.15-0.25 F1. The most plausible explanation:

- **MLM trains the backbone to be *invariant* to local nucleotide
  identity** — its job is to predict the masked nucleotide from
  context, so it learns context-conditioned representations that
  ignore the actual letter.
- **Breakpoint detection needs the *opposite*** — sensitivity to where
  R differs from each parent at each position.
- The "difference" features `h_R - h_P_i` are noisy under pretrained
  embeddings (similar contexts collapse to similar vectors) but
  preserve raw cross-sequence difference under random projections.

**Implication: M1.2 is NOT a useful foundation for M3.** Start M3 from
random init, or change the pretraining objective.

---

## What's at the 200-event scale (unfinished)

We tried to scale the M1.2-vs-random fine-tune comparison from 40 events
to 200 events to verify the gap holds. The random-init run NaN'd at
epoch 3 (fp16 + B=1 + high pos_weight on untrained activations) and got
stuck at trivial F1 0.28. M1.2 also reached 0.28 but stably (well-conditioned
activations).

So at 200 events the comparison is **broken**, not informative. To
definitively rule out a small-sample artifact:

- Switch fp16 → bf16 (wider dynamic range, no scaler).
- Lower LR for random (1e-5 instead of 1e-4).
- Add stronger gradient clipping.
- Re-run 200-event comparison.

This is ~1 hour of work and would close the open question.

---

## Files added this session

In `worktree-m12-mlm`:
- `pretrain_mlm.py` — M1.2 MLM training loop (committed earlier in session).
- `eval_mlm_per_shard.py` — CPU val per-shard to test distribution shift.
- `eval_positional_cheat.py` — CPU val per position bucket to test positional shortcut.
- `snapshot_ckpts.py` — Side process snapshotting per-stage ckpts (add this FROM THE START next time).
- `m13_linear_probe.py` — Linear/MLP probe on frozen backbone with triplet diff features.
- `m3_mini_finetune.py` — Same as M1.3 but with backbone unfrozen.
- `HANDOVER_M12_M13.md` — this file.

Models/snapshots (gitignored — won't transfer with branch):
- `models_test/snapshots/backbone_mlm_v1_e13_30k_gs292630.pt` (5040 MB) — the M1.2 ckpt to use as reference.
- `models_test/history_mlm_v1.json` — training curves.
- `models_test/m13_*.json` — probe results.
- `models_test/m3_mini_*.json` — fine-tune comparison results.

---

## Open decisions for the next session

### Decision 1: Should we resolve the 200-event question?

**Cost:** ~1 hour of GPU work with bf16 + lower LR for random init.

**Value:** Definitive answer on "does M1.2 hurt at scale, or only at 40 events?" If it hurts at 200 events too, M1.2 is conclusively useless. If it catches up at 200 events, M1.2 might be salvageable.

**Recommendation:** Yes, run this first. Cheap insurance.

### Decision 2: What's the M3 pretraining strategy?

Three plausible paths:

**(A) M3 from random init, no pretraining.** Build a proper end-to-end breakpoint detector from a fresh Hyena backbone. Train on thousands of events with stability fixes. Accepts that M1.2 didn't pay off but moves forward cleanly. Lowest risk to deliver something.

**(B) M3 with a NEW pretraining objective that preserves cross-sequence signal.** Options:
- *Contrastive triplet learning:* given (R, P1, P2), predict per-position which parent R matches.
- *Sliding-window parent-assignment:* given fixed-length R chunks + P1/P2 contexts, classify parental origin.
- These pretraining objectives directly train the cross-sequence comparison signal that MLM destroyed.

**(C) M3 with M1.2 ckpt but a richer feature representation.** Maybe the issue was `h_R - h_P_i` as the cross-sequence feature. Try cosine similarity, |diff|², concat-with-product. If feature engineering brings M1.2 to ~0.40, the ckpt is salvageable.

**My read:** (A) is the safest path to a working M3. (B) is the most exciting research direction but adds days of pretraining. (C) is a quick test we can do in ~1 hour.

### Decision 3: Stop chasing HyenaDNA-style pretraining entirely?

Worth considering: the legacy CNN architecture in `CNN.ipynb` got F1 0.421 on SANTA UnseenTestSet and **F1 0.533 on LANL real-HIV CRFs** (the deployment baseline). The whole reason we went to HyenaDNA was to drop the hand-crafted MaxChi features and build a "sequence-only" model.

If the random-Hyena probe at F1 0.40 with NO pretraining is already approaching the legacy CNN's 0.42 (on similar SANTA-style data), then maybe the right move is:

- Keep HyenaDNA architecture (it works fine)
- Skip MLM pretraining (it actively hurts)
- Train M3 end-to-end from random init
- Compare LANL F1 to legacy CNN's 0.533 → that's the real deployment test

This is essentially option (A) with the philosophical framing that "MLM pretraining is the wrong move for this specific task."

---

## What to NOT do

1. **Do NOT delete the M1.2 snapshot ckpt** (`backbone_mlm_v1_e13_30k_gs292630.pt`).
   We may still want to compare against it. It's also the only artifact of
   5 days of GPU time.

2. **Do NOT immediately restart M1.2 pretraining with a "fix."** Spend
   the day on probe/feature/architecture work first. More pretraining
   isn't the answer if the objective is wrong.

3. **Do NOT push to origin without explicit user approval.** Branch is
   `worktree-m12-mlm`, several commits ahead.

4. **Do NOT trust subagent reports without verification.** This was true
   last session and remains true.

5. **Do NOT chase the 0.645 lost peak ckpt by retraining M1.2.** The
   marginal value is ~0.024 MLM val, which doesn't translate to
   downstream F1 (we tested this; pretraining itself is the wrong
   direction).

---

## Suggested first moves for the next session

1. Read this file. Read `project_m13_pretraining_hurts.md`. Read `MASTER_PLAN.md`.
2. Pick a path among Decisions 1/2/3 with the user, with explicit trade-offs
   surfaced as concrete time/cost numbers.
3. If going with (A) — design a real M3 training run from random init with
   thousands of events, stability fixes from the start, proper
   per-epoch ckpt snapshotting, and stage-level rollback. Then launch.
4. Update `MASTER_PLAN.md` with the M1.2/M1.3 outcomes — the master
   plan still says M1.2 will hit 0.6 and feed into M3. That story is
   now half wrong; the val side worked but the downstream-transfer
   side didn't.

---

## User's working style (still relevant)

- "Ultrathink" requests deserve depth. Use the advisor tool.
- Anti-friction on clarifying questions — make reasonable defaults clear.
- Challenges premature framings — bring concrete numbers, don't argue from intuition.
- Authorizes destructive actions individually.
- Appreciates honest reporting of unexpected results — don't soft-pedal the M1.3 finding.

Good luck. The hard work is figuring out what M3 should actually be.
