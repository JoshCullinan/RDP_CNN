# Pre-pivot archive

`IDEAS.md` and `TODO.md` were written while the project was still iterating on the
Keras dilated-CNN against the F1≈0.17 plateau. They are kept for historical
context.

**Do not act on their queues** — they are inconsistent with the current direction:

- `IDEAS.md` Tier-A (esp. A1 RDP-Gaussian channels) is the foundation runB2 was
  built on. The committed direction is **sequence-only inputs** for multi-virus
  transfer (see `project_backbone_replacement_direction.md` in memory), so the
  engineered RDP channels are deliberately dropped, not added.
- `TODO.md`'s north star is "beat MaxChi/RDP/GeneConv on simulated UnseenTestSet";
  the live north star is multi-virus deployment from raw nucleotide sequences.
- Most "Already implemented" cell-level checklists refer to `CNN.ipynb`, which is
  now legacy.

**Live plan:** [`MASTER_PLAN.md`](../../MASTER_PLAN.md) at repo root.
