# archive/ — superseded code and outputs

Everything here is **historical / dead**. It is preserved for traceability (and
because several items are well-characterized *negative results* worth keeping),
but **none of it is part of the active pipeline** and **no active code imports
it**. Do not act on recommendations found in these files. For the live project,
see the root `README.md`.

| Folder | What it is | Why it's archived |
|---|---|---|
| `keras_era/` | The pre-pivot Keras dilated-CNN (`CNN.ipynb`, `Autoencoder.ipynb`) and the run29–43 one-off scripts, drivers, and data-prep helpers. | The project pivoted off the Keras CNN to the PyTorch M3 pipeline. The legacy `runB2_sig10` baseline (LANL F1 0.533, a *fusion* model needing RDP inputs) lives here. |
| `hyenadna_era/` | M1.x: HyenaDNA backbone probes (`m12_*`), MLM-pretraining diagnostics (`m13_*`), the SANTA-realism investigation (`m14_*`), the abandoned Mamba attempt (`backbone_mamba.py`), the M1.2 ckpt snapshotter. | HyenaDNA + MLM pretraining **hurt** downstream breakpoint detection (memory `project_m13_pretraining_hurts`). The whole transformer-backbone direction was dropped in favour of the plain dilated CNN. |
| `superseded_m3/` | Early M3 attempts (`m3_mini_finetune.py`, `m3_train.py`). | Negative results — frozen/linear heads on Hyena features collapsed at scale. Superseded by `m3_dilated.py`. |
| `phase0_audit/` | M0.4 baseline verification (`m04_*`) that confirmed the v2 cache reproduces the legacy CNN bit-for-bit. | One-time audit; its job is done. |
| `old_outputs/` | Stale result JSONs (`results_run*`, `results_cnn_*`, `results_rustrdp_*`), run logs, partial diagnostics, the old `monitor/` helper. | Outputs of the above dead runs. |

**Note:** `backbone_hyenadna.py` and `pretrain_mlm.py` are the HyenaDNA direction
too, but they are **not** here — they remain at the repo root because
`m3_dilated.py` imports them at module level (for the unused `hyena` feature
mode). They are dead in spirit but load-bearing as imports.
