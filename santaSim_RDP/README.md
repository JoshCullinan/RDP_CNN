# santaSim_RDP

SANTA (Sequence Analysis Tool for Adaptation) configuration + driver scripts
used to generate the training data in `dataRaw/`.

**What's tracked here** (~190 KB, small/source-only):
- `XMLs/{1..5}.xml` + `XMLs/Others/*.xml` — the simulation configurations
  referenced in `CLAUDE.md` and `HANDOVER_NEXT_AGENT.md`. These define each
  shard's virus/genome/mutation-rate/recomb-rate profile.
- `Test Set XML/sarbeco_high.xml` — the UnseenTestSet generator config.
- `scripts/*.sh`, `pipeline.bat` — HPC + local drivers.
- `Simulation.py` — Python entry-point.
- `XMLs/ORIGINAL_CLUSTER_PARAMETERS.txt` — original cluster parameter notes.

**What's gitignored** (binary / regenerable / large; see repo `.gitignore`):
- `santa.jar` (~14 MB) — external SANTA jar. Download separately when reproducing.
- `sarbeco.fas`, `sarbecoStart.fa` (~9 MB combined) — FASTA seed alignments.
- `outputs/` (~109 MB) — generated simulation runs.
- `.vscode/` — IDE config.

**Use in this project:** the XML configs are the canonical record of what each
shard of SANTA data contains. See `project_santa_realism_filter.md` in memory
for which shards were dropped by M0.5 and why.
