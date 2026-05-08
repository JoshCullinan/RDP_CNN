# Handover: explore and clean ~250GB of SANTA simulation data

> **Read this in full before doing anything.** You're inheriting a data
> archaeology task, not a modeling task. There is no training to run, no
> notebook to edit. The deliverable is a cleaned, well-documented,
> deduplicated dataset that the project's main pipeline (running on the
> WSL side, separately) can later consume.

---

## 1. Your mission, in one paragraph

A separate agent / pipeline trains a CNN that detects viral recombination
breakpoints in nucleotide alignments. Its current bottleneck (diagnosed
this week) is that the existing training data has positive labels
concentrated at sequence positions 0–19k, while the held-out test set
has positives spread 0–30k with 37% past position 20k. The model can't
detect breakpoints at positions it has never been trained at. **This
new data dump (~250GB of SANTA simulations from collaborating
researchers) is expected to provide the long-content / late-position
training samples the pipeline currently lacks.** Your job is to find
out what's actually in this archive, organize it, deduplicate it, and
report what's useful.

You will **not** train any models. You will **not** modify the
notebook or anything in `/home/joshcullinan/RDP_CNN/`. You operate
entirely in your own working directory on the Windows side, where the
data has been downloaded.

---

## 2. Project context (so you know what "useful" means)

- **Goal of the parent project**: beat the classical recombination
  detectors (RDP5, MaxChi, GeneConv) on per-position breakpoint detection
  in HIV-class viral genomes (~9.7 kb up to ~30 kb after gap insertion in
  alignments). Quality is the only metric that matters; compute is not
  a constraint.
- **Input format the model expects per training event**: a triplet of
  aligned sequences `(recombinant, parent1, parent2)` where the
  recombinant's identity is given (label-known), and the model has to
  localize the breakpoint(s) inside it.
- **Ground truth** comes from the simulator (SANTA) which records, for
  each simulated recombination event, the actual recombinant sequence
  ID and the actual breakpoint positions in alignment coordinates.
- **Auxiliary signal** from RDP5 detector: per-event predicted
  breakpoint positions, per-method p-values from 9 detectors (RDP,
  GENECONV, Bootscan, Maxchi, Chimaera, SiSscan, PhylPro, LARD, 3Seq),
  and per-hypothesis scoring stats. Already-validated as input
  channels in the model — RDP5 outputs are NOT labels, they're
  features. So files where RDP5 has been run carry richer signal than
  files where only SANTA output exists; both are useful but in
  different ways.

---

## 3. What the existing pipeline expects per simulation

The existing pipeline (which you do not need to touch, only mirror the
schema of) expects, for each simulated alignment, **up to four** files
that share a common base filename:

| File suffix                          | Required? | Contents |
|--------------------------------------|-----------|----------|
| `<base>.fa`                          | Yes       | FASTA alignment of ~100 sequences. Sequence IDs are integers. |
| `<base>.faSimVSRealCompare.csv`      | Yes (for ground truth) | Per-event ground truth: `RDPEvent`, `ActualRecomb` (integer ID of the recombinant in the FASTA), `SimBPStart`, `SimBPEnd` (true breakpoint positions in alignment coordinates), plus `PredBPStart`/`PredBPEnd` (RDP5's predicted breakpoints if RDP was run). |
| `<base>.faRecombIdentifyStats.csv`   | Useful    | 3 rows per event (3 candidate hypotheses for which sequence is the recombinant). Columns include `Event`, `StartBP`, `EndBP`, `ISeqs(A)` (the candidate sequence ID list), and ~40 stats columns ending in `(A)` (e.g. `dMax(A)`, `BadDists(A)`, `SubPhPrScore(A)`, `SimScore(A)`, `PhPrScore(A)`, `TrpScore(A)`, etc.). |
| `<base>.fa.csv`                      | Useful    | RDP5's full output table. Multi-row per event (first row has data, subsequent rows are sequence-ID continuations). Begins with 15 lines of header / table-key text, then a `Recombination Event Number, ...` header row, then data. Columns 11-19 carry the 9 per-method p-values (cols: RDP, GENECONV, Bootscan, Maxchi, Chimaera, SiSscan, PhylPro, LARD, 3Seq). Cells holding `"NS"` mean "not significant for this method". |

The user has noted **RDP has not been run on all files** in this new
dump. So you will likely see a mix of:
- Full sets `(.fa + SimVSReal + RecombIdentifyStats + .fa.csv)` ←
  most useful, drop-in compatible.
- SANTA-only sets `(.fa + SimVSReal)` ← still very useful, just
  missing the RDP-derived auxiliary CSVs. Existing pipeline can use
  these (it falls back gracefully when `.fa.csv` is missing).
- `.fa` alone with no ground-truth CSV ← only useful if the
  ground-truth events file lives elsewhere (e.g. a single combined CSV
  per simulation batch). This is the case you most need to investigate.
- "Filtered files containing only recombinant events in the final
  sample fasta" — the user has not yet confirmed what these are. Two
  hypotheses to test:
  - (a) Pre-filtered subset of the FASTA containing only the sequences
    the simulator labels as recombinants. Useful as a positive-event
    index.
  - (b) Detection-method output (RDP-like) listing detected
    recombinants. Useful as auxiliary signal, **not** as ground truth.
  Investigate which interpretation applies in this dump and report.

---

## 4. What you should produce — deliverables

When you're done, the user wants to be able to look at one place and
understand:

1. **A top-level inventory** (`INVENTORY.md` in your working directory):
   - Total file count and size by extension.
   - Number of distinct simulation runs (i.e. distinct base filenames).
   - Per-run completeness: how many runs have all four files vs only
     `.fa + SimVSReal` vs `.fa` alone.
   - Distribution of FASTA lengths (`MAX_SEQ_LEN` for the existing pipeline
     is 32000; runs with content > 32000 will be truncated). Bucket by
     content length: 0-5k, 5-10k, 10-15k, 15-20k, 20-25k, 25-30k, 30k+.
   - Distribution of breakpoint positions (parse `SimBPStart`/`SimBPEnd`
     from a sample of `.faSimVSRealCompare.csv` files).

2. **A relationship map** (`RELATIONSHIPS.md`):
   - How directories / sub-archives relate to each other (e.g. "directory
     `simrun_abc/` contains the SANTA outputs; `rdp_runs/abc/` contains
     the matching RDP5 outputs for the same base names").
   - Naming conventions across sub-archives (do filenames match exactly,
     or is there a transformation?).
   - Any orphaned files (CSVs without matching FASTA, FASTAs without
     matching ground truth).
   - Whether the "filtered files containing only recombinant events"
     are pre-filtered FASTA subsets (a) or detection-method outputs (b).

3. **A duplicate report** (`DUPLICATES.md`):
   - Files where the content matches an existing file in `dataRaw/XML-1..5/`
     or `dataRaw/UnseenTestSet/` (existing data should not be re-imported).
     Note: you may not have direct access to that directory. If you don't,
     skip this check and document the limitation; the user will run the
     final cross-check.
   - Internal duplicates within the new dump (same file content, multiple
     paths). Hash by SHA-256 of file contents.
   - Near-duplicates (same FASTA but different `.fa.csv` — e.g. RDP run
     twice with different parameters).

4. **A quarantine directory** (`quarantine/`):
   - **Do not delete anything.** Suspected duplicates and orphans go
     into this dir, **preserving the original directory structure**
     (so the move is reversible by `mv quarantine/* . -r`). Each
     quarantined file gets an entry in `quarantine/MANIFEST.md`
     explaining why (e.g. "SHA-256 match with `path/to/keeper.fa`",
     "no matching ground-truth CSV found within the same parent dir").
   - The user will review this manifest before any deletion happens.

5. **A relevance summary** (`RELEVANCE.md`):
   - Of the cleaned, deduplicated data: how many events have
     breakpoints in the 19k–30k position range? This is the key number
     for whether the new data fixes the diagnosed long-position
     bottleneck.
   - Distribution of `actual_len` (length of recombinant after
     stripping trailing gaps) for the cleaned set.
   - Per-XML or per-batch breakdown if there are obvious sub-corpora.
   - Recommended train/holdout split strategy (e.g. "stratify by
     content length so all splits cover the full 0-30k range").

6. **A "ready to import" subset** (`ready/` or just a list in
   `READY.md`):
   - The subset you'd recommend the user copy into `dataRaw/<new_dir>/`.
   - Excludes quarantined items, duplicates, files with broken or
     missing ground truth, anything that looks corrupted.
   - Use the existing `dataRaw/XML-1..5/` directory layout as the
     target shape (one flat directory of `.fa + .faSimVSRealCompare.csv
     [+ .faRecombIdentifyStats.csv [+ .fa.csv]]` triples/quads sharing
     a base name).

---

## 5. Concrete first steps (in order)

Don't try to design the cleanup before you've seen the data. Do these
in sequence:

1. **High-level inventory.** Run something like `find . -type f | head
   -100`, look at the directory tree depth, get a first taste of the
   naming conventions. Don't enumerate all 250GB worth of paths
   eagerly — sample first.

2. **Pick 5-10 representative files** by type/size/depth. Read their
   first ~100 lines (or the FASTA header for `.fa` files). Confirm
   they look like SANTA output. Look for README, INFO, or notes files.

3. **Identify the simulation-run grouping unit.** SANTA typically
   produces one `.fa` per parameter combination. Find what the unit is
   here — is it one directory per run, multiple runs per directory,
   batched into archives? This determines how you'll dedupe.

4. **Check for existing RDP5 outputs.** The user said "RDP has not been
   run on all files necessarily." Find the ratio. If RDP outputs exist
   for, say, 30% of the runs, the user will probably want a follow-up
   step to run RDP5 on the rest — but that's their call, not yours.
   Just report what fraction has RDP outputs.

5. **Check for the "filtered files containing only recombinant
   events"** — open one. Is it a FASTA (sequences only)? Is it a CSV
   (event listings)? Is it some other format? Report what you find,
   and resolve the (a)/(b) interpretation question above.

6. **Sample-parse 50 ground-truth CSVs** to get the position
   distribution. The headline number for the user is: how many events
   have `SimBPStart` or `SimBPEnd` past position 19k, and how many past
   25k? This is the relevance-to-the-bottleneck answer.

7. **Hash and dedupe.** SHA-256 every `.fa` file. Group by hash. If a
   hash appears more than once, all but the lexically first copy go to
   `quarantine/` with a manifest entry pointing at the keeper.

8. **Check orphans.** For every `.fa`, check whether a matching
   `<base>.faSimVSRealCompare.csv` exists. Files without the SimVSReal
   CSV have no ground truth and are candidates for quarantine
   (manifest reason: "no SimVSReal CSV found"). Same for the reverse
   (CSVs with no FASTA — those go to quarantine as "FASTA missing").

9. **Write the deliverables** (Section 4) progressively as you go.
   Don't hold them all to the end. The user wants to be able to peek
   at progress.

10. **Stop and ask if you find something weird.** Examples: a file
    extension you don't recognize, a directory structure that suggests
    these are model outputs not simulation outputs, a CSV with a
    schema you can't reconcile with the four file types in Section 3.
    Don't guess — ask.

---

## 6. Hard rules

1. **No deletions, ever.** Move to `quarantine/` with a manifest entry.
   The user will review and delete manually.
2. **Don't move anything in or out of the user's existing
   `dataRaw/` tree** (you may not even have access to it; if you do,
   leave it alone). Your work is on the new dump only.
3. **Don't run RDP5 / 3Seq / MaxChi / any classical detector**. If the
   user wants those run on the new data, that's a separate step they
   will scope.
4. **Don't decompress archives in place if it would more than double
   disk usage.** Stream-process tarballs/zips where possible. If you
   must decompress, decompress to a sibling directory and quarantine
   the original archive (don't delete it) until the user confirms.
5. **Don't write to `/mnt/c/Users/joshc/wsl_monitor/`** if you can see
   it — that's a separate utility directory.
6. **Treat any binary file you can't identify as quarantine-candidate**
   with manifest reason "unknown format" — let the user decide.

---

## 7. How to escalate

If you reach a decision point that's irreversible or affects file
organization in a way you can't easily back out of, **stop and
ask the user**. Examples that should trigger an ask:

- You found a giant tarball (>20GB) that needs decompression and
  you're unsure whether to expand it in place.
- You found files referenced across directories with absolute paths
  baked in (would moving them break references?).
- You found what looks like a different research project's data mixed
  in (not SANTA / not viral) and you're unsure whether to quarantine
  the whole subdir.
- You hit a parsing pattern that the four-file schema in Section 3
  doesn't cover, and you need to know whether to keep digging or
  document and move on.

The one thing **not** to ask about: when in doubt about whether a file
is a duplicate or an orphan, **always quarantine + manifest entry +
keep going**. The cost of an over-zealous quarantine is zero (the user
will pull it back if needed). The cost of a deletion is data loss.

---

## 8. What "done" looks like

You hand the user back the data dump in a state where:

- They can read `INVENTORY.md`, `RELATIONSHIPS.md`, `DUPLICATES.md`,
  `RELEVANCE.md`, and `READY.md` in 10 minutes and have a complete
  picture of what they got and what's worth importing.
- They can run `du -sh quarantine/` and see how much was set aside,
  flip through `quarantine/MANIFEST.md`, and either approve the
  removal or pull individual files back.
- They can `cp -r ready/* /home/joshcullinan/RDP_CNN/dataRaw/<new_dir>/`
  (manually, separately) and have a usable training corpus that
  matches the existing `XML-1..5` schema.

That's the bar. Don't optimize beyond it; you're feeding a separate
training pipeline that will do the heavy lifting on its own side.

---

## 9. Quick reference — example file headers

You'll see these often. Recognize them, don't get confused.

`.fa` (FASTA):
```
>1
ACGT---ACGT...
>2
ACGT---ACGT...
```
Sequence IDs are integers in the existing data.

`.faSimVSRealCompare.csv` (header + first data row):
```
RDPEvent,ActualRecomb,PredictRecomb,MatchNo,Matchscore,SimEVentNo,SimBPStart,SimBPEnd, PredBPStart, PredBPEnd
 1, 21, 21, 8, 1, 68090, 0, 1542, 1, 1542
```
- `ActualRecomb` = integer ID of the recombinant sequence in the `.fa`.
- `SimBPStart`/`SimBPEnd` = ground-truth breakpoint positions.
- `PredBPStart`/`PredBPEnd` = RDP5's predictions; may be empty/NaN if
  RDP wasn't run.

`.faRecombIdentifyStats.csv` (header):
```
Event, StartBP, EndBP, ISeqs(A),ListCorr(A),SimScoreB(A),SimScore(A),
PhPrScore(A),PhPrScore2(A),PhPrScore3(A),SubScore(A),SSDist(A),OUIndexA(A),
SubPhPrScore(A),SubScore2(A),SubPhPrScore2(A),SRCompatF(A),SRCompatS(A),
RCompat(A),RCompat2(A),RCompat3(A),RCompat4(A),RCompatS(A),RCompatS2(A),
RCompatS3(A),RCompatS4(A),RCompatXF(A),RCompatXS(A),RCompatC(A),RCompatD(A),
TrpScore(A),BadDists(A),OUList(A),ListCorr2(A),ListCorr3(A),Consensus(A:0),
Consensus(A:1),Consensus(A:2),OuCheck(A),SetTot(0:A),SetTot(1:A),
RankF(A:0),RankF(A:1),dMax(A),
```
3 rows per event. The (A) suffix denotes hypothesis-row stats.

`.fa.csv` (RDP5 output — first ~17 lines are table-key prose; data starts at line ~16):
```
... [15 lines of header / explanation ending with the column-headers row]
 1, 1, 2665, 3822, 2665, 3822, 2665, 3822, 13, 23, 41,  2.10684e-178, 5.67481e-172, ...
 1, , , , , , , , , , 27
```
Multi-row per event; only the first row of an event has the BP and
p-value data. Columns 11-19 are the per-method p-values; `NS` strings
mean "not significant for that method".

---

## 10. Final note

The ~250GB is large but most of it is going to be FASTA bulk that
compresses well and isn't actually new information per byte. The
*useful* output of your work is probably <50GB of cleaned, dedupe'd,
ground-truth-paired data. Don't be afraid to be aggressive with
quarantine — the user reviews it before anything is removed.

Good luck. Stop and ask the user if you hit anything weird.
