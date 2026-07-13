# Genomic-Spectrogram Identification (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-way recombinant-identification pipeline that turns an aligned triplet `(recombinant, parent1, parent2)` into an image and asks a 2-D CNN "which of the three is the mosaic?", honestly benchmarking the paper's Fourier/spectrogram representation against a natural positional-features control.

**Architecture:** One shared "permuted-channel 3-way" harness feeds three interchangeable representation encoders (A0 positional, A1 summed spectrogram, A2 unsummed spectrogram) into a backbone (ImageNet-pretrained / random-init / small-from-scratch floor). Confound diagnostics run first as blocking gates; evaluation is leave-one-CRF-out cross-validation with pre-registered McNemar + cluster-bootstrap significance.

**Tech Stack:** Python 3.12, PyTorch 2.6 (+cu124, already installed), `timm` (new), `scipy.signal` (installed), `statsmodels` (new), NumPy, the repo's `cache_v2_reader.CacheV2`.

Spec: `docs/superpowers/specs/2026-07-13-genomic-spectrogram-identification-design.md`.

## Global Constraints

- **Task = identification only.** 3-way "which channel is the recombinant." No breakpoint localization; no 4-way `{…,none}` head (documented as eventual goal, deferred).
- **Fixed length:** every aligned triplet is pre-padded/truncated to exactly **`SEQ_LEN = 10000`** positions **before any transform**. Never anisotropically rescale the position axis per-sample.
- **Shared transform grid:** the three channels of one image MUST share an identical transform grid (same `nperseg`, `noverlap`, pre-pad). Assert equal frame counts across channels.
- **Normalization:** symmetric **joint** normalization across the three channels together. Never per-channel z-norm; never ImageNet mean/std.
- **Gap encoding:** gaps/pad → 0 in indicator signals; byte-identical handling across SANTA and LANL.
- **OOM rule (repo-wide, non-negotiable):** never `np.array(X, copy=True)` or `X.astype(...)` on multi-GB cached tensors; encode per-triplet in a loop, cache images to disk. Hardware: RTX 3070, 8 GB VRAM, 30 GB RAM.
- **Encoding (SANTA, from `CacheV2`):** int8 `{A:0, T:1, G:2, C:3, -:4}` (`GAP_INT = 4`). All code uses this convention; LANL FASTA is converted into it.
- **Pinned hyperparameters (revisable only by editing this section):** `NPERSEG = 256`, `NOVERLAP = 32` (scipy default `nperseg // 8`), `IMG_SIZE = 224`, `SCALES = (50, 100, 200, 500)` bp (A0 multi-scale rows), backbone `convnext_base` via timm, batch size `16` with AMP.
- **Primary metric (pre-registered):** mean held-out-CRF top-1 identification accuracy across the leave-one-CRF-out folds (chance = 1/3).
- **Decision rule (pre-registered):** a Fourier arm (A1 or A2) beats A0 iff ALL of: (a) higher mean LOCO held-out accuracy than A0; (b) McNemar p < 0.05 on paired per-triplet predictions vs A0 (pooled across folds); (c) cluster-bootstrap-by-CRF 95% CI on the accuracy difference excludes 0. Confound guard: label-permutation null must sit at chance; the SANTA↔held-out gap is reported as a first-class number.

---

## File Structure

All new code lives under a new `spectrogram/` package (clean build, repo as home). Tests under `tests/spectrogram/`.

| File | Responsibility |
|---|---|
| `spectrogram/config.py` | Pinned constants (Global Constraints), single source of truth. |
| `spectrogram/data.py` | Load + unify SANTA (`CacheV2`) and LANL (FASTA) triplets into fixed-length int8 rows with metadata. |
| `spectrogram/encode.py` | The three representation encoders (A0/A1/A2) + shared-grid asserts. |
| `spectrogram/harness.py` | Permuted-channel 3-way `Dataset`, symmetric joint normalization, resize. |
| `spectrogram/models.py` | Backbone loader (timm, in-chans adaptation, 3-way head) + small-CNN floor. |
| `spectrogram/train.py` | AMP training loop, config-driven. |
| `spectrogram/probe.py` | Confound diagnostics: P2 own-parents gate, positional-scramble, divergence. |
| `spectrogram/eval.py` | LOCO folds, McNemar, cluster-bootstrap, label-permutation null, power calc, decision rule. |
| `spectrogram/expand_lanl.py` | Fetch multiple subtype reference strains → enumerate parent pairings per CRF. |
| `spectrogram/run_stage1.py` | Orchestration: pre-registration record → power calc → diagnostics gates → bake-off → LOCO eval → report. |

Test runner: `/home/joshc/Dev/RDP_CNN/.venv/bin/python -m pytest`. All commands below assume `PY=/home/joshc/Dev/RDP_CNN/.venv/bin/python` and are run from the worktree root.

---

## Task 0: Dependencies and package scaffold

**Files:**
- Create: `spectrogram/__init__.py`, `spectrogram/config.py`, `tests/spectrogram/__init__.py`, `tests/spectrogram/test_config.py`
- Modify: `requirements.txt` (append `timm==1.0.20`, `statsmodels==0.14.5`)

**Interfaces:**
- Produces: `spectrogram.config` module exposing `SEQ_LEN, NPERSEG, NOVERLAP, IMG_SIZE, SCALES, GAP_INT, NT_INDICATOR_ORDER, BACKBONE, BATCH_SIZE, LANL_TRIPLET_DIR, CACHE_IMG_DIR`.

- [ ] **Step 1: Install new deps**

Run: `PY=/home/joshc/Dev/RDP_CNN/.venv/bin/python; $PY -m pip install "timm==1.0.20" "statsmodels==0.14.5"`
Expected: successful install; `$PY -c "import timm, statsmodels; print(timm.__version__, statsmodels.__version__)"` prints versions.

- [ ] **Step 2: Append deps to requirements.txt**

Add these two lines under the scientific-stack block:
```
timm==1.0.20
statsmodels==0.14.5
```

- [ ] **Step 3: Write config module**

Create `spectrogram/config.py`:
```python
"""Single source of truth for Stage-1 spectrogram-identification constants.
See docs/superpowers/specs/2026-07-13-genomic-spectrogram-identification-design.md
and the plan's Global Constraints."""
from pathlib import Path

SEQ_LEN = 10_000            # fixed pre-pad length (positions) before any transform
NPERSEG = 256               # STFT window (paper's value)
NOVERLAP = 32               # scipy default nperseg // 8
IMG_SIZE = 224              # square backbone input
SCALES = (50, 100, 200, 500)  # A0 multi-scale rows (bp), matches repo MaxChi windows
GAP_INT = 4                 # {A:0,T:1,G:2,C:3,-:4}
NT_INDICATOR_ORDER = (0, 1, 2, 3)  # A,T,G,C ints for indicator signals (gap excluded)
BACKBONE = "convnext_base"
BATCH_SIZE = 16
LANL_TRIPLET_DIR = Path("data/lanl_crf/triplets")
CACHE_IMG_DIR = Path("cache/spectrogram_v1")
```

- [ ] **Step 4: Write the failing test**

Create `tests/spectrogram/test_config.py`:
```python
from spectrogram import config

def test_pinned_constants():
    assert config.SEQ_LEN == 10_000
    assert config.NPERSEG == 256
    assert config.NOVERLAP == config.NPERSEG // 8
    assert config.GAP_INT == 4
    assert config.NT_INDICATOR_ORDER == (0, 1, 2, 3)
    assert config.SCALES == (50, 100, 200, 500)
```

- [ ] **Step 5: Run test**

Run: `$PY -m pytest tests/spectrogram/test_config.py -v`
Expected: PASS (create empty `spectrogram/__init__.py` and `tests/spectrogram/__init__.py` if collection fails).

- [ ] **Step 6: Commit**

```bash
git add spectrogram/ tests/spectrogram/ requirements.txt
git commit -m "feat(spectrogram): package scaffold + pinned Stage-1 config"
```

---

## Task 1: Unified triplet loading (SANTA + LANL)

**Files:**
- Create: `spectrogram/data.py`, `tests/spectrogram/test_data.py`

**Interfaces:**
- Consumes: `cache_v2_reader.CacheV2` (`.shards`, `CacheV2Shard.get_triplet(event_idx) -> dict` with int8 `R,P1,P2` + `source_file`), `spectrogram.config`.
- Produces:
  - `dataclass Triplet(rows: np.ndarray, recomb_idx: int, source: str, group: str)` where `rows` is int8 shape `(3, SEQ_LEN)` in the order the loader chose (before harness permutation), `recomb_idx` is the row index (0,1,2) of the recombinant, `source` is a provenance tag (`"santa"`/`"lanl"`), `group` is the clustering key (SANTA shard name or CRF name).
  - `fix_length(row: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray` (pad with `GAP_INT`, or truncate).
  - `load_santa_triplets(cache: CacheV2, limit: int | None = None) -> list[Triplet]`.
  - `load_lanl_triplets(triplet_dir: Path = LANL_TRIPLET_DIR) -> list[Triplet]`.
  - `FASTA_NT_TO_INT: dict[str,int]` mapping `A,T,G,C,-` and ambiguity/`N`→`GAP_INT`.

- [ ] **Step 1: Write the failing tests**

Create `tests/spectrogram/test_data.py`:
```python
import numpy as np
from spectrogram import config
from spectrogram.data import fix_length, load_lanl_triplets, Triplet

def test_fix_length_pads_with_gap():
    row = np.array([0, 1, 2], dtype=np.int8)
    out = fix_length(row, seq_len=6)
    assert out.shape == (6,)
    assert out[:3].tolist() == [0, 1, 2]
    assert out[3:].tolist() == [config.GAP_INT] * 3

def test_fix_length_truncates():
    row = np.arange(20, dtype=np.int8)
    assert fix_length(row, seq_len=5).shape == (5,)

def test_lanl_loads_four_crfs():
    trips = load_lanl_triplets()
    assert len(trips) == 4                      # the 4 built CRF families
    groups = {t.group for t in trips}
    assert groups == {"CRF02_AG", "CRF07_BC", "CRF08_BC", "CRF12_BF"}
    for t in trips:
        assert isinstance(t, Triplet)
        assert t.rows.shape == (3, config.SEQ_LEN)
        assert t.rows.dtype == np.int8
        assert t.recomb_idx in (0, 1, 2)
        assert t.source == "lanl"
        assert set(np.unique(t.rows)).issubset({0, 1, 2, 3, 4})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/spectrogram/test_data.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `spectrogram/data.py`**

```python
"""Load and unify SANTA (CacheV2) and LANL (FASTA) triplets into fixed-length
int8 rows {A:0,T:1,G:2,C:3,-:4}."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from spectrogram.config import SEQ_LEN, GAP_INT, LANL_TRIPLET_DIR

FASTA_NT_TO_INT = {"A": 0, "T": 1, "G": 2, "C": 3, "-": GAP_INT}

@dataclass
class Triplet:
    rows: np.ndarray      # int8 (3, SEQ_LEN); row order is loader's, recomb at recomb_idx
    recomb_idx: int       # which row (0/1/2) is the recombinant
    source: str           # "santa" | "lanl"
    group: str            # clustering key: SANTA shard or CRF name

def fix_length(row: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    row = np.asarray(row, dtype=np.int8)
    if row.shape[0] >= seq_len:
        return row[:seq_len].copy()
    out = np.full(seq_len, GAP_INT, dtype=np.int8)
    out[: row.shape[0]] = row
    return out

def _encode_fasta_seq(seq: str) -> np.ndarray:
    up = seq.upper()
    arr = np.fromiter((FASTA_NT_TO_INT.get(c, GAP_INT) for c in up),
                      dtype=np.int8, count=len(up))
    return fix_length(arr)

def load_lanl_triplets(triplet_dir: Path = LANL_TRIPLET_DIR) -> list[Triplet]:
    """Each CRF FASTA is 3 records in order (recomb, parent1, parent2)."""
    from Bio import SeqIO
    out: list[Triplet] = []
    for fa in sorted(Path(triplet_dir).glob("*.fa")):
        recs = list(SeqIO.parse(str(fa), "fasta"))
        assert len(recs) == 3, f"{fa} has {len(recs)} records, expected 3"
        rows = np.stack([_encode_fasta_seq(str(r.seq)) for r in recs])
        out.append(Triplet(rows=rows, recomb_idx=0, source="lanl", group=fa.stem))
    return out

def load_santa_triplets(cache, limit: int | None = None) -> list[Triplet]:
    out: list[Triplet] = []
    for shard_name, shard in cache.shards.items():
        for ev in range(len(shard)):
            t = shard.get_triplet(ev)
            rows = np.stack([fix_length(t["R"]), fix_length(t["P1"]), fix_length(t["P2"])])
            out.append(Triplet(rows=rows, recomb_idx=0, source="santa", group=shard_name))
            if limit is not None and len(out) >= limit:
                return out
    return out
```

Note: `recomb_idx=0` by loader convention (R is always row 0 here); the harness (Task 3) is what permutes and hides this. `load_lanl_triplets` assumes the built FASTA record order matches `build_lanl_triplets.py` (recomb first). If a future FASTA reorders, add an explicit id check — out of scope now.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/spectrogram/test_data.py -v`
Expected: PASS. (If the LANL FASTA record order is not recomb-first, the assertion in the test will catch it — fix the loader to select by accession before proceeding.)

- [ ] **Step 5: Commit**

```bash
git add spectrogram/data.py tests/spectrogram/test_data.py
git commit -m "feat(spectrogram): unified SANTA+LANL fixed-length triplet loader"
```

---

## Task 2: Representation encoders (A0 / A1 / A2)

**Files:**
- Create: `spectrogram/encode.py`, `tests/spectrogram/test_encode.py`

**Interfaces:**
- Consumes: `spectrogram.config` (`NPERSEG, NOVERLAP, SCALES, SEQ_LEN, NT_INDICATOR_ORDER, GAP_INT`), `scipy.signal.spectrogram`.
- Produces (all take one fixed-length int8 row or a pair, return float32):
  - `nt_indicator(row, nt) -> np.ndarray` shape `(SEQ_LEN,)` — 1.0 where `row==nt`, else 0.0 (gap is never any nt → 0).
  - `spectrogram_one(row) -> np.ndarray` shape `(F, T)` — summed over the 4 nucleotide spectrograms (A1 per-sequence plane).
  - `spectrogram_unsummed(row) -> np.ndarray` shape `(4, F, T)` — one plane per nucleotide (A2 per-sequence planes).
  - `pairwise_identity_multiscale(row_i, row_j) -> np.ndarray` shape `(len(SCALES), T)` — per-scale boxcar-smoothed positional identity, sampled onto the same `T` grid as the spectrograms (A0 building block).
  - `encode_triplet(rows, arm) -> np.ndarray` — arm ∈ {"A0","A1","A2"}; returns channels-first float32:
    - A0: `(3, len(SCALES), T)` → reshaped to `(3, len(SCALES), T)` (3 pairwise channels).
    - A1: `(3, F, T)` (one summed plane per sequence).
    - A2: `(12, F, T)` (4 planes × 3 sequences).
  - `grid_frame_count() -> int` — the shared `T` from an all-zero probe, used to assert equal frame counts.

- [ ] **Step 1: Write the failing tests**

Create `tests/spectrogram/test_encode.py`:
```python
import numpy as np
from scipy.signal import spectrogram
from spectrogram import config
from spectrogram.encode import (nt_indicator, spectrogram_one, spectrogram_unsummed,
                                 pairwise_identity_multiscale, encode_triplet, grid_frame_count)

def _rows():
    rng = np.random.default_rng(0)
    r = rng.integers(0, 4, size=config.SEQ_LEN).astype(np.int8)
    p1 = r.copy(); p2 = r.copy()
    # make a mosaic: R == P1 on first half, R == P2 on second half
    half = config.SEQ_LEN // 2
    p1[half:] = (r[half:] + 1) % 4     # P1 differs from R on 2nd half
    p2[:half] = (r[:half] + 1) % 4     # P2 differs from R on 1st half
    return np.stack([r, p1, p2]).astype(np.int8)

def test_indicator_is_binary_and_gap_zero():
    row = np.array([0, 1, 2, 3, config.GAP_INT], dtype=np.int8)
    ind = nt_indicator(row, 0)
    assert ind.tolist() == [1, 0, 0, 0, 0]
    assert nt_indicator(row, config.GAP_INT).sum() == 0  # gap is not an nt

def test_summed_equals_sum_of_unsummed():
    row = _rows()[0]
    s = spectrogram_one(row)
    u = spectrogram_unsummed(row)
    assert np.allclose(s, u.sum(axis=0), atol=1e-5)

def test_shared_grid_frame_counts_equal():
    rows = _rows()
    a1 = encode_triplet(rows, "A1")
    a2 = encode_triplet(rows, "A2")
    a0 = encode_triplet(rows, "A0")
    T = grid_frame_count()
    assert a1.shape[-1] == T and a2.shape[-1] == T and a0.shape[-1] == T

def test_encode_shapes():
    rows = _rows()
    T = grid_frame_count()
    F = config.NPERSEG // 2 + 1
    assert encode_triplet(rows, "A1").shape == (3, F, T)
    assert encode_triplet(rows, "A2").shape == (12, F, T)
    assert encode_triplet(rows, "A0").shape == (3, len(config.SCALES), T)

def test_a0_encodes_mosaic_switch():
    # id(R,P1) high on 1st half, id(R,P2) high on 2nd half => the two pairwise
    # channels involving R must differ in opposite halves.
    rows = _rows()
    a0 = encode_triplet(rows, "A0")   # channels: (R,P1),(R,P2),(P1,P2) by convention
    left = a0[:, :, : a0.shape[-1] // 2].mean(axis=(1, 2))
    right = a0[:, :, a0.shape[-1] // 2:].mean(axis=(1, 2))
    assert left[0] > left[1]     # (R,P1) more identical than (R,P2) on the left
    assert right[1] > right[0]   # (R,P2) more identical than (R,P1) on the right
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/spectrogram/test_encode.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `spectrogram/encode.py`**

```python
"""Three representation encoders for the identification bake-off.
All share one STFT grid so channels correspond position-for-position."""
from __future__ import annotations
import numpy as np
from scipy.signal import spectrogram
from spectrogram.config import (NPERSEG, NOVERLAP, SCALES, SEQ_LEN,
                                NT_INDICATOR_ORDER)

def nt_indicator(row: np.ndarray, nt: int) -> np.ndarray:
    return (np.asarray(row) == nt).astype(np.float32)

def _stft(sig: np.ndarray) -> np.ndarray:
    _, _, Sxx = spectrogram(sig, fs=1.0, nperseg=NPERSEG, noverlap=NOVERLAP,
                            mode="magnitude")
    return np.log1p(Sxx).astype(np.float32)   # (F, T)

def grid_frame_count() -> int:
    return _stft(np.zeros(SEQ_LEN, dtype=np.float32)).shape[1]

def spectrogram_unsummed(row: np.ndarray) -> np.ndarray:
    return np.stack([_stft(nt_indicator(row, nt)) for nt in NT_INDICATOR_ORDER])  # (4,F,T)

def spectrogram_one(row: np.ndarray) -> np.ndarray:
    return spectrogram_unsummed(row).sum(axis=0)  # (F,T)

def _boxcar(sig: np.ndarray, w: int) -> np.ndarray:
    k = np.ones(w, dtype=np.float32) / w
    return np.convolve(sig, k, mode="same").astype(np.float32)

def pairwise_identity_multiscale(row_i: np.ndarray, row_j: np.ndarray) -> np.ndarray:
    ident = (np.asarray(row_i) == np.asarray(row_j)).astype(np.float32)  # (SEQ_LEN,)
    T = grid_frame_count()
    # sample the smoothed identity onto the shared T-grid (linear indices)
    idx = np.linspace(0, SEQ_LEN - 1, T).astype(int)
    return np.stack([_boxcar(ident, w)[idx] for w in SCALES])  # (len(SCALES), T)

def encode_triplet(rows: np.ndarray, arm: str) -> np.ndarray:
    r, p1, p2 = rows[0], rows[1], rows[2]
    if arm == "A1":
        return np.stack([spectrogram_one(r), spectrogram_one(p1), spectrogram_one(p2)])
    if arm == "A2":
        return np.concatenate([spectrogram_unsummed(r), spectrogram_unsummed(p1),
                               spectrogram_unsummed(p2)], axis=0)  # (12,F,T)
    if arm == "A0":
        return np.stack([pairwise_identity_multiscale(r, p1),
                         pairwise_identity_multiscale(r, p2),
                         pairwise_identity_multiscale(p1, p2)])   # (3, S, T)
    raise ValueError(f"unknown arm {arm!r}")
```

Note on A0 channel semantics: channels are `(R,P1), (R,P2), (P1,P2)` **only at loader order**; the harness permutes *rows before encoding* (Task 3), so at train time the model never sees a fixed "R is row 0." The A0 pairwise convention is defined over whatever row order the harness hands in.

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/spectrogram/test_encode.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spectrogram/encode.py tests/spectrogram/test_encode.py
git commit -m "feat(spectrogram): A0/A1/A2 encoders on a shared STFT grid"
```

---

## Task 3: Permuted-channel dataset + symmetric normalization

**Files:**
- Create: `spectrogram/harness.py`, `tests/spectrogram/test_harness.py`

**Interfaces:**
- Consumes: `spectrogram.data.Triplet`, `spectrogram.encode.encode_triplet`, `spectrogram.config` (`IMG_SIZE`).
- Produces:
  - `permute_rows(rows, recomb_idx, rng) -> tuple[np.ndarray, int]` — returns row-permuted copy and the new index of the recombinant.
  - `joint_normalize(img) -> np.ndarray` — subtract a single scalar mean and divide by a single scalar std computed over the WHOLE image (all channels together), so per-channel relationships are preserved.
  - `class IdentificationDataset(torch.utils.data.Dataset)` — ctor `(triplets, arm, rng_seed, scramble=False)`; `__getitem__` returns `(tensor (C, IMG_SIZE, IMG_SIZE) float32, label int)` where `label` is the permuted recombinant channel-group index (0/1/2). For A2 (12 planes) the label indexes the 3 sequence-groups, and permutation is applied at the sequence level (blocks of 4 planes move together). `scramble=True` independently shuffles each channel's position (last) axis AFTER encoding (positional-scramble control).

- [ ] **Step 1: Write the failing tests**

Create `tests/spectrogram/test_harness.py`:
```python
import numpy as np, torch
from spectrogram import config
from spectrogram.data import Triplet
from spectrogram.harness import permute_rows, joint_normalize, IdentificationDataset

def _triplet():
    rng = np.random.default_rng(1)
    rows = rng.integers(0, 4, size=(3, config.SEQ_LEN)).astype(np.int8)
    return Triplet(rows=rows, recomb_idx=0, source="santa", group="XML-2")

def test_permute_tracks_recomb():
    rows = np.array([[0]*4, [1]*4, [2]*4], dtype=np.int8)
    rng = np.random.default_rng(3)
    for _ in range(20):
        pr, new_idx = permute_rows(rows, 0, rng)
        assert (pr[new_idx] == 0).all()      # recomb row still found at new_idx

def test_permutation_is_roughly_uniform():
    rows = np.zeros((3, 2), dtype=np.int8)
    rng = np.random.default_rng(4)
    counts = [0, 0, 0]
    for _ in range(600):
        _, idx = permute_rows(rows, 0, rng)
        counts[idx] += 1
    assert all(120 < c < 280 for c in counts)   # ~200 each, not collapsed

def test_joint_normalize_is_symmetric():
    img = np.stack([np.full((4, 4), 2.0), np.full((4, 4), 4.0), np.full((4, 4), 6.0)])
    out = joint_normalize(img.astype(np.float32))
    # a single global mean/std => channel ordering/relative magnitude preserved
    assert out[1].mean() > out[0].mean() and out[2].mean() > out[1].mean()

def test_dataset_item_shape_and_label():
    ds = IdentificationDataset([_triplet()], arm="A1", rng_seed=0)
    x, y = ds[0]
    assert x.shape == (3, config.IMG_SIZE, config.IMG_SIZE)
    assert x.dtype == torch.float32
    assert y in (0, 1, 2)

def test_scramble_changes_image_but_not_shape():
    t = _triplet()
    plain = IdentificationDataset([t], arm="A0", rng_seed=0)[0][0]
    scr = IdentificationDataset([t], arm="A0", rng_seed=0, scramble=True)[0][0]
    assert plain.shape == scr.shape
    assert not torch.allclose(plain, scr)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/spectrogram/test_harness.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `spectrogram/harness.py`**

```python
"""Permuted-channel 3-way identification dataset."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from spectrogram.config import IMG_SIZE
from spectrogram.encode import encode_triplet

def permute_rows(rows: np.ndarray, recomb_idx: int, rng: np.random.Generator):
    perm = rng.permutation(3)
    new_rows = rows[perm]
    new_idx = int(np.where(perm == recomb_idx)[0][0])
    return new_rows, new_idx

def joint_normalize(img: np.ndarray) -> np.ndarray:
    m = float(img.mean()); s = float(img.std()) + 1e-6
    return ((img - m) / s).astype(np.float32)

def _scramble_positions(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    for c in range(out.shape[0]):
        out[c] = out[c][:, rng.permutation(out.shape[-1])]
    return out

class IdentificationDataset(Dataset):
    def __init__(self, triplets, arm, rng_seed=0, scramble=False):
        self.triplets = list(triplets)
        self.arm = arm
        self.scramble = scramble
        self.rng_seed = rng_seed

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, i):
        t = self.triplets[i]
        rng = np.random.default_rng(self.rng_seed * 1_000_003 + i)
        rows, label = permute_rows(t.rows, t.recomb_idx, rng)
        img = encode_triplet(rows, self.arm)          # (C, H, W)
        if self.scramble:
            img = _scramble_positions(img, rng)
        img = joint_normalize(img)
        x = torch.from_numpy(img).unsqueeze(0)        # (1,C,H,W) for interpolate
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                          align_corners=False).squeeze(0)
        return x, label
```

Note: because all triplets are pre-padded to `SEQ_LEN`, every image has the same native `(C,F,T)` before resize, so the uniform bilinear resize to `IMG_SIZE` is consistent across samples (the anisotropy hazard the spec warns about only arises with variable native widths).

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/spectrogram/test_harness.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spectrogram/harness.py tests/spectrogram/test_harness.py
git commit -m "feat(spectrogram): permuted-channel dataset + symmetric joint norm"
```

---

## Task 4: Models — timm backbone + small-CNN floor

**Files:**
- Create: `spectrogram/models.py`, `tests/spectrogram/test_models.py`

**Interfaces:**
- Consumes: `timm`, `spectrogram.config` (`BACKBONE`).
- Produces:
  - `build_backbone(in_ch, pretrained, name=BACKBONE, n_classes=3) -> torch.nn.Module` — timm model with `in_chans=in_ch`, `num_classes=n_classes`; when `pretrained and in_ch != 3` timm adapts the stem by summing/replicating pretrained stem weights (its default). 3-way head.
  - `SmallCNN(in_ch, n_classes=3) -> torch.nn.Module` — 4 conv-BN-ReLU blocks + global average pool + linear; the from-scratch capacity floor.
  - `in_channels_for(arm) -> int` — A0→3, A1→3, A2→12.

- [ ] **Step 1: Write the failing tests**

Create `tests/spectrogram/test_models.py`:
```python
import torch
from spectrogram.models import build_backbone, SmallCNN, in_channels_for

def test_in_channels_map():
    assert in_channels_for("A0") == 3
    assert in_channels_for("A1") == 3
    assert in_channels_for("A2") == 12

def test_backbone_forward_3ch():
    m = build_backbone(in_ch=3, pretrained=False)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)

def test_backbone_forward_12ch():
    m = build_backbone(in_ch=12, pretrained=False)
    out = m(torch.randn(2, 12, 224, 224))
    assert out.shape == (2, 3)

def test_smallcnn_forward():
    m = SmallCNN(in_ch=3)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/spectrogram/test_models.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `spectrogram/models.py`**

```python
"""Backbone loader + small from-scratch capacity floor."""
from __future__ import annotations
import timm
import torch
import torch.nn as nn
from spectrogram.config import BACKBONE

def in_channels_for(arm: str) -> int:
    return {"A0": 3, "A1": 3, "A2": 12}[arm]

def build_backbone(in_ch: int, pretrained: bool, name: str = BACKBONE, n_classes: int = 3):
    return timm.create_model(name, pretrained=pretrained, in_chans=in_ch,
                             num_classes=n_classes)

class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 3):
        super().__init__()
        chs = [in_ch, 32, 64, 128, 128]
        blocks = []
        for a, b in zip(chs[:-1], chs[1:]):
            blocks += [nn.Conv2d(a, b, 3, stride=2, padding=1),
                       nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(chs[-1], n_classes)

    def forward(self, x):
        x = self.features(x).mean(dim=(2, 3))
        return self.head(x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/spectrogram/test_models.py -v`
Expected: PASS (first run downloads timm weights only if `pretrained=True`; tests use `pretrained=False`, no network).

- [ ] **Step 5: Commit**

```bash
git add spectrogram/models.py tests/spectrogram/test_models.py
git commit -m "feat(spectrogram): timm backbone (in-chans adapt) + small-CNN floor"
```

---

## Task 5: Training loop (AMP) + smoke test

**Files:**
- Create: `spectrogram/train.py`, `tests/spectrogram/test_train_smoke.py`

**Interfaces:**
- Consumes: `spectrogram.harness.IdentificationDataset`, `spectrogram.models`.
- Produces:
  - `train_model(model, train_ds, val_ds, *, epochs, batch_size=BATCH_SIZE, lr=1e-4, device=None, amp=True) -> dict` returning `{"model": model, "history": [...], "best_val_acc": float}`.
  - `predict(model, ds, device=None) -> np.ndarray` — per-sample predicted class (0/1/2), order matching `ds`.
  - `accuracy(preds, ds) -> float`.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/spectrogram/test_train_smoke.py`:
```python
import numpy as np
from spectrogram.data import Triplet
from spectrogram.harness import IdentificationDataset
from spectrogram.models import SmallCNN, in_channels_for
from spectrogram.train import train_model, predict, accuracy
from spectrogram import config

def _mosaic_triplets(n=24, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        r = rng.integers(0, 4, size=config.SEQ_LEN).astype(np.int8)
        half = config.SEQ_LEN // 2
        p1 = r.copy(); p1[half:] = (r[half:] + 1) % 4
        p2 = r.copy(); p2[:half] = (r[:half] + 1) % 4
        out.append(Triplet(np.stack([r, p1, p2]).astype(np.int8), 0, "santa", "XML-2"))
    return out

def test_training_runs_and_learns_on_easy_signal():
    ds = IdentificationDataset(_mosaic_triplets(), arm="A0", rng_seed=0)
    m = SmallCNN(in_ch=in_channels_for("A0"))
    res = train_model(m, ds, ds, epochs=3, batch_size=8, amp=False)
    assert "best_val_acc" in res
    preds = predict(res["model"], ds)
    assert preds.shape == (len(ds),)
    # easy separable mosaic on A0: floor CNN should beat chance within 3 epochs
    assert accuracy(preds, ds) > 0.34
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/spectrogram/test_train_smoke.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `spectrogram/train.py`**

```python
"""Config-driven AMP training + prediction for the 3-way identification task."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader
from spectrogram.config import BATCH_SIZE

def _device(d=None):
    return torch.device(d or ("cuda" if torch.cuda.is_available() else "cpu"))

def train_model(model, train_ds, val_ds, *, epochs, batch_size=BATCH_SIZE, lr=1e-4,
                device=None, amp=True):
    dev = _device(device); model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")
    lossf = torch.nn.CrossEntropyLoss()
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    history, best = [], 0.0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp and dev.type == "cuda"):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        va = accuracy(predict(model, val_ds, dev), val_ds)
        history.append({"epoch": ep, "val_acc": va}); best = max(best, va)
    return {"model": model, "history": history, "best_val_acc": best}

@torch.no_grad()
def predict(model, ds, device=None):
    dev = _device(device); model.to(dev).eval()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    out = []
    for x, _ in dl:
        out.append(model(x.to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def accuracy(preds, ds) -> float:
    labels = np.array([ds[i][1] for i in range(len(ds))])
    return float((preds == labels).mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/spectrogram/test_train_smoke.py -v`
Expected: PASS (CPU, a few seconds). If flaky near the 0.34 bound, raise epochs to 5 — the signal is deliberately easy.

- [ ] **Step 5: Commit**

```bash
git add spectrogram/train.py tests/spectrogram/test_train_smoke.py
git commit -m "feat(spectrogram): AMP training loop + predict/accuracy + smoke test"
```

---

## Task 6: Significance + evaluation utilities

**Files:**
- Create: `spectrogram/eval.py`, `tests/spectrogram/test_eval.py`

**Interfaces:**
- Consumes: `statsmodels.stats.contingency_tables.mcnemar`, `spectrogram.data.Triplet`.
- Produces:
  - `loco_folds(lanl_triplets) -> list[tuple[str, list, list]]` — `(held_out_crf, train_triplets, test_triplets)` per CRF (test = that CRF's triplets, train = the rest).
  - `mcnemar_pvalue(correct_a, correct_b) -> float` — paired boolean arrays (per-sample correctness of arm-A vs arm-B), exact McNemar.
  - `cluster_bootstrap_diff(correct_a, correct_b, groups, n=2000, seed=0) -> tuple[float,float]` — 95% CI on mean(acc_a) − mean(acc_b), resampling whole `groups` (CRF clusters).
  - `label_permutation_null(preds_fixed, labels, n=1000, seed=0) -> float` — accuracy quantile of the model vs shuffled-label chance distribution.
  - `power_mde(n_pairs, discordant_rate=0.3, alpha=0.05, power=0.8) -> float` — minimum detectable paired accuracy difference under McNemar for a given number of paired test samples (simulation-based).
  - `decides_win(acc_a0, acc_arm, mcnemar_p, ci_low, ci_high) -> bool` — the pre-registered rule.

- [ ] **Step 1: Write the failing tests**

Create `tests/spectrogram/test_eval.py`:
```python
import numpy as np
from spectrogram.data import Triplet
from spectrogram.eval import (loco_folds, mcnemar_pvalue, cluster_bootstrap_diff,
                              decides_win, power_mde)

def _lanl():
    return [Triplet(np.zeros((3, 4), np.int8), 0, "lanl", c)
            for c in ("CRF02_AG", "CRF07_BC", "CRF08_BC", "CRF12_BF")]

def test_loco_leaves_one_crf_out():
    folds = loco_folds(_lanl())
    assert len(folds) == 4
    for held, train, test in folds:
        assert all(t.group == held for t in test)
        assert all(t.group != held for t in train)

def test_mcnemar_detects_clear_difference():
    a = np.ones(100, bool)                 # arm A always right
    b = np.zeros(100, bool); b[:5] = True  # arm B mostly wrong
    assert mcnemar_pvalue(a, b) < 0.001

def test_mcnemar_no_difference():
    a = np.array([True, False] * 50); b = a.copy()
    assert mcnemar_pvalue(a, b) == 1.0 or mcnemar_pvalue(a, b) > 0.99

def test_cluster_bootstrap_ci_orders():
    a = np.ones(40, bool); b = np.zeros(40, bool)
    groups = np.repeat(["c1", "c2", "c3", "c4"], 10)
    lo, hi = cluster_bootstrap_diff(a, b, groups, n=500, seed=1)
    assert lo > 0 and hi >= lo

def test_decision_rule_requires_all_three():
    assert decides_win(0.4, 0.6, mcnemar_p=0.01, ci_low=0.05, ci_high=0.3) is True
    assert decides_win(0.4, 0.6, mcnemar_p=0.20, ci_low=0.05, ci_high=0.3) is False  # p fails
    assert decides_win(0.4, 0.6, mcnemar_p=0.01, ci_low=-0.02, ci_high=0.3) is False # CI includes 0
    assert decides_win(0.6, 0.4, mcnemar_p=0.01, ci_low=0.05, ci_high=0.3) is False  # not higher

def test_power_mde_shrinks_with_n():
    assert power_mde(20) > power_mde(200)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/spectrogram/test_eval.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `spectrogram/eval.py`**

```python
"""Leave-one-CRF-out folds + pre-registered significance discipline."""
from __future__ import annotations
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def loco_folds(lanl_triplets):
    crfs = sorted({t.group for t in lanl_triplets})
    folds = []
    for held in crfs:
        test = [t for t in lanl_triplets if t.group == held]
        train = [t for t in lanl_triplets if t.group != held]
        folds.append((held, train, test))
    return folds

def mcnemar_pvalue(correct_a, correct_b) -> float:
    a = np.asarray(correct_a, bool); b = np.asarray(correct_b, bool)
    n01 = int(((~a) & b).sum())   # A wrong, B right
    n10 = int((a & (~b)).sum())   # A right, B wrong
    table = [[0, n01], [n10, 0]]
    exact = (n01 + n10) < 25
    return float(mcnemar(table, exact=exact).pvalue)

def cluster_bootstrap_diff(correct_a, correct_b, groups, n=2000, seed=0):
    a = np.asarray(correct_a, float); b = np.asarray(correct_b, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.concatenate([np.where(groups == g)[0] for g in pick])
        diffs.append(a[mask].mean() - b[mask].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)

def label_permutation_null(preds_fixed, labels, n=1000, seed=0) -> float:
    labels = np.asarray(labels); rng = np.random.default_rng(seed)
    obs = float((np.asarray(preds_fixed) == labels).mean())
    null = [float((np.asarray(preds_fixed) == rng.permutation(labels)).mean())
            for _ in range(n)]
    return float((np.array(null) >= obs).mean())   # p-value; want small if real

def power_mde(n_pairs, discordant_rate=0.3, alpha=0.05, power=0.8, seed=0):
    """Smallest paired accuracy diff detectable at `power` via McNemar, by sim."""
    rng = np.random.default_rng(seed)
    for delta in np.linspace(0.02, 0.6, 30):
        hits = 0
        p_b_right_given_discordant = 0.5 + delta / (2 * discordant_rate)
        p_b_right_given_discordant = min(max(p_b_right_given_discordant, 0), 1)
        for _ in range(300):
            disc = rng.random(n_pairs) < discordant_rate
            b_right = rng.random(n_pairs) < p_b_right_given_discordant
            n01 = int((disc & b_right).sum()); n10 = int((disc & ~b_right).sum())
            if (n01 + n10) == 0:
                continue
            p = float(mcnemar([[0, n01], [n10, 0]], exact=True).pvalue)
            hits += p < alpha
        if hits / 300 >= power:
            return float(delta)
    return float("inf")

def decides_win(acc_a0, acc_arm, mcnemar_p, ci_low, ci_high) -> bool:
    return bool(acc_arm > acc_a0 and mcnemar_p < 0.05 and ci_low > 0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/spectrogram/test_eval.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spectrogram/eval.py tests/spectrogram/test_eval.py
git commit -m "feat(spectrogram): LOCO folds + McNemar + cluster-bootstrap + power/decision rule"
```

---

## Task 7: Confound diagnostics (blocking gates)

**Files:**
- Create: `spectrogram/probe.py`, `tests/spectrogram/test_probe.py`

**Interfaces:**
- Consumes: `spectrogram.encode`, `spectrogram.harness`, `spectrogram.train`, `spectrogram.data.Triplet`.
- Produces:
  - `own_parents_dataset(triplets, arm, rng_seed=0) -> IdentificationDataset`-like binary set: each item is a single sequence's plane(s), label = is-recombinant (1) vs is-a-parent (0), balanced. Used for the **P2 hard gate**.
  - `run_p2_gate(triplets, arm, epochs=5) -> dict` → `{"auc": float, "leak": bool}` (`leak = auc > 0.6`).
  - `pairwise_divergence(triplet) -> float` — mean over positions of fraction of the 3 pairs that mismatch (a divergence proxy).
  - `divergence_vs_prediction(triplets, preds, correct) -> float` — point-biserial correlation between per-triplet divergence and correctness (near 0 = not keying on divergence).

- [ ] **Step 1: Write the failing tests**

Create `tests/spectrogram/test_probe.py`:
```python
import numpy as np
from spectrogram.data import Triplet
from spectrogram.probe import pairwise_divergence, divergence_vs_prediction

def test_divergence_zero_for_identical():
    rows = np.zeros((3, 100), np.int8)
    assert pairwise_divergence(Triplet(rows, 0, "santa", "g")) == 0.0

def test_divergence_positive_when_different():
    rows = np.stack([np.zeros(100), np.ones(100), np.full(100, 2)]).astype(np.int8)
    assert pairwise_divergence(Triplet(rows, 0, "santa", "g")) > 0.9

def test_divergence_vs_prediction_detects_dependence():
    # correctness perfectly tracks divergence => strong correlation
    trips, preds, correct = [], [], []
    for k in range(20):
        v = k / 20.0
        rows = np.zeros((3, 100), np.int8)
        rows[1, : int(v * 100)] = 1
        trips.append(Triplet(rows, 0, "santa", "g"))
        correct.append(v > 0.5)
    corr = divergence_vs_prediction(trips, np.zeros(20), np.array(correct))
    assert abs(corr) > 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/spectrogram/test_probe.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `spectrogram/probe.py`**

```python
"""Confound diagnostics: own-parents leak gate, divergence analysis.
Positional-scramble is exercised via IdentificationDataset(scramble=True)."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from spectrogram.encode import encode_triplet, spectrogram_one
from spectrogram.harness import joint_normalize
from spectrogram.config import IMG_SIZE
import torch.nn.functional as F

def pairwise_divergence(triplet) -> float:
    r = triplet.rows
    pairs = [(0, 1), (0, 2), (1, 2)]
    mm = [(r[i] != r[j]).mean() for i, j in pairs]
    return float(np.mean(mm))

def divergence_vs_prediction(triplets, preds, correct) -> float:
    div = np.array([pairwise_divergence(t) for t in triplets])
    c = np.asarray(correct, float)
    if c.std() == 0 or div.std() == 0:
        return 0.0
    return float(np.corrcoef(div, c)[0, 1])

class _OwnParentsDataset(Dataset):
    """Single-sequence binary: recombinant (1) vs one of its parents (0)."""
    def __init__(self, triplets, rng_seed=0):
        self.items = []
        rng = np.random.default_rng(rng_seed)
        for t in triplets:
            self.items.append((t.rows[t.recomb_idx], 1))
            par = [k for k in range(3) if k != t.recomb_idx][rng.integers(2)]
            self.items.append((t.rows[par], 0))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        row, y = self.items[i]
        img = joint_normalize(spectrogram_one(row)[None])   # (1,F,T)
        x = F.interpolate(torch.from_numpy(img)[None], size=(IMG_SIZE, IMG_SIZE),
                          mode="bilinear", align_corners=False).squeeze(0)
        return x, y

def run_p2_gate(triplets, epochs=5, device=None) -> dict:
    from spectrogram.models import SmallCNN
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = _OwnParentsDataset(triplets)
    m = SmallCNN(in_ch=1, n_classes=2).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    for _ in range(epochs):
        m.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(m(x), y); loss.backward(); opt.step()
    m.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=16):
            ps.append(torch.softmax(m(x.to(dev)), 1)[:, 1].cpu().numpy()); ys.append(y.numpy())
    auc = float(roc_auc_score(np.concatenate(ys), np.concatenate(ps)))
    return {"auc": auc, "leak": auc > 0.6}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/spectrogram/test_probe.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add spectrogram/probe.py tests/spectrogram/test_probe.py
git commit -m "feat(spectrogram): confound diagnostics (P2 gate, divergence, scramble hooks)"
```

---

## Task 8: LANL parent-pairing expansion (grow eval N, with fallback)

**Files:**
- Create: `spectrogram/expand_lanl.py`, `tests/spectrogram/test_expand_lanl.py`
- Reference (do not import Entrez logic wholesale): `build_lanl_triplets.py`

**Interfaces:**
- Consumes: NCBI Entrez (network), MAFFT (`/usr/bin/mafft`), the existing `data/lanl_crf/genbank_cache/`.
- Produces:
  - `SUBTYPE_REPS: dict[str, list[str]]` — multiple accessions per subtype (extend the single-rep set in `build_lanl_triplets.py`).
  - `enumerate_pairings(crf) -> list[tuple[str,str,str]]` — `(recomb_acc, pA_acc, pB_acc)` over the cartesian product of the two parent subtypes' reps.
  - `build_expanded_triplets(out_dir=data/lanl_crf/triplets_expanded) -> int` — writes one aligned 3-seq FASTA per pairing; returns count. Uses MAFFT + HXB2-strip exactly as `build_lanl_triplets.py` does.

- [ ] **Step 1: Write the failing test (offline parts only)**

Create `tests/spectrogram/test_expand_lanl.py`:
```python
from spectrogram.expand_lanl import SUBTYPE_REPS, enumerate_pairings

def test_multiple_reps_per_subtype():
    # at least the parent subtypes we need must have >1 rep to grow N
    for st in ("A1", "B", "C", "F1", "G"):
        assert st in SUBTYPE_REPS and len(SUBTYPE_REPS[st]) >= 2

def test_enumerate_pairings_is_product():
    p = enumerate_pairings("CRF02_AG")   # parents A1 x G
    assert len(p) == len(SUBTYPE_REPS["A1"]) * len(SUBTYPE_REPS["G"])
    for recomb, pa, pb in p:
        assert recomb == "L39106"        # CRF02_AG recombinant accession
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/spectrogram/test_expand_lanl.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `spectrogram/expand_lanl.py`**

Reuse the alignment machinery from `build_lanl_triplets.py` (MAFFT `--auto` on recomb + 2 parents + HXB2, then strip HXB2). Extend the reference set — pick widely-cited additional subtype references (verify each accession resolves before committing them; the list below is the *structure*, and the implementer confirms accessions against NCBI):

```python
"""Grow the real-HIV identification set by pairing each CRF's recombinant with
MULTIPLE reference strains per parent subtype. Same MAFFT+HXB2-strip pipeline
as build_lanl_triplets.py. Independent unit stays the CRF family (4)."""
from __future__ import annotations
from pathlib import Path
from itertools import product

# >=2 accessions per subtype. Implementer verifies each resolves on NCBI and is
# a full-genome, non-recombinant subtype reference before committing.
SUBTYPE_REPS = {
    "A1": ["U51190", "AF004885"],
    "B":  ["K03455", "AY423387"],
    "C":  ["U46016", "AF067155"],
    "F1": ["AF005494", "AF077336"],
    "G":  ["AF061641", "AF084936"],
}
CRF_PARENTS = {
    "CRF02_AG": ("L39106",   ("A1", "G")),
    "CRF07_BC": ("AF286226", ("B", "C")),
    "CRF08_BC": ("AY008715", ("B", "C")),
    "CRF12_BF": ("AF385936", ("B", "F1")),
}

def enumerate_pairings(crf: str):
    recomb, (sa, sb) = CRF_PARENTS[crf]
    return [(recomb, pa, pb) for pa, pb in product(SUBTYPE_REPS[sa], SUBTYPE_REPS[sb])]

def build_expanded_triplets(out_dir: Path = Path("data/lanl_crf/triplets_expanded")) -> int:
    # Fetch (cache) each accession via Entrez, then per pairing run MAFFT with HXB2,
    # strip HXB2, write a 3-record FASTA (recomb, pA, pB) named <crf>__<pa>__<pb>.fa.
    # Mirror the exact fetch+align+strip steps in build_lanl_triplets.py.
    raise NotImplementedError("port fetch+MAFFT+strip from build_lanl_triplets.py")
```

Then port the concrete fetch/align/strip from `build_lanl_triplets.py` into `build_expanded_triplets`, writing `data/lanl_crf/triplets_expanded/<crf>__<pa>__<pb>.fa` with the `group` = CRF name preserved.

- [ ] **Step 4: Run the offline test**

Run: `$PY -m pytest tests/spectrogram/test_expand_lanl.py -v`
Expected: PASS (tests only touch `SUBTYPE_REPS`/`enumerate_pairings`, no network).

- [ ] **Step 5: Build the expanded set (network + MAFFT, run manually)**

Run: `$PY -m spectrogram.expand_lanl`
Expected: writes N ≈ Σ over CRFs of |repsA|×|repsB| FASTAs (≈ 2×2×4 = 16 with two reps each). **Fallback:** if NCBI/MAFFT is unavailable or an accession is recombinant/partial, drop that rep and log it; the pipeline must still run on the original 4 triplets. `load_lanl_triplets` (Task 1) should be pointed at `triplets_expanded/` when it exists, else the original 4 — add a `dir` arg default that prefers the expanded dir.

- [ ] **Step 6: Commit**

```bash
git add spectrogram/expand_lanl.py tests/spectrogram/test_expand_lanl.py
git commit -m "feat(spectrogram): LANL parent-pairing expansion to grow eval N (with fallback)"
```

---

## Task 9: Orchestration + pre-registration + report

**Files:**
- Create: `spectrogram/run_stage1.py`, `tests/spectrogram/test_run_stage1.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `write_preregistration(path) -> None` — dumps the pinned primary metric + decision rule + power statement to `results_spectrogram_prereg.json` BEFORE any modelling.
  - `run_power_check(lanl_triplets) -> dict` — computes `power_mde` for the actual expanded N; the go/no-go gate.
  - `run_bakeoff(santa, lanl, arms=("A0","A1","A2"), inits=("imagenet","random","floor"), epochs) -> dict` — for each (arm, init): LOCO train on SANTA + 3 CRFs, predict held-out CRF, collect per-triplet correctness + group labels; apply diagnostics.
  - `main()` — orchestrates: prereg → power check → P2 gate → bake-off (with positional-scramble control for the winning arm) → decision rule → write `results_spectrogram_stage1.json`.

- [ ] **Step 1: Write the failing test (orchestration wiring, tiny data)**

Create `tests/spectrogram/test_run_stage1.py`:
```python
import json
from pathlib import Path
from spectrogram.run_stage1 import write_preregistration

def test_prereg_written(tmp_path):
    p = tmp_path / "prereg.json"
    write_preregistration(p)
    d = json.loads(Path(p).read_text())
    assert d["primary_metric"].startswith("mean held-out-CRF")
    assert "decision_rule" in d and "power_statement" in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/spectrogram/test_run_stage1.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `spectrogram/run_stage1.py`**

```python
"""Stage-1 orchestration: pre-register, power-check, diagnostics, bake-off, decide."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from spectrogram.config import BACKBONE
from spectrogram.data import load_santa_triplets, load_lanl_triplets
from spectrogram.harness import IdentificationDataset
from spectrogram.models import build_backbone, SmallCNN, in_channels_for
from spectrogram.train import train_model, predict, accuracy
from spectrogram.eval import (loco_folds, mcnemar_pvalue, cluster_bootstrap_diff,
                              decides_win, power_mde, label_permutation_null)
from spectrogram.probe import run_p2_gate, divergence_vs_prediction

def write_preregistration(path) -> None:
    Path(path).write_text(json.dumps({
        "primary_metric": "mean held-out-CRF top-1 identification accuracy (chance=1/3)",
        "decision_rule": ("A Fourier arm (A1/A2) beats A0 iff higher mean LOCO acc "
                          "AND McNemar p<0.05 vs A0 AND cluster-bootstrap-by-CRF 95% CI "
                          "on the diff excludes 0."),
        "power_statement": "power_mde computed on the actual expanded N before modelling.",
        "backbone": BACKBONE, "arms": ["A0", "A1", "A2"],
        "inits": ["imagenet", "random", "floor"],
    }, indent=2))

def _make_model(arm, init):
    ch = in_channels_for(arm)
    if init == "floor":
        return SmallCNN(in_ch=ch)
    return build_backbone(in_ch=ch, pretrained=(init == "imagenet"))

def run_power_check(lanl):
    n = len(lanl)
    return {"n_test_triplets": n, "mde": power_mde(n)}

def run_bakeoff(santa, lanl, arms=("A0", "A1", "A2"),
                inits=("imagenet", "random", "floor"), epochs=15):
    results = {}
    for arm in arms:
        for init in inits:
            per_correct, per_group = [], []
            for held, train_lanl, test_lanl in loco_folds(lanl):
                train = santa + train_lanl
                m = _make_model(arm, init)
                tr = IdentificationDataset(train, arm, rng_seed=0)
                te = IdentificationDataset(test_lanl, arm, rng_seed=1)
                fit = train_model(m, tr, te, epochs=epochs)
                preds = predict(fit["model"], te)
                labels = np.array([te[i][1] for i in range(len(te))])
                per_correct.append(preds == labels)
                per_group.append(np.array([t.group for t in test_lanl]))
            results[(arm, init)] = {
                "correct": np.concatenate(per_correct),
                "groups": np.concatenate(per_group),
                "acc": float(np.concatenate(per_correct).mean()),
            }
    return results

def main(epochs=15):
    write_preregistration("results_spectrogram_prereg.json")
    santa = load_santa_triplets(_cache(), limit=20_000)
    lanl = load_lanl_triplets()          # prefers expanded dir if present
    power = run_power_check(lanl)
    p2 = run_p2_gate(santa[:2000])
    bake = run_bakeoff(santa, lanl, epochs=epochs)
    a0 = bake[("A0", "floor")]
    decisions = {}
    for (arm, init), r in bake.items():
        if arm == "A0":
            continue
        p = mcnemar_pvalue(r["correct"], a0["correct"])
        lo, hi = cluster_bootstrap_diff(r["correct"], a0["correct"], r["groups"])
        decisions[f"{arm}:{init}"] = {
            "acc": r["acc"], "a0_acc": a0["acc"], "mcnemar_p": p,
            "ci": [lo, hi], "win": decides_win(a0["acc"], r["acc"], p, lo, hi),
        }
    out = {"power": power, "p2_gate": p2,
           "accs": {f"{a}:{i}": v["acc"] for (a, i), v in bake.items()},
           "decisions": decisions}
    Path("results_spectrogram_stage1.json").write_text(json.dumps(out, indent=2, default=float))
    return out

def _cache():
    from cache_v2_reader import CacheV2
    return CacheV2()

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/spectrogram/test_run_stage1.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full Stage-1 pipeline (GPU, long — run in background)**

Run: `$PY -u -m spectrogram.run_stage1 > stage1_run.log 2>&1 &`
Expected: writes `results_spectrogram_prereg.json` first, then `results_spectrogram_stage1.json` with power/P2/accs/decisions. Watch RSS (Global Constraints OOM rule); if VRAM-bound, drop `BATCH_SIZE` in config or switch `BACKBONE` to `convnext_tiny`.

- [ ] **Step 6: Commit**

```bash
git add spectrogram/run_stage1.py tests/spectrogram/test_run_stage1.py
git commit -m "feat(spectrogram): Stage-1 orchestration (prereg, power, diagnostics, bake-off)"
```

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:
- §4.1 shared harness → Task 3 (permutation, fixed length via Task 1, shared grid asserts Task 2, joint norm Task 3).
- §4.2 backbone + floor → Task 4.
- §4.3 bake-off A0/A1/A2 → Task 2 (encoders) + Task 9 (run).
- §4.4 init A/B → Task 4 + Task 9 (`inits`).
- §5 diagnostics (P2, positional-scramble, divergence) → Task 7 (P2, divergence) + Task 3 (`scramble=True`) + Task 9 (wiring).
- §6 LOCO CV + McNemar + cluster-bootstrap + power + label-permutation null → Task 6 + Task 9.
- §7 file list → File Structure (matches).
- Data (§3) incl. the n≈4 problem and parent-pairing growth → Task 1 + Task 8.
- §2 deferred `none` head → out of scope by design; noted, not implemented.

**2. Placeholder scan** — the one deliberate `NotImplementedError` (Task 8 `build_expanded_triplets`) is a *port* instruction with the exact source (`build_lanl_triplets.py`) and output contract given; it is a network/MAFFT step, not a code gap. Accession lists in `SUBTYPE_REPS` are flagged for the implementer to verify against NCBI. No other placeholders.

**3. Type consistency** — `Triplet(rows, recomb_idx, source, group)` used identically in Tasks 1/3/6/7/9; `encode_triplet(rows, arm)` signature consistent Tasks 2/3; `predict`/`accuracy`/`train_model` signatures consistent Tasks 5/9; `decides_win`/`mcnemar_pvalue`/`cluster_bootstrap_diff` signatures consistent Tasks 6/9; `in_channels_for` consistent Tasks 4/9.

## Known limitations (carry into results, per spec §9)

- **Statistical power is intrinsically capped by ~4 independent CRF families.** Parent-pairing expansion (Task 8) grows per-fold sample count but not the number of independent recombinants; cluster-bootstrap over 4 clusters is weak. The `power_mde` gate (Task 9) will likely report a large MDE — if so, the honest outcome is "exploratory, underpowered," and the real lever is adding more CRF *families* (e.g. other LANL CRFs with defined non-recombinant parents), which is a follow-on data task, not part of Stage 1.
- **The most likely result** (per the Fable-5 red-team) is A1 losing to A0 — a real, reportable finding about the paper's summed-spectrogram representation, with A2-vs-A1 isolating whether nucleotide-summing is the culprit.
