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
