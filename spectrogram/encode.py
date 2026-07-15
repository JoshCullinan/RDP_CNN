"""Three representation encoders for the identification bake-off.
All share one STFT grid so channels correspond position-for-position."""
from __future__ import annotations
import numpy as np
from scipy.signal import spectrogram
from spectrogram.config import (NPERSEG, NOVERLAP, SCALES, SEQ_LEN,
                                NT_INDICATOR_ORDER, GAP_INT)

def nt_indicator(row: np.ndarray, nt: int) -> np.ndarray:
    # gap is never a nucleotide, so even if nt == GAP_INT, return all zeros
    row_arr = np.asarray(row)
    if nt == GAP_INT:
        return np.zeros(len(row_arr), dtype=np.float32)
    return (row_arr == nt).astype(np.float32)

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

# --- splice-explicit arms (Fable-5 advisor, 2026-07-13) -----------------------
# A3/A4 encode the CROSS-sequence match structure directly (not each sequence
# independently), so the only cheap signal is the splice, not composition.

def match_track(row_i: np.ndarray, row_j: np.ndarray) -> np.ndarray:
    """Per-position identity of two aligned rows (same convention as A0)."""
    return (np.asarray(row_i) == np.asarray(row_j)).astype(np.float32)  # (SEQ_LEN,)

def diff_spectrogram_triplet(rows: np.ndarray) -> np.ndarray:
    """A3: STFT of each pairwise match track. A splice (recombinant switching
    which parent it matches) makes a localized transient; composition is
    stationary. One channel per pair -> (3, F, T)."""
    r, p1, p2 = rows[0], rows[1], rows[2]
    return np.stack([_stft(match_track(r, p1)),
                     _stft(match_track(r, p2)),
                     _stft(match_track(p1, p2))])  # (3, F, T)

def dotplot_pair(row_i: np.ndarray, row_j: np.ndarray, n: int = 128) -> np.ndarray:
    """Coarse n x n block-identity map. dot[u,v] = mean position-wise identity
    between block u of row_i and block v of row_j. The diagonal is local
    identity; a mosaic recombinant's r-vs-parent diagonal is bright on the block
    it matches and dim elsewhere -> a splice is a bright-then-dim diagonal."""
    L = (SEQ_LEN // n) * n
    a = np.asarray(row_i)[:L].reshape(n, -1)   # (n, w)
    b = np.asarray(row_j)[:L].reshape(n, -1)   # (n, w)
    eq = (a[:, None, :] == b[None, :, :]).astype(np.float32)  # (n, n, w)
    return eq.mean(axis=2)  # (n, n)

def dotplot_triplet(rows: np.ndarray, n: int = 128) -> np.ndarray:
    """A4: three pairwise dot-plots stacked -> (3, n, n)."""
    r, p1, p2 = rows[0], rows[1], rows[2]
    return np.stack([dotplot_pair(r, p1, n), dotplot_pair(r, p2, n),
                     dotplot_pair(p1, p2, n)])

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
    if arm == "A3":
        return diff_spectrogram_triplet(rows)   # (3, F, T)
    if arm == "A4":
        return dotplot_triplet(rows)            # (3, n, n)
    raise ValueError(f"unknown arm {arm!r}")
