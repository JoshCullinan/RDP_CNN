"""Tests for the splice-explicit arms A3 (difference-spectrogram) and A4 (dot-plot)."""
import numpy as np
from spectrogram import config
from spectrogram.encode import (match_track, diff_spectrogram_triplet,
                                 dotplot_pair, dotplot_triplet, encode_triplet,
                                 grid_frame_count)
from spectrogram.models import in_channels_for


def _mosaic_rows():
    rng = np.random.default_rng(0)
    r = rng.integers(0, 4, size=config.SEQ_LEN).astype(np.int8)
    half = config.SEQ_LEN // 2
    p1 = r.copy(); p1[half:] = (r[half:] + 1) % 4   # P1 differs from R on 2nd half
    p2 = r.copy(); p2[:half] = (r[:half] + 1) % 4   # P2 differs from R on 1st half
    return np.stack([r, p1, p2]).astype(np.int8)


def test_in_channels_new_arms():
    assert in_channels_for("A3") == 3
    assert in_channels_for("A4") == 3


def test_match_track_tracks_the_splice():
    rows = _mosaic_rows()
    m = match_track(rows[0], rows[1])   # match(R, P1): 1 on first half, 0 on second
    half = config.SEQ_LEN // 2
    assert m[:half].mean() > 0.95 and m[half:].mean() < 0.5


def test_a3_shape_shared_grid():
    rows = _mosaic_rows()
    T = grid_frame_count()
    F = config.NPERSEG // 2 + 1
    a3 = encode_triplet(rows, "A3")
    assert a3.shape == (3, F, T)


def test_a4_dotplot_shape_and_diagonal_splice():
    rows = _mosaic_rows()
    a4 = encode_triplet(rows, "A4")
    n = 128
    assert a4.shape == (3, n, n)
    # channel 0 = dotplot(R, P1): diagonal bright on first-half blocks (R matches
    # P1 there), dim on second-half blocks -> a splice signature.
    diag = np.diagonal(a4[0])
    assert diag[: n // 2].mean() > diag[n // 2:].mean() + 0.3


def test_a4_dotplot_pair_diagonal_is_identity():
    # identical rows -> diagonal all 1.0
    row = np.arange(config.SEQ_LEN, dtype=np.int8) % 4
    dp = dotplot_pair(row, row, n=64)
    assert np.allclose(np.diagonal(dp), 1.0)
