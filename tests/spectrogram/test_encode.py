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
