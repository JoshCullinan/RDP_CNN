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
