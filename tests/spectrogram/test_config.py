from spectrogram import config

def test_pinned_constants():
    assert config.SEQ_LEN == 10_000
    assert config.NPERSEG == 256
    assert config.NOVERLAP == config.NPERSEG // 8
    assert config.GAP_INT == 4
    assert config.NT_INDICATOR_ORDER == (0, 1, 2, 3)
    assert config.SCALES == (50, 100, 200, 500)
