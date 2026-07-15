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
