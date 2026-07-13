import numpy as np
import torch
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
    np.random.seed(0)
    torch.manual_seed(0)
    ds = IdentificationDataset(_mosaic_triplets(), arm="A0", rng_seed=0)
    m = SmallCNN(in_ch=in_channels_for("A0"))
    res = train_model(m, ds, ds, epochs=8, batch_size=8, amp=False)
    assert "best_val_acc" in res
    preds = predict(res["model"], ds)
    assert preds.shape == (len(ds),)
    # easy separable mosaic on A0: floor CNN should beat chance within 3 epochs
    assert accuracy(preds, ds) > 0.34
