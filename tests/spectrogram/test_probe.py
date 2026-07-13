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
