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
