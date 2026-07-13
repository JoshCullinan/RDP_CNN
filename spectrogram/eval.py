"""Leave-one-CRF-out folds + pre-registered significance discipline."""
from __future__ import annotations
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def loco_folds(lanl_triplets):
    crfs = sorted({t.group for t in lanl_triplets})
    folds = []
    for held in crfs:
        test = [t for t in lanl_triplets if t.group == held]
        train = [t for t in lanl_triplets if t.group != held]
        folds.append((held, train, test))
    return folds

def mcnemar_pvalue(correct_a, correct_b) -> float:
    a = np.asarray(correct_a, bool); b = np.asarray(correct_b, bool)
    n01 = int(((~a) & b).sum())   # A wrong, B right
    n10 = int((a & (~b)).sum())   # A right, B wrong
    table = [[0, n01], [n10, 0]]
    exact = (n01 + n10) < 25
    return float(mcnemar(table, exact=exact).pvalue)

def cluster_bootstrap_diff(correct_a, correct_b, groups, n=2000, seed=0):
    a = np.asarray(correct_a, float); b = np.asarray(correct_b, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.concatenate([np.where(groups == g)[0] for g in pick])
        diffs.append(a[mask].mean() - b[mask].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)

def label_permutation_null(preds_fixed, labels, n=1000, seed=0) -> float:
    labels = np.asarray(labels); rng = np.random.default_rng(seed)
    obs = float((np.asarray(preds_fixed) == labels).mean())
    null = [float((np.asarray(preds_fixed) == rng.permutation(labels)).mean())
            for _ in range(n)]
    return float((np.array(null) >= obs).mean())   # p-value; want small if real

def power_mde(n_pairs, discordant_rate=0.3, alpha=0.05, power=0.8, seed=0):
    """Smallest paired accuracy diff detectable at `power` via McNemar, by sim."""
    rng = np.random.default_rng(seed)
    for delta in np.linspace(0.02, 0.6, 30):
        hits = 0
        p_b_right_given_discordant = 0.5 + delta / (2 * discordant_rate)
        p_b_right_given_discordant = min(max(p_b_right_given_discordant, 0), 1)
        for _ in range(300):
            disc = rng.random(n_pairs) < discordant_rate
            b_right = rng.random(n_pairs) < p_b_right_given_discordant
            n01 = int((disc & b_right).sum()); n10 = int((disc & ~b_right).sum())
            if (n01 + n10) == 0:
                continue
            p = float(mcnemar([[0, n01], [n10, 0]], exact=True).pvalue)
            hits += p < alpha
        if hits / 300 >= power:
            return float(delta)
    return float("inf")

def decides_win(acc_a0, acc_arm, mcnemar_p, ci_low, ci_high) -> bool:
    return bool(acc_arm > acc_a0 and mcnemar_p < 0.05 and ci_low > 0)
