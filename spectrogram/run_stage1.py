"""Stage-1 orchestration: pre-register, power-check, diagnostics, bake-off, decide."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from spectrogram.config import BACKBONE
from spectrogram.data import load_santa_triplets, load_lanl_triplets
from spectrogram.harness import IdentificationDataset
from spectrogram.models import build_backbone, SmallCNN, in_channels_for
from spectrogram.train import train_model, predict, accuracy
from spectrogram.eval import (loco_folds, mcnemar_pvalue, cluster_bootstrap_diff,
                              decides_win, power_mde, label_permutation_null)
from spectrogram.probe import run_p2_gate, divergence_vs_prediction

def write_preregistration(path) -> None:
    Path(path).write_text(json.dumps({
        "primary_metric": "mean held-out-CRF top-1 identification accuracy (chance=1/3)",
        "decision_rule": ("A Fourier arm (A1/A2) beats A0 iff higher mean LOCO acc "
                          "AND McNemar p<0.05 vs A0 AND cluster-bootstrap-by-CRF 95% CI "
                          "on the diff excludes 0."),
        "power_statement": "power_mde computed on the actual expanded N before modelling.",
        "backbone": BACKBONE, "arms": ["A0", "A1", "A2"],
        "inits": ["imagenet", "random", "floor"],
    }, indent=2))

def _make_model(arm, init):
    ch = in_channels_for(arm)
    if init == "floor":
        return SmallCNN(in_ch=ch)
    return build_backbone(in_ch=ch, pretrained=(init == "imagenet"))

def run_power_check(lanl):
    n = len(lanl)
    return {"n_test_triplets": n, "mde": power_mde(n)}

def run_bakeoff(santa, lanl, arms=("A0", "A1", "A2"),
                inits=("imagenet", "random", "floor"), epochs=15):
    results = {}
    for arm in arms:
        for init in inits:
            per_correct, per_group = [], []
            for held, train_lanl, test_lanl in loco_folds(lanl):
                train = santa + train_lanl
                m = _make_model(arm, init)
                tr = IdentificationDataset(train, arm, rng_seed=0)
                te = IdentificationDataset(test_lanl, arm, rng_seed=1)
                fit = train_model(m, tr, te, epochs=epochs)
                preds = predict(fit["model"], te)
                labels = np.array([te[i][1] for i in range(len(te))])
                per_correct.append(preds == labels)
                per_group.append(np.array([t.group for t in test_lanl]))
            results[(arm, init)] = {
                "correct": np.concatenate(per_correct),
                "groups": np.concatenate(per_group),
                "acc": float(np.concatenate(per_correct).mean()),
            }
    return results

def main(epochs=15):
    write_preregistration("results_spectrogram_prereg.json")
    santa = load_santa_triplets(_cache(), limit=20_000)
    lanl = load_lanl_triplets()          # prefers expanded dir if present
    power = run_power_check(lanl)
    p2 = run_p2_gate(santa[:2000])
    bake = run_bakeoff(santa, lanl, epochs=epochs)
    a0 = bake[("A0", "floor")]
    decisions = {}
    for (arm, init), r in bake.items():
        if arm == "A0":
            continue
        p = mcnemar_pvalue(r["correct"], a0["correct"])
        lo, hi = cluster_bootstrap_diff(r["correct"], a0["correct"], r["groups"])
        decisions[f"{arm}:{init}"] = {
            "acc": r["acc"], "a0_acc": a0["acc"], "mcnemar_p": p,
            "ci": [lo, hi], "win": decides_win(a0["acc"], r["acc"], p, lo, hi),
        }
    out = {"power": power, "p2_gate": p2,
           "accs": {f"{a}:{i}": v["acc"] for (a, i), v in bake.items()},
           "decisions": decisions}
    Path("results_spectrogram_stage1.json").write_text(json.dumps(out, indent=2, default=float))
    return out

def _cache():
    from cache_v2_reader import CacheV2
    return CacheV2()

if __name__ == "__main__":
    main()
