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
                inits=("imagenet", "random", "floor"), epochs=15,
                scramble=False, santa_val=None):
    results = {}
    for arm in arms:
        for init in inits:
            per_correct, per_group, per_preds, per_labels = [], [], [], []
            per_triplets, per_santa_val_acc = [], []
            for held, train_lanl, test_lanl in loco_folds(lanl):
                train = santa + train_lanl
                m = _make_model(arm, init)
                tr = IdentificationDataset(train, arm, rng_seed=0, scramble=scramble)
                te = IdentificationDataset(test_lanl, arm, rng_seed=1, scramble=scramble)
                fit = train_model(m, tr, te, epochs=epochs)
                preds = predict(fit["model"], te)
                labels = np.array([te[i][1] for i in range(len(te))])
                per_correct.append(preds == labels)
                per_group.append(np.array([t.group for t in test_lanl]))
                per_preds.append(preds)
                per_labels.append(labels)
                per_triplets.extend(test_lanl)
                if santa_val is not None:
                    sv = IdentificationDataset(santa_val, arm, rng_seed=2)
                    per_santa_val_acc.append(accuracy(predict(fit["model"], sv), sv))
            results[(arm, init)] = {
                "correct": np.concatenate(per_correct),
                "groups": np.concatenate(per_group),
                "acc": float(np.concatenate(per_correct).mean()),
                "preds": np.concatenate(per_preds),
                "labels": np.concatenate(per_labels),
                "test_triplets": per_triplets,
                "santa_val_acc": float(np.mean(per_santa_val_acc)) if santa_val is not None else None,
            }
    return results

def main(epochs=15):
    write_preregistration("results_spectrogram_prereg.json")
    santa = load_santa_triplets(_cache(), limit=20_000)
    lanl = load_lanl_triplets()          # prefers expanded dir if present
    SANTA_VAL_N = 500
    santa_val = santa[-SANTA_VAL_N:]
    santa_train = santa[:-SANTA_VAL_N]
    power = run_power_check(lanl)
    p2 = run_p2_gate(santa[:2000])
    bake = run_bakeoff(santa_train, lanl, epochs=epochs, santa_val=santa_val)
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

    # Winning Fourier arm (highest LOCO acc among A1/A2) gets the confound gates.
    win_key = max((k for k in bake if k[0] in ("A1", "A2")), key=lambda k: bake[k]["acc"])
    win = bake[win_key]
    # Scrambled control: same arm/init, positions shuffled within each channel
    # (destroys mosaic alignment, preserves marginals) -- winning arm/init only.
    scr = run_bakeoff(santa_train, lanl, arms=(win_key[0],), inits=(win_key[1],),
                      epochs=epochs, scramble=True, santa_val=None)
    scr_acc = scr[win_key]["acc"]
    diagnostics = {
        "winning_arm": f"{win_key[0]}:{win_key[1]}",
        "label_permutation_null_p": label_permutation_null(win["preds"], win["labels"]),
        "divergence_corr": divergence_vs_prediction(win["test_triplets"], win["preds"], win["correct"]),
        "santa_heldout_gap": ((win.get("santa_val_acc") - win["acc"])
                              if win.get("santa_val_acc") is not None else None),
        # If scrambled_acc stays near unscrambled_acc (small delta), the model
        # is reading composition, not mosaic structure -> confounded.
        "scramble_control": {"scrambled_acc": scr_acc, "unscrambled_acc": win["acc"],
                             "delta": win["acc"] - scr_acc},
    }

    out = {"power": power, "p2_gate": p2,
           "accs": {f"{a}:{i}": v["acc"] for (a, i), v in bake.items()},
           "santa_val_accs": {f"{a}:{i}": v["santa_val_acc"] for (a, i), v in bake.items()},
           "decisions": decisions,
           "diagnostics": diagnostics}
    Path("results_spectrogram_stage1.json").write_text(json.dumps(out, indent=2, default=float))
    return out

def _cache():
    from cache_v2_reader import CacheV2
    return CacheV2()

if __name__ == "__main__":
    main()
