#!/usr/bin/env python3
"""Powered in-domain diagnostic: stratify SANTA-val which-of-3 accuracy by
recombinant match-sharpness (GLOBAL and INFORMATIVE-SITE), and locate real-HIV
(LANL) recombinants on those axes. Uses a saved whichof3 ckpt; NO retrain.

Two sharpness metrics per triplet (mean of max(R==P1, R==P2)):
  - GLOBAL: over all recomb-content positions (R non-gap).
  - INFORMATIVE: over positions where the parents DIFFER (P1!=P2) & R non-gap.
    This is the metric that matters for which-of-3 -- the discriminative signal
    lives only at informative sites; global sharpness is dominated by the ~87%
    non-informative sites where R trivially matches both parents.

Secondary axis: parent-parent pairwise divergence.

Verdict discriminates: (softness gap, measured properly at informative sites)
vs (construction-specific residual after matching on observables). Matched
comparison uses SANTA restricted to LANL's band on BOTH informative-sharpness
AND parent-divergence.

Eval runs in fp32 (autocast OFF) so the invariant head is EXACTLY invariant.
OOM rule: per-triplet streaming; never astype/copy the memmap.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_whichof3 import (
    build_net_from_ckpt, hypothesis_features, content_len, trunc_rows,
    load_lanl_triplets, softmax_np, GAP,
)
from m3_dilated import event_plan, load_event, _rss_watchdog, _rss
from m3_divergence_gate import pairwise_divergence
from cache_v2_reader import CacheV2


def sharpness_global(R, P1, P2) -> float:
    mask = (R != GAP)
    if not mask.any():
        return float("nan")
    best = np.maximum(R == P1, R == P2)
    return float(best[mask].mean())


def sharpness_informative(R, P1, P2) -> tuple[float, int]:
    """max(R==P1,R==P2) averaged over INFORMATIVE positions: (P1!=P2) & R non-gap.
    Returns (sharpness, n_informative)."""
    mask = (P1 != P2) & (R != GAP)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    best = np.maximum(R == P1, R == P2)
    return float(best[mask].mean()), n


@torch.no_grad()
def predict_fp32(net, rows, device) -> np.ndarray:
    cl = content_len(rows)
    X = hypothesis_features(rows).to(device).float()
    return net(X, cl).float().cpu().numpy()


def which_of_3(net, rows, true_idx, device) -> tuple[int, np.ndarray]:
    logits = predict_fp32(net, rows, device)
    return int(int(logits.argmax()) == true_idx), logits


def bin_curve(x, corr, n_bins):
    x = np.asarray(x, float); corr = np.asarray(corr, float)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)).copy()
    edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"bin": b, "n": int(m.sum()), "x_lo": float(edges[b]),
                    "x_hi": float(edges[b + 1]), "x_mean": float(x[m].mean()),
                    "acc": float(corr[m].mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models_test/whichof3_s0.pt"))
    ap.add_argument("--val-shards", nargs="+", default=["XML-5"])
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=0)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--lanl-dir", type=Path, default=Path("data/lanl_crf/triplets"))
    ap.add_argument("--lanl-expanded-dir", type=Path,
                    default=Path("data/lanl_crf/triplets_expanded"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("models_test/whichof3_strat.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, ck = build_net_from_ckpt(args.ckpt, device)
    max_len = args.max_len or int(ck["train_cfg"]["max_len"])
    print(f"[{time.strftime('%H:%M:%S')}] loaded {args.ckpt} epoch={ck.get('epoch')} "
          f"santa_val_acc={ck.get('santa_val_acc')} max_len={max_len}", flush=True)

    cache = CacheV2()
    val_plan = event_plan(cache, args.val_shards, args.n_val,
                          random.Random(args.seed), max_len)
    print(f"  SANTA val triplets: {len(val_plan)}", flush=True)

    per = []
    t0 = time.time()
    for k, (sh, ev) in enumerate(val_plan):
        e = load_event(cache, sh, ev)
        rows = trunc_rows(np.stack([e["R"], e["P1"], e["P2"]]), max_len)
        R, P1, P2 = rows[0], rows[1], rows[2]
        sg = sharpness_global(R, P1, P2)
        si, ninf = sharpness_informative(R, P1, P2)
        pdiv = pairwise_divergence(P1, P2)
        corr, _ = which_of_3(net, rows, 0, device)
        per.append({"sharp_global": sg, "sharp_inf": si, "n_informative": ninf,
                    "parent_div": pdiv, "correct": corr})
        if (k + 1) % 500 == 0:
            print(f"    {k+1}/{len(val_plan)} ({time.time()-t0:.0f}s, "
                  f"RSS {_rss()/2**30:.1f}GB)", flush=True)
        _rss_watchdog(label=f"strat {k}")
    print(f"  SANTA eval done in {time.time()-t0:.0f}s", flush=True)

    sg = np.array([p["sharp_global"] for p in per], float)
    si = np.array([p["sharp_inf"] for p in per], float)
    dv = np.array([p["parent_div"] for p in per], float)
    co = np.array([p["correct"] for p in per], float)
    v = ~(np.isnan(sg) | np.isnan(si))
    sg, si, dv, co = sg[v], si[v], dv[v], co[v]

    def q(a, p): return float(np.quantile(a, p))
    santa = {"n": int(len(si)), "overall_acc": float(co.mean()),
             "global": {"min": float(sg.min()), "p25": q(sg, .25),
                        "median": float(np.median(sg)), "max": float(sg.max())},
             "informative": {"min": float(si.min()), "p05": q(si, .05),
                             "p25": q(si, .25), "median": float(np.median(si)),
                             "p75": q(si, .75), "max": float(si.max())},
             "parent_div": {"min": float(dv.min()), "median": float(np.median(dv)),
                            "max": float(dv.max())}}

    curve_global = bin_curve(sg, co, args.n_bins)
    curve_inf = bin_curve(si, co, args.n_bins)
    curve_div = bin_curve(dv, co, args.n_bins)

    # ---- LANL ----
    def lanl_set(d, tag):
        out = []
        for crf, rows, ridx, rec_ids in load_lanl_triplets(Path(d)):
            R = rows[ridx]; others = [rows[j] for j in range(3) if j != ridx]
            g = sharpness_global(R, others[0], others[1])
            i, ninf = sharpness_informative(R, others[0], others[1])
            pv = pairwise_divergence(others[0], others[1])
            c, logits = which_of_3(net, rows, ridx, device)
            smx = softmax_np(logits)
            out.append({"set": tag, "crf": crf, "rec_ids": rec_ids, "recomb_idx": ridx,
                        "sharp_global": g, "sharp_inf": i, "n_informative": ninf,
                        "parent_div": pv, "correct": c,
                        "softmax": [float(x) for x in smx], "true_prob": float(smx[ridx])})
        return out

    lanl = lanl_set(args.lanl_dir, "base")
    if args.lanl_expanded_dir.exists():
        lanl += lanl_set(args.lanl_expanded_dir, "expanded")
    li = np.array([x["sharp_inf"] for x in lanl], float)
    lg = np.array([x["sharp_global"] for x in lanl], float)
    ld = np.array([x["parent_div"] for x in lanl], float)
    lanl_acc = float(np.mean([x["correct"] for x in lanl]))

    # ---- matched comparisons (SANTA restricted to LANL bands) ----
    band_inf = (si >= li.min()) & (si <= li.max())
    band_div = (dv >= ld.min()) & (dv <= ld.max())
    matched = {
        "lanl_acc": lanl_acc, "lanl_n": len(lanl),
        "lanl_inf_sharp": {"min": float(li.min()), "median": float(np.median(li)),
                           "max": float(li.max())},
        "lanl_global_sharp_median": float(np.median(lg)),
        "lanl_parent_div": {"min": float(ld.min()), "max": float(ld.max())},
        "lanl_inf_median_santa_pctile": float((si < np.median(li)).mean()),
        "santa_acc_matched_inf_only": float(co[band_inf].mean()) if band_inf.any() else float("nan"),
        "santa_n_matched_inf_only": int(band_inf.sum()),
        "santa_acc_matched_inf_and_div": float(co[band_inf & band_div].mean()) if (band_inf & band_div).any() else float("nan"),
        "santa_n_matched_inf_and_div": int((band_inf & band_div).sum()),
    }

    # ---- verdict ----
    lo_i, hi_i = curve_inf[0], curve_inf[-1]
    crater_inf = hi_i["acc"] - lo_i["acc"]
    santa_median_inf = santa["informative"]["median"]
    santa_p25_inf = santa["informative"]["p25"]
    lanl_med_inf = float(np.median(li))
    dm_acc = matched["santa_acc_matched_inf_and_div"]
    residual = (dm_acc - lanl_acc) if not np.isnan(dm_acc) else float("nan")

    lanl_soft_inf = lanl_med_inf < santa_p25_inf
    if lanl_soft_inf and crater_inf >= 0.15:
        verdict = ("SOFTNESS GAP (informative-site): LANL recombinants are soft "
                   "at INFORMATIVE sites (median inf-sharpness below SANTA p25) and "
                   "SANTA accuracy craters on the informative-soft bin. Fix = "
                   "soft-mosaic augmentation.")
    elif not np.isnan(residual) and abs(residual) < 0.15:
        verdict = ("MOSTLY EXPLAINED BY OBSERVABLES: after matching SANTA to LANL "
                   "on BOTH informative-sharpness and parent-divergence, SANTA acc "
                   f"({dm_acc:.2f}) ~= LANL acc ({lanl_acc:.2f}) (residual "
                   f"{residual:+.2f}). LANL sits in a HARD region (low divergence + "
                   "moderate informative-softness) that SANTA also fails; there is "
                   "NO large separate construction-specific gap. Both softness AND "
                   "low informative content contribute.")
    else:
        verdict = (f"CONSTRUCTION-SPECIFIC RESIDUAL: even doubly-matched on "
                   f"informative-sharpness AND divergence, SANTA ({dm_acc:.2f}) "
                   f"beats LANL ({lanl_acc:.2f}) by {residual:+.2f} -- a residual "
                   "not explained by these observables.")

    report = {"ckpt": str(args.ckpt), "ckpt_epoch": ck.get("epoch"),
              "max_len": max_len, "n_bins": args.n_bins, "santa": santa,
              "curve_global_sharpness": curve_global,
              "curve_informative_sharpness": curve_inf,
              "curve_parent_div": curve_div,
              "lanl": lanl, "matched": matched,
              "verdict_stats": {"crater_inf_low_to_high": crater_inf,
                                "santa_median_inf": santa_median_inf,
                                "santa_p25_inf": santa_p25_inf,
                                "lanl_median_inf": lanl_med_inf,
                                "lanl_soft_at_informative": bool(lanl_soft_inf),
                                "doubly_matched_residual": residual},
              "verdict": verdict, "per_triplet": per}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # ---- console ----
    print(f"\n=== SANTA which-of-3 n={santa['n']} overall acc {santa['overall_acc']:.3f} ===")
    g = santa["global"]; i = santa["informative"]
    print(f"  GLOBAL sharpness:      min {g['min']:.3f} p25 {g['p25']:.3f} "
          f"median {g['median']:.3f} max {g['max']:.3f}")
    print(f"  INFORMATIVE sharpness: min {i['min']:.3f} p05 {i['p05']:.3f} "
          f"p25 {i['p25']:.3f} median {i['median']:.3f} max {i['max']:.3f}")
    print(f"\n  ACC vs INFORMATIVE-SITE SHARPNESS (deciles):")
    print(f"    {'bin':>3} {'n':>5} {'lo':>7} {'hi':>7} {'mean':>7} {'acc':>6}")
    for b in curve_inf:
        print(f"    {b['bin']:>3} {b['n']:>5} {b['x_lo']:>7.3f} {b['x_hi']:>7.3f} "
              f"{b['x_mean']:>7.3f} {b['acc']:>6.3f}")
    print(f"\n  ACC vs GLOBAL sharpness (deciles):")
    for b in curve_global:
        print(f"    bin{b['bin']:>2} n={b['n']:>4} mean {b['x_mean']:.3f} acc {b['acc']:.3f}")
    print(f"\n=== LANL (n={len(lanl)}: 4 base + expanded), which-of-3 acc {lanl_acc:.3f} ===")
    print(f"  LANL INFORMATIVE sharpness: min {li.min():.3f} median {np.median(li):.3f} "
          f"max {li.max():.3f}  (SANTA inf p25={i['p25']:.3f} median={i['median']:.3f})")
    print(f"  LANL GLOBAL sharpness median {np.median(lg):.3f}  parent_div "
          f"[{ld.min():.3f},{ld.max():.3f}]")
    print(f"  LANL inf-sharpness median sits at SANTA percentile "
          f"{matched['lanl_inf_median_santa_pctile']*100:.1f}%")
    print(f"\n  MATCHED SANTA accuracy vs LANL {lanl_acc:.3f}:")
    print(f"    inf-sharpness band only      : {matched['santa_acc_matched_inf_only']:.3f} "
          f"(n={matched['santa_n_matched_inf_only']})")
    print(f"    inf-sharpness AND div bands  : {matched['santa_acc_matched_inf_and_div']:.3f} "
          f"(n={matched['santa_n_matched_inf_and_div']})  <- doubly-matched")
    print(f"\n  Per-LANL (sorted by informative sharpness):")
    for x in sorted(lanl, key=lambda z: z["sharp_inf"]):
        print(f"    {x['set']:>8} {x['crf']:<10} inf {x['sharp_inf']:.3f} glob "
              f"{x['sharp_global']:.3f} div {x['parent_div']:.3f} true_prob "
              f"{x['true_prob']:.2f} {'OK' if x['correct'] else 'X'}")
    print(f"\n=== VERDICT ===\n  {verdict}")
    print(f"\n  report -> {args.out}")


if __name__ == "__main__":
    main()
