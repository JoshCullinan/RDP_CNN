"""M3 v4 auxiliary-head evaluation — the criterion-4 harness.

The aux head outputs P(this triplet is a recombinant). It GATES the BP head at
deployment. This script measures whether the gate generalizes — specifically
whether it keys on recombination *structure* rather than the SANTA-vs-real
provenance confound (see project_m3_v3_failed + the v4 handover).

It computes aux_prob over five groups:

  REAL positives    : LANL HIV CRF triplets        (true recombinants → want HIGH)
  REAL negatives    : Ebola + Zika panel triplets  (cross-species   → want LOW)
  SANTA positives   : held-out XML-5 recomb events (in-distribution pos)
  SANTA negatives   : XML-5 non-recombinant triplets (in-distribution neg)

and reports several AUROCs:

  LANL-pos vs Ebola-neg          ← PRIMARY (criterion #4): all-real, the
                                    deployment-relevant pair. LANL is NEVER
                                    trained on, so a high value means unseen real
                                    recombinants score above cross-species.
  LANL-pos vs (Ebola+Zika)-neg   ← broader real test
  SANTA-pos vs SANTA-neg         ← in-distribution sanity (should be ~1.0)
  SANTA-pos vs (Ebola+Zika)-neg  ← CONFOUND DETECTOR: if this is high while
                                    LANL-vs-Ebola is low, the head learned
                                    "SANTA vs real" not "recombinant vs not".

Finally it suggests a gate threshold (keep all LANL positives; Youden-optimal on
LANL-vs-Ebola) and reports the confusion at that threshold.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3_dilated import (
    raw_features, event_plan, load_event,
    build_santa_neg_pool, sample_santa_negative,
)
from cache_v2_reader import CacheV2
from m3_eval_lanl import load_trained_head, head_forward


NT_TO_INT = {'A': 0, 'T': 1, 'G': 2, 'C': 3, '-': 4}


def seq_to_int8(s: str) -> np.ndarray:
    arr = np.empty(len(s), dtype=np.int8)
    for i, ch in enumerate(s.upper()):
        arr[i] = NT_TO_INT.get(ch, 4)
    return arr


def auroc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUROC: P(pos_score > neg_score), ties = 0.5."""
    p = np.asarray(pos, dtype=np.float64)
    n = np.asarray(neg, dtype=np.float64)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    cmp = p[:, None] - n[None, :]
    wins = float((cmp > 0).sum()) + 0.5 * float((cmp == 0).sum())
    return wins / (len(p) * len(n))


def aux_prob_for_triplet(head, head_mode, R, P1, P2, device) -> float:
    feats = raw_features(R, P1, P2).to(device)
    _, aux = head_forward(head, head_mode, feats, device)
    return aux


# ---------- group loaders ----------

def lanl_positives(head, head_mode, device, triplet_dir: Path) -> list[float]:
    out = []
    for fa in sorted(triplet_dir.glob("*.fa")):
        seqs = list(SeqIO.parse(fa, "fasta"))
        if len(seqs) != 3:
            continue
        R, P1, P2 = (seq_to_int8(str(s.seq)) for s in seqs)
        out.append((fa.stem, aux_prob_for_triplet(head, head_mode, R, P1, P2, device)))
    return out


def panel_negatives(head, head_mode, device, panel: str,
                    exclude: set[str]) -> list[float]:
    fa = Path(f"/home/joshc/Dev/RDP_CNN/data/real_recombinants/{panel}/aligned.fa")
    seqs = {r.id: seq_to_int8(str(r.seq)) for r in SeqIO.parse(fa, "fasta")
            if r.id not in exclude}
    ids = list(seqs)
    out = []
    for trio in combinations(ids, 3):
        R, P1, P2 = (seqs[i] for i in trio)
        out.append(aux_prob_for_triplet(head, head_mode, R, P1, P2, device))
    return out


def santa_group(head, head_mode, device, cache, shards, n, max_len,
                rng, positive: bool) -> list[float]:
    out = []
    if positive:
        plan = event_plan(cache, shards, n, rng, max_len)
        for sh, ev_idx in plan:
            ev = load_event(cache, sh, ev_idx)
            R, P1, P2 = ev["R"][:max_len], ev["P1"][:max_len], ev["P2"][:max_len]
            out.append(aux_prob_for_triplet(head, head_mode, R, P1, P2, device))
    else:
        pool = build_santa_neg_pool(cache, shards)
        for _ in range(n):
            neg = sample_santa_negative(cache, pool, rng, max_len)
            if neg is None:
                continue
            out.append(aux_prob_for_triplet(
                head, head_mode, neg["R"], neg["P1"], neg["P2"], device))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("models_test/m3d_v4_snaps/m3d_best.pt"))
    ap.add_argument("--triplet-dir", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/triplets"))
    ap.add_argument("--val-shards", nargs="+", default=["XML-5"])
    ap.add_argument("--n-santa", type=int, default=150)
    ap.add_argument("--max-len", type=int, default=30500)
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--head-blocks", type=int, default=6)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--head-mode", choices=["auto", "single", "multi"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("models_test/m3_eval_aux.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    head, head_mode, ck = load_trained_head(
        args.ckpt, device, args.head_hidden, args.head_blocks, args.head_dropout,
        args.head_mode)
    print(f"[{time.strftime('%H:%M:%S')}] device={device}  ckpt={args.ckpt}  "
          f"head_mode={head_mode}  epoch={ck.get('epoch')}", flush=True)
    if head_mode != "multi":
        raise SystemExit("aux eval requires a multi-head (v4) checkpoint")

    cache = CacheV2()

    print("  scoring LANL positives (real HIV recombinants)...", flush=True)
    lanl = lanl_positives(head, head_mode, device, args.triplet_dir)
    lanl_probs = [p for _, p in lanl]
    print("  scoring Ebola negatives (cross-species)...", flush=True)
    ebola = panel_negatives(head, head_mode, device, "ebola", set())
    print("  scoring Zika negatives...", flush=True)
    zika = panel_negatives(head, head_mode, device, "zika", set())
    print(f"  scoring {args.n_santa} SANTA positives (XML-5 events)...", flush=True)
    santa_pos = santa_group(head, head_mode, device, cache, args.val_shards,
                            args.n_santa, args.max_len, rng, positive=True)
    print(f"  scoring {args.n_santa} SANTA negatives (XML-5 non-recomb)...", flush=True)
    santa_neg = santa_group(head, head_mode, device, cache, args.val_shards,
                            args.n_santa, args.max_len, rng, positive=False)

    def summ(name, xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            print(f"    {name:>26}: (empty)")
            return
        print(f"    {name:>26}: n={len(xs):>3}  "
              f"mean={np.mean(xs):.3f}  min={min(xs):.3f}  "
              f"med={np.median(xs):.3f}  max={max(xs):.3f}")

    print(f"\n=== aux_prob distributions ===")
    summ("LANL pos (real HIV)", lanl_probs)
    summ("Ebola neg (cross-species)", ebola)
    summ("Zika neg", zika)
    summ("SANTA pos (XML-5)", santa_pos)
    summ("SANTA neg (XML-5)", santa_neg)
    print(f"  per-CRF LANL aux_prob: " +
          ", ".join(f"{c}={p:.3f}" for c, p in lanl))

    real_neg = ebola + zika
    aurocs = {
        "LANL_pos_vs_Ebola_neg": auroc(lanl_probs, ebola),
        "LANL_pos_vs_EbolaZika_neg": auroc(lanl_probs, real_neg),
        "SANTA_pos_vs_SANTA_neg": auroc(santa_pos, santa_neg),
        "SANTA_pos_vs_EbolaZika_neg": auroc(santa_pos, real_neg),
    }
    print(f"\n=== AUROC ===")
    print(f"  PRIMARY (criterion #4)  LANL-pos vs Ebola-neg     : "
          f"{aurocs['LANL_pos_vs_Ebola_neg']:.3f}   (target >= 0.85)")
    print(f"          LANL-pos vs (Ebola+Zika)-neg              : "
          f"{aurocs['LANL_pos_vs_EbolaZika_neg']:.3f}")
    print(f"  in-dist SANTA-pos vs SANTA-neg                    : "
          f"{aurocs['SANTA_pos_vs_SANTA_neg']:.3f}")
    print(f"  CONFOUND SANTA-pos vs (Ebola+Zika)-neg            : "
          f"{aurocs['SANTA_pos_vs_EbolaZika_neg']:.3f}")
    print(f"  (confound check: if SANTA-vs-real AUROC is high but "
          f"LANL-vs-Ebola is LOW, the aux head learned provenance, not structure)")

    # Suggested gate threshold: Youden-optimal on LANL-pos vs Ebola-neg, plus a
    # conservative "keep all LANL" threshold.
    cand = sorted(set(lanl_probs + ebola + zika))
    best_j, best_thr = -1.0, 0.5
    for t in cand:
        tpr = np.mean([p >= t for p in lanl_probs]) if lanl_probs else 0.0
        fpr = np.mean([p >= t for p in real_neg]) if real_neg else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr = j, float(t)
    keep_all_thr = min(lanl_probs) if lanl_probs else 0.5
    print(f"\n=== Suggested gate threshold ===")
    print(f"  Youden-optimal (LANL vs Ebola+Zika): thr={best_thr:.3f}  J={best_j:.3f}")
    print(f"  'keep all LANL' thr (= min LANL aux_prob): {keep_all_thr:.3f}")
    for label, thr in [("youden", best_thr), ("keep_all_LANL", keep_all_thr)]:
        lanl_keep = np.mean([p >= thr for p in lanl_probs]) if lanl_probs else float("nan")
        eb_supp = np.mean([p < thr for p in ebola]) if ebola else float("nan")
        zk_supp = np.mean([p < thr for p in zika]) if zika else float("nan")
        print(f"  @thr={thr:.3f} [{label}]: LANL kept {lanl_keep:.2f}  "
              f"Ebola suppressed {eb_supp:.2f}  Zika suppressed {zk_supp:.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "ckpt": str(args.ckpt), "epoch": ck.get("epoch"),
        "aurocs": aurocs,
        "gate_youden": best_thr, "gate_keep_all_lanl": keep_all_thr,
        "lanl_per_crf": dict(lanl),
        "dist": {
            "lanl_pos": lanl_probs, "ebola_neg": ebola, "zika_neg": zika,
            "santa_pos": santa_pos, "santa_neg": santa_neg,
        },
    }, indent=2))
    print(f"\n  report → {args.out}")


if __name__ == "__main__":
    main()
