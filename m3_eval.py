"""Evaluate an M3-dilated ckpt against canonical SANTA + real-HIV baselines.

Loads `models_test/m3d_snapshots/m3d_best.pt` (or any compatible ckpt) and
runs find_peaks F1 evaluation over:

  - SANTA `UnseenTestSet` (5539 events, ~30 kb sequences, truncated to
    max_len for the model). Direct comparison with legacy CNN
    runB2_sig10 F1 0.421 on this test set.

  - LANL real-HIV CRFs (`data/lanl_crf/triplets/`) if requested. Direct
    comparison with legacy CNN F1 0.533 (aggregated across 4 CRFs).

Reports threshold sweep + best F1 + precision/recall.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_v2_reader import CacheV2
from m3_dilated import (
    DilatedHead,
    raw_features,
    triplet_features,
    extract_peaks,
    event_f1,
    load_event,
)
from backbone_hyenadna import SequenceBackbone, BackboneConfig, DEFAULT_HF_NAME
from pretrain_mlm import BidirMLM


THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
              0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def features_for_event(feature_mode: str, ev: dict, backbone, device: str
                       ) -> torch.Tensor:
    if feature_mode == "raw":
        return raw_features(ev["R"], ev["P1"], ev["P2"]).to(device)
    if feature_mode == "hyena":
        return triplet_features(backbone, ev["R"], ev["P1"], ev["P2"], device)
    if feature_mode == "combined":
        rf = raw_features(ev["R"], ev["P1"], ev["P2"]).to(device)
        hf = triplet_features(backbone, ev["R"], ev["P1"], ev["P2"], device)
        return torch.cat([rf, hf], dim=-1)
    raise ValueError(feature_mode)


def eval_shard(head, backbone, cache: CacheV2, shard_name: str,
               feature_mode: str, max_len: int, device: str,
               amp_dtype: torch.dtype, max_events: int | None = None
               ) -> dict:
    head.eval()
    shard = cache.shards[shard_name]
    n_total = len(shard.events)
    indices = list(range(n_total))
    if max_events is not None and max_events < n_total:
        indices = indices[:max_events]
    per_thr = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in THRESHOLDS}
    n_events = 0
    t0 = time.time()
    for ev_idx in indices:
        ev = load_event(cache, shard_name, ev_idx)
        if ev["L"] > max_len:
            ev["R"] = ev["R"][:max_len]; ev["P1"] = ev["P1"][:max_len]
            ev["P2"] = ev["P2"][:max_len]; ev["L"] = max_len
            ev["bps"] = [b for b in ev["bps"] if b < max_len]
        if not ev["bps"]:
            continue
        feats = features_for_event(feature_mode, ev, backbone, device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                logits = head(feats[None]).squeeze(0)
        p = torch.sigmoid(logits.float()).cpu().numpy()
        for thr in THRESHOLDS:
            peaks = extract_peaks(p, thr)
            tp, fp, fn = event_f1(ev["bps"], peaks)
            per_thr[thr]["tp"] += tp
            per_thr[thr]["fp"] += fp
            per_thr[thr]["fn"] += fn
        n_events += 1
        if n_events % 500 == 0:
            rate = n_events / max(time.time() - t0, 1e-3)
            print(f"    {shard_name}  {n_events}/{len(indices)}  ({rate:.1f} ev/s)",
                  flush=True)
    out = []
    for thr, c in per_thr.items():
        prec = c["tp"] / max(1, c["tp"] + c["fp"])
        rec = c["tp"] / max(1, c["tp"] + c["fn"])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        out.append({"thr": thr, "precision": prec, "recall": rec, "f1": f1, **c})
    out.sort(key=lambda d: -d["f1"])
    return {"shard": shard_name, "n_events": n_events,
            "elapsed_s": time.time() - t0,
            "best": out[0], "all_thresholds": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("models_test/m3d_snapshots/m3d_best.pt"))
    ap.add_argument("--feature-mode", choices=["raw", "hyena", "combined"],
                    default="raw")
    ap.add_argument("--backbone-mode", choices=["random", "m12", "hf"], default="random",
                    help="only used for hyena/combined feature modes")
    ap.add_argument("--m12-ckpt", type=Path,
                    default=Path("models_test/snapshots/backbone_mlm_v1_e13_30k_gs292630.pt"))
    ap.add_argument("--shards", nargs="+", default=["UnseenTestSet"])
    ap.add_argument("--max-len", type=int, default=11_000)
    ap.add_argument("--max-events", type=int, default=None)
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--head-blocks", type=int, default=6)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--out", type=Path, default=Path("models_test/m3_eval.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # Build backbone if needed (for hyena/combined)
    backbone = None
    d_model = 256
    if args.feature_mode in ("hyena", "combined"):
        pretrained_hf = args.backbone_mode in ("m12", "hf")
        bb = SequenceBackbone(BackboneConfig(hf_name=DEFAULT_HF_NAME), pretrained=pretrained_hf)
        d_model = bb.cfg.d_model
        bidir = BidirMLM(bb, d_model=d_model, n_classes=5).to(device)
        if args.backbone_mode == "m12":
            ck = torch.load(args.m12_ckpt, map_location=device, weights_only=False)
            bidir.load_state_dict(ck["model_state"])
        for p in bidir.parameters():
            p.requires_grad = False
        bidir.eval()
        backbone = bidir.backbone

    # Compute feature dim
    if args.feature_mode == "raw":
        feat_dim = 22
    elif args.feature_mode == "hyena":
        feat_dim = 3 * (2 * d_model)
    else:
        feat_dim = 22 + 3 * (2 * d_model)

    head = DilatedHead(in_channels=feat_dim, hidden=args.head_hidden,
                       n_blocks=args.head_blocks, dropout=args.head_dropout).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    head.load_state_dict(ck["head_state"])
    print(f"[{time.strftime('%H:%M:%S')}] device={device}  feature_mode={args.feature_mode}",
          flush=True)
    print(f"  loaded ckpt: epoch={ck.get('epoch')}  gs={ck.get('global_step'):,}  "
          f"best_val_f1_during_train={ck.get('best_val_f1', 'N/A')}", flush=True)
    print(f"  head params: {sum(p.numel() for p in head.parameters()):,}  "
          f"feat_dim={feat_dim}", flush=True)

    cache = CacheV2()
    results = {}
    for sh in args.shards:
        if sh not in cache.shards:
            print(f"  SKIP {sh}: not in cache", flush=True)
            continue
        print(f"\n=== Evaluating on {sh} ===", flush=True)
        r = eval_shard(head, backbone, cache, sh, args.feature_mode,
                       args.max_len, device, amp_dtype, args.max_events)
        results[sh] = r
        print(f"  {sh}: n={r['n_events']:,}  best F1 {r['best']['f1']:.3f} "
              f"@ thr={r['best']['thr']:.2f}  P {r['best']['precision']:.3f}  "
              f"R {r['best']['recall']:.3f}  ({r['elapsed_s']:.0f}s)", flush=True)

    print(f"\n=== Summary ===")
    for sh, r in results.items():
        print(f"  {sh:>20}  F1 {r['best']['f1']:.3f}  P {r['best']['precision']:.3f}  "
              f"R {r['best']['recall']:.3f}  @ thr={r['best']['thr']:.2f}  "
              f"(n={r['n_events']})")
    print("\nKey baselines:")
    print("  Legacy CNN runB2_sig10 on SANTA UnseenTestSet  : F1 0.421 @ EB=25")
    print("  Legacy CNN runB2_sig10 on LANL CRFs           : F1 0.533 (aggregated)")
    print("  Classical RDP on LANL CRFs                    : F1 0.519")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        sh: {"best_f1": r["best"]["f1"],
              "best_thr": r["best"]["thr"],
              "precision": r["best"]["precision"],
              "recall": r["best"]["recall"],
              "n_events": r["n_events"],
              "all_thresholds": r["all_thresholds"]}
        for sh, r in results.items()
    }, indent=2))
    print(f"  report → {args.out}")


if __name__ == "__main__":
    main()
