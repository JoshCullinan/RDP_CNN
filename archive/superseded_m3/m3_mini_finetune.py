"""M3-mini — end-to-end fine-tune comparison: M1.2-start vs random-start.

M1.3's linear probe showed random-init Hyena beats our M1.2 pretrained backbone
for breakpoint detection (F1 0.40 vs 0.22). But a linear probe freezes the
backbone. The pretrained backbone might still win when allowed to adapt via
gradient flow.

This script:
  - Builds BidirMLM backbone in either 'm12' or 'random' mode
  - UNFREEZES the backbone (this is the key difference from m13_linear_probe)
  - Adds the same triplet-feature linear head as M1.3
  - Fine-tunes end-to-end with weighted BCE on Gaussian-soft breakpoint targets
  - Evaluates F1 each epoch via find_peaks

If M1.2-start converges higher → pretraining helps when allowed to flow
gradients; M3 proper should use M1.2 ckpt.

If random-start converges higher → pretraining actively hurts even
end-to-end; M3 proper should skip MLM and train from scratch.

Memory: backbone in train mode + gradient checkpointing required to fit
on 8 GB. B=1 events processed one at a time (the events vary in length
anyway). Optimizer state for the full ~3.3M params adds modest cost.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_hyenadna import (
    SequenceBackbone,
    BackboneConfig,
    DEFAULT_HF_NAME,
    v2_to_hyena_ids,
)
from cache_v2_reader import CacheV2
from pretrain_mlm import BidirMLM, _V2_RC
from m13_linear_probe import (
    gaussian_soft_label,
    extract_peaks,
    event_f1,
    sample_events,
    _current_rss_bytes,
    _rss_watchdog,
)


def hidden_with_grad(backbone: SequenceBackbone, v2_seq: np.ndarray,
                     device: str) -> torch.Tensor:
    """Forward + RC; gradients flow back to backbone."""
    v2 = torch.from_numpy(v2_seq[None, :].astype(np.int64))
    fwd_hy = v2_to_hyena_ids(v2).to(device)
    v2_rc = _V2_RC[v2.clamp(0, 4)].flip(dims=[1])
    rc_hy = v2_to_hyena_ids(v2_rc).to(device)
    h_fwd = backbone(fwd_hy, is_hyena_ids=True)
    h_rc = backbone(rc_hy, is_hyena_ids=True).flip(dims=[1])
    return torch.cat([h_fwd, h_rc], dim=-1)[0]                    # (L, 2D)


def triplet_features_with_grad(backbone: SequenceBackbone, R, P1, P2,
                                device: str) -> torch.Tensor:
    h_R = hidden_with_grad(backbone, R, device)
    h_P1 = hidden_with_grad(backbone, P1, device)
    h_P2 = hidden_with_grad(backbone, P2, device)
    return torch.cat([h_R, h_R - h_P1, h_R - h_P2], dim=-1)       # (L, 6D)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] device={device}  mode={args.backbone_mode}",
          flush=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    cache = CacheV2()

    # Backbone
    pretrained_hf = args.backbone_mode in ("m12", "hf")
    backbone = SequenceBackbone(BackboneConfig(hf_name=DEFAULT_HF_NAME),
                                 pretrained=pretrained_hf)
    d_model = backbone.cfg.d_model
    bidir = BidirMLM(backbone, d_model=d_model, n_classes=5).to(device)
    if args.backbone_mode == "m12":
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        bidir.load_state_dict(ck["model_state"])
        print(f"  loaded M1.2 ckpt: epoch={ck['epoch']} gs={ck['global_step']:,}",
              flush=True)
    else:
        print(f"  no M1.2 ckpt loaded (mode={args.backbone_mode})", flush=True)

    # KEY DIFFERENCE FROM M1.3: backbone is unfrozen.
    for p in bidir.backbone.parameters():
        p.requires_grad = True
    # Gradient checkpointing to fit training in 8 GB at L=10k.
    bidir.backbone.hyena.gradient_checkpointing_enable()
    print(f"  backbone UNFROZEN, gradient_checkpointing ON", flush=True)

    feature_dim = 3 * (2 * d_model)
    head = nn.Linear(feature_dim, 1).to(device)
    pw = torch.tensor([args.pos_weight], device=device)

    # Only the backbone needs amp; head is tiny
    optim = torch.optim.AdamW(
        list(bidir.backbone.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Sample events
    train_events = sample_events(cache, args.train_shards, args.n_train, rng,
                                  max_len=args.max_len)
    val_events = sample_events(cache, args.val_shards, args.n_val, rng,
                                max_len=args.max_len)
    print(f"  TRAIN: {len(train_events)} events; VAL: {len(val_events)} events",
          flush=True)

    threshold_sweep = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    history: list[dict] = []
    best_f1 = -1.0

    for ep in range(1, args.epochs + 1):
        bidir.train()
        head.train()
        order = list(range(len(train_events)))
        rng.shuffle(order)
        epoch_loss = 0.0
        epoch_n = 0
        t_ep = time.time()
        for step_in_epoch, i in enumerate(order):
            ev = train_events[i]
            y = torch.from_numpy(gaussian_soft_label(ev["L"], ev["bps"])).to(device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16,
                                     enabled=(device == "cuda")):
                feats = triplet_features_with_grad(
                    bidir.backbone, ev["R"], ev["P1"], ev["P2"], device)
                logits = head(feats).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    list(bidir.backbone.parameters()) + list(head.parameters()),
                    args.grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            epoch_loss += float(loss.item()) * ev["L"]
            epoch_n += ev["L"]
            _rss_watchdog(label=f"train ep{ep} step{step_in_epoch}")

        train_loss = epoch_loss / max(1, epoch_n)

        # eval
        bidir.eval()
        head.eval()
        per_thr = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in threshold_sweep}
        with torch.no_grad():
            for ev in val_events:
                with torch.amp.autocast("cuda", dtype=torch.float16,
                                         enabled=(device == "cuda")):
                    feats = triplet_features_with_grad(
                        bidir.backbone, ev["R"], ev["P1"], ev["P2"], device)
                    logits = head(feats).squeeze(-1)
                p = torch.sigmoid(logits.float()).cpu().numpy()
                for thr in threshold_sweep:
                    peaks = extract_peaks(p, thr)
                    tp, fp, fn = event_f1(ev["bps"], peaks)
                    per_thr[thr]["tp"] += tp
                    per_thr[thr]["fp"] += fp
                    per_thr[thr]["fn"] += fn
        thr_results = []
        for thr, c in per_thr.items():
            prec = c["tp"] / max(1, c["tp"] + c["fp"])
            rec = c["tp"] / max(1, c["tp"] + c["fn"])
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            thr_results.append({"thr": thr, "precision": prec, "recall": rec, "f1": f1, **c})
        thr_results.sort(key=lambda d: -d["f1"])
        best = thr_results[0]
        elapsed = time.time() - t_ep
        gpu = (torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0)
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
        print(f"  epoch {ep:>2}  train_loss {train_loss:.4f}  "
              f"best F1 {best['f1']:.3f} @ thr={best['thr']:.2f}  "
              f"P {best['precision']:.3f}  R {best['recall']:.3f}  "
              f"({elapsed:.0f}s, RSS {_current_rss_bytes()/2**30:.1f} GB, "
              f"GPU peak {gpu:.2f} GB)", flush=True)
        history.append({
            "epoch": ep, "train_loss": train_loss,
            "best_f1": best["f1"], "best_thr": best["thr"],
            "precision": best["precision"], "recall": best["recall"],
        })
        best_f1 = max(best_f1, best["f1"])

    print(f"\n=== M3-mini fine-tune result (mode={args.backbone_mode}) ===")
    print(f"  best F1 = {best_f1:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "backbone_mode": args.backbone_mode,
        "best_f1": best_f1,
        "history": history,
    }, indent=2))
    print(f"  report → {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("models_test/snapshots/backbone_mlm_v1_e13_30k_gs292630.pt"))
    ap.add_argument("--backbone-mode", choices=["m12", "hf", "random"], default="m12")
    ap.add_argument("--train-shards", nargs="+", default=["XML-2", "XML-6"])
    ap.add_argument("--val-shards", nargs="+", default=["XML-5"])
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=11_500)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)         # lower than probe — fine-tuning
    ap.add_argument("--pos-weight", type=float, default=200.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("models_test/m3_mini.json"))
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
