"""Path I — combined-feature probe.

Tests whether the M1.2 pretrained backbone has COMPLEMENTARY value when
paired with raw MaxChi-style features. The M1.3 result showed M1.2-alone
features lose to random-alone, but didn't test the combination.

Three feature configurations on the same train/val event split:

    raw         22 channels (legacy-CNN-style):
                  R one-hot (5)            channels 0-4
                  P1 one-hot (5)           channels 5-9
                  P2 one-hot (5)           channels 10-14
                  match_p1, match_p2, informative (3)  channels 15-17
                  windowed MaxChi @ {50, 100, 200, 500} bp (4)  channels 18-21

    m12         1536 channels (frozen M1.2 hidden states):
                  [h_R, h_R - h_P1, h_R - h_P2]  3 × 2 × d_model

    combined    1558 channels (raw + m12)

If `combined` >> `raw`: M1.2 has complementary conservation/context signal.
                       M3 should use M1.2 backbone + raw features.
If `raw` ≈ `combined`: M1.2 adds nothing; M3 should skip pretraining and
                       use raw MaxChi features (essentially the legacy CNN
                       approach).
If `combined` < `raw`: M1.2 is actively dragging the probe down; same as
                       above, drop the backbone.

Same train/val split as the original M1.3 probe so numbers are directly
comparable.
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_hyenadna import (
    SequenceBackbone,
    BackboneConfig,
    DEFAULT_HF_NAME,
)
from cache_v2_reader import CacheV2
from pretrain_mlm import BidirMLM
from m13_linear_probe import (
    gaussian_soft_label,
    extract_peaks,
    event_f1,
    sample_events,
    backbone_hidden,
    _current_rss_bytes,
    _rss_watchdog,
)


# ---------- raw 22-channel feature extraction ------------------------------

MAXCHI_WINDOWS = (50, 100, 200, 500)


def raw_features(R: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """22 channels matching the legacy CNN's encode_triplet output.

    Args:
        R, P1, P2: int8 (L,) — v2 nucleotide codes {A:0, T:1, G:2, C:3, gap:4}.

    Returns:
        (L, 22) float32 array. Layout per docstring above.
    """
    L = len(R)
    out = np.zeros((L, 22), dtype=np.float32)

    # One-hots: channels 0-4, 5-9, 10-14
    for v, base in ((R, 0), (P1, 5), (P2, 10)):
        for tok in range(5):
            out[v == tok, base + tok] = 1.0

    # Comparison flags: channels 15, 16, 17
    match_p1 = (R == P1).astype(np.float32)
    match_p2 = (R == P2).astype(np.float32)
    informative = (P1 != P2).astype(np.float32)
    out[:, 15] = match_p1
    out[:, 16] = match_p2
    out[:, 17] = informative

    # Windowed MaxChi signals: channels 18-21
    parental = match_p1 - match_p2          # ∈ {-1, 0, 1}
    cumsum = np.concatenate([[0.0], np.cumsum(parental)]).astype(np.float64)
    for ch, w in enumerate(MAXCHI_WINDOWS):
        # right window: mean(parental[i:i+w]), left window: mean(parental[i-w:i])
        i = np.arange(L)
        right_lo = np.minimum(i, L)
        right_hi = np.minimum(i + w, L)
        left_lo = np.maximum(i - w, 0)
        left_hi = np.maximum(i, 0)
        right_sum = cumsum[right_hi] - cumsum[right_lo]
        right_n = (right_hi - right_lo).astype(np.float64)
        right_mean = np.where(right_n > 0, right_sum / np.maximum(right_n, 1.0), 0.0)
        left_sum = cumsum[left_hi] - cumsum[left_lo]
        left_n = (left_hi - left_lo).astype(np.float64)
        left_mean = np.where(left_n > 0, left_sum / np.maximum(left_n, 1.0), 0.0)
        out[:, 18 + ch] = (right_mean - left_mean).astype(np.float32)

    return out


@torch.no_grad()
def m12_triplet_features(backbone: SequenceBackbone, R: np.ndarray,
                          P1: np.ndarray, P2: np.ndarray, device: str
                          ) -> torch.Tensor:
    """Return (L, 6D) features per position: [h_R, h_R - h_P1, h_R - h_P2]."""
    h_R = backbone_hidden(backbone, R, device)
    h_P1 = backbone_hidden(backbone, P1, device)
    h_P2 = backbone_hidden(backbone, P2, device)
    return torch.cat([h_R, h_R - h_P1, h_R - h_P2], dim=-1).cpu()


# ---------- probe training -------------------------------------------------

def train_probe(train_cached: list[dict], val_cached: list[dict],
                feature_dim: int, *, epochs: int, lr: float, pos_weight: float,
                device: str, threshold_sweep: list[float],
                rng: random.Random) -> dict:
    head = nn.Linear(feature_dim, 1).to(device)
    optim = torch.optim.Adam(head.parameters(), lr=lr)
    pw = torch.tensor([pos_weight], device=device)

    best_f1 = -1.0
    history: list[dict] = []
    for ep in range(1, epochs + 1):
        head.train()
        order = list(range(len(train_cached)))
        rng.shuffle(order)
        epoch_loss, epoch_n = 0.0, 0
        for i in order:
            ev = train_cached[i]
            x = ev["feats"].to(device)
            y = ev["y"].to(device)
            logits = head(x).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            optim.zero_grad(); loss.backward(); optim.step()
            epoch_loss += float(loss.item()) * x.shape[0]
            epoch_n += x.shape[0]
        train_loss = epoch_loss / max(1, epoch_n)

        head.eval()
        per_thr = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in threshold_sweep}
        with torch.no_grad():
            for ev in val_cached:
                x = ev["feats"].to(device)
                p = torch.sigmoid(head(x).squeeze(-1)).cpu().numpy()
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
        print(f"  epoch {ep:>2}  train_loss {train_loss:.4f}  "
              f"best F1 {best['f1']:.3f} @ thr={best['thr']:.2f}  "
              f"P {best['precision']:.3f}  R {best['recall']:.3f}", flush=True)
        history.append({
            "epoch": ep, "train_loss": train_loss,
            "best_f1": best["f1"], "best_thr": best["thr"],
            "precision": best["precision"], "recall": best["recall"],
        })
        best_f1 = max(best_f1, best["f1"])
    return {"best_f1": best_f1, "history": history}


def cache_for_mode(mode: str, events: list[dict], backbone: SequenceBackbone | None,
                   device: str, label: str = "") -> list[dict]:
    """Pre-compute features once per event according to mode."""
    out: list[dict] = []
    t0 = time.time()
    for i, ev in enumerate(events):
        raw = torch.from_numpy(raw_features(ev["R"], ev["P1"], ev["P2"]))  # (L, 22)
        if mode == "raw":
            feats = raw
        elif mode == "m12":
            feats = m12_triplet_features(backbone, ev["R"], ev["P1"], ev["P2"], device)
        elif mode == "combined":
            m12 = m12_triplet_features(backbone, ev["R"], ev["P1"], ev["P2"], device)
            feats = torch.cat([raw, m12], dim=-1)
        else:
            raise ValueError(mode)
        y = torch.from_numpy(gaussian_soft_label(ev["L"], ev["bps"]))
        out.append({**ev, "feats": feats, "y": y})
        _rss_watchdog(label=f"feature cache {label} #{i}")
        if (i + 1) % 10 == 0 or i + 1 == len(events):
            rate = (i + 1) / max(time.time() - t0, 1e-3)
            print(f"    {label} cached {i+1}/{len(events)}  ({rate:.2f} ev/s)  "
                  f"RSS {_current_rss_bytes()/2**30:.1f} GB  feat_dim={feats.shape[-1]}",
                  flush=True)
    return out


# ---------- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("models_test/snapshots/backbone_mlm_v1_e13_30k_gs292630.pt"))
    ap.add_argument("--backbone-mode", choices=["m12", "hf", "random"], default="m12",
                    help="backbone weights for m12/combined feature modes")
    ap.add_argument("--train-shards", nargs="+", default=["XML-2", "XML-6"])
    ap.add_argument("--val-shards", nargs="+", default=["XML-5"])
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=11_500)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-modes", nargs="+",
                    default=["raw", "m12", "combined"],
                    help="any subset of {raw, m12, combined}")
    ap.add_argument("--out", type=Path, default=Path("models_test/m13_combined_probe.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] device={device}", flush=True)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    cache = CacheV2()

    # Load backbone once (used for m12 and combined modes)
    backbone = None
    if any(m in args.feature_modes for m in ("m12", "combined")):
        print(f"[{time.strftime('%H:%M:%S')}] building backbone "
              f"(mode={args.backbone_mode})", flush=True)
        pretrained_hf = args.backbone_mode in ("m12", "hf")
        bb = SequenceBackbone(BackboneConfig(hf_name=DEFAULT_HF_NAME),
                              pretrained=pretrained_hf)
        d_model = bb.cfg.d_model
        bidir = BidirMLM(bb, d_model=d_model, n_classes=5).to(device)
        if args.backbone_mode == "m12":
            ck = torch.load(args.ckpt, map_location=device, weights_only=False)
            bidir.load_state_dict(ck["model_state"])
            print(f"  loaded M1.2 ckpt: epoch={ck['epoch']} gs={ck['global_step']:,}",
                  flush=True)
        else:
            print(f"  no M1.2 ckpt loaded (backbone_mode={args.backbone_mode})",
                  flush=True)
        for p in bidir.parameters():
            p.requires_grad = False
        bidir.eval()
        backbone = bidir.backbone

    # Sample events ONCE — every mode uses the same split (apples-to-apples)
    train_events = sample_events(cache, args.train_shards, args.n_train, rng,
                                  max_len=args.max_len)
    val_events = sample_events(cache, args.val_shards, args.n_val, rng,
                                max_len=args.max_len)
    print(f"  TRAIN: {len(train_events)} events; VAL: {len(val_events)} events",
          flush=True)

    threshold_sweep = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    results: dict = {}
    feature_dims = {"raw": 22, "m12": 3 * 256 * 2, "combined": 22 + 3 * 256 * 2}

    for mode in args.feature_modes:
        print(f"\n[{time.strftime('%H:%M:%S')}] === mode={mode} "
              f"(feature_dim={feature_dims[mode]}) ===", flush=True)
        train_cached = cache_for_mode(mode, train_events, backbone, device, label="train")
        val_cached = cache_for_mode(mode, val_events, backbone, device, label="val")
        # Fresh probe rng each mode for fairness
        mode_rng = random.Random(args.seed + hash(mode) % 100)
        torch.manual_seed(args.seed + 1)
        r = train_probe(train_cached, val_cached, feature_dims[mode],
                         epochs=args.epochs, lr=args.lr,
                         pos_weight=args.pos_weight, device=device,
                         threshold_sweep=threshold_sweep, rng=mode_rng)
        results[mode] = {"best_f1": r["best_f1"], "history": r["history"]}
        print(f"  mode={mode}  best F1 = {r['best_f1']:.3f}", flush=True)
        # Free the cached features before next mode
        del train_cached, val_cached

    print(f"\n=== Path I summary ===")
    for mode in args.feature_modes:
        print(f"  {mode:>10}: best F1 = {results[mode]['best_f1']:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "modes": args.feature_modes,
        "n_train": len(train_events),
        "n_val": len(val_events),
        "results": results,
    }, indent=2))
    print(f"  report → {args.out}")


if __name__ == "__main__":
    main()
