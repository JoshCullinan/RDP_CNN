"""M1.3 — Linear probe on the pretrained HyenaDNA backbone.

Tests whether the M1.2 pretrained backbone has learned breakpoint-relevant
features by training only a tiny linear classifier on top of frozen
per-position hidden states.

Per-position features (1536-dim) for each event triplet (R, P1, P2):
    h_R                      512  — bidirectional context at position i
    h_R - h_P1               512  — divergence of R from parent 1
    h_R - h_P2               512  — divergence of R from parent 2

The two "difference" channels are the key — they encode where R looks
different from each parent. Backbone has never seen triplets during M1.2
pretraining (single-sequence MLM only), so the linear probe is asking:
"do the MLM-learned representations contain enough signal to localise a
parental switch when we hand it the comparison via subtraction?"

Outcome semantics:
    F1 ≳ 0.10  -- backbone features are useful; M3 (full fine-tune) justified
    F1 ≈ 0.00  -- backbone features lack breakpoint signal; M3 risky

Comparison baselines (downstream-task F1):
    Random baseline                 : ~0.000
    Legacy CNN runB2 σ=20           :  0.448 on SANTA UnseenTestSet @ EB=25
    Legacy CNN runB2_sig10          :  0.421 on SANTA UnseenTestSet @ EB=25
    Best classical (RDP)            :  0.519 on LANL real-HIV CRFs
    Legacy CNN deployment baseline  :  0.533 on LANL real-HIV CRFs

The probe is held-out at the EVENT level (TRAIN events from XML-2/-4/-6
TRAIN shards, eval events from XML-5/long_content_30k_001 VAL shards).
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
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_hyenadna import (
    SequenceBackbone,
    BackboneConfig,
    DEFAULT_HF_NAME,
    v2_to_hyena_ids,
)
from cache_v2_reader import CacheV2
from pretrain_mlm import BidirMLM, _V2_RC


# ---------- RSS watchdog (same as pretrain) ---------------------------------

_RSS_CEILING_BYTES = 26 * 1024 * 1024 * 1024


def _current_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _rss_watchdog(label: str = "") -> None:
    rss = _current_rss_bytes()
    if rss > _RSS_CEILING_BYTES:
        raise MemoryError(
            f"RSS watchdog tripped {label}: {rss / 2**30:.1f} GB > "
            f"{_RSS_CEILING_BYTES / 2**30:.0f} GB cap"
        )


# ---------- target + eval ---------------------------------------------------

TOLERANCE = 200
DEFAULT_SIGMA = 10
EDGE_BUFFER = 25       # suppress peaks within first/last 25 bp (per legacy runB2)


def gaussian_soft_label(L: int, bps: list[int], sigma: int = DEFAULT_SIGMA) -> np.ndarray:
    """Per-position soft Gaussian label at each true breakpoint."""
    y = np.zeros(L, dtype=np.float32)
    for bp in bps:
        if not (0 <= bp < L):
            continue
        lo = max(0, bp - 3 * sigma)
        hi = min(L, bp + 3 * sigma + 1)
        idx = np.arange(lo, hi)
        y[idx] = np.maximum(y[idx], np.exp(-0.5 * ((idx - bp) / sigma) ** 2))
    return y


def extract_peaks(p: np.ndarray, threshold: float, edge_buffer: int = EDGE_BUFFER
                  ) -> np.ndarray:
    """find_peaks → suppress edges → return predicted breakpoint indices."""
    L = len(p)
    p = p.copy()
    p[:edge_buffer] = 0.0
    p[L - edge_buffer:] = 0.0
    peaks, _ = find_peaks(p, height=threshold, distance=TOLERANCE)
    return peaks


def event_f1(true_bps: list[int], pred_peaks: np.ndarray,
             tolerance: int = TOLERANCE) -> tuple[int, int, int]:
    """Greedy nearest-first matching. Returns (tp, fp, fn)."""
    matched_true = set()
    matched_pred = set()
    pairs: list[tuple[int, int, int]] = []  # (dist, true_idx, pred_idx)
    for ti, tbp in enumerate(true_bps):
        for pi, ppk in enumerate(pred_peaks):
            d = abs(int(ppk) - int(tbp))
            if d <= tolerance:
                pairs.append((d, ti, pi))
    pairs.sort()
    for d, ti, pi in pairs:
        if ti in matched_true or pi in matched_pred:
            continue
        matched_true.add(ti)
        matched_pred.add(pi)
    tp = len(matched_true)
    fp = len(pred_peaks) - tp
    fn = len(true_bps) - tp
    return tp, fp, fn


# ---------- feature extraction ---------------------------------------------

@torch.no_grad()
def backbone_hidden(backbone: SequenceBackbone, v2_seq: np.ndarray,
                    device: str) -> torch.Tensor:
    """Run backbone forward + RC; return concatenated hidden states (L, 2*d_model).

    Same bidirectional scheme as BidirMLM: h_fwd concat h_rc.flip(L).
    """
    v2 = torch.from_numpy(v2_seq[None, :].astype(np.int64))           # (1, L)
    fwd_hy = v2_to_hyena_ids(v2).to(device)
    v2_rc = _V2_RC[v2.clamp(0, 4)].flip(dims=[1])
    rc_hy = v2_to_hyena_ids(v2_rc).to(device)
    h_fwd = backbone(fwd_hy, is_hyena_ids=True)                       # (1, L, D)
    h_rc = backbone(rc_hy, is_hyena_ids=True).flip(dims=[1])
    return torch.cat([h_fwd, h_rc], dim=-1)[0]                        # (L, 2D)


@torch.no_grad()
def triplet_features(backbone: SequenceBackbone, R: np.ndarray, P1: np.ndarray,
                     P2: np.ndarray, device: str) -> torch.Tensor:
    """Return (L, 3 * 2D) features per position: [h_R, h_R - h_P1, h_R - h_P2]."""
    h_R = backbone_hidden(backbone, R, device)
    h_P1 = backbone_hidden(backbone, P1, device)
    h_P2 = backbone_hidden(backbone, P2, device)
    return torch.cat([h_R, h_R - h_P1, h_R - h_P2], dim=-1).cpu()     # (L, 6D)


# ---------- event sampling --------------------------------------------------

def sample_events(cache: CacheV2, shard_names: list[str], n: int,
                  rng: random.Random, max_len: int | None = None) -> list[dict]:
    """Sample up to n random events from the listed shards (uniform across shards).
    Returns dicts with R, P1, P2, bps."""
    pool: list[tuple[str, int]] = []
    for sh in shard_names:
        if sh not in cache.shards:
            continue
        shard = cache.shards[sh]
        for ev_idx in range(len(shard.events)):
            ev = shard.events[ev_idx]
            if max_len is not None and ev["seq_len"] if "seq_len" in ev.dtype.names \
                    else False:
                pass
            pool.append((sh, ev_idx))
    rng.shuffle(pool)

    events: list[dict] = []
    for sh, ev_idx in pool:
        if len(events) >= n:
            break
        shard = cache.shards[sh]
        try:
            ev = shard.get_triplet(ev_idx)
        except Exception:
            continue
        L = ev["seq_len"]
        if max_len is not None and L > max_len:
            continue
        if ev["bp_start"] < 0 or ev["bp_end"] > L:
            continue
        # Some events have bp_end == seq_len (recombinant ends at alignment boundary).
        # Clamp into the valid index range so the Gaussian label includes a peak
        # at the very last position instead of silently dropping it.
        bps = sorted({
            max(0, min(int(ev["bp_start"]), L - 1)),
            max(0, min(int(ev["bp_end"]), L - 1)),
        })
        events.append({
            "shard": sh,
            "ev_idx": ev_idx,
            "L": L,
            "R": np.asarray(ev["R"], dtype=np.int8),
            "P1": np.asarray(ev["P1"], dtype=np.int8),
            "P2": np.asarray(ev["P2"], dtype=np.int8),
            "bps": bps,
        })
    return events


# ---------- train + evaluate ------------------------------------------------

def precompute_features(backbone: SequenceBackbone, events: list[dict],
                        device: str, label: str = "") -> list[dict]:
    """Run backbone over every event once; cache per-position features on CPU."""
    out: list[dict] = []
    t0 = time.time()
    for i, ev in enumerate(events):
        feats = triplet_features(backbone, ev["R"], ev["P1"], ev["P2"], device)
        y = gaussian_soft_label(ev["L"], ev["bps"])
        out.append({**ev, "feats": feats, "y": torch.from_numpy(y)})
        _rss_watchdog(label=f"feature cache {label} #{i}")
        if (i + 1) % 5 == 0 or i + 1 == len(events):
            rate = (i + 1) / max(time.time() - t0, 1e-3)
            print(f"    {label} cached {i+1}/{len(events)}  ({rate:.2f} ev/s)  "
                  f"RSS {_current_rss_bytes()/2**30:.1f} GB", flush=True)
    return out


def train_probe(train_events: list[dict], val_events: list[dict],
                feature_dim: int, *, epochs: int, lr: float, pos_weight: float,
                device: str, threshold_sweep: list[float],
                rng: random.Random, head_type: str = "linear",
                mlp_hidden: int = 128) -> dict:
    if head_type == "linear":
        head = nn.Linear(feature_dim, 1).to(device)
    elif head_type == "mlp":
        head = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        ).to(device)
    else:
        raise ValueError(f"unknown head_type {head_type!r}")
    optim = torch.optim.Adam(head.parameters(), lr=lr)
    pw = torch.tensor([pos_weight], device=device)

    history: list[dict] = []
    best_f1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        head.train()
        order = list(range(len(train_events)))
        rng.shuffle(order)
        epoch_loss = 0.0
        epoch_n = 0
        for i in order:
            ev = train_events[i]
            x = ev["feats"].to(device)                                # (L, F)
            y = ev["y"].to(device)                                    # (L,)
            logits = head(x).squeeze(-1)                              # (L,)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item()) * x.shape[0]
            epoch_n += x.shape[0]
        train_loss = epoch_loss / max(1, epoch_n)

        # eval
        head.eval()
        per_thr = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in threshold_sweep}
        with torch.no_grad():
            for ev in val_events:
                x = ev["feats"].to(device)
                logits = head(x).squeeze(-1)
                p = torch.sigmoid(logits).cpu().numpy()
                for thr in threshold_sweep:
                    peaks = extract_peaks(p, thr)
                    tp, fp, fn = event_f1(ev["bps"], peaks)
                    per_thr[thr]["tp"] += tp
                    per_thr[thr]["fp"] += fp
                    per_thr[thr]["fn"] += fn

        # F1 per threshold
        thr_results = []
        for thr, c in per_thr.items():
            prec = c["tp"] / max(1, c["tp"] + c["fp"])
            rec = c["tp"] / max(1, c["tp"] + c["fn"])
            f1 = 2 * prec * rec / max(1e-9, prec + rec)
            thr_results.append({
                "thr": thr, "precision": prec, "recall": rec, "f1": f1, **c
            })
        thr_results.sort(key=lambda d: -d["f1"])
        best = thr_results[0]
        print(f"  epoch {ep:>2}  train_loss {train_loss:.4f}  "
              f"best F1 {best['f1']:.3f} @ thr={best['thr']:.2f}  "
              f"P {best['precision']:.3f}  R {best['recall']:.3f}  "
              f"(tp/fp/fn = {best['tp']}/{best['fp']}/{best['fn']})", flush=True)
        history.append({
            "epoch": ep,
            "train_loss": train_loss,
            "thresholds": thr_results,
            "best_thr": best["thr"],
            "best_f1": best["f1"],
        })
        if best["f1"] > best_f1:
            best_f1 = best["f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    return {
        "history": history,
        "best_f1": best_f1,
        "best_head_state": best_state,
        "n_train_events": len(train_events),
        "n_val_events": len(val_events),
    }


# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("models_test/snapshots/backbone_mlm_v1_e13_30k_gs292630.pt"))
    ap.add_argument("--backbone-mode", choices=["m12", "hf", "random"], default="m12",
                    help="m12 = our pretrained ckpt; hf = HuggingFace pretrained (no MLM); "
                         "random = no pretraining at all")
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--train-shards", nargs="+",
                    default=["XML-2", "XML-6"])     # short genomes, fast forward
    ap.add_argument("--val-shards", nargs="+",
                    default=["XML-5"])              # held-out HIV-like
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=11_500)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("models_test/m13_linear_probe.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] device={device}", flush=True)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    cache = CacheV2()

    # Build backbone according to mode:
    #   m12    -- HuggingFace-pretrained + our M1.2 MLM weights from ckpt
    #   hf     -- HuggingFace-pretrained on human genome, no MLM training
    #   random -- no pretraining at all (random init of HyenaDNA architecture)
    print(f"[{time.strftime('%H:%M:%S')}] building backbone (mode={args.backbone_mode})",
          flush=True)
    pretrained = args.backbone_mode in ("m12", "hf")
    backbone = SequenceBackbone(BackboneConfig(hf_name=DEFAULT_HF_NAME),
                                 pretrained=pretrained)
    d_model = backbone.cfg.d_model
    bidir = BidirMLM(backbone, d_model=d_model, n_classes=5).to(device)
    if args.backbone_mode == "m12":
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        bidir.load_state_dict(ck["model_state"])
        print(f"  loaded M1.2 ckpt: epoch={ck['epoch']} "
              f"stage_idx_done={ck.get('stage_idx_done')} gs={ck['global_step']:,}",
              flush=True)
    else:
        print(f"  no M1.2 ckpt loaded (ablation control)", flush=True)
    # Freeze the backbone (and the MLM head, which we don't use anyway).
    for p in bidir.parameters():
        p.requires_grad = False
    bidir.eval()

    print(f"[{time.strftime('%H:%M:%S')}] sampling events", flush=True)
    train_events = sample_events(cache, args.train_shards, args.n_train, rng,
                                  max_len=args.max_len)
    val_events = sample_events(cache, args.val_shards, args.n_val, rng,
                                max_len=args.max_len)
    print(f"  TRAIN: {len(train_events)} events from {args.train_shards}",
          flush=True)
    print(f"  VAL:   {len(val_events)} events from {args.val_shards}", flush=True)
    if not train_events or not val_events:
        raise SystemExit("not enough events sampled; check shard names")

    feature_dim = 3 * (2 * d_model)
    print(f"[{time.strftime('%H:%M:%S')}] precomputing features (dim={feature_dim})",
          flush=True)
    print(f"  train cache size estimate: "
          f"{sum(e['L'] for e in train_events) * feature_dim * 4 / 2**30:.2f} GB",
          flush=True)
    print(f"  val   cache size estimate: "
          f"{sum(e['L'] for e in val_events) * feature_dim * 4 / 2**30:.2f} GB",
          flush=True)

    # NB: feature extraction wants the SequenceBackbone wrapper (forward signature).
    train_cached = precompute_features(bidir.backbone, train_events, device, label="train")
    val_cached = precompute_features(bidir.backbone, val_events, device, label="val")

    threshold_sweep = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    print(f"[{time.strftime('%H:%M:%S')}] training linear probe "
          f"({args.epochs} epochs)", flush=True)
    result = train_probe(train_cached, val_cached, feature_dim,
                          epochs=args.epochs, lr=args.lr,
                          pos_weight=args.pos_weight, device=device,
                          threshold_sweep=threshold_sweep, rng=rng,
                          head_type=args.head, mlp_hidden=args.mlp_hidden)

    print(f"\n=== M1.3 linear probe result ===")
    print(f"  best F1 = {result['best_f1']:.3f}")
    print(f"  trained on {result['n_train_events']} TRAIN events")
    print(f"  evaluated on {result['n_val_events']} VAL events")

    # Persist a JSON-serialisable report.
    out_doc = {
        "ckpt": str(args.ckpt),
        "train_shards": args.train_shards,
        "val_shards": args.val_shards,
        "n_train_events": result["n_train_events"],
        "n_val_events": result["n_val_events"],
        "feature_dim": feature_dim,
        "best_f1": result["best_f1"],
        "epochs": [{
            "epoch": h["epoch"],
            "train_loss": h["train_loss"],
            "best_thr": h["best_thr"],
            "best_f1": h["best_f1"],
        } for h in result["history"]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2))
    print(f"  report → {args.out}")


if __name__ == "__main__":
    main()
