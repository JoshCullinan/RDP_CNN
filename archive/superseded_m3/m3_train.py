"""M3 — production breakpoint detector training.

Verdict from M1.2 + M1.3 + Path I (see memory/project_m13_*.md):
  - MLM pretraining (M1.2) is structurally wrong for breakpoint detection.
    Pretrained Hyena features lose the cross-sequence comparison signal.
  - Random-init Hyena + triplet diff features hit F1 0.41 in a 40-event
    linear probe — already 95% of the legacy CNN's 0.42 on SANTA.
  - Adding raw 22-channel MaxChi features didn't help random Hyena, so
    we feed the backbone one-hot sequences and let it learn the
    integration. The Hyena convs already do windowed comparison
    naturally.

This script:
  1. Initializes HyenaDNA-small-32k from random (no pretraining).
  2. For each event triplet (R, P1, P2), forwards all three through the
     SAME backbone (forward + reverse-complement, same as M1.2 bidir).
  3. Builds per-position features `[h_R, h_R - h_P1, h_R - h_P2]`.
  4. Predicts per-position breakpoint probability via a small head
     (linear or 1-hidden-layer MLP).
  5. Trains with Gaussian-soft σ=10 targets, weighted BCE.
  6. Evaluates F1 via find_peaks + ±200 bp tolerance.

Stability infra (the M1.2 + M1.3 lessons):
  - bf16 autocast (no GradScaler — eliminates the NaN that hit fp16 at
    scale on random init).
  - Conservative LR with linear warmup + cosine decay to 10% of peak.
  - Gradient clipping at 0.5.
  - Gradient checkpointing on the backbone (fits B=4 at L=10k).
  - Per-epoch checkpoint + snapshot from epoch 1 (don't lose peaks like
    M1.2 did).
  - RSS watchdog at 26 GB.

Success bar: SANTA UnseenTestSet F1 > 0.421 (runB2_sig10 baseline).
Stretch: LANL CRF agg F1 > 0.533 (legacy deployment baseline).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import resource
import shutil
import sys
import time
from dataclasses import dataclass
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


# ---------- RSS watchdog ---------------------------------------------------

_RSS_CEILING_BYTES = 26 * 1024 * 1024 * 1024


def _rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _rss_watchdog(label: str = "") -> None:
    rss = _rss()
    if rss > _RSS_CEILING_BYTES:
        raise MemoryError(
            f"RSS watchdog tripped {label}: {rss / 2**30:.1f} GB > "
            f"{_RSS_CEILING_BYTES / 2**30:.0f} GB cap"
        )


# ---------- targets + eval -------------------------------------------------

TOLERANCE = 200
LABEL_SIGMA = 10
EDGE_BUFFER = 25


def gaussian_soft_label(L: int, bps: list[int], sigma: int = LABEL_SIGMA) -> np.ndarray:
    y = np.zeros(L, dtype=np.float32)
    for bp in bps:
        if not (0 <= bp < L):
            continue
        lo = max(0, bp - 3 * sigma)
        hi = min(L, bp + 3 * sigma + 1)
        idx = np.arange(lo, hi)
        y[idx] = np.maximum(y[idx], np.exp(-0.5 * ((idx - bp) / sigma) ** 2))
    return y


def extract_peaks(p: np.ndarray, threshold: float,
                   edge_buffer: int = EDGE_BUFFER) -> np.ndarray:
    L = len(p)
    p2 = p.copy()
    p2[:edge_buffer] = 0.0
    p2[L - edge_buffer:] = 0.0
    peaks, _ = find_peaks(p2, height=threshold, distance=TOLERANCE)
    return peaks


def event_f1(true_bps: list[int], pred_peaks: np.ndarray,
              tolerance: int = TOLERANCE) -> tuple[int, int, int]:
    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    pairs = []
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


# ---------- streaming event sampler ---------------------------------------

def event_plan(cache: CacheV2, shard_names: list[str], n: int,
                rng: random.Random, max_len: int | None = None) -> list[tuple[str, int]]:
    """Return shuffled (shard, ev_idx) plan. Doesn't materialise events.

    `seq_len` lives in `align_idx`, not in the events row — we look it up via
    the per-file mapping.
    """
    pool = []
    for sh in shard_names:
        if sh not in cache.shards:
            continue
        shard = cache.shards[sh]
        for ev_idx in range(len(shard.events)):
            pool.append((sh, ev_idx))
    rng.shuffle(pool)
    if max_len is None:
        return pool[:n]
    out = []
    for sh, ev_idx in pool:
        if len(out) >= n:
            break
        shard = cache.shards[sh]
        ev = shard.events[ev_idx]
        fi = int(ev["file_idx"])
        L = int(shard._by_file[fi]["seq_len"])
        if L > max_len:
            continue
        if int(ev["bp_start"]) < 0 or int(ev["bp_end"]) > L:
            continue
        out.append((sh, ev_idx))
    return out


def load_event(cache: CacheV2, sh: str, ev_idx: int) -> dict:
    ev = cache.shards[sh].get_triplet(ev_idx)
    L = ev["seq_len"]
    bps = sorted({
        max(0, min(int(ev["bp_start"]), L - 1)),
        max(0, min(int(ev["bp_end"]), L - 1)),
    })
    return {
        "shard": sh, "ev_idx": ev_idx, "L": L,
        "R": np.asarray(ev["R"], dtype=np.int8),
        "P1": np.asarray(ev["P1"], dtype=np.int8),
        "P2": np.asarray(ev["P2"], dtype=np.int8),
        "bps": bps,
    }


# ---------- model ----------------------------------------------------------

def hidden(backbone: SequenceBackbone, v2_seq: np.ndarray, device: str) -> torch.Tensor:
    """Forward + reverse-complement, concat hidden states. Gradients flow."""
    v2 = torch.from_numpy(v2_seq[None, :].astype(np.int64))
    fwd_hy = v2_to_hyena_ids(v2).to(device)
    v2_rc = _V2_RC[v2.clamp(0, 4)].flip(dims=[1])
    rc_hy = v2_to_hyena_ids(v2_rc).to(device)
    h_fwd = backbone(fwd_hy, is_hyena_ids=True)
    h_rc = backbone(rc_hy, is_hyena_ids=True).flip(dims=[1])
    return torch.cat([h_fwd, h_rc], dim=-1)[0]                    # (L, 2D)


def triplet_features(backbone: SequenceBackbone, R, P1, P2, device: str
                      ) -> torch.Tensor:
    h_R = hidden(backbone, R, device)
    h_P1 = hidden(backbone, P1, device)
    h_P2 = hidden(backbone, P2, device)
    return torch.cat([h_R, h_R - h_P1, h_R - h_P2], dim=-1)        # (L, 6D)


def build_head(head_type: str, feature_dim: int, hidden_dim: int) -> nn.Module:
    if head_type == "linear":
        return nn.Linear(feature_dim, 1)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    raise ValueError(f"unknown head_type {head_type!r}")


# ---------- LR schedule ---------------------------------------------------

def linear_warmup_cosine(step: int, warmup: int, total: int,
                         min_frac: float = 0.1) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return min_frac + (1.0 - min_frac) * cos


# ---------- evaluation ----------------------------------------------------

def evaluate(bidir, head, cache, val_plan, max_len, device, amp_dtype,
             threshold_sweep, rng) -> dict:
    bidir.eval()
    head.eval()
    per_thr = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in threshold_sweep}
    n_events = 0
    n_positions = 0
    t0 = time.time()
    with torch.no_grad():
        for sh, ev_idx in val_plan:
            ev = load_event(cache, sh, ev_idx)
            if ev["L"] > max_len:
                ev["R"] = ev["R"][:max_len]; ev["P1"] = ev["P1"][:max_len]
                ev["P2"] = ev["P2"][:max_len]; ev["L"] = max_len
                ev["bps"] = [bp for bp in ev["bps"] if bp < max_len]
            with torch.amp.autocast("cuda", dtype=amp_dtype,
                                     enabled=(device == "cuda")):
                feats = triplet_features(bidir.backbone, ev["R"], ev["P1"], ev["P2"], device)
                logits = head(feats).squeeze(-1)
            p = torch.sigmoid(logits.float()).cpu().numpy()
            for thr in threshold_sweep:
                peaks = extract_peaks(p, thr)
                tp, fp, fn = event_f1(ev["bps"], peaks)
                per_thr[thr]["tp"] += tp
                per_thr[thr]["fp"] += fp
                per_thr[thr]["fn"] += fn
            n_events += 1
            n_positions += ev["L"]
            _rss_watchdog(label=f"val event {n_events}")
    thr_results = []
    for thr, c in per_thr.items():
        prec = c["tp"] / max(1, c["tp"] + c["fp"])
        rec = c["tp"] / max(1, c["tp"] + c["fn"])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        thr_results.append({"thr": thr, "precision": prec, "recall": rec, "f1": f1, **c})
    thr_results.sort(key=lambda d: -d["f1"])
    return {
        "n_events": n_events, "n_positions": n_positions,
        "elapsed_s": time.time() - t0,
        "best": thr_results[0], "all_thresholds": thr_results,
    }


# ---------- training loop -------------------------------------------------

@dataclass
class TrainConfig:
    train_shards: list[str]
    val_shards: list[str]
    n_train: int
    n_val: int
    max_len: int
    epochs: int
    lr: float
    pos_weight: float
    warmup_steps: int
    grad_clip: float
    bf16: bool
    seed: int
    head_type: str
    head_hidden: int
    backbone_mode: str            # 'random' | 'm12' | 'hf'
    ckpt: Path
    ckpt_out: Path
    snapshots_dir: Path
    history_out: Path
    log_every: int
    val_every_epoch: int


def train(cfg: TrainConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)

    print(f"[{time.strftime('%H:%M:%S')}] device={device}  mode={cfg.backbone_mode}",
          flush=True)
    print(f"  shards: train={cfg.train_shards}  val={cfg.val_shards}", flush=True)
    print(f"  events: train={cfg.n_train}  val={cfg.n_val}  max_len={cfg.max_len}",
          flush=True)

    cache = CacheV2()

    # backbone
    pretrained_hf = cfg.backbone_mode in ("m12", "hf")
    bb = SequenceBackbone(BackboneConfig(hf_name=DEFAULT_HF_NAME), pretrained=pretrained_hf)
    d_model = bb.cfg.d_model
    bidir = BidirMLM(bb, d_model=d_model, n_classes=5).to(device)
    if cfg.backbone_mode == "m12":
        ck = torch.load(cfg.ckpt, map_location=device, weights_only=False)
        bidir.load_state_dict(ck["model_state"])
        print(f"  loaded M1.2 ckpt: epoch={ck['epoch']} gs={ck['global_step']:,}",
              flush=True)
    bidir.backbone.hyena.gradient_checkpointing_enable()
    print(f"  gradient_checkpointing: ON", flush=True)

    feature_dim = 3 * (2 * d_model)
    head = build_head(cfg.head_type, feature_dim, cfg.head_hidden).to(device)
    print(f"  feature_dim={feature_dim}  head={cfg.head_type}", flush=True)
    print(f"  params: backbone={sum(p.numel() for p in bidir.backbone.parameters()):,}  "
          f"head={sum(p.numel() for p in head.parameters()):,}", flush=True)

    optim_params = list(bidir.backbone.parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(optim_params, lr=cfg.lr, weight_decay=0.01)
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    print(f"  AMP dtype: {amp_dtype}  (bf16={cfg.bf16})", flush=True)

    train_plan = event_plan(cache, cfg.train_shards, cfg.n_train, rng, cfg.max_len)
    val_plan = event_plan(cache, cfg.val_shards, cfg.n_val, rng, cfg.max_len)
    if not train_plan or not val_plan:
        raise SystemExit("empty plan; check shard names + max_len")
    print(f"  train_plan: {len(train_plan)}  val_plan: {len(val_plan)}", flush=True)

    total_steps = cfg.epochs * len(train_plan)
    pw = torch.tensor([cfg.pos_weight], device=device)
    threshold_sweep = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    cfg.snapshots_dir.mkdir(parents=True, exist_ok=True)
    history: dict = {"epochs": [], "config": {k: (str(v) if isinstance(v, (Path,)) else v)
                                              for k, v in cfg.__dict__.items()}}

    global_step = 0
    best_val_f1 = -1.0
    best_epoch = -1

    for ep in range(1, cfg.epochs + 1):
        bidir.train()
        head.train()
        order = list(range(len(train_plan)))
        rng.shuffle(order)
        ep_loss = 0.0
        ep_n = 0
        t_ep = time.time()

        for step_in_ep, idx in enumerate(order):
            sh, ev_idx = train_plan[idx]
            ev = load_event(cache, sh, ev_idx)
            if ev["L"] > cfg.max_len:
                ev["R"] = ev["R"][:cfg.max_len]; ev["P1"] = ev["P1"][:cfg.max_len]
                ev["P2"] = ev["P2"][:cfg.max_len]; ev["L"] = cfg.max_len
                ev["bps"] = [bp for bp in ev["bps"] if bp < cfg.max_len]
            y = torch.from_numpy(gaussian_soft_label(ev["L"], ev["bps"])).to(device)

            # LR step
            lr_mult = linear_warmup_cosine(global_step, cfg.warmup_steps, total_steps)
            for g in optim.param_groups:
                g["lr"] = cfg.lr * lr_mult

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                feats = triplet_features(bidir.backbone, ev["R"], ev["P1"], ev["P2"], device)
                logits = head(feats).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, cfg.grad_clip)
            optim.step()
            ep_loss += float(loss.item()) * ev["L"]
            ep_n += ev["L"]
            global_step += 1
            if global_step % cfg.log_every == 0:
                gpu = (torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0)
                print(f"    step {global_step:>6}  loss {loss.item():.4f}  "
                      f"lr {cfg.lr*lr_mult:.2e}  RSS {_rss()/2**30:.1f}GB  "
                      f"GPU {gpu:.2f}GB", flush=True)
            _rss_watchdog(label=f"train ep{ep} step{step_in_ep}")

        train_loss = ep_loss / max(1, ep_n)
        elapsed = time.time() - t_ep

        # Eval
        if ep % cfg.val_every_epoch == 0 or ep == cfg.epochs:
            val = evaluate(bidir, head, cache, val_plan, cfg.max_len, device, amp_dtype,
                           threshold_sweep, rng)
            print(f"  epoch {ep:>2}  train_loss {train_loss:.4f}  "
                  f"VAL F1 {val['best']['f1']:.3f} @ thr={val['best']['thr']:.2f}  "
                  f"P {val['best']['precision']:.3f}  R {val['best']['recall']:.3f}  "
                  f"({elapsed:.0f}s train, {val['elapsed_s']:.0f}s val)", flush=True)
            history["epochs"].append({
                "epoch": ep, "train_loss": train_loss,
                "val_f1": val["best"]["f1"], "val_thr": val["best"]["thr"],
                "val_precision": val["best"]["precision"],
                "val_recall": val["best"]["recall"],
                "elapsed_s": elapsed,
            })
            # Save ckpt + snapshot
            ck_state = {
                "backbone_state": bidir.backbone.state_dict(),
                "head_state": head.state_dict(),
                "optim_state": optim.state_dict(),
                "epoch": ep, "global_step": global_step,
                "best_val_f1": best_val_f1,
                "cfg": history["config"],
            }
            torch.save(ck_state, cfg.ckpt_out)
            snap_name = f"m3_ep{ep:02d}_f1{val['best']['f1']:.3f}.pt".replace(" ", "")
            shutil.copy2(cfg.ckpt_out, cfg.snapshots_dir / snap_name)
            print(f"  ckpt → {cfg.ckpt_out}  snapshot → {cfg.snapshots_dir / snap_name}",
                  flush=True)
            if val["best"]["f1"] > best_val_f1:
                best_val_f1 = val["best"]["f1"]
                best_epoch = ep
                # Also save a "best so far" copy
                shutil.copy2(cfg.ckpt_out, cfg.snapshots_dir / "m3_best.pt")
                print(f"  new best: F1 {best_val_f1:.3f} @ epoch {ep} → m3_best.pt",
                      flush=True)
            with cfg.history_out.open("w") as f:
                json.dump(history, f, indent=2)
        else:
            print(f"  epoch {ep:>2}  train_loss {train_loss:.4f}  "
                  f"(no val this epoch)  ({elapsed:.0f}s)", flush=True)
            history["epochs"].append({
                "epoch": ep, "train_loss": train_loss,
                "val_f1": None, "elapsed_s": elapsed,
            })

    print(f"\n=== M3 training done ===")
    print(f"  best VAL F1: {best_val_f1:.3f} @ epoch {best_epoch}")
    print(f"  ckpt: {cfg.snapshots_dir / 'm3_best.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-mode", choices=["random", "m12", "hf"], default="random",
                    help="random (default, recommended) | m12 | hf")
    ap.add_argument("--ckpt", type=Path,
                    default=Path("models_test/snapshots/backbone_mlm_v1_e13_30k_gs292630.pt"))
    ap.add_argument("--train-shards", nargs="+", default=["XML-2", "XML-6"])
    ap.add_argument("--val-shards", nargs="+", default=["XML-5"])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=11_000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--pos-weight", type=float, default=200.0)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--grad-clip", type=float, default=0.5)
    ap.add_argument("--no-bf16", action="store_true",
                    help="use fp16 instead of bf16 (default uses bf16)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--ckpt-out", type=Path, default=Path("models_test/m3.pt"))
    ap.add_argument("--snapshots-dir", type=Path,
                    default=Path("models_test/m3_snapshots"))
    ap.add_argument("--history-out", type=Path,
                    default=Path("models_test/m3_history.json"))
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every-epoch", type=int, default=1)
    args = ap.parse_args()

    cfg = TrainConfig(
        train_shards=args.train_shards, val_shards=args.val_shards,
        n_train=args.n_train, n_val=args.n_val, max_len=args.max_len,
        epochs=args.epochs, lr=args.lr, pos_weight=args.pos_weight,
        warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
        bf16=not args.no_bf16, seed=args.seed,
        head_type=args.head, head_hidden=args.head_hidden,
        backbone_mode=args.backbone_mode, ckpt=args.ckpt,
        ckpt_out=args.ckpt_out, snapshots_dir=args.snapshots_dir,
        history_out=args.history_out,
        log_every=args.log_every, val_every_epoch=args.val_every_epoch,
    )
    train(cfg)


if __name__ == "__main__":
    main()
