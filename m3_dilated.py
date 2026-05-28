"""M3 with deep dilated-CNN head (the synthesis).

The story arc:
  - M1.2 MLM pretraining: backbone learned nucleotide invariance — opposite
    of breakpoint sensitivity.
  - M1.3 + M3-mini at 40 events: random Hyena + linear head looked like
    F1 0.41 — but that was small-sample noise.
  - At 200+ events: frozen random + linear head collapses to F1 0.28
    (trivial baseline). Linear can't integrate cross-position patterns.

Insight: breakpoint detection needs *windowed integration across positions*
(detect transitions in cross-sequence similarity), which is exactly what
the legacy CNN's dilated conv stack does. The legacy CNN gets F1 0.42 on
SANTA precisely because of this depth.

This script combines:
  - Frozen random-init HyenaDNA backbone (provides per-position context
    from forward + reverse-complement passes — no MLM baggage).
  - Triplet diff features `[h_R, h_R - h_P1, h_R - h_P2]` (1536-dim).
  - **Dilated 1D CNN head** with 6 residual blocks at dilations
    {1, 2, 4, 8, 16, 32}, giving ~511 bp receptive field — enough to
    detect MaxChi-style transitions at window scales {50, 100, 200, 500}.
  - Per-position sigmoid output, weighted BCE on Gaussian-soft σ=10
    targets, find_peaks F1 eval with ±200 bp tolerance.

Backbone is FROZEN (no gradient flow) so we don't destroy the random
projections that preserve the cross-sequence signal. Head trains in bf16.

Success bar: F1 > 0.30 means depth helps; F1 > 0.42 means we beat the
legacy CNN; F1 > 0.50 means we beat classical RDP.
"""

from __future__ import annotations

import argparse
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


# ---------- RSS watchdog ----------

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


# ---------- target + eval ----------

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


# ---------- dilated CNN head ----------

class DilatedHead(nn.Module):
    """Per-position breakpoint probability via a residual dilated conv stack.

    Mirrors the legacy CNN's design philosophy: multiple receptive-field
    scales {3, 5, 9, 17, 33, 65, 129} via dilations {1, 2, 4, 8, 16, 32},
    so the head can detect MaxChi-style windowed transitions.

    Input:  (B, L, C_in)  — per-position features from backbone
    Output: (B, L)        — per-position logits
    """

    def __init__(self, in_channels: int = 1536, hidden: int = 128,
                 n_blocks: int = 6, kernel: int = 3,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            d = 2 ** i
            pad = d * (kernel // 2)
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=kernel,
                          padding=pad, dilation=d),
                nn.GELU(),
                nn.GroupNorm(8, hidden),
                nn.Dropout1d(dropout),
            ))
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C_in) → (B, C_in, L)
        x = x.transpose(1, 2)
        h = self.proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h).squeeze(1)         # (B, L)


class M3MultiHead(nn.Module):
    """Two-head model (M3 v4): per-position BP logits + a single per-triplet
    "is this a recombinant?" logit.

    The trunk is the SAME residual dilated conv stack as ``DilatedHead`` (kept
    as a separate class so v2 single-head checkpoints still load via
    ``DilatedHead``). The BP head is identical to v2's final conv.

    The aux head reads THREE pooled signals over positions:
      - mean / max of the trunk features (transition-energy structure), and
      - the per-channel mean of the RAW 22-channel input (``x.mean`` over L).

    The raw-input mean is the decisive addition for cross-species safety: each
    block's GroupNorm centres the trunk activations, so absolute parent-match
    *level* — high for a true recombinant (a mosaic of its parents, matching one
    of them everywhere), low for a cross-species triplet (matches neither) — is
    weakened in the trunk. Feeding ``mean(match_p1), mean(match_p2), ...``
    directly gives the aux head the divergence signal that separates an HIV
    recombinant from an Ebola cross-species comparison. This is consistent with
    M3's existing feature philosophy (the raw 22 channels already include
    MaxChi-style statistics).

    forward(x: (B, L, C_in)) → (bp_logits: (B, L), aux_logit: (B,))
    """

    def __init__(self, in_channels: int = 22, hidden: int = 128,
                 n_blocks: int = 6, kernel: int = 3,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.proj = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            d = 2 ** i
            pad = d * (kernel // 2)
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=kernel,
                          padding=pad, dilation=d),
                nn.GELU(),
                nn.GroupNorm(8, hidden),
                nn.Dropout1d(dropout),
            ))
        self.bp_head = nn.Conv1d(hidden, 1, kernel_size=1)
        self.aux_head = nn.Sequential(
            nn.Linear(2 * hidden + in_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C_in) → (B, C_in, L)
        x = x.transpose(1, 2)
        h = self.proj(x)
        for block in self.blocks:
            h = h + block(h)
        return h                                # (B, hidden, L)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)                       # (B, hidden, L)
        bp_logits = self.bp_head(h).squeeze(1)              # (B, L)
        mean_pool = h.mean(dim=-1)                          # (B, hidden)
        max_pool = h.amax(dim=-1)                           # (B, hidden)
        raw_mean = x.mean(dim=1).to(mean_pool.dtype)        # (B, C_in) — over L
        pooled = torch.cat([mean_pool, max_pool, raw_mean], dim=-1)  # (B, 2H+C)
        aux_logit = self.aux_head(pooled).squeeze(-1)       # (B,)
        return bp_logits, aux_logit


def build_head(head_mode: str, feat_dim: int, hidden: int, blocks: int,
               dropout: float) -> nn.Module:
    """Construct the BP head for the requested mode (used by training + eval)."""
    if head_mode == "multi":
        return M3MultiHead(feat_dim, hidden, blocks, dropout=dropout)
    if head_mode == "single":
        return DilatedHead(feat_dim, hidden, blocks, dropout=dropout)
    raise ValueError(head_mode)


# ---------- feature extraction (frozen backbone) ----------

@torch.no_grad()
def backbone_hidden(backbone: SequenceBackbone, v2_seq: np.ndarray,
                    device: str) -> torch.Tensor:
    v2 = torch.from_numpy(v2_seq[None, :].astype(np.int64))
    fwd_hy = v2_to_hyena_ids(v2).to(device)
    v2_rc = _V2_RC[v2.clamp(0, 4)].flip(dims=[1])
    rc_hy = v2_to_hyena_ids(v2_rc).to(device)
    h_fwd = backbone(fwd_hy, is_hyena_ids=True)
    h_rc = backbone(rc_hy, is_hyena_ids=True).flip(dims=[1])
    return torch.cat([h_fwd, h_rc], dim=-1)[0]  # (L, 2D)


@torch.no_grad()
def triplet_features(backbone, R, P1, P2, device):
    h_R = backbone_hidden(backbone, R, device)
    h_P1 = backbone_hidden(backbone, P1, device)
    h_P2 = backbone_hidden(backbone, P2, device)
    return torch.cat([h_R, h_R - h_P1, h_R - h_P2], dim=-1)  # (L, 6D)


# ---------- raw legacy-CNN-style features (no backbone) -------------------

MAXCHI_WINDOWS = (50, 100, 200, 500)


def raw_features(R: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> torch.Tensor:
    """22 channels: 3×one-hot + match flags + windowed MaxChi. Matches the legacy
    CNN's `encode_triplet` output. Returns (L, 22) float32."""
    L = len(R)
    out = np.zeros((L, 22), dtype=np.float32)
    for v, base in ((R, 0), (P1, 5), (P2, 10)):
        for tok in range(5):
            out[v == tok, base + tok] = 1.0
    match_p1 = (R == P1).astype(np.float32)
    match_p2 = (R == P2).astype(np.float32)
    out[:, 15] = match_p1
    out[:, 16] = match_p2
    out[:, 17] = (P1 != P2).astype(np.float32)
    parental = match_p1 - match_p2
    cumsum = np.concatenate([[0.0], np.cumsum(parental)]).astype(np.float64)
    i = np.arange(L)
    for ch, w in enumerate(MAXCHI_WINDOWS):
        right_lo = np.minimum(i, L); right_hi = np.minimum(i + w, L)
        left_lo = np.maximum(i - w, 0); left_hi = np.maximum(i, 0)
        right_n = np.maximum((right_hi - right_lo).astype(np.float64), 1.0)
        left_n = np.maximum((left_hi - left_lo).astype(np.float64), 1.0)
        out[:, 18 + ch] = ((cumsum[right_hi] - cumsum[right_lo]) / right_n
                            - (cumsum[left_hi] - cumsum[left_lo]) / left_n).astype(np.float32)
    return torch.from_numpy(out)


def features_for_event(feature_mode: str, ev: dict, backbone, device: str
                       ) -> tuple[torch.Tensor, int]:
    """Returns (features (L, D), feature_dim) according to mode."""
    if feature_mode == "raw":
        return raw_features(ev["R"], ev["P1"], ev["P2"]).to(device), 22
    if feature_mode == "hyena":
        f = triplet_features(backbone, ev["R"], ev["P1"], ev["P2"], device)
        return f, f.shape[-1]
    if feature_mode == "combined":
        rf = raw_features(ev["R"], ev["P1"], ev["P2"]).to(device)
        hf = triplet_features(backbone, ev["R"], ev["P1"], ev["P2"], device)
        return torch.cat([rf, hf], dim=-1), rf.shape[-1] + hf.shape[-1]
    raise ValueError(feature_mode)


# ---------- event sampling (lookups via align_idx) ----------

def event_plan(cache: CacheV2, shard_names: list[str], n: int,
                rng: random.Random, max_len: int) -> list[tuple[str, int]]:
    pool = []
    for sh in shard_names:
        if sh not in cache.shards:
            continue
        for ev_idx in range(len(cache.shards[sh].events)):
            pool.append((sh, ev_idx))
    rng.shuffle(pool)
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


def load_negative_panels(panels: tuple[str, ...]) -> dict[str, list[np.ndarray]]:
    """Load real-virus reference panels as int8 sequences for negative-control
    triplet sampling. Keyed by panel name."""
    from Bio import SeqIO
    out: dict[str, list[np.ndarray]] = {}
    for panel in panels:
        fa = Path(f"/home/joshc/Dev/RDP_CNN/data/real_recombinants/{panel}/aligned.fa")
        if not fa.exists():
            print(f"  WARN: negative panel {panel} missing at {fa}", flush=True)
            continue
        recs = []
        for r in SeqIO.parse(fa, "fasta"):
            # Skip known recombinants — we want NEGATIVES only.
            if "recombinant" in r.id.lower() or "_xbb_" in r.id.lower() and "parent" not in r.id.lower():
                continue
            s = str(r.seq).upper()
            arr = np.empty(len(s), dtype=np.int8)
            for i, ch in enumerate(s):
                arr[i] = {"A": 0, "T": 1, "G": 2, "C": 3, "-": 4}.get(ch, 4)
            recs.append(arr)
        out[panel] = recs
        print(f"  loaded negative panel {panel}: {len(recs)} sequences", flush=True)
    return out


def sample_negative(neg_panels: dict[str, list[np.ndarray]],
                    rng: random.Random, max_len: int) -> dict | None:
    """Sample a 3-sequence negative-control triplet from the loaded panels.
    Returns event-shaped dict with empty bps."""
    if not neg_panels:
        return None
    panel = rng.choice(list(neg_panels.keys()))
    recs = neg_panels[panel]
    if len(recs) < 3:
        return None
    R_idx, P1_idx, P2_idx = rng.sample(range(len(recs)), 3)
    R, P1, P2 = recs[R_idx], recs[P1_idx], recs[P2_idx]
    L = min(len(R), len(P1), len(P2), max_len)
    return {
        "shard": f"neg:{panel}", "ev_idx": -1, "L": L,
        "R": R[:L].copy(), "P1": P1[:L].copy(), "P2": P2[:L].copy(),
        "bps": [],          # negative control: no breakpoints
        "is_recomb": 0,
    }


def build_santa_neg_pool(cache: CacheV2, shard_names: list[str]
                         ) -> list[tuple[str, int]]:
    """List (shard, file_idx) pairs in the train shards that have >=3
    non-recombinant sequences — the pool for SANTA-internal negatives."""
    pool: list[tuple[str, int]] = []
    for sh in shard_names:
        if sh not in cache.shards:
            continue
        shard = cache.shards[sh]
        for fi, ids in shard._non_recomb_ids.items():
            if len(ids) >= 3:
                pool.append((sh, int(fi)))
    return pool


def sample_santa_negative(cache: CacheV2, neg_file_pool: list[tuple[str, int]],
                          rng: random.Random, max_len: int) -> dict | None:
    """Sample a NON-recombinant SANTA triplet: 3 non-recombinant sequences from
    the same alignment (coordinate-aligned, no registered mosaic). This puts
    SANTA provenance on BOTH sides of the aux label so the aux head is forced to
    learn recombination *structure*, not "is this simulator output?". Returns an
    event-shaped dict with empty bps."""
    if not neg_file_pool:
        return None
    sh, fi = neg_file_pool[rng.randrange(len(neg_file_pool))]
    shard = cache.shards[sh]
    ids = shard.sample_non_recombinant_ids(fi, 3, rng)
    if len(ids) < 3:
        return None
    align = shard.get_alignment(fi)                 # memmap view (n, L)
    L = min(int(align.shape[1]), max_len)
    R = np.asarray(align[ids[0] - 1][:L], dtype=np.int8).copy()
    P1 = np.asarray(align[ids[1] - 1][:L], dtype=np.int8).copy()
    P2 = np.asarray(align[ids[2] - 1][:L], dtype=np.int8).copy()
    return {
        "shard": f"santaneg:{sh}", "ev_idx": -1, "L": L,
        "R": R, "P1": P1, "P2": P2,
        "bps": [],
        "is_recomb": 0,
    }


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
        "is_recomb": 1,
    }


# ---------- LR schedule ----------

def linear_warmup_cosine(step: int, warmup: int, total: int,
                         min_frac: float = 0.1) -> float:
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return min_frac + (1.0 - min_frac) * cos


# ---------- eval ----------

def evaluate(backbone, head, cache, val_plan, max_len, device, amp_dtype,
              thresholds, feature_mode: str, head_mode: str = "single") -> dict:
    head.eval()
    per_thr = {thr: {"tp": 0, "fp": 0, "fn": 0} for thr in thresholds}
    n_events = 0
    aux_probs: list[float] = []
    t0 = time.time()
    for sh, ev_idx in val_plan:
        ev = load_event(cache, sh, ev_idx)
        if ev["L"] > max_len:
            ev["R"] = ev["R"][:max_len]; ev["P1"] = ev["P1"][:max_len]
            ev["P2"] = ev["P2"][:max_len]; ev["L"] = max_len
            ev["bps"] = [b for b in ev["bps"] if b < max_len]
        feats, _ = features_for_event(feature_mode, ev, backbone, device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device=="cuda")):
                out = head(feats[None])
        if head_mode == "multi":
            bp_logits, aux_logit = out
            logits = bp_logits                       # (1, L)
            aux_probs.append(float(torch.sigmoid(aux_logit.float())[0]))
        else:
            logits = out                             # (1, L)
        p = torch.sigmoid(logits[0].float()).cpu().numpy()
        for thr in thresholds:
            peaks = extract_peaks(p, thr)
            tp, fp, fn = event_f1(ev["bps"], peaks)
            per_thr[thr]["tp"] += tp
            per_thr[thr]["fp"] += fp
            per_thr[thr]["fn"] += fn
        n_events += 1
        _rss_watchdog(label=f"val ev{n_events}")
    out = []
    for thr, c in per_thr.items():
        prec = c["tp"] / max(1, c["tp"] + c["fp"])
        rec = c["tp"] / max(1, c["tp"] + c["fn"])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        out.append({"thr": thr, "precision": prec, "recall": rec, "f1": f1, **c})
    out.sort(key=lambda d: -d["f1"])
    aux_mean = float(np.mean(aux_probs)) if aux_probs else float("nan")
    return {"n_events": n_events, "elapsed_s": time.time() - t0,
            "best": out[0], "all_thresholds": out,
            "aux_prob_mean": aux_mean}


# ---------- training ----------

@dataclass
class Cfg:
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
    head_hidden: int
    head_blocks: int
    head_dropout: float
    backbone_mode: str        # 'random' | 'm12' | 'hf'
    feature_mode: str         # 'hyena' | 'raw' | 'combined'
    head_mode: str            # 'single' (v2) | 'multi' (v4: + aux gate head)
    lambda_aux: float         # weight on the aux "is recombinant?" BCE term
    neg_santa_frac: float     # of each NEGATIVE step, fraction that is a
                              # SANTA-internal non-recombinant triplet (vs a
                              # real-virus-panel triplet). Breaks the
                              # SANTA-vs-real provenance confound.
    ckpt_in: Path
    ckpt_out: Path
    snapshots_dir: Path
    history_out: Path
    log_every: int
    neg_frac: float = 0.0     # fraction of each epoch's training steps that
                              # use a NEGATIVE-CONTROL triplet (real virus
                              # panel, no recombination, target = all zeros).
                              # Teaches the model to suppress predictions on
                              # constant-high-divergence inputs.
    neg_panels: tuple[str, ...] = (
        "ebola", "zika", "sarscov2_full",
    )


def train(cfg: Cfg) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    print(f"[{time.strftime('%H:%M:%S')}] device={device}  mode={cfg.backbone_mode}",
          flush=True)

    cache = CacheV2()

    # ---- backbone (only loaded if features need it) ----
    backbone = None
    d_model = 256
    if cfg.feature_mode in ("hyena", "combined"):
        pretrained_hf = cfg.backbone_mode in ("m12", "hf")
        bb = SequenceBackbone(BackboneConfig(hf_name=DEFAULT_HF_NAME),
                              pretrained=pretrained_hf)
        d_model = bb.cfg.d_model
        bidir = BidirMLM(bb, d_model=d_model, n_classes=5).to(device)
        if cfg.backbone_mode == "m12":
            ck = torch.load(cfg.ckpt_in, map_location=device, weights_only=False)
            bidir.load_state_dict(ck["model_state"])
            print(f"  loaded M1.2 ckpt: ep={ck['epoch']} gs={ck['global_step']:,}",
                  flush=True)
        for p in bidir.parameters():
            p.requires_grad = False
        bidir.eval()
        backbone = bidir.backbone

    # ---- head ----
    if cfg.feature_mode == "raw":
        feat_dim = 22
    elif cfg.feature_mode == "hyena":
        feat_dim = 3 * (2 * d_model)
    elif cfg.feature_mode == "combined":
        feat_dim = 22 + 3 * (2 * d_model)
    else:
        raise ValueError(cfg.feature_mode)
    head = build_head(cfg.head_mode, feat_dim, cfg.head_hidden,
                      cfg.head_blocks, cfg.head_dropout).to(device)
    print(f"  feature_mode={cfg.feature_mode}  feat_dim={feat_dim}  "
          f"head_mode={cfg.head_mode}", flush=True)
    n_head = sum(p.numel() for p in head.parameters())
    n_bb = sum(p.numel() for p in backbone.parameters()) if backbone is not None else 0
    print(f"  backbone params (frozen): {n_bb:,}  head params (trainable): {n_head:,}",
          flush=True)

    optim = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=0.01)
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    print(f"  AMP dtype: {amp_dtype}", flush=True)

    train_plan = event_plan(cache, cfg.train_shards, cfg.n_train, rng, cfg.max_len)
    val_plan = event_plan(cache, cfg.val_shards, cfg.n_val, rng, cfg.max_len)
    if not train_plan or not val_plan:
        raise SystemExit("empty plan")
    print(f"  train: {len(train_plan)} events  val: {len(val_plan)} events", flush=True)

    # Negative-control sources (only used if neg_frac > 0):
    #   - SANTA-internal: 3 non-recombinant seqs from one alignment (breaks the
    #     SANTA-vs-real provenance confound — see sample_santa_negative).
    #   - real-virus panels: cross-species anchors (ebola/zika/sarscov2).
    neg_panels_data: dict[str, list[np.ndarray]] = {}
    santa_neg_pool: list[tuple[str, int]] = []
    if cfg.neg_frac > 0:
        santa_neg_pool = build_santa_neg_pool(cache, cfg.train_shards)
        print(f"  neg_frac={cfg.neg_frac} (santa_frac={cfg.neg_santa_frac}): "
              f"{len(santa_neg_pool)} SANTA files with >=3 non-recombinants",
              flush=True)
        if cfg.neg_santa_frac < 1.0:
            print(f"  loading real negative panels {cfg.neg_panels}", flush=True)
            neg_panels_data = load_negative_panels(cfg.neg_panels)
        if not santa_neg_pool and not neg_panels_data:
            print(f"  WARN: no negative sources available, neg_frac disabled",
                  flush=True)
            cfg.neg_frac = 0.0

    total_steps = cfg.epochs * len(train_plan)
    pw = torch.tensor([cfg.pos_weight], device=device)
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    cfg.snapshots_dir.mkdir(parents=True, exist_ok=True)
    history: dict = {"epochs": [], "config": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}}

    global_step = 0
    best_val_f1 = -1.0
    best_epoch = -1

    for ep in range(1, cfg.epochs + 1):
        head.train()
        order = list(range(len(train_plan)))
        rng.shuffle(order)
        ep_loss = 0.0; ep_n = 0
        t_ep = time.time()
        ep_bp = 0.0; ep_aux = 0.0; aux_correct = 0; aux_total = 0
        for step_in_ep, idx in enumerate(order):
            # With probability neg_frac, swap in a NEGATIVE-CONTROL triplet
            # instead of the planned positive event. Negatives are SANTA-internal
            # (with prob neg_santa_frac) or real-virus-panel triplets.
            if cfg.neg_frac > 0 and rng.random() < cfg.neg_frac:
                use_santa = santa_neg_pool and (
                    not neg_panels_data or rng.random() < cfg.neg_santa_frac)
                if use_santa:
                    neg = sample_santa_negative(cache, santa_neg_pool, rng, cfg.max_len)
                else:
                    neg = sample_negative(neg_panels_data, rng, cfg.max_len)
                if neg is not None:
                    ev = neg
                else:
                    sh, ev_idx = train_plan[idx]
                    ev = load_event(cache, sh, ev_idx)
            else:
                sh, ev_idx = train_plan[idx]
                ev = load_event(cache, sh, ev_idx)
            if ev["L"] > cfg.max_len:
                ev["R"] = ev["R"][:cfg.max_len]; ev["P1"] = ev["P1"][:cfg.max_len]
                ev["P2"] = ev["P2"][:cfg.max_len]; ev["L"] = cfg.max_len
                ev["bps"] = [b for b in ev["bps"] if b < cfg.max_len]
            y = torch.from_numpy(gaussian_soft_label(ev["L"], ev["bps"])).to(device)

            lr_mult = linear_warmup_cosine(global_step, cfg.warmup_steps, total_steps)
            for g in optim.param_groups:
                g["lr"] = cfg.lr * lr_mult

            feats, _ = features_for_event(cfg.feature_mode, ev, backbone, device)
            is_recomb = int(ev.get("is_recomb", 1))

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device=="cuda")):
                if cfg.head_mode == "multi":
                    bp_logits, aux_logit = head(feats[None])
                    bp_logits = bp_logits.squeeze(0)     # (L,)
                    aux_logit = aux_logit.squeeze(0)     # scalar
                    # BP loss only on POSITIVES — never push BP→0 on negatives
                    # (that was the v3 failure). Negatives shape the trunk only
                    # via the aux gradient.
                    if is_recomb:
                        l_bp = F.binary_cross_entropy_with_logits(
                            bp_logits, y, pos_weight=pw)
                    else:
                        l_bp = bp_logits.new_zeros(())
                    aux_tgt = aux_logit.new_full((), float(is_recomb))
                    l_aux = F.binary_cross_entropy_with_logits(aux_logit, aux_tgt)
                    loss = l_bp + cfg.lambda_aux * l_aux
                else:
                    logits = head(feats[None]).squeeze(0)   # (L,)
                    loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
            optim.step()

            if cfg.head_mode == "multi":
                ep_bp += float(l_bp) * (ev["L"] if is_recomb else 0)
                ep_aux += float(l_aux)
                aux_total += 1
                aux_correct += int((aux_logit.item() > 0) == bool(is_recomb))
            ep_loss += float(loss.item()) * ev["L"]
            ep_n += ev["L"]
            global_step += 1
            if global_step % cfg.log_every == 0:
                gpu = (torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0)
                print(f"    step {global_step:>6}  loss {loss.item():.4f}  "
                      f"lr {cfg.lr*lr_mult:.2e}  RSS {_rss()/2**30:.1f}GB  "
                      f"GPU {gpu:.2f}GB", flush=True)
            _rss_watchdog(label=f"train ep{ep}/s{step_in_ep}")
        train_loss = ep_loss / max(1, ep_n)
        elapsed = time.time() - t_ep

        # Val
        val = evaluate(backbone, head, cache, val_plan, cfg.max_len, device, amp_dtype,
                       thresholds, cfg.feature_mode, cfg.head_mode)
        aux_msg = ""
        if cfg.head_mode == "multi":
            train_aux_acc = aux_correct / max(1, aux_total)
            aux_msg = (f"  [aux: train_loss {ep_aux / max(1, aux_total):.3f} "
                       f"acc {train_aux_acc:.2f}  val_pos_prob {val['aux_prob_mean']:.2f}]")
        print(f"  epoch {ep:>2}  train_loss {train_loss:.4f}  "
              f"VAL F1 {val['best']['f1']:.3f} @ thr={val['best']['thr']:.2f}  "
              f"P {val['best']['precision']:.3f}  R {val['best']['recall']:.3f}  "
              f"({elapsed:.0f}s train, {val['elapsed_s']:.0f}s val){aux_msg}", flush=True)

        history["epochs"].append({
            "epoch": ep, "train_loss": train_loss,
            "val_f1": val["best"]["f1"], "val_thr": val["best"]["thr"],
            "val_precision": val["best"]["precision"],
            "val_recall": val["best"]["recall"],
            "elapsed_s": elapsed,
            **({"train_aux_loss": ep_aux / max(1, aux_total),
                "train_aux_acc": aux_correct / max(1, aux_total),
                "val_aux_pos_prob": val["aux_prob_mean"]}
               if cfg.head_mode == "multi" else {}),
        })

        ck_state = {"head_state": head.state_dict(), "optim_state": optim.state_dict(),
                    "epoch": ep, "global_step": global_step, "best_val_f1": best_val_f1,
                    "cfg": history["config"]}
        torch.save(ck_state, cfg.ckpt_out)
        snap = cfg.snapshots_dir / f"m3d_ep{ep:02d}_f1{val['best']['f1']:.3f}.pt"
        shutil.copy2(cfg.ckpt_out, snap)
        if val["best"]["f1"] > best_val_f1:
            best_val_f1 = val["best"]["f1"]
            best_epoch = ep
            shutil.copy2(cfg.ckpt_out, cfg.snapshots_dir / "m3d_best.pt")
            print(f"  new best F1 {best_val_f1:.3f} → m3d_best.pt", flush=True)
        with cfg.history_out.open("w") as f:
            json.dump(history, f, indent=2)

    print(f"\n=== M3 dilated training done ===")
    print(f"  best VAL F1: {best_val_f1:.3f} @ epoch {best_epoch}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-mode", choices=["random", "m12", "hf"], default="random")
    ap.add_argument("--feature-mode", choices=["hyena", "raw", "combined"], default="hyena",
                    help="hyena (frozen Hyena triplet diffs) | raw (legacy 22ch) | combined")
    ap.add_argument("--ckpt-in", type=Path,
                    default=Path("models_test/snapshots/backbone_mlm_v1_e13_30k_gs292630.pt"))
    ap.add_argument("--train-shards", nargs="+", default=["XML-2", "XML-6"])
    ap.add_argument("--val-shards", nargs="+", default=["XML-5"])
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=11_000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pos-weight", type=float, default=200.0)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head-hidden", type=int, default=128)
    ap.add_argument("--head-blocks", type=int, default=6)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--ckpt-out", type=Path, default=Path("models_test/m3d.pt"))
    ap.add_argument("--snapshots-dir", type=Path, default=Path("models_test/m3d_snapshots"))
    ap.add_argument("--history-out", type=Path, default=Path("models_test/m3d_history.json"))
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--head-mode", choices=["single", "multi"], default="single",
                    help="single = v2 BP-only head; multi = v4 BP head + aux "
                         "'is recombinant?' gate head")
    ap.add_argument("--lambda-aux", type=float, default=0.5,
                    help="weight on the aux BCE term (multi head only)")
    ap.add_argument("--neg-frac", type=float, default=0.0,
                    help="fraction of training steps that use a negative-control "
                         "triplet (no recombination, BP loss skipped)")
    ap.add_argument("--neg-santa-frac", type=float, default=0.6,
                    help="of each negative step, fraction that is a SANTA-internal "
                         "non-recombinant triplet (vs a real-virus-panel triplet)")
    ap.add_argument("--neg-panels", nargs="+",
                    default=["ebola", "zika", "sarscov2_full"],
                    help="real virus panels to source negative triplets from")
    args = ap.parse_args()

    cfg = Cfg(
        train_shards=args.train_shards, val_shards=args.val_shards,
        n_train=args.n_train, n_val=args.n_val, max_len=args.max_len,
        epochs=args.epochs, lr=args.lr, pos_weight=args.pos_weight,
        warmup_steps=args.warmup_steps, grad_clip=args.grad_clip,
        bf16=not args.no_bf16, seed=args.seed,
        head_hidden=args.head_hidden, head_blocks=args.head_blocks,
        head_dropout=args.head_dropout, backbone_mode=args.backbone_mode,
        feature_mode=args.feature_mode,
        head_mode=args.head_mode, lambda_aux=args.lambda_aux,
        neg_santa_frac=args.neg_santa_frac,
        ckpt_in=args.ckpt_in, ckpt_out=args.ckpt_out,
        snapshots_dir=args.snapshots_dir, history_out=args.history_out,
        log_every=args.log_every,
        neg_frac=args.neg_frac,
        neg_panels=tuple(args.neg_panels),
    )
    train(cfg)


if __name__ == "__main__":
    main()
