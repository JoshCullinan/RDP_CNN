"""M1.2 — MLM pretraining for HyenaDNA on filtered SANTA.

Pretrains HyenaDNA-small-32k with bidirectional masked language modeling
on splits/v2_filtered_split.json TRAIN, evaluates on VAL. The released
HyenaDNA backbone is causal (autoregressive); we get bidirectional
context by running two passes — forward and reverse-complement — and
concatenating hidden states `[h_fwd[i] || h_rc[L-1-i]]` before a fresh
5-way head over {A, T, G, C, gap}.

Masking is BERT-style: 15% of non-gap positions, with 80% → [MASK]=3,
10% → random nucleotide, 10% unchanged. Gaps are never masked because
"predict gap at gap" trains alignment topology, not nucleotide stats.

Critical: the released causal `lm_head` is NOT reused. The new MLM head
is initialized fresh because the zero-shot probe confirmed HyenaDNA's
pretraining has no MLM ability (token 3 = [MASK] has essentially
untrained embedding from the causal pretraining).

The smoke-test mode (`--smoke`) runs 1 epoch on a 4 kb subset (~30 min
target on the 3070). The decision gate for the full ~10-day run is:

  - val MLM acc moves 0.31 → 0.45+  : data has entropy headroom, launch
    full curriculum.
  - val MLM acc stays pinned ~0.31   : MLM is structurally blocked on
    this backbone, pivot to causal LM (option a from the handover).

The 0.308 zero-shot probe ceiling is the majority-class A-prior, not
an entropy ceiling — a fine-tuned model should clear it easily. If it
doesn't, that's a real signal, not noise.

Curriculum (full run):
  Epochs 1-5  : 4 kb only       (XML-2 + XML-4)
  Epochs 6-15 : + 10 kb         (+ XML-6)
  Epochs 16-30: + 30 kb         (+ long_content_30k_002)

Memory discipline: per-event streaming, never materialise the full
TRAIN tensor in RAM. RSS watchdog at 26 GB (see CLAUDE.md /
feedback_padding_mask_oom.md).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_hyenadna import (
    DEFAULT_HF_NAME,
    SequenceBackbone,
    BackboneConfig,
    v2_to_hyena_ids,
)
from cache_v2_reader import CacheV2


# ---------- RSS watchdog ----------------------------------------------------

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


# ---------- vocab ----------------------------------------------------------

# v2 cache encoding: A=0 T=1 G=2 C=3 gap=4 (5-way target classes).
# HyenaDNA vocab: [CLS]=0 [SEP]=1 [BOS]=2 [MASK]=3 [PAD]=4 [RES]=5
#                 [UNK]=6 A=7 C=8 G=9 T=10 N=11.
HY_MASK = 3
HY_PAD = 4
HY_A, HY_C, HY_G, HY_T = 7, 8, 9, 10
HY_NT_TOKENS = (HY_A, HY_C, HY_G, HY_T, HY_PAD)  # gap-as-PAD is class 4 here

# v2 reverse-complement lookup over {A,T,G,C,gap} → {T,A,C,G,gap}.
_V2_RC = torch.tensor([1, 0, 3, 2, 4], dtype=torch.long)

# HyenaDNA reverse-complement lookup over the full 12-vocab; identity
# elsewhere so [MASK] stays [MASK], [PAD] stays [PAD].
_HY_RC = torch.arange(12, dtype=torch.long)
_HY_RC[HY_A] = HY_T
_HY_RC[HY_T] = HY_A
_HY_RC[HY_G] = HY_C
_HY_RC[HY_C] = HY_G


def hyena_to_v2_class(hy_ids: torch.Tensor) -> torch.Tensor:
    """Map HyenaDNA ids back to v2 5-class targets (A=0,T=1,G=2,C=3,gap=4).

    Anything outside the nt set goes to -100 (ignore_index in CE).
    """
    out = torch.full_like(hy_ids, -100, dtype=torch.long)
    out[hy_ids == HY_A] = 0
    out[hy_ids == HY_T] = 1
    out[hy_ids == HY_G] = 2
    out[hy_ids == HY_C] = 3
    out[hy_ids == HY_PAD] = 4
    return out


# ---------- model ----------------------------------------------------------

class BidirMLM(nn.Module):
    """HyenaDNA + reverse-complement dual-pass + fresh 5-way MLM head.

    Output: (B, L, 5) logits over {A,T,G,C,gap}.

    Bidirectional context is obtained by running the causal backbone twice:
    once on the forward sequence, once on the reverse-complemented sequence.
    Hidden states from the RC pass are flipped along L so position i in both
    streams sees the same nucleotide's left/right context.
    """

    def __init__(self, backbone: SequenceBackbone, d_model: int = 256,
                 n_classes: int = 5) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(2 * d_model, n_classes)

    def forward(self, fwd_ids: torch.Tensor, rc_ids: torch.Tensor
                ) -> torch.Tensor:
        h_fwd = self.backbone(fwd_ids, is_hyena_ids=True)            # (B, L, D)
        h_rc = self.backbone(rc_ids, is_hyena_ids=True)              # (B, L, D)
        h_rc = h_rc.flip(dims=[1])                                   # align frames
        combined = torch.cat([h_fwd, h_rc], dim=-1)                  # (B, L, 2D)
        return self.head(combined)


# ---------- masking --------------------------------------------------------

def make_mlm_batch(v2_ids: np.ndarray, mask_prob: float,
                   rng: np.random.Generator, device: str
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a single MLM batch.

    Inputs
        v2_ids: (B, L) int8 in {0..4} — v2 cache nucleotides.

    Returns
        fwd_ids:    (B, L) int64 HyenaDNA ids, masked with 80/10/10 BERT scheme.
        rc_ids:     (B, L) int64 HyenaDNA ids, reverse-complement, masked.
        targets:    (B, L) int64 — true 5-class label at masked positions,
                    -100 elsewhere (CE ignore_index).
        loss_mask:  (B, L) bool — True where targets is real (== mask positions).

    Mask is over non-gap positions only. Same mask positions for fwd and rc
    passes (mirrored along L); independent 80/10/10 draws are fine.
    """
    v2_t = torch.from_numpy(np.asarray(v2_ids, dtype=np.int64))      # (B, L)
    B, L = v2_t.shape

    # Map v2 → HyenaDNA for both forward and RC sequences.
    fwd_hy = v2_to_hyena_ids(v2_t)                                   # (B, L) int64
    v2_rc = _V2_RC[v2_t.clamp(0, 4)].flip(dims=[1])                  # (B, L)
    rc_hy = v2_to_hyena_ids(v2_rc)

    non_gap = (v2_t != 4).numpy()                                    # (B, L)
    targets_v2 = v2_t.clone()                                        # for targets

    # Pick masked positions: per-sequence, sample mask_prob * non_gap count.
    mask_bool = np.zeros((B, L), dtype=bool)
    for b in range(B):
        idx = np.where(non_gap[b])[0]
        if len(idx) == 0:
            continue
        n_mask = max(1, int(round(mask_prob * len(idx))))
        choice = rng.choice(idx, size=n_mask, replace=False)
        mask_bool[b, choice] = True
    mask_t = torch.from_numpy(mask_bool)                             # (B, L)
    rc_mask_t = mask_t.flip(dims=[1])                                # mirror for RC frame

    # 80/10/10 perturbation, independent for fwd and rc.
    def _perturb(ids: torch.Tensor, mpos: torch.Tensor) -> torch.Tensor:
        out = ids.clone()
        if not mpos.any():
            return out
        u = torch.from_numpy(rng.random(ids.shape).astype(np.float32))
        mask_pos = mpos & (u < 0.8)
        rand_pos = mpos & (u >= 0.8) & (u < 0.9)
        # 80%: set to [MASK]
        out[mask_pos] = HY_MASK
        # 10%: set to random nucleotide (uniform over A/C/G/T — keep gap-as-PAD out)
        if rand_pos.any():
            n_r = int(rand_pos.sum().item())
            choices = torch.tensor([HY_A, HY_C, HY_G, HY_T], dtype=torch.long)
            picks = choices[torch.from_numpy(rng.integers(0, 4, size=n_r))]
            out[rand_pos] = picks
        # 10% unchanged: nothing to do
        return out

    fwd_ids = _perturb(fwd_hy, mask_t).to(device)
    rc_ids = _perturb(rc_hy, rc_mask_t).to(device)

    targets = torch.full((B, L), -100, dtype=torch.long)
    targets[mask_t] = targets_v2[mask_t]                             # already in {0..4}
    targets = targets.to(device)
    loss_mask = mask_t.to(device)
    return fwd_ids, rc_ids, targets, loss_mask


# ---------- data ----------------------------------------------------------

@dataclass
class CurriculumStage:
    name: str
    shards: list[str]
    max_len: int
    batch_size: int = 2                  # B=1 at 30 kb on 8 GB VRAM; B=2 fits at ≤19 kb


def stages_for_curriculum(stage: str) -> list[CurriculumStage]:
    """Stages match the measured per-shard seq_len:
        10 kb: XML-2 (9181) + XML-6 (10000) — TRAIN; XML-5 (10800) + XML-6 — VAL
        19 kb: + XML-4 (19025)
        30 kb: + long_content_30k_002 / _001
    'smoke' is a small 10 kb stage cap.
    """
    # Per-stage batch sizes: keep L * B token budget ≈ constant so VRAM stays in
    # safe range at 19 k and 30 k (B=8 OOMs at 19 k once allocator working set fills).
    if stage == "smoke" or stage == "A":
        return [CurriculumStage("10k", ["XML-2", "XML-6"], max_len=11000, batch_size=8)]
    if stage == "B":
        return [
            CurriculumStage("10k", ["XML-2", "XML-6"], max_len=11000, batch_size=8),
            CurriculumStage("19k", ["XML-4"], max_len=19500, batch_size=4),
        ]
    if stage == "C":
        return [
            CurriculumStage("10k", ["XML-2", "XML-6"], max_len=11000, batch_size=8),
            CurriculumStage("19k", ["XML-4"], max_len=19500, batch_size=4),
            CurriculumStage("30k", ["long_content_30k_002"], max_len=30500, batch_size=2),
        ]
    raise ValueError(f"unknown stage: {stage!r}")


def val_shards_for_stage(stage: CurriculumStage) -> list[str]:
    """VAL shards that mirror the TRAIN stage in length."""
    if stage.name == "10k":
        return ["XML-5", "XML-6"]
    if stage.name == "19k":
        return ["XML-5", "XML-6"]            # no 19 kb VAL shard exists; use 10 kb proxies
    if stage.name == "30k":
        return ["long_content_30k_001"]
    return []


def iter_aligned_blocks(cache: CacheV2, files_by_shard: dict[str, list[str]],
                        max_len: int, max_files: int | None,
                        rng: random.Random):
    """Yield (shard_name, file_idx, alignment_int8) — one alignment per call.

    Memmap-backed access; no full-tensor materialisation. Caller batches.
    Files are visited in shuffled order.
    """
    plan: list[tuple[str, int]] = []
    for shard_name, fn_list in files_by_shard.items():
        if shard_name not in cache.shards:
            continue
        shard = cache.shards[shard_name]
        for fn in fn_list:
            try:
                file_idx = shard.files.index(fn)
            except ValueError:
                continue
            plan.append((shard_name, file_idx))
    rng.shuffle(plan)
    if max_files is not None:
        plan = plan[:max_files]

    for shard_name, file_idx in plan:
        shard = cache.shards[shard_name]
        a = shard.get_alignment(file_idx)
        if a.shape[1] > max_len + 200:
            # length-bucketed: skip alignments well outside this stage's bucket
            continue
        if a.shape[1] > max_len:
            a = a[:, :max_len]
        yield shard_name, file_idx, a


# ---------- training step --------------------------------------------------

def _crop_to_max_L(seqs: list[np.ndarray], max_L: int) -> list[np.ndarray]:
    out = []
    for s in seqs:
        if len(s) > max_L:
            out.append(s[:max_L])
        else:
            out.append(s)
    return out


def _pad_batch(seqs: list[np.ndarray]) -> np.ndarray:
    """Pad list of 1-D int8 sequences to (B, max_L) with gap (=4)."""
    max_L = max(len(s) for s in seqs)
    out = np.full((len(seqs), max_L), 4, dtype=np.int8)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


@dataclass
class TrainConfig:
    splits_path: Path = Path("splits/v2_filtered_split.json")
    cache_root: Path | None = None       # None → CacheV2 auto-discovers sibling of dataRaw/
    out_dir: Path = Path("models_test")
    history_path: Path = Path("models_test/history_mlm_v1.json")
    ckpt_path: Path = Path("models_test/backbone_mlm_v1.pt")

    stage: str = "smoke"               # 'smoke' | 'A' | 'B' | 'C'
    epochs: int = 1
    batch_size: int = 4                # for 4k. Drops to 1 at 30k.
    seqs_per_file: int = 16            # MLM tasks per FASTA per epoch (random rows)
    mask_prob: float = 0.15
    lr: float = 1e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    mixed_precision: bool = True
    seed: int = 0
    max_files: int | None = None       # cap for smoke / debug
    val_max_files: int | None = 32
    val_seqs_per_file: int = 8
    log_every: int = 25

    pretrained: bool = True
    hf_name: str = DEFAULT_HF_NAME
    resume: bool = False                  # load model/optim/epoch from ckpt_path if it exists
    grad_ckpt: bool = True                # gradient checkpointing on the HyenaDNA backbone
    bf16: bool = False                    # use bf16 instead of fp16 (no GradScaler needed)


def linear_warmup_cosine(step: int, warmup: int, total: int,
                         min_frac: float = 0.1) -> float:
    """Linear warmup → cosine decay to min_frac of peak LR (so we never floor at 0)."""
    if step < warmup:
        return step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return min_frac + (1.0 - min_frac) * cos


def evaluate_mlm(model: BidirMLM, cache: CacheV2, val_files: dict[str, list[str]],
                 stage: CurriculumStage, cfg: TrainConfig, device: str,
                 rng: random.Random, np_rng: np.random.Generator) -> dict:
    model.eval()
    total_correct = 0
    total_seen = 0
    total_loss = 0.0
    total_loss_n = 0
    n_files = 0
    t0 = time.time()
    with torch.no_grad():
        for shard_name, file_idx, align in iter_aligned_blocks(
                cache, val_files, stage.max_len, cfg.val_max_files, rng):
            n_files += 1
            n_rows = align.shape[0]
            picks = list(range(n_rows))
            rng.shuffle(picks)
            picks = picks[: cfg.val_seqs_per_file]

            bs = cfg.batch_size
            for b_start in range(0, len(picks), bs):
                batch_rows = picks[b_start:b_start + bs]
                seqs = [align[r] for r in batch_rows]
                seqs = _crop_to_max_L(seqs, stage.max_len)
                v2_ids = _pad_batch(seqs)
                fwd, rc, tgt, _ = make_mlm_batch(v2_ids, cfg.mask_prob,
                                                  np_rng, device)
                logits = model(fwd, rc)                              # (B, L, 5)
                loss = F.cross_entropy(logits.reshape(-1, 5),
                                       tgt.reshape(-1),
                                       ignore_index=-100, reduction="sum")
                n_scored = int((tgt != -100).sum().item())
                if n_scored:
                    pred = logits.argmax(dim=-1)
                    mask = tgt != -100
                    correct = int(((pred == tgt) & mask).sum().item())
                    total_correct += correct
                    total_seen += n_scored
                    total_loss += float(loss.item())
                    total_loss_n += n_scored
            _rss_watchdog(label=f"val after file {n_files}")
    elapsed = time.time() - t0
    acc = total_correct / total_seen if total_seen else float("nan")
    mean_loss = total_loss / total_loss_n if total_loss_n else float("nan")
    return {
        "val_acc": acc,
        "val_loss": mean_loss,
        "val_n_positions": total_seen,
        "val_n_files": n_files,
        "val_elapsed_s": elapsed,
    }


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    rng = random.Random(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] device={device}", flush=True)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    splits = json.load(cfg.splits_path.open())
    train_files = {
        sh: sd.get("files", []) for sh, sd in splits["splits"]["TRAIN"]["dirs"].items()
    }
    val_files = {
        sh: sd.get("files", []) for sh, sd in splits["splits"]["VAL"]["dirs"].items()
    }
    print(f"  loaded {sum(len(v) for v in train_files.values()):,} TRAIN files / "
          f"{sum(len(v) for v in val_files.values()):,} VAL files", flush=True)

    cache = CacheV2(cfg.cache_root)
    print(f"  cache shards: {sorted(cache.shards.keys())}", flush=True)

    print(f"  building model (pretrained={cfg.pretrained})", flush=True)
    backbone = SequenceBackbone(BackboneConfig(hf_name=cfg.hf_name),
                                pretrained=cfg.pretrained)
    d_model = backbone.cfg.d_model
    model = BidirMLM(backbone, d_model=d_model, n_classes=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"  params: total={n_params:,}  head={n_head:,}", flush=True)

    if cfg.grad_ckpt and hasattr(model.backbone.hyena, "gradient_checkpointing_enable"):
        # use_reentrant=False is the modern recommended path; HF defaults to it.
        model.backbone.hyena.gradient_checkpointing_enable()
        print(f"  gradient_checkpointing: ENABLED on HyenaDNA backbone", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    use_scaler = cfg.mixed_precision and device == "cuda" and not cfg.bf16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if cfg.mixed_precision and device == "cuda":
        print(f"  AMP: dtype={amp_dtype}  scaler={'on' if use_scaler else 'off'}",
              flush=True)

    stages = stages_for_curriculum(cfg.stage)
    print(f"  curriculum stages: {[s.name for s in stages]}", flush=True)

    history: dict = {
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in cfg.__dict__.items()},
        "epochs": [],
    }

    # Better total-steps estimate using per-stage shard sizes (drives the LR schedule).
    stage_steps_per_epoch: dict[str, int] = {}
    epochs_per_stage_letter = {"smoke": [1], "A": [1] * cfg.epochs,
                               "B": [1] * cfg.epochs, "C": [1] * cfg.epochs}
    total_steps_calc = 0
    for st in stages:
        files_in_stage = sum(len(train_files.get(sh, [])) for sh in st.shards)
        if cfg.max_files is not None:
            files_in_stage = min(files_in_stage, cfg.max_files)
        batches_per_file = max(1, cfg.seqs_per_file // st.batch_size)
        steps_per_epoch_this_stage = files_in_stage * batches_per_file
        stage_steps_per_epoch[st.name] = steps_per_epoch_this_stage
        total_steps_calc += steps_per_epoch_this_stage * cfg.epochs
    total_steps_est = max(1, total_steps_calc)
    print(f"  per-stage steps/epoch: {stage_steps_per_epoch}", flush=True)

    global_step = 0
    start_epoch = 1
    if cfg.resume and cfg.ckpt_path.exists():
        ck = torch.load(cfg.ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optim.load_state_dict(ck["optim_state"])
        start_epoch = int(ck["epoch"]) + 1
        global_step = int(ck["global_step"])
        if cfg.history_path.exists():
            history = json.load(cfg.history_path.open())
        print(f"  RESUMED from {cfg.ckpt_path}: epoch={start_epoch} "
              f"global_step={global_step}", flush=True)
    print(f"  total_steps estimate: {total_steps_est:,}", flush=True)

    for epoch in range(start_epoch, cfg.epochs + 1):
        for stage_idx, stage in enumerate(stages):
            # Clear cached memory before each new stage — sequence length jumps
            # change the allocator's working set, and reserved-but-unused blocks
            # from the previous stage cause fragmentation OOMs.
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            print(f"\n[{time.strftime('%H:%M:%S')}] === epoch {epoch} stage {stage.name} "
                  f"(shards: {stage.shards}, max_len={stage.max_len}) ===", flush=True)
            model.train()
            t_epoch = time.time()
            epoch_loss_sum = 0.0
            epoch_loss_n = 0
            epoch_correct = 0
            epoch_seen = 0
            n_files = 0

            stage_train_files = {sh: train_files.get(sh, []) for sh in stage.shards}
            for shard_name, file_idx, align in iter_aligned_blocks(
                    cache, stage_train_files, stage.max_len, cfg.max_files, rng):
                n_files += 1
                n_rows = align.shape[0]
                picks = list(range(n_rows))
                rng.shuffle(picks)
                picks = picks[: cfg.seqs_per_file]

                bs = stage.batch_size
                for b_start in range(0, len(picks), bs):
                    batch_rows = picks[b_start:b_start + bs]
                    seqs = [align[r] for r in batch_rows]
                    seqs = _crop_to_max_L(seqs, stage.max_len)
                    v2_ids = _pad_batch(seqs)
                    fwd, rc, tgt, _ = make_mlm_batch(
                        v2_ids, cfg.mask_prob, np_rng, device)

                    # LR schedule
                    lr_mult = linear_warmup_cosine(global_step, cfg.warmup_steps,
                                                   total_steps_est)
                    for g in optim.param_groups:
                        g["lr"] = cfg.lr * lr_mult

                    optim.zero_grad(set_to_none=True)
                    amp_ctx = (torch.amp.autocast("cuda", dtype=amp_dtype)
                               if cfg.mixed_precision and device == "cuda"
                               else contextlib.nullcontext())
                    with amp_ctx:
                        logits = model(fwd, rc)
                        loss = F.cross_entropy(logits.reshape(-1, 5),
                                               tgt.reshape(-1),
                                               ignore_index=-100)
                    if use_scaler:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        scaler.step(optim)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        optim.step()

                    with torch.no_grad():
                        pred = logits.argmax(dim=-1)
                        m = tgt != -100
                        n_pos = int(m.sum().item())
                        if n_pos:
                            c = int(((pred == tgt) & m).sum().item())
                            epoch_correct += c
                            epoch_seen += n_pos
                            epoch_loss_sum += float(loss.item()) * n_pos
                            epoch_loss_n += n_pos

                    global_step += 1
                    if global_step % cfg.log_every == 0:
                        train_acc = epoch_correct / max(1, epoch_seen)
                        rss = _current_rss_bytes() / 2**30
                        gpu = (torch.cuda.max_memory_allocated() / 2**30
                               if device == "cuda" else 0.0)
                        print(f"    step {global_step:6d}  loss {loss.item():.4f}  "
                              f"train_acc {train_acc:.3f}  lr {cfg.lr * lr_mult:.2e}  "
                              f"RSS {rss:.1f} GB  GPU {gpu:.2f} GB",
                              flush=True)
                _rss_watchdog(label=f"train after file {n_files}")

            train_acc = epoch_correct / max(1, epoch_seen)
            train_loss = epoch_loss_sum / max(1, epoch_loss_n)
            elapsed = time.time() - t_epoch
            print(f"  [stage {stage.name} done] files={n_files} steps_this_stage~"
                  f"{global_step} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"elapsed={elapsed:.1f}s", flush=True)

            # Validate on matching VAL shards
            stage_val_files = {sh: val_files.get(sh, []) for sh in val_shards_for_stage(stage)}
            val_metrics = evaluate_mlm(model, cache, stage_val_files, stage, cfg,
                                       device, rng, np_rng)
            print(f"  VAL: acc={val_metrics['val_acc']:.3f}  "
                  f"loss={val_metrics['val_loss']:.4f}  "
                  f"n_pos={val_metrics['val_n_positions']:,}  "
                  f"n_files={val_metrics['val_n_files']}  "
                  f"({val_metrics['val_elapsed_s']:.1f}s)", flush=True)

            history["epochs"].append({
                "epoch": epoch,
                "stage": stage.name,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "global_step": global_step,
                "elapsed_s": elapsed,
                **val_metrics,
            })
            with cfg.history_path.open("w") as f:
                json.dump(history, f, indent=2)

            # Save per-stage so a transition-OOM doesn't waste an hour.
            torch.save({
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "epoch": epoch,
                "stage_idx_done": stage_idx,
                "global_step": global_step,
                "cfg": history["config"],
            }, cfg.ckpt_path)
            print(f"  ckpt saved → {cfg.ckpt_path} "
                  f"(epoch {epoch} stage {stage.name})", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] training complete.", flush=True)


# ---------- CLI ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("splits/v2_filtered_split.json"))
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="defaults to CacheV2 auto-discovery (sibling of dataRaw/)")
    ap.add_argument("--out-dir", type=Path, default=Path("models_test"))
    ap.add_argument("--history", type=Path, default=Path("models_test/history_mlm_v1.json"))
    ap.add_argument("--ckpt", type=Path, default=Path("models_test/backbone_mlm_v1.pt"))
    ap.add_argument("--stage", choices=["smoke", "A", "B", "C"], default="smoke")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seqs-per-file", type=int, default=16)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--val-max-files", type=int, default=32)
    ap.add_argument("--val-seqs-per-file", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed-precision (debug)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick smoke: stage=smoke, max_files=20, epochs=1")
    ap.add_argument("--resume", action="store_true",
                    help="resume from ckpt_path if it exists (model/optim/epoch)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing on the HyenaDNA backbone")
    ap.add_argument("--bf16", action="store_true",
                    help="use bf16 autocast instead of fp16 (no GradScaler needed)")
    args = ap.parse_args()

    if args.smoke:
        args.stage = "smoke"
        args.epochs = 1
        if args.max_files is None:
            args.max_files = 20
        args.val_max_files = 8

    cfg = TrainConfig(
        splits_path=args.splits,
        cache_root=args.cache_root,
        out_dir=args.out_dir,
        history_path=args.history,
        ckpt_path=args.ckpt,
        stage=args.stage,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seqs_per_file=args.seqs_per_file,
        mask_prob=args.mask_prob,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        max_files=args.max_files,
        val_max_files=args.val_max_files,
        val_seqs_per_file=args.val_seqs_per_file,
        log_every=args.log_every,
        pretrained=not args.no_pretrained,
        mixed_precision=not args.no_amp,
        resume=args.resume,
        grad_ckpt=not args.no_grad_ckpt,
        bf16=args.bf16,
    )
    train(cfg)


if __name__ == "__main__":
    main()
