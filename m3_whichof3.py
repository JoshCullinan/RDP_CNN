#!/usr/bin/env python3
"""M3 which-of-3: permutation-equivariant Deep Sets head on the M3 dilated trunk.

Step 1.1 (director-approved design). Decides WHICH of 3 aligned sequences is the
recombinant, invariant to input ordering. Extends M3 (a breakpoint localizer
given a KNOWN recombinant) toward *identifying* the recombinant.

Design:
  - Enumerate 3 hypotheses (each candidate = "recomb", other two = "parents"),
    and symmetrize over the 2 parent orders (6-pass, EXACT) because
    ``raw_features`` is asymmetric under P1<->P2 (ch 5-9<->10-14, 15<->16, and
    a MaxChi sign flip on ch 18-21). 6-pass average makes the whole thing
    invariant to all 3! input orderings by construction.
  - Shared trunk (``M3MultiHead.trunk``, reused) per hypothesis.
  - CONTENT-MASKED mean pool + max pool + raw-channel mean (mask gap-only cols
    past content_end) -> reuse ``M3MultiHead.{mean_norm,max_norm,aux_head}`` as
    the shared per-candidate readout -> one logit per hypothesis.
  - Average the 2 parent-order logits per candidate -> 3 candidate logits ->
    softmax -> plain CrossEntropy vs true recombinant index.
  - Random-init trunk, trained jointly with the head. (Warm-start from a
    BASELINE_M3 trunk is experiment #2, not here.)

Why this avoids the M3-v4 aux-gate confound: all 3 candidates share provenance
(same SANTA file / same LANL CRF alignment), so provenance is common-mode across
the 3 candidate logits and cancels in the softmax -- it cannot bias the argmax.

Optimizer: BASELINE recipe (AdamW lr 1e-3, wd 0.01, bf16 autocast,
linear-warmup-cosine, grad-clip 1.0). Loss = plain CE (pos_weight / LABEL_SIGMA
do NOT apply here).

OOM rule: per-triplet streaming (batch = the 6 hypotheses of ONE triplet); never
astype/copy the memmap.
"""
from __future__ import annotations

import argparse
import itertools
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
from m3_dilated import (
    raw_features, M3MultiHead, event_plan, load_event,
    linear_warmup_cosine, _rss_watchdog, _rss,
)
from cache_v2_reader import CacheV2

GAP = 4
NT_TO_INT = {"A": 0, "T": 1, "G": 2, "C": 3, "-": 4}
PERMS = list(itertools.permutations(range(3)))   # the 3! = 6 orderings


def softmax_np(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------- feature construction ----------

def content_len(rows: np.ndarray) -> int:
    """Last content column = min over the 3 seqs of (last non-gap index + 1).

    Positions >= this are trailing gaps and are masked out of the pools
    (matches m3_eval_lanl's content_end convention). Internal gaps are kept.
    """
    L = int(rows.shape[1])
    ce = L
    for r in rows:
        nz = np.where(r != GAP)[0]
        if len(nz):
            ce = min(ce, int(nz[-1]) + 1)
    return max(1, ce)


def hypothesis_features(rows: np.ndarray) -> torch.Tensor:
    """rows: int8 (3, L). Returns (6, L, 22).

    For candidate c in {0,1,2} and parent order o in {0,1}:
      raw_features(rows[c], parents in order o).
    Hypothesis index = c*2 + o; candidate c corresponds to rows[c].
    """
    feats = []
    for c in range(3):
        others = [rows[j] for j in range(3) if j != c]
        feats.append(raw_features(rows[c], others[0], others[1]))   # order 0
        feats.append(raw_features(rows[c], others[1], others[0]))   # order 1
    return torch.stack(feats, dim=0)   # (6, L, 22)


def trunc_rows(rows: np.ndarray, max_len: int) -> np.ndarray:
    if rows.shape[1] > max_len:
        return rows[:, :max_len].copy()
    return rows


# ---------- Deep Sets which-of-3 head ----------

class WhichOf3Net(nn.Module):
    """Permutation-equivariant which-of-3 head.

    Reuses M3MultiHead's trunk + pool norms + aux readout unchanged. The
    ``bp_head`` inside M3MultiHead is unused here (trivial params, kept so the
    module is byte-compatible for a possible warm-start later).

    forward(X: (6, L, 22), cl: int) -> logits (3,)
    """

    def __init__(self, hidden: int = 128, blocks: int = 6, dropout: float = 0.1):
        super().__init__()
        self.core = M3MultiHead(in_channels=22, hidden=hidden,
                                n_blocks=blocks, dropout=dropout)
        self.hidden = hidden

    def _pool(self, h: torch.Tensor, x: torch.Tensor, cl: int) -> torch.Tensor:
        # h: (6, hidden, L)  x: (6, L, 22)  cl: content length
        core = self.core
        cl = min(int(cl), h.shape[-1])
        h_c = h[..., :cl]                                    # (6, hidden, cl)
        mean_pool = core.mean_norm(h_c.mean(dim=-1))         # (6, hidden)
        max_pool = core.max_norm(h_c.amax(dim=-1))           # (6, hidden)
        raw_mean = x[:, :cl, :].mean(dim=1).to(mean_pool.dtype)   # (6, 22)
        return torch.cat([mean_pool, max_pool, raw_mean], dim=-1)  # (6, 2H+22)

    def forward(self, X: torch.Tensor, cl: int) -> torch.Tensor:
        h = self.core.trunk(X)                       # (6, hidden, L)
        pooled = self._pool(h, X, cl)                # (6, 2H+22)
        ell6 = self.core.aux_head(pooled).squeeze(-1)   # (6,)
        ell3 = ell6.view(3, 2).mean(dim=1)           # (3,)  parent-order average
        return ell3


# ---------- checkpoint save / load ----------

def ckpt_dict(net: WhichOf3Net, args, epoch: int, santa_val_acc, global_step: int) -> dict:
    return {
        "net_state": net.state_dict(),
        "arch": {"hidden": args.hidden, "blocks": args.blocks, "dropout": args.dropout},
        "train_cfg": {
            "max_len": args.max_len, "n_train": args.n_train, "epochs": args.epochs,
            "lr": args.lr, "wd": args.wd, "seed": args.seed,
            "train_shards": list(args.train_shards), "val_shards": list(args.val_shards),
            "permute_train": bool(args.permute_train),
            "six_pass": True, "content_masked_pool": True,
        },
        "epoch": epoch, "santa_val_acc": santa_val_acc, "global_step": global_step,
    }


def save_ckpt(net, args, path, epoch, santa_val_acc, global_step) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_dict(net, args, epoch, santa_val_acc, global_step), path)


def best_ckpt_path(ckpt_out: Path) -> Path:
    ckpt_out = Path(ckpt_out)
    return ckpt_out.parent / (ckpt_out.stem + "_best" + ckpt_out.suffix)


def build_net_from_ckpt(ckpt_path: Path, device: str) -> tuple[WhichOf3Net, dict]:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["arch"]
    net = WhichOf3Net(hidden=a["hidden"], blocks=a["blocks"],
                      dropout=a.get("dropout", 0.1)).to(device)
    net.load_state_dict(ck["net_state"])
    net.eval()
    return net, ck


# ---------- inference helper ----------

@torch.no_grad()
def predict_logits(net, rows: np.ndarray, device: str, amp_dtype) -> np.ndarray:
    cl = content_len(rows)
    X = hypothesis_features(rows).to(device)
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
        logits = net(X, cl)
    return logits.float().cpu().numpy()   # (3,)


# ---------- SANTA eval ----------

def eval_santa(net, cache, val_plan, max_len, device, amp_dtype, seed=1234) -> dict:
    """Which-of-3 accuracy on held-out SANTA. Each triplet is given a
    deterministic per-triplet permutation so the recombinant is NOT always at
    index 0 (an always-0 predictor would otherwise score 1.0)."""
    net.eval()
    correct = 0
    n = 0
    for k, (sh, ev) in enumerate(val_plan):
        e = load_event(cache, sh, ev)
        rows = trunc_rows(np.stack([e["R"], e["P1"], e["P2"]]), max_len)
        perm = np.random.default_rng(seed + k).permutation(3)
        label = int(np.where(perm == 0)[0][0])   # position of R (canonical 0)
        pred = int(predict_logits(net, rows[perm], device, amp_dtype).argmax())
        correct += int(pred == label)
        n += 1
        _rss_watchdog(label=f"santa_eval {k}")
    return {"acc": correct / max(1, n), "n": n}


# ---------- GATE (b): permutation invariance across all 6 orderings ----------

def eval_invariance(net, cache, val_plan, max_len, device, amp_dtype) -> dict:
    """Run all 3! = 6 input orderings on every val triplet; report per-ordering
    which-of-3 accuracy and the max-min swing. Should be ~0 by construction."""
    net.eval()
    per_perm_correct = [0] * len(PERMS)
    n = 0
    for k, (sh, ev) in enumerate(val_plan):
        e = load_event(cache, sh, ev)
        rows = trunc_rows(np.stack([e["R"], e["P1"], e["P2"]]), max_len)
        for pi, perm in enumerate(PERMS):
            label = perm.index(0)     # position of R (canonical idx 0) after perm
            pred = int(predict_logits(net, rows[list(perm)], device, amp_dtype).argmax())
            per_perm_correct[pi] += int(pred == label)
        n += 1
        _rss_watchdog(label=f"inv_eval {k}")
    accs = [c / max(1, n) for c in per_perm_correct]
    return {"per_perm_acc": accs, "swing": (max(accs) - min(accs)) if accs else 0.0,
            "n": n}


# ---------- LANL eval (GATE c) + parse check ----------

def _find_recomb_idx(rec_ids: list[str], crf_name: str) -> int:
    """Ported from spectrogram/data.py. CRF02/08/12 use a 'recomb_' prefix;
    CRF07_BC (OpenRDP source) names the recombinant '07_BC' at index 2."""
    for i, rid in enumerate(rec_ids):
        if "recomb" in rid.lower():
            return i
    if crf_name == "CRF07_BC":
        for i, rid in enumerate(rec_ids):
            if "07_BC" in rid or "07BC" in rid or rid == "07_BC":
                return i
        return len(rec_ids) - 1
    return 0


def seq_to_int8(s: str) -> np.ndarray:
    arr = np.empty(len(s), dtype=np.int8)
    su = s.upper()
    for i, ch in enumerate(su):
        arr[i] = NT_TO_INT.get(ch, 4)
    return arr


def load_lanl_triplets(triplet_dir: Path) -> list[tuple]:
    from Bio import SeqIO
    out = []
    for fa in sorted(Path(triplet_dir).glob("*.fa")):
        recs = list(SeqIO.parse(str(fa), "fasta"))
        if len(recs) != 3:
            continue
        crf = fa.stem.split("__")[0]
        rec_ids = [r.id for r in recs]
        ridx = _find_recomb_idx(rec_ids, crf)
        rows = np.stack([seq_to_int8(str(r.seq)) for r in recs])
        out.append((crf, rows, ridx, rec_ids))
    return out


def eval_lanl(net, triplet_dir, device, amp_dtype) -> dict:
    """GATE (c). Per CRF, record argmax AND the full 3-way softmax over
    candidates in SEQUENCE-IDENTITY order (the file's row order), with the true
    index marked. Near-uniform (~0.33 each) => no signal / features washed out;
    confident-but-wrong (~0.7 on a parent) => active mis-identification."""
    net.eval()
    trips = load_lanl_triplets(Path(triplet_dir))
    correct = 0
    per_perm_correct = [0] * len(PERMS)
    details = []
    for crf, rows, ridx, rec_ids in trips:
        logits = predict_logits(net, rows, device, amp_dtype)   # sequence-identity order
        sm = softmax_np(logits)
        pred = int(logits.argmax())
        ok = int(pred == ridx)
        correct += ok
        for pi, perm in enumerate(PERMS):
            label = perm.index(ridx)
            p = int(predict_logits(net, rows[list(perm)], device, amp_dtype).argmax())
            per_perm_correct[pi] += int(p == label)
        details.append({
            "crf": crf,
            "recomb_idx": ridx,                       # true index (into rec_ids order)
            "pred": pred,
            "correct": ok,
            "rec_ids": rec_ids,                       # sequence identities, in order
            "softmax": [float(v) for v in sm],        # candidate probs, same order
            "logits": [float(v) for v in logits],
            "true_prob": float(sm[ridx]),             # prob mass on the true recombinant
        })
    n = len(trips)
    accs = [c / max(1, n) for c in per_perm_correct]
    return {"acc": correct / max(1, n), "n": n,
            "per_perm_acc": accs, "swing": (max(accs) - min(accs)) if accs else 0.0,
            "details": details}


# ---------- training ----------

def rng_perm(rng: random.Random) -> tuple:
    p = [0, 1, 2]
    rng.shuffle(p)
    return tuple(p)


def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    amp_dtype = torch.bfloat16 if not args.no_bf16 else torch.float16
    print(f"[{time.strftime('%H:%M:%S')}] mode={args.mode} device={device} "
          f"amp={amp_dtype}", flush=True)

    cache = CacheV2()
    train_plan = event_plan(cache, args.train_shards, args.n_train, rng, args.max_len)
    val_plan = event_plan(cache, args.val_shards, args.n_val, rng, args.max_len)
    if not train_plan:
        raise SystemExit("empty train plan")
    print(f"  train: {len(train_plan)} triplets  val: {len(val_plan)} triplets",
          flush=True)

    net = WhichOf3Net(hidden=args.hidden, blocks=args.blocks,
                      dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    n_train_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  net params: {n_params:,}  trainable: {n_train_p:,}", flush=True)

    optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train_plan)
    warmup = args.warmup_steps if args.warmup_steps >= 0 else max(1, total_steps // 10)

    history = {"epochs": [], "config": {
        "mode": args.mode, "n_train": len(train_plan), "n_val": len(val_plan),
        "max_len": args.max_len, "epochs": args.epochs, "lr": args.lr,
        "wd": args.wd, "hidden": args.hidden, "blocks": args.blocks,
        "seed": args.seed, "six_pass": True, "content_masked_pool": True,
        "permute_train": bool(args.permute_train)}}

    best_val_acc = -1.0
    best_epoch = -1
    best_path = best_ckpt_path(args.ckpt_out) if args.ckpt_out else None
    global_step = 0
    epoch_times = []
    for ep in range(1, args.epochs + 1):
        net.train()
        order = list(range(len(train_plan)))
        rng.shuffle(order)
        ep_loss = 0.0
        ep_correct = 0
        ep_n = 0
        t_ep = time.time()
        for step_in_ep, idx in enumerate(order):
            sh, ev = train_plan[idx]
            e = load_event(cache, sh, ev)
            rows = trunc_rows(np.stack([e["R"], e["P1"], e["P2"]]), args.max_len)
            if args.permute_train:
                perm = rng_perm(rng)
                rows = rows[list(perm)]
                label = int(perm.index(0))
            else:
                label = 0

            lr_mult = linear_warmup_cosine(global_step, warmup, total_steps)
            for g in optim.param_groups:
                g["lr"] = args.lr * lr_mult

            cl = content_len(rows)
            X = hypothesis_features(rows).to(device)
            target = torch.tensor([label], device=device)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                logits = net(X, cl)                       # (3,)
                loss = F.cross_entropy(logits[None], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optim.step()

            ep_loss += float(loss.item())
            ep_correct += int(logits.detach().float().argmax().item() == label)
            ep_n += 1
            global_step += 1
            if global_step % args.log_every == 0:
                gpu = (torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0)
                print(f"    step {global_step:>6}  loss {loss.item():.4f}  "
                      f"lr {args.lr*lr_mult:.2e}  RSS {_rss()/2**30:.1f}GB  "
                      f"GPU {gpu:.2f}GB", flush=True)
            _rss_watchdog(label=f"train ep{ep}/s{step_in_ep}")

        elapsed = time.time() - t_ep
        epoch_times.append(elapsed)
        train_loss = ep_loss / max(1, ep_n)
        train_acc = ep_correct / max(1, ep_n)
        print(f"  epoch {ep:>2}  train_loss {train_loss:.4f}  train_acc "
              f"{train_acc:.3f}  ({elapsed:.0f}s, {elapsed/max(1,ep_n)*1000:.0f} ms/triplet)",
              flush=True)
        ep_rec = {"epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
                  "elapsed_s": elapsed}

        # Per-epoch SANTA val for model selection + best snapshot.
        if val_plan and args.val_every and (ep % args.val_every == 0):
            vs = eval_santa(net, cache, val_plan, args.max_len, device, amp_dtype)
            ep_rec["santa_val_acc"] = vs["acc"]
            print(f"    [val] SANTA which-of-3 acc = {vs['acc']:.3f} (n={vs['n']})",
                  flush=True)
            if vs["acc"] > best_val_acc:
                best_val_acc = vs["acc"]
                best_epoch = ep
                if best_path:
                    save_ckpt(net, args, best_path, ep, vs["acc"], global_step)
                    print(f"    new best SANTA val acc {best_val_acc:.3f} -> {best_path}",
                          flush=True)

        history["epochs"].append(ep_rec)
        if args.history_out:
            Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.history_out, "w") as f:
                json.dump(history, f, indent=2)

    # ---- save FINAL (end-of-training) checkpoint ----
    if args.ckpt_out:
        save_ckpt(net, args, args.ckpt_out, args.epochs,
                  best_val_acc if best_val_acc >= 0 else None, global_step)
        print(f"\n  saved FINAL ckpt -> {args.ckpt_out}", flush=True)
        if best_path and best_epoch > 0:
            print(f"  best SANTA-val ckpt (ep{best_epoch}, acc {best_val_acc:.3f}) "
                  f"-> {best_path}", flush=True)

    # ---- bounded validations (on the FINAL net) ----
    report = run_evals(net, cache, val_plan, args, device, amp_dtype,
                       epoch_times=epoch_times, history=history["epochs"])
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  report -> {args.report_out}", flush=True)


# ---------- shared eval driver (train end + eval-ckpt mode) ----------

def run_evals(net, cache, val_plan, args, device, amp_dtype,
              epoch_times=None, history=None) -> dict:
    report = {}
    if epoch_times is not None:
        report["epoch_times_s"] = epoch_times
        report["mean_epoch_s"] = float(np.mean(epoch_times)) if epoch_times else None
    if history is not None:
        report["history"] = history

    if args.eval_santa and val_plan:
        t0 = time.time()
        s = eval_santa(net, cache, val_plan, args.max_len, device, amp_dtype)
        s["elapsed_s"] = time.time() - t0
        report["santa"] = s
        print(f"\n[GATE a] SANTA which-of-3 acc = {s['acc']:.3f} "
              f"(n={s['n']}, chance=0.333, {s['elapsed_s']:.0f}s)", flush=True)

    if args.eval_invariance and val_plan:
        t0 = time.time()
        inv = eval_invariance(net, cache, val_plan, args.max_len, device, amp_dtype)
        inv["elapsed_s"] = time.time() - t0
        report["invariance"] = inv
        print(f"[GATE b] permutation invariance: per-ordering acc = "
              f"{[round(a,4) for a in inv['per_perm_acc']]}", flush=True)
        print(f"[GATE b] max-min swing across 6 orderings = {inv['swing']:.5f} "
              f"(target <= 0.01; ~0 by construction)", flush=True)

    if args.eval_lanl:
        t0 = time.time()
        lanl = eval_lanl(net, args.lanl_dir, device, amp_dtype)
        lanl["elapsed_s"] = time.time() - t0
        report["lanl"] = lanl
        print(f"[GATE c] LANL which-of-3 acc = {lanl['acc']:.3f} "
              f"(n={lanl['n']}, swing={lanl['swing']:.5f})  "
              f"[n=4 => statistically UNINFORMATIVE alone]", flush=True)
        for d in lanl["details"]:
            sm = ", ".join(f"{i}:{p:.2f}{'*' if i == d['recomb_idx'] else ''}"
                           for i, p in enumerate(d["softmax"]))
            print(f"    {d['crf']}: true_idx={d['recomb_idx']} pred={d['pred']} "
                  f"{'OK' if d['correct'] else 'X'}  softmax[{sm}]  "
                  f"true_prob={d['true_prob']:.3f}  ids={d['rec_ids']}", flush=True)
    return report


def run_eval_ckpt(args) -> None:
    """Load a saved ckpt and run the requested evals WITHOUT retraining."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if not args.no_bf16 else torch.float16
    if not args.ckpt_in or not Path(args.ckpt_in).exists():
        raise SystemExit(f"--ckpt-in required and must exist (got {args.ckpt_in})")
    net, ck = build_net_from_ckpt(args.ckpt_in, device)
    print(f"[{time.strftime('%H:%M:%S')}] eval-ckpt device={device} amp={amp_dtype}",
          flush=True)
    print(f"  loaded {args.ckpt_in}  arch={ck['arch']}  epoch={ck.get('epoch')}  "
          f"santa_val_acc={ck.get('santa_val_acc')}", flush=True)

    # Default to LANL eval if the user asked for nothing specific.
    if not (args.eval_lanl or args.eval_santa or args.eval_invariance):
        args.eval_lanl = True

    cache = None
    val_plan = []
    if args.eval_santa or args.eval_invariance:
        cache = CacheV2()
        val_plan = event_plan(cache, args.val_shards, args.n_val,
                              random.Random(args.seed), args.max_len)
        print(f"  val: {len(val_plan)} triplets", flush=True)

    report = run_evals(net, cache, val_plan, args, device, amp_dtype)
    report["ckpt_in"] = str(args.ckpt_in)
    report["ckpt_epoch"] = ck.get("epoch")
    report["ckpt_santa_val_acc"] = ck.get("santa_val_acc")
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  report -> {args.report_out}", flush=True)


def check_lanl_parse(triplet_dir) -> None:
    """No-model sanity: confirm the 4 LANL triplets parse and recomb indices
    resolve (esp. the CRF07_BC index-2 special case)."""
    trips = load_lanl_triplets(Path(triplet_dir))
    print(f"[LANL parse check] {len(trips)} triplets in {triplet_dir}")
    for crf, rows, ridx, rec_ids in trips:
        L = rows.shape[1]
        cl = content_len(rows)
        print(f"    {crf}: L={L} content_len={cl} recomb_idx={ridx} "
              f"(-> '{rec_ids[ridx]}')  ids={rec_ids}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke",
                    choices=["smoke", "timing", "train", "eval-ckpt", "lanl-parse"])
    ap.add_argument("--train-shards", nargs="+", default=["XML-1", "XML-2", "XML-3", "XML-4"])
    ap.add_argument("--val-shards", nargs="+", default=["XML-5"])
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--max-len", type=int, default=11_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=-1)   # -1 => total//10
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=1,
                    help="run SANTA val (model selection + best snapshot) every N "
                         "epochs; 0 disables per-epoch val")
    ap.add_argument("--permute-train", action="store_true",
                    help="random row permutation during training (a no-op for "
                         "invariance; light regularizer)")
    ap.add_argument("--eval-santa", action="store_true")
    ap.add_argument("--eval-invariance", action="store_true")
    ap.add_argument("--eval-lanl", action="store_true")
    ap.add_argument("--lanl-dir", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/triplets"))
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-out", type=Path, default=Path("models_test/whichof3_s0.pt"),
                    help="final-model ckpt path; best-SANTA-val snapshot goes to "
                         "<stem>_best<suffix>")
    ap.add_argument("--ckpt-in", type=Path, default=None,
                    help="for --mode eval-ckpt: load this ckpt and eval without training")
    ap.add_argument("--history-out", type=Path, default=None)
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args()

    if args.mode == "lanl-parse":
        check_lanl_parse(args.lanl_dir)
        return
    if args.mode == "eval-ckpt":
        run_eval_ckpt(args)
        return
    if args.mode == "smoke":
        args.eval_santa = True
        args.eval_invariance = True
    if args.mode == "timing":
        args.epochs = 1

    train(args)


if __name__ == "__main__":
    main()
