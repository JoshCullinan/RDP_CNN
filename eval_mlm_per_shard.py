"""Per-shard MLM val on CPU — does NOT touch the running training process.

Loads the latest per-stage checkpoint from `models_test/backbone_mlm_v1.pt`,
reconstructs the BidirMLM model, and validates separately on:

    - XML-5             (10 kb,  VAL-only, no matched TRAIN sibling)
    - XML-6 (val files) (10 kb,  matched TRAIN distribution — only file-level held out)
    - long_content_30k_001 (30 kb, sister config of train's _30k_002)

If matched-distribution val (XML-6) accuracy is much higher than mismatched
(XML-5), the headline val asymptote is suppressed by distribution shift.

CPU-only intentionally — the GPU is pegged by training; we don't want to
fight for kernels. The model is small enough this completes in 10-15 min.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_hyenadna import SequenceBackbone, BackboneConfig, DEFAULT_HF_NAME
from cache_v2_reader import CacheV2
from pretrain_mlm import BidirMLM, make_mlm_batch


def run_shard(model: BidirMLM, cache: CacheV2, shard_name: str,
              files: list[str], n_files: int, n_seqs: int,
              mask_prob: float, max_len: int, device: str,
              rng: random.Random, np_rng: np.random.Generator) -> dict:
    shard = cache.shards[shard_name]
    rng.shuffle(files)
    files = files[:n_files]

    total_correct = 0
    total_seen = 0
    total_loss = 0.0
    total_loss_n = 0
    files_used = 0
    seqs_seen = 0

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for fn in files:
            try:
                file_idx = shard.files.index(fn)
            except ValueError:
                continue
            a = shard.get_alignment(file_idx)
            if a.shape[1] > max_len:
                a = a[:, :max_len]
            files_used += 1
            n_rows = a.shape[0]
            picks = list(range(n_rows))
            rng.shuffle(picks)
            picks = picks[:n_seqs]

            for r in picks:
                v2_ids = a[r:r+1]
                fwd, rc, tgt, _ = make_mlm_batch(v2_ids, mask_prob, np_rng, device)
                logits = model(fwd, rc)
                loss = F.cross_entropy(logits.reshape(-1, 5),
                                       tgt.reshape(-1),
                                       ignore_index=-100, reduction="sum")
                n_scored = int((tgt != -100).sum().item())
                if n_scored:
                    pred = logits.argmax(dim=-1)
                    m = tgt != -100
                    correct = int(((pred == tgt) & m).sum().item())
                    total_correct += correct
                    total_seen += n_scored
                    total_loss += float(loss.item())
                    total_loss_n += n_scored
                seqs_seen += 1

    elapsed = time.time() - t0
    return {
        "shard": shard_name,
        "files_used": files_used,
        "seqs_seen": seqs_seen,
        "n_scored": total_seen,
        "acc": total_correct / max(1, total_seen),
        "loss": total_loss / max(1, total_loss_n),
        "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models_test/backbone_mlm_v1.pt"))
    ap.add_argument("--splits", type=Path, default=Path("splits/v2_filtered_split.json"))
    ap.add_argument("--n-files", type=int, default=8)
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cpu"
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    print(f"[{time.strftime('%H:%M:%S')}] loading splits + cache + ckpt", flush=True)
    import json
    splits = json.load(args.splits.open())
    cache = CacheV2()
    val_files = {sh: sd.get("files", []) for sh, sd in splits["splits"]["VAL"]["dirs"].items()}

    backbone = SequenceBackbone(BackboneConfig(hf_name=DEFAULT_HF_NAME), pretrained=False)
    d_model = backbone.cfg.d_model
    model = BidirMLM(backbone, d_model=d_model, n_classes=5).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    print(f"  ckpt: epoch {ck['epoch']} stage_idx_done {ck.get('stage_idx_done')} "
          f"global_step {ck['global_step']:,}", flush=True)

    plan = [
        ("XML-5",                "10 kb — no matched TRAIN sibling (HIV-like, mut=0.005)", 11_000),
        ("XML-6",                "10 kb — MATCHED TRAIN (same simulator config, file-level holdout)", 11_000),
        ("long_content_30k_001", "30 kb — sibling of TRAIN's _002 (similar config, different seeds)", 30_500),
    ]

    print(f"\nPer-shard val ({args.n_files} files × {args.n_seqs} seqs each, "
          f"mask_prob={args.mask_prob}, CPU):", flush=True)
    print(f"{'shard':>25}  {'acc':>6}  {'loss':>6}  {'n_scored':>9}  {'time':>5}  notes")
    results = []
    for sh, note, max_len in plan:
        files = val_files.get(sh, [])
        if not files:
            print(f"{sh:>25}  --- no files in VAL split ---")
            continue
        r = run_shard(model, cache, sh, list(files), args.n_files, args.n_seqs,
                      args.mask_prob, max_len, device, rng, np_rng)
        print(f"{r['shard']:>25}  {r['acc']:>6.3f}  {r['loss']:>6.3f}  "
              f"{r['n_scored']:>9,}  {r['elapsed_s']:>4.0f}s  {note}", flush=True)
        results.append(r)

    if len(results) >= 2:
        xml5_acc = next((r["acc"] for r in results if r["shard"] == "XML-5"), None)
        xml6_acc = next((r["acc"] for r in results if r["shard"] == "XML-6"), None)
        lc001_acc = next((r["acc"] for r in results if r["shard"] == "long_content_30k_001"), None)
        print(f"\nDiagnostic:")
        if xml5_acc is not None and xml6_acc is not None:
            delta_10k = xml6_acc - xml5_acc
            print(f"  10 kb tier: matched XML-6 val acc = {xml6_acc:.3f}  vs  "
                  f"unmatched XML-5 val acc = {xml5_acc:.3f}  (Δ = {delta_10k:+.3f})")
            if delta_10k > 0.04:
                print(f"    → distribution shift is meaningfully suppressing val acc "
                      f"on XML-5 (Δ > 0.04).")
            elif abs(delta_10k) < 0.02:
                print(f"    → no meaningful shift between matched and unmatched 10 kb.")
            else:
                print(f"    → small shift effect, not load-bearing.")
        if lc001_acc is not None:
            print(f"  30 kb tier: long_content_30k_001 = {lc001_acc:.3f} "
                  f"(no matched-config TRAIN sibling — only file-family sibling _30k_002)")


if __name__ == "__main__":
    main()
