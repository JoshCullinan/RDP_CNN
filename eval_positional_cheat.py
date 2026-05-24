"""Test whether the 30 kb val acc is inflated by positional shortcutting.

Hypothesis: HyenaDNA's long convolutions are inherently position-aware. In
our training data, positions past 19,025 only appear in long_content shards.
The model could be learning "if past position 19k, I'm in long_content
territory, predict from its nucleotide priors" rather than learning
position-invariant DNA structure.

Test: validate on long_content_30k_001 but bucket scored positions by
where they fall in the genome:
    early (0-9999)        — shared with all 10 kb shards in train
    mid   (10000-19024)   — shared with XML-4 + long_content
    late  (19025-29901)   — long_content ONLY in train

If early-position acc ≈ 10 kb val acc (~0.40), and late-position acc
is much higher (~0.65+), the positional shortcut is real.

Also bucket XML-5 (10 kb val) by position to compare matched-position
predictability.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backbone_hyenadna import SequenceBackbone, BackboneConfig, DEFAULT_HF_NAME
from cache_v2_reader import CacheV2
from pretrain_mlm import BidirMLM, make_mlm_batch


# Position-range buckets to track.
# Late bucket extends past 30,001 to catch the long_content tail (29,902 max).
BUCKETS = [
    ("0-9.9k",       0,      10_000),
    ("10k-19k",      10_000, 19_025),
    ("19k-30k",      19_025, 30_001),
]


def run_shard(model: BidirMLM, cache: CacheV2, shard_name: str,
              files: list[str], n_files: int, n_seqs: int,
              mask_prob: float, max_len: int, device: str,
              rng: random.Random, np_rng: np.random.Generator) -> dict:
    shard = cache.shards[shard_name]
    rng.shuffle(files)
    files = files[:n_files]

    bucket_correct: dict[str, int] = defaultdict(int)
    bucket_seen: dict[str, int] = defaultdict(int)
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
            n_rows = a.shape[0]
            picks = list(range(n_rows))
            rng.shuffle(picks)
            picks = picks[:n_seqs]

            for r in picks:
                v2_ids = a[r:r+1]
                fwd, rc, tgt, _ = make_mlm_batch(v2_ids, mask_prob, np_rng, device)
                logits = model(fwd, rc)
                pred = logits.argmax(dim=-1)              # (1, L)
                m = (tgt != -100)                          # (1, L)
                pred_np = pred[0].cpu().numpy()
                tgt_np = tgt[0].cpu().numpy()
                mask_np = m[0].cpu().numpy()
                if not mask_np.any():
                    continue
                # Position indices of every masked position in this sequence
                positions = np.where(mask_np)[0]
                correct = (pred_np[positions] == tgt_np[positions])
                for bkt_name, lo, hi in BUCKETS:
                    in_bkt = (positions >= lo) & (positions < hi)
                    if not in_bkt.any():
                        continue
                    bucket_seen[bkt_name] += int(in_bkt.sum())
                    bucket_correct[bkt_name] += int(correct[in_bkt].sum())
                seqs_seen += 1

    elapsed = time.time() - t0
    by_bucket = {
        bkt_name: {
            "n": bucket_seen[bkt_name],
            "acc": bucket_correct[bkt_name] / max(1, bucket_seen[bkt_name]),
        }
        for bkt_name, _, _ in BUCKETS
        if bucket_seen[bkt_name] > 0
    }
    overall_correct = sum(bucket_correct.values())
    overall_seen = sum(bucket_seen.values())
    return {
        "shard": shard_name,
        "seqs_seen": seqs_seen,
        "overall_acc": overall_correct / max(1, overall_seen),
        "overall_n": overall_seen,
        "by_bucket": by_bucket,
        "elapsed_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models_test/backbone_mlm_v1.pt"))
    ap.add_argument("--splits", type=Path, default=Path("splits/v2_filtered_split.json"))
    ap.add_argument("--n-files", type=int, default=16)
    ap.add_argument("--n-seqs", type=int, default=8)
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
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
        ("XML-5",                11_000,  "10 kb (max position 10.8k)"),
        ("long_content_30k_001", 30_500,  "30 kb (max position 29.9k) — the headline shard"),
    ]

    print(f"\nPer-position bucket val ({args.n_files} files × {args.n_seqs} seqs each, "
          f"mask_prob={args.mask_prob}, CPU):", flush=True)
    for sh, max_len, note in plan:
        files = val_files.get(sh, [])
        if not files:
            print(f"\n[{sh}] no files in VAL")
            continue
        r = run_shard(model, cache, sh, list(files), args.n_files, args.n_seqs,
                      args.mask_prob, max_len, device, rng, np_rng)
        print(f"\n[{sh}]  overall={r['overall_acc']:.3f}  n_scored={r['overall_n']:,}  "
              f"{r['elapsed_s']:.0f}s  ({note})", flush=True)
        for bkt_name, lo, hi in BUCKETS:
            if bkt_name in r["by_bucket"]:
                b = r["by_bucket"][bkt_name]
                print(f"    position bucket {bkt_name:>10}: acc={b['acc']:.3f}  n={b['n']:,}",
                      flush=True)

    print(f"\n--- interpretation ---", flush=True)
    print("If long_content_30k_001 buckets differ substantially "
          "(e.g. 0-9.9k acc ≈ XML-5 acc, but 19k-30k acc >> that),", flush=True)
    print("the model is exploiting position-as-shard-id instead of learning "
          "position-invariant DNA structure.", flush=True)


if __name__ == "__main__":
    main()
