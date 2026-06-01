"""M1.2 zero-shot probe — does HyenaDNA's human-genome pretraining
transfer to SANTA-simulated HIV-like sequences?

HyenaDNA-small was pretrained via causal language modelling on the
human genome at 32k context. Before committing to full from-scratch
MLM pretraining (~10 days on the 3070), measure next-token accuracy
on SANTA validation sequences with the existing checkpoint frozen.

Decision impact:
  acc >= 0.45 → M1.2 can collapse to a short domain-adaptation fine-tune
  0.30 <= acc < 0.45 → partial transfer, need real MLM but maybe shorter
  acc < 0.30 → near-baseline, full from-scratch MLM as originally planned

Baselines:
  random over 5 nt classes:  0.200
  majority-class HIV-ish:    ~0.30 (AT-rich)

Probe scope: ~30 sequences sampled per length scale (XML-5 ~4 kb,
XML-6 ~10 kb, long_content_30k_001 ~30 kb), all from VAL split.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# RSS watchdog: same pattern as m04_verify_inference.py.
_RSS_CEILING_BYTES = 26 * 1024 * 1024 * 1024


def _current_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _rss_watchdog(label: str = "") -> None:
    rss = _current_rss_bytes()
    if rss > _RSS_CEILING_BYTES:
        raise MemoryError(
            f"RSS watchdog tripped {label}: {rss/2**30:.1f} GB > "
            f"{_RSS_CEILING_BYTES/2**30:.0f} GB cap"
        )


import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_v2_reader import CacheV2
from backbone_hyenadna import v2_to_hyena_ids, DEFAULT_HF_NAME

# HyenaDNA vocab indices for nucleotide tokens.
HY_A, HY_C, HY_G, HY_T, HY_N = 7, 8, 9, 10, 11
HY_PAD = 4  # gap → PAD
NT_TOKENS = (HY_A, HY_C, HY_G, HY_T, HY_PAD)
TOKEN_NAME = {HY_A: "A", HY_C: "C", HY_G: "G", HY_T: "T", HY_PAD: "gap"}


def sample_sequences(cache: CacheV2, shard_name: str,
                     n_seqs: int, rng: random.Random) -> list[np.ndarray]:
    shard = cache.shards[shard_name]
    out: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    file_count = shard.n_files
    if file_count == 0:
        return out
    attempts = 0
    while len(out) < n_seqs and attempts < n_seqs * 20:
        attempts += 1
        fi = rng.randrange(file_count)
        align = shard.get_alignment(fi)
        n, _ = align.shape
        row = rng.randrange(n)
        key = (fi, row)
        if key in seen:
            continue
        seen.add(key)
        seq = align[row]
        if (seq != 4).sum() < 100:  # skip mostly-gap rows
            continue
        out.append(np.asarray(seq, dtype=np.int8))
    return out


def evaluate_next_token(model: torch.nn.Module, sequences: list[np.ndarray],
                        device: str, batch_size: int = 2) -> dict:
    model.eval()
    total_correct = 0
    total_seen = 0
    per_class_correct: Counter = Counter()
    per_class_seen: Counter = Counter()
    per_bucket_correct: dict[str, int] = defaultdict(int)
    per_bucket_seen: dict[str, int] = defaultdict(int)
    nt_set = set(NT_TOKENS)

    n = len(sequences)
    t0 = time.time()
    for b_start in range(0, n, batch_size):
        b_end = min(b_start + batch_size, n)
        batch = sequences[b_start:b_end]
        max_L = max(len(s) for s in batch)
        ids_np = np.full((len(batch), max_L), HY_PAD, dtype=np.int64)
        seq_lens = []
        for i, s in enumerate(batch):
            ids_np[i, :len(s)] = v2_to_hyena_ids(torch.from_numpy(s)).numpy()
            seq_lens.append(len(s))
        ids = torch.from_numpy(ids_np).to(device)

        with torch.no_grad():
            out = model(input_ids=ids)
        logits = out.logits  # (B, L, V)
        pred = logits.argmax(dim=-1)  # (B, L)

        for i, L in enumerate(seq_lens):
            if L < 2:
                continue
            p = pred[i, : L - 1].cpu().numpy()
            t = ids[i, 1:L].cpu().numpy()
            mask = np.isin(t, list(nt_set))
            if not mask.any():
                continue
            p_m = p[mask]
            t_m = t[mask]
            correct = (p_m == t_m)
            total_correct += int(correct.sum())
            total_seen += int(mask.sum())
            for cls in NT_TOKENS:
                cls_mask = t_m == cls
                if cls_mask.any():
                    per_class_seen[cls] += int(cls_mask.sum())
                    per_class_correct[cls] += int(correct[cls_mask].sum())
            positions = np.arange(L - 1)[mask]
            if L > 0:
                rel = positions / max(L - 1, 1)
                for bkt, lo, hi in (("0-25%", 0.0, 0.25),
                                    ("25-75%", 0.25, 0.75),
                                    ("75-100%", 0.75, 1.0 + 1e-9)):
                    bm = (rel >= lo) & (rel < hi)
                    per_bucket_seen[bkt] += int(bm.sum())
                    per_bucket_correct[bkt] += int(correct[bm].sum())

        _rss_watchdog(label=f"after batch ending at {b_end}")
        if (b_end // batch_size) % 5 == 0 or b_end == n:
            rate = b_end / max(time.time() - t0, 1e-3)
            rss = _current_rss_bytes() / 2**30
            print(f"    [{b_end}/{n}]  {rate:.1f} seq/s  RSS {rss:.1f} GB",
                  flush=True)

    acc = total_correct / total_seen if total_seen else float("nan")
    per_class = {
        TOKEN_NAME[cls]: {
            "n": per_class_seen[cls],
            "correct": per_class_correct[cls],
            "acc": (per_class_correct[cls] / per_class_seen[cls])
                   if per_class_seen[cls] else 0.0,
        }
        for cls in NT_TOKENS
    }
    per_bucket = {
        bkt: {
            "n": per_bucket_seen[bkt],
            "correct": per_bucket_correct[bkt],
            "acc": (per_bucket_correct[bkt] / per_bucket_seen[bkt])
                   if per_bucket_seen[bkt] else 0.0,
        }
        for bkt in ("0-25%", "25-75%", "75-100%")
    }
    return {
        "n_seqs": n,
        "n_positions_scored": total_seen,
        "next_token_accuracy": acc,
        "per_class": per_class,
        "per_bucket": per_bucket,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/cache/v2"))
    ap.add_argument("--n-per-scale", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--hf-name", type=str, default=DEFAULT_HF_NAME)
    ap.add_argument("--out", type=Path, default=Path("m12_zeroshot_probe.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] device={device}", flush=True)

    cache = CacheV2(args.cache_root)
    rng = random.Random(args.seed)

    print(f"[{time.strftime('%H:%M:%S')}] loading "
          f"AutoModelForCausalLM {args.hf_name}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.hf_name,
                                                  trust_remote_code=True)
    model.to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    val_scales = [
        ("XML-5",                "short (~4 kb)"),
        ("XML-6",                "medium (~10 kb)"),
        ("long_content_30k_001", "long (~30 kb)"),
    ]

    report: dict = {
        "hf_name": args.hf_name,
        "seed": args.seed,
        "n_per_scale": args.n_per_scale,
        "scales": {},
    }
    for shard_name, label in val_scales:
        if shard_name not in cache.shards:
            print(f"  skip {shard_name}: shard not in cache")
            continue
        print(f"\n=== {shard_name} — {label} ===", flush=True)
        seqs = sample_sequences(cache, shard_name, args.n_per_scale, rng)
        if not seqs:
            print(f"  no sequences sampled — skipping")
            continue
        avg_len = int(np.mean([len(s) for s in seqs]))
        print(f"  sampled {len(seqs)} sequences, avg_len={avg_len}",
              flush=True)
        res = evaluate_next_token(model, seqs, device, batch_size=args.batch)
        print(f"  next-token accuracy: {res['next_token_accuracy']:.3f} "
              f"(n={res['n_positions_scored']:,})", flush=True)
        for cls, v in res["per_class"].items():
            print(f"    {cls}: acc={v['acc']:.3f}  (n={v['n']:,})", flush=True)
        for bkt, v in res["per_bucket"].items():
            print(f"    pos {bkt}: acc={v['acc']:.3f}  (n={v['n']:,})",
                  flush=True)
        report["scales"][shard_name] = {
            "label": label, "avg_len": avg_len, **res
        }

    accs = [s["next_token_accuracy"] for s in report["scales"].values()]
    if accs:
        report["overall_accuracy_mean"] = float(np.mean(accs))
    with args.out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.out}")

    if accs:
        mean_acc = report["overall_accuracy_mean"]
        print(f"\n=== M1.2 zero-shot probe summary ===")
        print(f"mean accuracy: {mean_acc:.3f}")
        if mean_acc >= 0.45:
            print("DECISION: HyenaDNA priors transfer strongly → M1.2 can be "
                  "a short domain-adaptation fine-tune, not from-scratch MLM.")
        elif mean_acc >= 0.30:
            print("DECISION: partial transfer → M1.2 needs MLM training but "
                  "may converge faster than expected (warm-start the encoder).")
        else:
            print("DECISION: weak transfer → run M1.2 from-scratch MLM as "
                  "originally planned (10-day budget).")


if __name__ == "__main__":
    main()
