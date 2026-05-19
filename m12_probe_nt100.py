"""M1.2 zero-shot MLM probe — does Nucleotide Transformer 100M Multi-Species
(NT-v2 100M MS) pretraining transfer to SANTA-simulated viral sequences?

This is a counterpart to m12_zeroshot_probe.py which scored HyenaDNA-small
at 0.308 next-token accuracy (essentially majority-class baseline).

NT differs from HyenaDNA in important ways:
  - bidirectional MLM objective (BERT-like, not causal LM)
  - 6-mer non-overlapping tokenization (vocab 4107, 4096 6-mers + specials)
  - max_position_embeddings = 2050 tokens (~12 kb), but we chunk at 6 kb to
    match the InstaDeepAI 1000-token max input.

Metric: token-level MLM accuracy on positions we randomly masked (15% rate).
  Random baseline over the 4096 6-mer vocab ≈ 0.00024.
  A "high" score (>0.10) would indicate genuine viral-sequence understanding.

Probe scope: 30 sequences per length scale (XML-5 ~10kb, XML-6 ~10 kb,
long_content_30k_001 ~30 kb), all from VAL shards via CacheV2.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_v2_reader import CacheV2

# RSS watchdog (the user reported an earlier OOM took down a desktop session).
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


# v2 cache int8 → ASCII. Gaps (4) are mapped to N as instructed.
V2_TO_ASCII = np.array([ord('A'), ord('T'), ord('G'), ord('C'), ord('N')],
                       dtype=np.uint8)


def decode_v2_to_str(arr: np.ndarray) -> str:
    return V2_TO_ASCII[arr.astype(np.int8)].tobytes().decode("ascii")


def sample_sequences(cache: CacheV2, shard_name: str,
                     n_seqs: int, rng: random.Random) -> list[np.ndarray]:
    """Same sampler as m12_zeroshot_probe.py — skips mostly-gap rows."""
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


def chunk_sequence(seq_arr: np.ndarray, chunk_bp: int) -> list[str]:
    """Decode and split into non-overlapping bp chunks. Drops trailing chunks
    shorter than 6 bp (would tokenize to <2 tokens and be uninformative)."""
    s = decode_v2_to_str(seq_arr)
    chunks: list[str] = []
    for i in range(0, len(s), chunk_bp):
        c = s[i:i + chunk_bp]
        if len(c) >= 12:  # need at least 2 6-mers for the probe to be useful
            chunks.append(c)
    return chunks


def evaluate_mlm(model, tokenizer, sequences: list[np.ndarray],
                 device: str, chunk_bp: int, mask_rate: float,
                 rng: random.Random, batch_size: int) -> dict:
    """Score MLM accuracy on randomly-masked 6-mer tokens.

    For each chunk:
      - tokenize (deterministically padded by HF)
      - identify nucleotide-only token positions (exclude specials and any
        single-nt fallback tokens produced by stray N characters)
      - sample mask_rate of those positions, replace with <mask>
      - forward pass, take argmax over the full vocab, compare to original ids
    """
    model.eval()
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    special_ids = set(tokenizer.all_special_ids)

    # Single-nt fallback tokens (A/T/G/C/N) — exclude from masking pool because
    # they are not the 6-mer prediction objective.
    single_nt_ids = set()
    for ch in ("A", "C", "G", "T", "N"):
        try:
            tid = tokenizer.convert_tokens_to_ids(ch)
            if tid is not None and tid != tokenizer.unk_token_id:
                single_nt_ids.add(int(tid))
        except Exception:
            pass

    total_correct = 0
    total_masked = 0
    per_bucket_correct: dict[str, int] = defaultdict(int)
    per_bucket_seen: dict[str, int] = defaultdict(int)
    chunks_seen = 0
    nucleotide_token_match_correct = 0
    nucleotide_token_match_seen = 0

    # Pre-build all chunks across sequences, then batch them uniformly.
    all_chunks: list[tuple[int, str]] = []  # (seq_idx, chunk_str)
    for si, seq in enumerate(sequences):
        for c in chunk_sequence(seq, chunk_bp):
            all_chunks.append((si, c))

    n_chunks = len(all_chunks)
    print(f"    {n_chunks} chunks of up to {chunk_bp} bp", flush=True)
    t0 = time.time()

    for b_start in range(0, n_chunks, batch_size):
        b_end = min(b_start + batch_size, n_chunks)
        batch_strs = [c for _, c in all_chunks[b_start:b_end]]

        enc = tokenizer(batch_strs, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        B, L = input_ids.shape
        original_ids = input_ids.clone()

        # Build a mask of positions eligible for masking: not padding,
        # not special, not single-nt fallback.
        ids_cpu = input_ids.cpu().numpy()
        eligible = np.ones_like(ids_cpu, dtype=bool)
        for sid in special_ids:
            eligible &= (ids_cpu != sid)
        for sid in single_nt_ids:
            eligible &= (ids_cpu != sid)

        mlm_mask = np.zeros_like(eligible, dtype=bool)
        for i in range(B):
            pos = np.flatnonzero(eligible[i])
            if pos.size == 0:
                continue
            k = max(1, int(round(pos.size * mask_rate)))
            chosen = rng.sample(list(pos), k=min(k, len(pos)))
            mlm_mask[i, chosen] = True

        n_mask_total = int(mlm_mask.sum())
        if n_mask_total == 0:
            continue

        mlm_mask_t = torch.from_numpy(mlm_mask).to(device)
        masked_input = input_ids.clone()
        masked_input[mlm_mask_t] = mask_id

        with torch.no_grad():
            out = model(input_ids=masked_input, attention_mask=attn)
        logits = out.logits  # (B, L, V)
        pred = logits.argmax(dim=-1)  # (B, L)

        pred_cpu = pred.cpu().numpy()
        orig_cpu = original_ids.cpu().numpy()

        # Per-position metrics.
        for i in range(B):
            mpos = np.flatnonzero(mlm_mask[i])
            if mpos.size == 0:
                continue
            p_m = pred_cpu[i, mpos]
            t_m = orig_cpu[i, mpos]
            correct = (p_m == t_m)
            total_correct += int(correct.sum())
            total_masked += int(mpos.size)

            # Soft nucleotide match: did the predicted token map back to a
            # 6-mer with the same nucleotides? Predicted token must also be a
            # 6-mer (excluding specials / single-nt fallbacks).
            for j_idx, pos in enumerate(mpos):
                tt = int(t_m[j_idx])
                pp = int(p_m[j_idx])
                if pp in special_ids or pp in single_nt_ids:
                    continue
                t_tok = tokenizer.convert_ids_to_tokens([tt])[0]
                p_tok = tokenizer.convert_ids_to_tokens([pp])[0]
                if len(t_tok) != 6 or len(p_tok) != 6:
                    continue
                nucleotide_token_match_seen += 6
                nucleotide_token_match_correct += sum(
                    1 for a, b in zip(t_tok, p_tok) if a == b
                )

            # Positional bucket (relative position inside chunk).
            chunk_len = max(int(attn[i].sum().item()) - 1, 1)
            rel = mpos / chunk_len
            for bkt, lo, hi in (("0-25%", 0.0, 0.25),
                                ("25-75%", 0.25, 0.75),
                                ("75-100%", 0.75, 1.0 + 1e-9)):
                bm = (rel >= lo) & (rel < hi)
                per_bucket_seen[bkt] += int(bm.sum())
                per_bucket_correct[bkt] += int(correct[bm].sum())

        chunks_seen += B
        _rss_watchdog(label=f"after chunk batch ending at {b_end}")
        if (b_end // batch_size) % 5 == 0 or b_end == n_chunks:
            rate = b_end / max(time.time() - t0, 1e-3)
            rss = _current_rss_bytes() / 2**30
            gpu_mem = (torch.cuda.max_memory_allocated() / 2**30
                       if device == "cuda" else 0.0)
            print(f"    [{b_end}/{n_chunks}] {rate:.1f} chunks/s "
                  f"RSS {rss:.1f} GB  GPU {gpu_mem:.2f} GB", flush=True)

    acc = total_correct / total_masked if total_masked else float("nan")
    nt_acc = (nucleotide_token_match_correct / nucleotide_token_match_seen
              if nucleotide_token_match_seen else float("nan"))
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
        "n_sequences": len(sequences),
        "n_chunks": chunks_seen,
        "n_tokens_masked": total_masked,
        "mlm_token_accuracy": acc,
        "nucleotide_level_accuracy": nt_acc,
        "per_bucket": per_bucket,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path,
                    default=Path("/home/joshc/Dev/RDP_CNN/cache/v2"))
    ap.add_argument("--n-per-scale", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-bp", type=int, default=6000,
                    help="6000 bp = 1000 6-mer tokens, matches NT's stated "
                         "max-input length.")
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--hf-name", type=str,
                    default="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species")
    ap.add_argument("--out", type=Path,
                    default=Path("m12_probe_nt100.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] device={device}", flush=True)

    cache = CacheV2(args.cache_root)
    rng = random.Random(args.seed)

    print(f"[{time.strftime('%H:%M:%S')}] loading "
          f"AutoModelForMaskedLM {args.hf_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_name,
                                              trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(args.hf_name,
                                                 trust_remote_code=True)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}", flush=True)

    # Verify tokenizer behavior on a known string.
    probe_str = "ACGT" * 1500  # 6000 bp -> should give 1001 tokens (cls + 1000)
    test_ids = tokenizer(probe_str, return_tensors="pt")["input_ids"]
    print(f"  tokenizer self-test: 6000 bp -> {test_ids.shape[1]} tokens "
          f"(expected 1001)", flush=True)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_start = time.time()
    val_scales = [
        ("XML-5",                "short (~10 kb)"),
        ("XML-6",                "medium (~10 kb)"),
        ("long_content_30k_001", "long (~30 kb)"),
    ]

    report: dict = {
        "hf_name": args.hf_name,
        "n_params": n_params,
        "seed": args.seed,
        "n_per_scale": args.n_per_scale,
        "chunk_bp": args.chunk_bp,
        "mask_rate": args.mask_rate,
        "batch": args.batch,
        "scales": {},
    }
    for shard_name, label in val_scales:
        if shard_name not in cache.shards:
            print(f"  skip {shard_name}: shard not in cache", flush=True)
            continue
        print(f"\n=== {shard_name} — {label} ===", flush=True)
        seqs = sample_sequences(cache, shard_name, args.n_per_scale, rng)
        if not seqs:
            print(f"  no sequences sampled — skipping", flush=True)
            continue
        avg_len = int(np.mean([len(s) for s in seqs]))
        print(f"  sampled {len(seqs)} sequences, avg_len={avg_len}",
              flush=True)
        res = evaluate_mlm(model, tokenizer, seqs, device,
                           chunk_bp=args.chunk_bp,
                           mask_rate=args.mask_rate,
                           rng=rng, batch_size=args.batch)
        print(f"  MLM token accuracy: {res['mlm_token_accuracy']:.4f}  "
              f"(n_masked={res['n_tokens_masked']:,})", flush=True)
        print(f"  nucleotide-level accuracy: "
              f"{res['nucleotide_level_accuracy']:.4f}", flush=True)
        for bkt, v in res["per_bucket"].items():
            print(f"    pos {bkt}: acc={v['acc']:.4f}  (n={v['n']:,})",
                  flush=True)
        report["scales"][shard_name] = {
            "label": label, "avg_len": avg_len, **res
        }

    accs = [s["mlm_token_accuracy"] for s in report["scales"].values()]
    nt_accs = [s["nucleotide_level_accuracy"]
               for s in report["scales"].values()]
    if accs:
        report["overall_mlm_token_accuracy_mean"] = float(np.mean(accs))
        report["overall_nucleotide_accuracy_mean"] = float(np.mean(nt_accs))
    report["wall_time_sec"] = time.time() - t_start
    if device == "cuda":
        report["peak_gpu_memory_gb"] = (
            torch.cuda.max_memory_allocated() / 2**30)

    with args.out.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.out}", flush=True)

    if accs:
        mean_acc = report["overall_mlm_token_accuracy_mean"]
        nt_mean = report["overall_nucleotide_accuracy_mean"]
        rand_baseline = 1.0 / 4096
        print(f"\n=== NT-v2 100M MS zero-shot MLM probe summary ===")
        print(f"mean 6-mer token accuracy:    {mean_acc:.4f}")
        print(f"mean nucleotide-level acc:    {nt_mean:.4f}")
        print(f"random 6-mer baseline:        {rand_baseline:.4f}")
        print(f"HyenaDNA-small (causal LM):   0.308 next-token accuracy")
        if mean_acc >= 0.10:
            print("DECISION: NT shows clear viral-sequence understanding; "
                  "consider NT warm-start over from-scratch MLM.")
        elif mean_acc >= 0.02:
            print("DECISION: NT shows medium signal — measurable transfer "
                  "but well below typical fine-tuned MLM accuracies.")
        else:
            print("DECISION: NT zero-shot is near random — multispecies "
                  "pretraining does not transfer to SANTA without "
                  "adaptation.")


if __name__ == "__main__":
    main()
