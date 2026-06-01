"""M1.3 — SANTA vs real-HIV statistical comparison (Plan-E precursor).

Two lines of evidence to answer "is SANTA close to real viruses?":

Line 1: HyenaDNA-small next-token accuracy on real HIV. Direct
   comparison vs the cached SANTA baseline (XML-6 ≈ 0.285, mean ≈ 0.308).

Line 2: statistical fingerprints — k-mer JSD, GC content (mean+sliding),
   period-3 spectral peak, pairwise within-alignment divergence.

Run:
    .venv/bin/python m13_santa_vs_real.py
"""

from __future__ import annotations

import json
import math
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cache_v2_reader import CacheV2  # noqa: E402
from backbone_hyenadna import v2_to_hyena_ids, DEFAULT_HF_NAME  # noqa: E402

# RSS watchdog
_RSS_CEILING_BYTES = 26 * 1024 * 1024 * 1024


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _rss_watchdog(label: str = "") -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if rss > _RSS_CEILING_BYTES:
        raise MemoryError(
            f"RSS watchdog tripped {label}: {rss/2**30:.1f} GB cap exceeded"
        )


# HyenaDNA vocab indices
HY_A, HY_C, HY_G, HY_T, HY_PAD = 7, 8, 9, 10, 4
NT_TOKENS = (HY_A, HY_C, HY_G, HY_T, HY_PAD)
TOKEN_NAME = {HY_A: "A", HY_C: "C", HY_G: "G", HY_T: "T", HY_PAD: "gap"}

# v2 cache encoding: A=0 T=1 G=2 C=3 gap=4
NT_TO_V2 = {"A": 0, "T": 1, "G": 2, "C": 3, "-": 4}


# --------------------------------------------------------------------------
# Real HIV FASTA loading
# --------------------------------------------------------------------------

def read_fasta_records(path: Path) -> list[tuple[str, str]]:
    """Minimal FASTA reader. Returns [(header, seq_uppercase), ...]."""
    out: list[tuple[str, str]] = []
    header = None
    chunks: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    out.append((header, "".join(chunks).upper()))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            out.append((header, "".join(chunks).upper()))
    return out


def seq_to_v2_int8(seq: str, drop_gaps: bool = True,
                   drop_ambig: bool = True) -> np.ndarray:
    """Convert nucleotide string to v2 int8 array.

    drop_gaps=True: filter '-' (alignment gaps) out entirely.
    drop_ambig=True: filter non-{A,T,G,C,-} bases (N, R, Y, W, ...) out.
    Output is int8 in {0,1,2,3} (or {0..4} if drop_gaps=False).
    """
    s = seq
    if drop_gaps:
        s = s.replace("-", "")
    arr = np.empty(len(s), dtype=np.int8)
    n = 0
    for c in s:
        v = NT_TO_V2.get(c)
        if v is None:
            if drop_ambig:
                continue
            else:
                # ambiguous nucleotide → treat as gap so the probe masks it
                arr[n] = 4
                n += 1
                continue
        arr[n] = v
        n += 1
    return arr[:n].copy()


def load_real_hiv_sequences(
    triplets_dir: Path, genbank_dir: Path,
) -> list[tuple[str, np.ndarray]]:
    """Return [(label, v2_int8_array), ...] for all real HIV sources."""
    out: list[tuple[str, np.ndarray]] = []
    for fa in sorted(triplets_dir.glob("*.fa")):
        for hdr, seq in read_fasta_records(fa):
            arr = seq_to_v2_int8(seq, drop_gaps=True, drop_ambig=True)
            if len(arr) < 100:
                continue
            out.append((f"{fa.stem}|{hdr}", arr))
    for fa in sorted(genbank_dir.glob("*.fa")):
        for hdr, seq in read_fasta_records(fa):
            arr = seq_to_v2_int8(seq, drop_gaps=True, drop_ambig=True)
            if len(arr) < 100:
                continue
            out.append((f"genbank|{fa.stem}", arr))
    return out


# --------------------------------------------------------------------------
# SANTA sampling (from XML-6 — closest length match to real HIV ~9.7 kb)
# --------------------------------------------------------------------------

def sample_santa_xml6(cache: CacheV2, n_seqs: int,
                      rng: random.Random) -> list[np.ndarray]:
    shard = cache.shards["XML-6"]
    out: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(out) < n_seqs and attempts < n_seqs * 30:
        attempts += 1
        fi = rng.randrange(shard.n_files)
        align = shard.get_alignment(fi)
        n, _ = align.shape
        row = rng.randrange(n)
        key = (fi, row)
        if key in seen:
            continue
        seen.add(key)
        seq = align[row]
        if (seq != 4).sum() < 100:
            continue
        # Drop gaps for parity with real HIV (real HIV inputs are ungapped)
        s = np.asarray(seq, dtype=np.int8)
        s = s[s != 4]
        if len(s) < 100:
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------
# Line 1: HyenaDNA next-token probe
# --------------------------------------------------------------------------

def evaluate_next_token(model, sequences, device, batch_size=2,
                        max_L_cap=32_000):
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
        # cap per-seq length
        batch = [s[:max_L_cap] for s in batch]
        max_L = max(len(s) for s in batch)
        ids_np = np.full((len(batch), max_L), HY_PAD, dtype=np.int64)
        seq_lens = []
        for i, s in enumerate(batch):
            ids_np[i, :len(s)] = v2_to_hyena_ids(torch.from_numpy(s)).numpy()
            seq_lens.append(len(s))
        ids = torch.from_numpy(ids_np).to(device)
        with torch.no_grad():
            out = model(input_ids=ids)
        logits = out.logits
        pred = logits.argmax(dim=-1)
        for i, L in enumerate(seq_lens):
            if L < 2:
                continue
            p = pred[i, :L - 1].cpu().numpy()
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
            rel = positions / max(L - 1, 1)
            for bkt, lo, hi in (("0-25%", 0.0, 0.25),
                                ("25-75%", 0.25, 0.75),
                                ("75-100%", 0.75, 1.0 + 1e-9)):
                bm = (rel >= lo) & (rel < hi)
                per_bucket_seen[bkt] += int(bm.sum())
                per_bucket_correct[bkt] += int(correct[bm].sum())
        _rss_watchdog(f"batch end {b_end}")
        rate = b_end / max(time.time() - t0, 1e-3)
        print(f"    [{b_end}/{n}]  {rate:.2f} seq/s  RSS {_rss_gb():.1f} GB",
              flush=True)
    acc = total_correct / total_seen if total_seen else float("nan")
    return {
        "n_seqs": n,
        "n_positions_scored": total_seen,
        "next_token_accuracy": acc,
        "per_class": {
            TOKEN_NAME[cls]: {
                "n": per_class_seen[cls], "correct": per_class_correct[cls],
                "acc": (per_class_correct[cls] / per_class_seen[cls])
                       if per_class_seen[cls] else 0.0,
            } for cls in NT_TOKENS
        },
        "per_bucket": {
            bkt: {
                "n": per_bucket_seen[bkt], "correct": per_bucket_correct[bkt],
                "acc": (per_bucket_correct[bkt] / per_bucket_seen[bkt])
                       if per_bucket_seen[bkt] else 0.0,
            } for bkt in ("0-25%", "25-75%", "75-100%")
        },
    }


# --------------------------------------------------------------------------
# Line 2: statistical fingerprints
# --------------------------------------------------------------------------

# k-mer enumeration over {A,T,G,C} only
NT4_INT = {0: "A", 1: "T", 2: "G", 3: "C"}


def kmer_freqs(seq_v2: np.ndarray, k: int) -> dict[str, int]:
    """Count k-mers over {A,T,G,C}; skip windows containing gap (4)."""
    counts: Counter = Counter()
    s = seq_v2
    # build a string repr to count via dict — keeps it simple
    # but for memory, iterate
    L = len(s)
    if L < k:
        return counts
    # use a sliding integer code
    valid_run = 0
    code = 0
    base = 4 ** (k - 1)
    for i in range(L):
        v = int(s[i])
        if v >= 4:
            valid_run = 0
            code = 0
            continue
        code = (code % base) * 4 + v if valid_run >= k - 1 else code * 4 + v
        valid_run += 1
        if valid_run >= k:
            # decode to string only if needed — store integer key
            counts[code] += 1
    # convert int → string mer
    out: dict[str, int] = {}
    for ckey, cnt in counts.items():
        mer = ""
        c = ckey
        for _ in range(k):
            mer = NT4_INT[c % 4] + mer
            c //= 4
        out[mer] = cnt
    return out


def aggregate_kmer_distribution(seqs: list[np.ndarray], k: int) -> np.ndarray:
    """Return a length-4**k probability vector (canonical lex order)."""
    total = Counter()
    for s in seqs:
        for mer, c in kmer_freqs(s, k).items():
            total[mer] += c
    # canonical ordering
    mers = sorted(total.keys() | _all_kmers(k))
    probs = np.array([total[m] for m in mers], dtype=np.float64)
    if probs.sum() == 0:
        return probs
    return probs / probs.sum()


def _all_kmers(k: int) -> set[str]:
    """Generate all 4**k k-mers in canonical order."""
    out = {""}
    for _ in range(k):
        out = {prev + c for prev in out for c in "ACGT"}
    return out


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits, base 2."""
    p = p + 1e-12
    q = q + 1e-12
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def gc_content_stats(seqs: list[np.ndarray]) -> dict:
    """Mean GC, std, and sliding-window mean trace (averaged across seqs)."""
    gcs = []
    for s in seqs:
        # ignore gaps (v2=4)
        valid = s[s != 4]
        if len(valid) == 0:
            continue
        gc = ((valid == 2) | (valid == 3)).mean()
        gcs.append(float(gc))
    return {
        "n_seqs": len(gcs),
        "mean_gc": float(np.mean(gcs)) if gcs else float("nan"),
        "std_gc": float(np.std(gcs)) if gcs else float("nan"),
        "min_gc": float(np.min(gcs)) if gcs else float("nan"),
        "max_gc": float(np.max(gcs)) if gcs else float("nan"),
    }


def period3_strength(seqs: list[np.ndarray], window: int = 3000) -> dict:
    """Compute the spectral peak at frequency 1/3 in the GC indicator.

    For each sequence, take an ungapped GC-indicator series (1 if G/C, 0
    otherwise), centred (subtract mean), and compute the squared FFT
    magnitude at frequency 1/3 normalised by total power. Report mean +
    std across sequences. A pure codon-usage signal would push the
    period-3 fraction up; neutral evolution should leave it at background.

    To get a stable estimate, use first `window` bases of each seq.
    """
    p3_fractions = []
    p3_zscores = []
    for s in seqs:
        valid = s[s != 4]
        if len(valid) < window:
            continue
        x = ((valid[:window] == 2) | (valid[:window] == 3)).astype(np.float64)
        x = x - x.mean()
        if x.std() == 0:
            continue
        F = np.fft.rfft(x)
        power = (F * F.conj()).real
        freqs = np.fft.rfftfreq(len(x))
        # find bin closest to freq=1/3
        idx3 = int(np.argmin(np.abs(freqs - 1.0 / 3.0)))
        total = power[1:].sum()  # exclude DC
        if total <= 0:
            continue
        frac = float(power[idx3] / total)
        # z-score: peak vs local background (10 bins on each side, skip self)
        lo = max(idx3 - 10, 1)
        hi = min(idx3 + 11, len(power))
        local = np.concatenate([power[lo:idx3], power[idx3 + 1:hi]])
        if len(local) >= 4 and local.std() > 0:
            z = float((power[idx3] - local.mean()) / local.std())
        else:
            z = float("nan")
        p3_fractions.append(frac)
        p3_zscores.append(z)
    return {
        "n_seqs": len(p3_fractions),
        "window": window,
        "mean_p3_power_fraction": float(np.mean(p3_fractions))
                                  if p3_fractions else float("nan"),
        "std_p3_power_fraction": float(np.std(p3_fractions))
                                  if p3_fractions else float("nan"),
        "mean_p3_zscore": float(np.nanmean(p3_zscores))
                          if p3_zscores else float("nan"),
        "std_p3_zscore": float(np.nanstd(p3_zscores))
                          if p3_zscores else float("nan"),
    }


def pairwise_divergence(seqs: list[np.ndarray], n_pairs: int = 50,
                        rng: random.Random | None = None) -> dict:
    """Sample random pairs, compute Hamming distance over aligned positions.

    For unaligned (gap-stripped) sequences, this is computed over the
    common prefix length (min len), which is a rough but consistent
    measure across SANTA and real HIV (and real HIV CRF alignments are
    already aligned to the same coordinate system).
    """
    if rng is None:
        rng = random.Random(0)
    if len(seqs) < 2:
        return {"n_pairs": 0, "mean_hamming": float("nan")}
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        i, j = rng.sample(range(len(seqs)), 2)
        a = seqs[i]; b = seqs[j]
        L = min(len(a), len(b))
        if L < 500:
            continue
        a = a[:L]; b = b[:L]
        # ignore positions where either is gap (shouldn't happen post strip,
        # but defensive)
        mask = (a != 4) & (b != 4)
        if mask.sum() < 100:
            continue
        d = float((a[mask] != b[mask]).mean())
        pairs.append(d)
    if not pairs:
        return {"n_pairs": 0, "mean_hamming": float("nan")}
    return {
        "n_pairs": len(pairs),
        "mean_hamming": float(np.mean(pairs)),
        "std_hamming": float(np.std(pairs)),
        "min_hamming": float(np.min(pairs)),
        "max_hamming": float(np.max(pairs)),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    out_path = Path("/home/joshc/Dev/RDP_CNN/m13_santa_vs_real.json")
    triplets_dir = Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/triplets")
    genbank_dir = Path("/home/joshc/Dev/RDP_CNN/data/lanl_crf/genbank_cache")
    cache_root = Path("/home/joshc/Dev/RDP_CNN/cache/v2")

    seed = 0
    rng = random.Random(seed)

    print(f"[{time.strftime('%H:%M:%S')}] loading real HIV sequences",
          flush=True)
    real = load_real_hiv_sequences(triplets_dir, genbank_dir)
    print(f"  loaded {len(real)} real HIV sequences", flush=True)
    for lbl, arr in real:
        print(f"    {lbl}: len={len(arr)}", flush=True)
    real_seqs = [arr for _, arr in real]

    print(f"\n[{time.strftime('%H:%M:%S')}] sampling SANTA XML-6 sequences",
          flush=True)
    cache = CacheV2(cache_root)
    santa_seqs = sample_santa_xml6(cache, n_seqs=30, rng=rng)
    print(f"  sampled {len(santa_seqs)} SANTA seqs, "
          f"avg_len={int(np.mean([len(s) for s in santa_seqs]))}", flush=True)

    report: dict = {
        "seed": seed,
        "n_real": len(real_seqs),
        "n_santa": len(santa_seqs),
        "real_labels": [lbl for lbl, _ in real],
    }

    # -------- Line 2 first (cheap, no GPU) --------
    print(f"\n=== Line 2: statistical fingerprints ===", flush=True)
    for k in (3, 6):
        p_real = aggregate_kmer_distribution(real_seqs, k)
        p_santa = aggregate_kmer_distribution(santa_seqs, k)
        d = jsd(p_real, p_santa)
        # baseline: real vs real (split in half) for context
        half = len(real_seqs) // 2
        p_real_a = aggregate_kmer_distribution(real_seqs[:half], k)
        p_real_b = aggregate_kmer_distribution(real_seqs[half:], k)
        d_within_real = jsd(p_real_a, p_real_b)
        half_s = len(santa_seqs) // 2
        p_santa_a = aggregate_kmer_distribution(santa_seqs[:half_s], k)
        p_santa_b = aggregate_kmer_distribution(santa_seqs[half_s:], k)
        d_within_santa = jsd(p_santa_a, p_santa_b)
        report[f"kmer_k{k}"] = {
            "jsd_real_vs_santa_bits": d,
            "jsd_within_real_bits": d_within_real,
            "jsd_within_santa_bits": d_within_santa,
        }
        print(f"  k={k}: JSD real↔SANTA = {d:.4f} bits   "
              f"(within-real {d_within_real:.4f}, "
              f"within-SANTA {d_within_santa:.4f})", flush=True)

    gc_real = gc_content_stats(real_seqs)
    gc_santa = gc_content_stats(santa_seqs)
    report["gc_real"] = gc_real
    report["gc_santa"] = gc_santa
    print(f"  GC real:  mean={gc_real['mean_gc']:.4f} "
          f"std={gc_real['std_gc']:.4f} "
          f"[{gc_real['min_gc']:.3f}, {gc_real['max_gc']:.3f}]", flush=True)
    print(f"  GC SANTA: mean={gc_santa['mean_gc']:.4f} "
          f"std={gc_santa['std_gc']:.4f} "
          f"[{gc_santa['min_gc']:.3f}, {gc_santa['max_gc']:.3f}]", flush=True)

    p3_real = period3_strength(real_seqs, window=3000)
    p3_santa = period3_strength(santa_seqs, window=3000)
    report["p3_real"] = p3_real
    report["p3_santa"] = p3_santa
    print(f"  Period-3 real:  "
          f"power_frac={p3_real['mean_p3_power_fraction']:.5f} "
          f"zscore={p3_real['mean_p3_zscore']:.2f} "
          f"(n={p3_real['n_seqs']})", flush=True)
    print(f"  Period-3 SANTA: "
          f"power_frac={p3_santa['mean_p3_power_fraction']:.5f} "
          f"zscore={p3_santa['mean_p3_zscore']:.2f} "
          f"(n={p3_santa['n_seqs']})", flush=True)

    pd_real = pairwise_divergence(real_seqs, n_pairs=50,
                                  rng=random.Random(seed))
    pd_santa = pairwise_divergence(santa_seqs, n_pairs=50,
                                   rng=random.Random(seed + 1))
    report["pairwise_divergence_real"] = pd_real
    report["pairwise_divergence_santa"] = pd_santa
    print(f"  Pairwise Hamming real:  mean={pd_real['mean_hamming']:.4f} "
          f"(n_pairs={pd_real['n_pairs']})", flush=True)
    print(f"  Pairwise Hamming SANTA: mean={pd_santa['mean_hamming']:.4f} "
          f"(n_pairs={pd_santa['n_pairs']})", flush=True)

    # save partial report so Line 2 is durable even if GPU fails
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[{time.strftime('%H:%M:%S')}] Line 2 saved to {out_path}",
          flush=True)

    # -------- Line 1: HyenaDNA probe --------
    print(f"\n=== Line 1: HyenaDNA-small next-token probe on REAL HIV ===",
          flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device={device}", flush=True)
    print(f"  loading {DEFAULT_HF_NAME}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_HF_NAME,
                                                  trust_remote_code=True)
    model.to(device)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    res_real = evaluate_next_token(model, real_seqs, device, batch_size=2)
    report["hyena_small_real_hiv"] = res_real
    print(f"  next-token accuracy on REAL HIV: "
          f"{res_real['next_token_accuracy']:.3f} "
          f"(n_positions={res_real['n_positions_scored']:,})", flush=True)
    for cls, v in res_real["per_class"].items():
        print(f"    {cls}: acc={v['acc']:.3f}  (n={v['n']:,})", flush=True)
    for bkt, v in res_real["per_bucket"].items():
        print(f"    pos {bkt}: acc={v['acc']:.3f}  (n={v['n']:,})",
              flush=True)

    # Direct comparison hooks — pull baselines from cached SANTA JSON
    santa_json = Path("/home/joshc/Dev/RDP_CNN/m12_zeroshot_probe.json")
    if santa_json.exists():
        with santa_json.open() as f:
            santa = json.load(f)
        report["santa_baseline"] = {
            "XML-5_acc": santa["scales"]["XML-5"]["next_token_accuracy"],
            "XML-6_acc": santa["scales"]["XML-6"]["next_token_accuracy"],
            "long30k_acc": santa["scales"]["long_content_30k_001"][
                                                "next_token_accuracy"],
            "mean_acc": santa["overall_accuracy_mean"],
        }
        print(f"\n  COMPARISON:", flush=True)
        print(f"    SANTA XML-6 (length-matched): "
              f"{santa['scales']['XML-6']['next_token_accuracy']:.3f}",
              flush=True)
        print(f"    SANTA mean: "
              f"{santa['overall_accuracy_mean']:.3f}", flush=True)
        print(f"    REAL HIV:    "
              f"{res_real['next_token_accuracy']:.3f}", flush=True)
        delta_xml6 = (res_real["next_token_accuracy"]
                      - santa["scales"]["XML-6"]["next_token_accuracy"])
        report["delta_real_minus_santa_xml6"] = delta_xml6
        print(f"    Δ(real - XML-6) = {delta_xml6:+.3f}", flush=True)

    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[{time.strftime('%H:%M:%S')}] full report saved to {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
