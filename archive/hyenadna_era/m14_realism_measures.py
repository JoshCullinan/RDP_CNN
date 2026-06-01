"""M1.4 SANTA realism audit — does SANTA-simulated training data look like
real recombinant viruses on per-XML basis?

Six measures per SANTA alignment (sampled per XML shard) and per real panel:
  1. Pairwise Hamming mean/std within alignment (50 random pairs)
  2. 6-mer JSD vs corresponding real panel
  3. BP geometry (fragment length, BP-position-relative) — SANTA only
  4. Conservation-vs-declared fitness-sites (Mann-Whitney) — SANTA only
  5. HyenaDNA-small next-token accuracy on sequence #1
  6. RustRDP min p-value across MaxChi/RDP/Chimaera/SiScan/Bootscan

Real panels (one fingerprint each):
  sarscov2_spike / sarscov2_orf1ab / ebola / zika / sarscov2_full /
  hiv_lanl (pooled LANL triplets)

Output: /home/joshc/Dev/RDP_CNN/m14_realism_measures.json (incremental save).
"""

from __future__ import annotations

import json
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from Bio import SeqIO

ROOT = Path("/home/joshc/Dev/RDP_CNN")
sys.path.insert(0, str(ROOT))

from cache_v2_reader import CacheV2  # noqa: E402

# ---- config ----------------------------------------------------------------
CACHE_ROOT = ROOT / "cache" / "v2"
OUT_JSON = ROOT / "m14_realism_measures.json"
PROGRESS_JSON = ROOT / "m14_realism_measures.partial.json"

REAL_PANELS = {
    "sarscov2_spike": ROOT / "data" / "real_recombinants" / "sarscov2_spike" / "aligned.fa",
    "sarscov2_orf1ab": ROOT / "data" / "real_recombinants" / "sarscov2_orf1ab" / "aligned.fa",
    "ebola": ROOT / "data" / "real_recombinants" / "ebola" / "aligned.fa",
    "zika": ROOT / "data" / "real_recombinants" / "zika" / "aligned.fa",
    "sarscov2_full": ROOT / "data" / "real_recombinants" / "sarscov2_full" / "aligned.fa",
    # hiv_lanl built by concatenating LANL triplet FASTAs
}
LANL_TRIPLETS_DIR = ROOT / "data" / "lanl_crf" / "triplets"

XML_TO_PANEL = {
    "XML-1": "sarscov2_spike",
    "XML-2": "hiv_lanl",
    "XML-3": "sarscov2_orf1ab",
    "XML-4": "ebola",
    "XML-5": "zika",
    "long_content_30k_001": "sarscov2_full",
    "long_content_30k_002": "sarscov2_full",
    "long_content_30k_003": "sarscov2_full",
}

XML_CONFIGS = {
    "XML-1": ROOT / "santaSim_RDP" / "XMLs" / "1.xml",
    "XML-2": ROOT / "santaSim_RDP" / "XMLs" / "2.xml",
    "XML-3": ROOT / "santaSim_RDP" / "XMLs" / "3.xml",
    "XML-4": ROOT / "santaSim_RDP" / "XMLs" / "4.xml",
    "XML-5": ROOT / "santaSim_RDP" / "XMLs" / "5.xml",
}

SAMPLE_PER_XML = 50
SAMPLE_LONG_CONTENT = 30  # per long_content shard
RUSTRDP_SUBSAMPLE = 50  # 3-seq triplet runs are fast (~1s), so do full
RUSTRDP_BIN = "/home/joshc/Dev/RustRDP/target/release/rustrdp"
RUSTRDP_METHODS = ("maxchi", "rdp", "chimaera", "siscan")  # bootscan too slow at length
RUSTRDP_TIMEOUT = 60
RUSTRDP_N_SEQS = 3  # use a single triplet (recomb + 2 parents) per alignment
HAMMING_N_PAIRS = 50
KMER_K = 6
SEED = 0

# RSS watchdog
_RSS_CEILING_BYTES = 26 * 1024 * 1024 * 1024


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**30


def _watchdog(label: str = "") -> None:
    rss = _rss_gb()
    if rss > _RSS_CEILING_BYTES / 2**30:
        raise MemoryError(f"RSS watchdog tripped at {label}: {rss:.1f} GB")


# ---- nucleotide encoding helpers ------------------------------------------
NT_TO_INT = {"A": 0, "T": 1, "G": 2, "C": 3, "-": 4}
INT_TO_NT = "ATGC-"


def str_to_int8(seq: str) -> np.ndarray:
    """Map a sequence string to int8 with A/T/G/C/- and N->4 (gap)."""
    s = seq.upper()
    arr = np.full(len(s), 4, dtype=np.int8)
    for c, i in NT_TO_INT.items():
        # vectorized: not needed for small alignments; do per-char in a numpy way
        pass
    # vectorized: use frombuffer
    buf = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    arr = np.full(buf.shape, 4, dtype=np.int8)
    arr[buf == ord("A")] = 0
    arr[buf == ord("T")] = 1
    arr[buf == ord("U")] = 1
    arr[buf == ord("G")] = 2
    arr[buf == ord("C")] = 3
    # everything else (gap, N, IUPAC) stays 4
    return arr


def load_fasta_as_int8(path: Path) -> np.ndarray:
    """Return (n, L) int8 matrix, padding to max length with gaps."""
    seqs = []
    max_len = 0
    for r in SeqIO.parse(str(path), "fasta"):
        arr = str_to_int8(str(r.seq))
        seqs.append(arr)
        if len(arr) > max_len:
            max_len = len(arr)
    if not seqs:
        return np.zeros((0, 0), dtype=np.int8)
    mat = np.full((len(seqs), max_len), 4, dtype=np.int8)
    for i, s in enumerate(seqs):
        mat[i, : len(s)] = s
    return mat


# ---- Measure 1: Hamming -----------------------------------------------------

def pairwise_hamming(align: np.ndarray, n_pairs: int, rng: random.Random) -> dict:
    """Mean/std Hamming over n_pairs random pairs from the alignment."""
    n = align.shape[0]
    if n < 2:
        return {"hamming_mean": float("nan"), "hamming_std": float("nan"), "n_pairs": 0}
    pairs = []
    seen = set()
    max_pairs = n * (n - 1) // 2
    target = min(n_pairs, max_pairs)
    while len(pairs) < target:
        i, j = rng.sample(range(n), 2)
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    dists = []
    for (i, j) in pairs:
        a, b = align[i], align[j]
        # ignore positions where either is gap
        mask = (a != 4) & (b != 4)
        if mask.sum() == 0:
            continue
        d = (a[mask] != b[mask]).mean()
        dists.append(float(d))
    if not dists:
        return {"hamming_mean": float("nan"), "hamming_std": float("nan"), "n_pairs": 0}
    return {
        "hamming_mean": float(np.mean(dists)),
        "hamming_std": float(np.std(dists)),
        "n_pairs": len(dists),
    }


# ---- Measure 2: 6-mer JSD --------------------------------------------------

def kmer_freq(align: np.ndarray, k: int = KMER_K) -> np.ndarray:
    """Return normalized k-mer freq vector over 4**k (A/T/G/C only; gaps skipped).
    Concatenate all sequences via the int8 array (gaps are skipped within each).
    """
    counts = np.zeros(4 ** k, dtype=np.float64)
    for row in align:
        # Drop gap and non-ATGC (still int8 with code <=3 are ATGC, 4 is gap)
        clean = row[row <= 3].astype(np.int64)
        if len(clean) < k:
            continue
        # encode k-mers via base-4 numbering
        idx = clean[: len(clean) - k + 1].copy()
        for off in range(1, k):
            idx = idx * 4 + clean[off : off + len(idx)]
        # bincount
        bc = np.bincount(idx, minlength=4 ** k)
        counts += bc
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base 2). Returns value in [0, 1]."""
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    m = 0.5 * (p + q)
    def kld(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * (kld(p, m) + kld(q, m))


# ---- Measure 4: conservation vs declared fitness sites ----------------------

_FITNESS_SITE_RE = re.compile(
    r"<purifyingFitness[^>]*>.*?<sites>(.*?)</sites>.*?</purifyingFitness>",
    re.DOTALL,
)


def parse_fitness_sites(xml_path: Path, genome_len: int) -> list[int]:
    """Return 0-indexed positions declared in <purifyingFitness><sites> blocks.
    XML uses 1-based, comma/space-separated, with ranges 'a-b' (inclusive).
    """
    text = xml_path.read_text()
    sites: set[int] = set()
    for m in _FITNESS_SITE_RE.finditer(text):
        for tok in re.split(r"[,\s]+", m.group(1).strip()):
            if not tok:
                continue
            if "-" in tok:
                a, b = tok.split("-", 1)
                try:
                    a_i, b_i = int(a), int(b)
                except ValueError:
                    continue
                for p in range(a_i, b_i + 1):
                    if 1 <= p <= genome_len:
                        sites.add(p - 1)
            else:
                try:
                    p = int(tok)
                except ValueError:
                    continue
                if 1 <= p <= genome_len:
                    sites.add(p - 1)
    return sorted(sites)


def per_position_entropy(align: np.ndarray) -> np.ndarray:
    """Shannon entropy across A/T/G/C (gaps excluded) per column."""
    n, L = align.shape
    ent = np.full(L, np.nan, dtype=np.float64)
    for j in range(L):
        col = align[:, j]
        col = col[col <= 3]
        if len(col) == 0:
            continue
        counts = np.bincount(col, minlength=4).astype(np.float64)
        p = counts / counts.sum()
        p_nz = p[p > 0]
        ent[j] = -float(np.sum(p_nz * np.log2(p_nz)))
    return ent


# ---- Measure 5: HyenaDNA next-token -----------------------------------------

_HYENA_MODEL = None
_HYENA_DEVICE = None


def get_hyena():
    global _HYENA_MODEL, _HYENA_DEVICE
    if _HYENA_MODEL is None:
        import torch
        from transformers import AutoModelForCausalLM
        _HYENA_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if _HYENA_DEVICE == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.8)
        print(f"  [hyena] loading on {_HYENA_DEVICE}...", flush=True)
        _HYENA_MODEL = AutoModelForCausalLM.from_pretrained(
            "LongSafari/hyenadna-small-32k-seqlen-hf",
            trust_remote_code=True,
        )
        _HYENA_MODEL.to(_HYENA_DEVICE)
        _HYENA_MODEL.eval()
        print(f"  [hyena] params: {sum(p.numel() for p in _HYENA_MODEL.parameters()):,}",
              flush=True)
    return _HYENA_MODEL, _HYENA_DEVICE


_V2_TO_HYENA = np.array([7, 10, 9, 8, 4], dtype=np.int64)  # A T G C gap


def hyena_next_token_acc(seq_int8: np.ndarray) -> float:
    """Next-token accuracy on one sequence (cap at 32k tokens)."""
    import torch
    model, device = get_hyena()
    seq = seq_int8[:32_000].clip(0, 4)  # truncate for safety
    ids_np = _V2_TO_HYENA[seq].astype(np.int64)
    ids = torch.from_numpy(ids_np[None]).to(device)
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits  # (1, L, V)
    pred = logits.argmax(dim=-1)[0].cpu().numpy()
    targ = ids_np[1:]
    pr = pred[:-1]
    # restrict to nt tokens 7..10
    mask = np.isin(targ, [7, 8, 9, 10])
    if not mask.any():
        return float("nan")
    acc = float((pr[mask] == targ[mask]).mean())
    del ids, out, logits, pred
    if device == "cuda":
        torch.cuda.empty_cache()
    return acc


# ---- Measure 6: RustRDP min p-value ----------------------------------------

def write_alignment_fasta(align: np.ndarray, path: Path, max_seqs: int = 20):
    """Write up to max_seqs sequences to FASTA from int8 array."""
    n = min(align.shape[0], max_seqs)
    table = np.array(list("ATGC-"), dtype="U1")
    with open(path, "w") as f:
        for i in range(n):
            row = align[i]
            # strip trailing gaps for slight speedup; RustRDP handles aligned data
            s = "".join(table[row.clip(0, 4)])
            f.write(f">seq_{i}\n{s}\n")


def rustrdp_min_pvalue(align: np.ndarray, label: str,
                       row_idxs: tuple[int, int, int] | None = None) -> dict:
    """Run RustRDP on a 3-seq triplet from the alignment.
    Triplet selection: explicit row_idxs (recomb, p1, p2), else first 3 rows.
    Returns min p-value across methods (maxchi/rdp/chimaera/siscan)."""
    with tempfile.NamedTemporaryFile(prefix="m14_", suffix=".fa", delete=False) as tf:
        fa = Path(tf.name)
    csv_out = fa.with_suffix(".csv")
    try:
        if row_idxs is not None:
            sub = align[list(row_idxs)]
        else:
            sub = align[: min(RUSTRDP_N_SEQS, align.shape[0])]
        write_alignment_fasta(sub, fa, max_seqs=RUSTRDP_N_SEQS)
        cmd = [
            RUSTRDP_BIN, "-i", str(fa),
            "-m", ",".join(RUSTRDP_METHODS),
            "-o", str(csv_out),
            "--max-pvalue", "1.0",
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=RUSTRDP_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"min_pvalue": float("nan"), "per_method": {}, "status": "timeout"}
        if res.returncode != 0:
            return {"min_pvalue": float("nan"), "per_method": {}, "status": f"err: {res.stderr[:200]}"}
        # parse CSV
        by_method: dict[str, float] = {}
        if not csv_out.exists():
            return {"min_pvalue": float("nan"), "per_method": {}, "status": "no_output"}
        with open(csv_out) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Method,"):
                    continue
                parts = line.split(",")
                if len(parts) < 7:
                    continue
                method = parts[0]
                try:
                    p = float(parts[6])
                except ValueError:
                    continue
                if method not in by_method or p < by_method[method]:
                    by_method[method] = p
        if not by_method:
            return {"min_pvalue": float("nan"), "per_method": {}, "status": "no_hits"}
        return {
            "min_pvalue": float(min(by_method.values())),
            "per_method": by_method,
            "status": "ok",
        }
    finally:
        for p in (fa, csv_out):
            try: p.unlink()
            except OSError: pass


# ---- main -------------------------------------------------------------------

def build_hiv_lanl_panel() -> np.ndarray:
    """Concatenate all 4 LANL CRF triplet FASTAs into one panel.
    Triplets are aligned within each file but different files have different
    lengths; pad each separately to its own max, then stack with gap-pad.
    """
    all_rows: list[np.ndarray] = []
    max_len = 0
    for fa in sorted(LANL_TRIPLETS_DIR.glob("*.fa")):
        for r in SeqIO.parse(str(fa), "fasta"):
            arr = str_to_int8(str(r.seq))
            all_rows.append(arr)
            if len(arr) > max_len:
                max_len = len(arr)
    if not all_rows:
        return np.zeros((0, 0), dtype=np.int8)
    mat = np.full((len(all_rows), max_len), 4, dtype=np.int8)
    for i, s in enumerate(all_rows):
        mat[i, : len(s)] = s
    return mat


def fingerprint_panel(name: str, align: np.ndarray, rng: random.Random) -> dict:
    print(f"  panel '{name}': {align.shape}", flush=True)
    out = {"n_seqs": int(align.shape[0]), "seq_len": int(align.shape[1])}
    out.update(pairwise_hamming(align, HAMMING_N_PAIRS, rng))
    out["kmer_freq"] = kmer_freq(align).tolist()  # store for later JSDs; pruned at save
    # Hyena on sequence #1
    try:
        if align.shape[0] >= 1:
            acc = hyena_next_token_acc(align[0])
            out["hyena_acc"] = acc
            print(f"    hyena: {acc:.3f}", flush=True)
    except Exception as e:
        out["hyena_acc"] = float("nan")
        out["hyena_err"] = str(e)[:200]
    # RustRDP
    try:
        r = rustrdp_min_pvalue(align, f"panel_{name}")
        out["rdp_min_pvalue"] = r["min_pvalue"]
        out["rdp_per_method"] = r["per_method"]
        out["rdp_status"] = r["status"]
        print(f"    rdp min p: {r['min_pvalue']}", flush=True)
    except Exception as e:
        out["rdp_min_pvalue"] = float("nan")
        out["rdp_status"] = f"err: {e}"
    return out


def fingerprint_santa_alignment(
    align: np.ndarray,
    bp_starts: list[int],
    bp_ends: list[int],
    seq_len: int,
    real_kmer: np.ndarray,
    fitness_sites: list[int],
    do_rustrdp: bool,
    rng: random.Random,
    event_triplet_rows: tuple[int, int, int] | None = None,
) -> dict:
    out: dict = {}
    out["n_seqs"] = int(align.shape[0])
    out["seq_len"] = int(align.shape[1])
    # Measure 1
    out.update(pairwise_hamming(align, HAMMING_N_PAIRS, rng))
    # Measure 2
    k = kmer_freq(align)
    out["6mer_jsd"] = jsd(k, real_kmer) if real_kmer is not None and real_kmer.sum() > 0 else float("nan")
    # Measure 3
    out["fragment_lengths"] = [int(e - s) for s, e in zip(bp_starts, bp_ends)]
    out["bp_positions_rel"] = [float(s) / seq_len if seq_len > 0 else float("nan")
                               for s in bp_starts]
    # Measure 4
    if fitness_sites:
        ent = per_position_entropy(align)
        L = ent.shape[0]
        # constrain to valid sites
        fs = [p for p in fitness_sites if 0 <= p < L]
        ns_mask = np.ones(L, dtype=bool)
        ns_mask[fs] = False
        fs_ent = ent[fs]
        ns_ent = ent[ns_mask]
        fs_ent = fs_ent[~np.isnan(fs_ent)]
        ns_ent = ns_ent[~np.isnan(ns_ent)]
        if len(fs_ent) > 0 and len(ns_ent) > 0:
            out["fitness_sites_entropy_mean"] = float(fs_ent.mean())
            out["non_fitness_sites_entropy_mean"] = float(ns_ent.mean())
            out["n_fitness_sites_used"] = int(len(fs_ent))
            # MWU
            try:
                from scipy.stats import mannwhitneyu
                stat, pval = mannwhitneyu(fs_ent, ns_ent, alternative="less")
                out["fitness_mwu_pvalue"] = float(pval)
            except Exception:
                out["fitness_mwu_pvalue"] = float("nan")
        else:
            out["fitness_sites_entropy_mean"] = float("nan")
            out["non_fitness_sites_entropy_mean"] = float("nan")
    # Measure 5
    try:
        if align.shape[0] >= 1:
            out["hyena_acc"] = hyena_next_token_acc(align[0])
    except Exception as e:
        out["hyena_acc"] = float("nan")
        out["hyena_err"] = str(e)[:200]
    # Measure 6 (subsampled)
    if do_rustrdp:
        try:
            r = rustrdp_min_pvalue(align, "santa", row_idxs=event_triplet_rows)
            out["rdp_min_pvalue"] = r["min_pvalue"]
            out["rdp_status"] = r["status"]
            out["rdp_per_method"] = r["per_method"]
        except Exception as e:
            out["rdp_min_pvalue"] = float("nan")
            out["rdp_status"] = f"err: {e}"
    else:
        out["rdp_min_pvalue"] = None
        out["rdp_status"] = "skipped"
    return out


def collect_santa_bp_data(shard, file_idx: int) -> tuple[list[int], list[int], tuple[int, int, int] | None]:
    """Get all bp_start, bp_end for events in this file, plus a representative
    (recomb, p1, p2) row-index triple from the first event (0-indexed)."""
    events = shard.events
    mask = events["file_idx"] == file_idx
    rows = events[mask]
    bp_s = [int(r["bp_start"]) for r in rows]
    bp_e = [int(r["bp_end"]) for r in rows]
    triplet = None
    if len(rows) > 0:
        r0 = rows[0]
        # event ids are 1-based in cache; convert to 0-based row indices
        triplet = (int(r0["recomb_id"]) - 1, int(r0["p1_id"]) - 1, int(r0["p2_id"]) - 1)
    return bp_s, bp_e, triplet


def main():
    t_start = time.time()
    rng = random.Random(SEED)

    # ----- step 1: load v2 cache -----
    print(f"[{time.strftime('%H:%M:%S')}] loading cache from {CACHE_ROOT}", flush=True)
    cache = CacheV2(CACHE_ROOT)
    print(f"  shards: {list(cache.shards.keys())}", flush=True)

    # ----- step 2: build/load real panels -----
    print(f"\n[{time.strftime('%H:%M:%S')}] building real panels...", flush=True)
    real_panel_aligns: dict[str, np.ndarray] = {}
    real_panel_kmers: dict[str, np.ndarray] = {}
    real_panel_fp: dict[str, dict] = {}
    for name, path in REAL_PANELS.items():
        if not path.exists():
            print(f"  MISSING: {path}", flush=True)
            continue
        align = load_fasta_as_int8(path)
        real_panel_aligns[name] = align
    # hiv_lanl
    hiv = build_hiv_lanl_panel()
    real_panel_aligns["hiv_lanl"] = hiv

    for name, align in real_panel_aligns.items():
        real_panel_kmers[name] = kmer_freq(align)
        fp = fingerprint_panel(name, align, rng)
        # JSD with self = 0
        fp["6mer_jsd_self"] = 0.0
        # drop the heavy kmer_freq array from JSON output
        fp.pop("kmer_freq", None)
        real_panel_fp[name] = fp
        _watchdog(label=f"after panel {name}")

    # ----- step 3: parse fitness sites -----
    print(f"\n[{time.strftime('%H:%M:%S')}] parsing XML fitness sites...", flush=True)
    fitness_sites_by_xml: dict[str, list[int]] = {}
    genome_lens = {"XML-1": 3822, "XML-2": 9181, "XML-3": 5800, "XML-4": 19025, "XML-5": 10727}
    for xml_name, xml_path in XML_CONFIGS.items():
        L = genome_lens[xml_name]
        sites = parse_fitness_sites(xml_path, L)
        fitness_sites_by_xml[xml_name] = sites
        print(f"  {xml_name}: {len(sites)} declared fitness positions", flush=True)

    # ----- step 4: per-shard sampling and fingerprints -----
    shard_plan = [
        ("XML-1", SAMPLE_PER_XML, RUSTRDP_SUBSAMPLE),
        ("XML-2", SAMPLE_PER_XML, RUSTRDP_SUBSAMPLE),
        ("XML-3", SAMPLE_PER_XML, RUSTRDP_SUBSAMPLE),
        ("XML-4", SAMPLE_PER_XML, RUSTRDP_SUBSAMPLE),
        ("XML-5", SAMPLE_PER_XML, RUSTRDP_SUBSAMPLE),
        ("long_content_30k_001", SAMPLE_LONG_CONTENT, SAMPLE_LONG_CONTENT),
        ("long_content_30k_002", SAMPLE_LONG_CONTENT, SAMPLE_LONG_CONTENT),
        ("long_content_30k_003", SAMPLE_LONG_CONTENT, SAMPLE_LONG_CONTENT),
    ]

    all_santa_records: list[dict] = []

    for xml_name, n_files, n_rdp in shard_plan:
        if xml_name not in cache.shards:
            print(f"  skip {xml_name}: shard not loaded", flush=True)
            continue
        shard = cache.shards[xml_name]
        panel_name = XML_TO_PANEL[xml_name]
        real_kmer = real_panel_kmers.get(panel_name)
        fitness_sites = fitness_sites_by_xml.get(xml_name, [])
        avail = shard.n_files
        # Fixed-seeded selection
        local_rng = random.Random(SEED + hash(xml_name) % 1000)
        sample_idxs = local_rng.sample(range(avail), min(n_files, avail))
        # pick RustRDP subset (first n_rdp of sample for stability)
        rdp_subset = set(sample_idxs[:n_rdp])
        print(f"\n[{time.strftime('%H:%M:%S')}] {xml_name}: sampling "
              f"{len(sample_idxs)} files (RustRDP on {len(rdp_subset)})", flush=True)
        t0 = time.time()
        for i, fi in enumerate(sample_idxs):
            try:
                align = shard.get_alignment(fi)
            except Exception as e:
                print(f"  err getting align {fi}: {e}", flush=True)
                continue
            bp_starts, bp_ends, triplet_rows = collect_santa_bp_data(shard, fi)
            seq_len = int(align.shape[1])
            do_rustrdp = fi in rdp_subset
            # validate triplet rows against alignment shape
            if triplet_rows is not None and any(r < 0 or r >= align.shape[0] for r in triplet_rows):
                triplet_rows = None
            try:
                rec = fingerprint_santa_alignment(
                    align=align,
                    bp_starts=bp_starts,
                    bp_ends=bp_ends,
                    seq_len=seq_len,
                    real_kmer=real_kmer,
                    fitness_sites=fitness_sites,
                    do_rustrdp=do_rustrdp,
                    rng=rng,
                    event_triplet_rows=triplet_rows,
                )
            except Exception as e:
                print(f"  err in fingerprint shard {xml_name} fi={fi}: {e}", flush=True)
                continue
            rec["xml_shard"] = xml_name
            rec["file_idx"] = int(fi)
            rec["source_file"] = shard.files[fi] if fi < len(shard.files) else ""
            rec["real_comparison"] = panel_name
            all_santa_records.append(rec)
            _watchdog(label=f"{xml_name} {i}/{len(sample_idxs)}")
            if (i + 1) % 5 == 0 or i + 1 == len(sample_idxs):
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                print(f"    [{i+1}/{len(sample_idxs)}] "
                      f"{rate:.2f} files/s  RSS {_rss_gb():.1f} GB", flush=True)
        # incremental save
        _save_progress(real_panel_fp, all_santa_records, fitness_sites_by_xml)

    # ----- step 5: summarize per XML -----
    summary = summarize(all_santa_records, real_panel_fp)

    out = {
        "config": {
            "sample_per_xml": SAMPLE_PER_XML,
            "sample_per_long_content": SAMPLE_LONG_CONTENT,
            "rustrdp_subsample_per_shard": RUSTRDP_SUBSAMPLE,
            "rustrdp_subsample_long_content": 10,
            "kmer_k": KMER_K,
            "hamming_n_pairs": HAMMING_N_PAIRS,
            "rustrdp_methods": list(RUSTRDP_METHODS),
            "rustrdp_max_pvalue": 1.0,
            "rustrdp_timeout_sec": RUSTRDP_TIMEOUT,
            "seed": SEED,
        },
        "real_panels": real_panel_fp,
        "fitness_sites_by_xml": {k: len(v) for k, v in fitness_sites_by_xml.items()},
        "santa_alignments": all_santa_records,
        "summary_per_xml": summary,
        "elapsed_sec": time.time() - t_start,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT_JSON} (elapsed {time.time()-t_start:.0f}s)", flush=True)
    print_headline(summary, real_panel_fp)


def _save_progress(real_panel_fp, santa_records, fitness_sites_by_xml):
    try:
        with open(PROGRESS_JSON, "w") as f:
            json.dump({
                "real_panels": real_panel_fp,
                "santa_alignments": santa_records,
                "fitness_sites_by_xml": {k: len(v) for k, v in fitness_sites_by_xml.items()},
            }, f, indent=2)
    except Exception as e:
        print(f"  progress save err: {e}", flush=True)


def summarize(records: list[dict], real_panel_fp: dict) -> dict:
    out = {}
    by_shard: dict[str, list[dict]] = {}
    for r in records:
        by_shard.setdefault(r["xml_shard"], []).append(r)
    for xml_name, recs in by_shard.items():
        def m(field):
            vals = [r.get(field) for r in recs
                    if r.get(field) is not None and not (isinstance(r.get(field), float) and np.isnan(r[field]))]
            return float(np.mean(vals)) if vals else None
        def frac_lt(field, thresh):
            vals = [r.get(field) for r in recs
                    if r.get(field) is not None and not (isinstance(r.get(field), float) and np.isnan(r[field]))]
            if not vals: return None
            return float(sum(1 for v in vals if v < thresh) / len(vals))
        # fitness sig: fraction of alignments where MWU p < 0.05 (conservation lower at sites)
        fitness_p = [r.get("fitness_mwu_pvalue") for r in recs
                     if r.get("fitness_mwu_pvalue") is not None and not (isinstance(r.get("fitness_mwu_pvalue"), float) and np.isnan(r["fitness_mwu_pvalue"]))]
        frac_fit_sig = (sum(1 for p in fitness_p if p < 0.05) / len(fitness_p)) if fitness_p else None
        all_frag = []
        all_relpos = []
        for r in recs:
            all_frag.extend(r.get("fragment_lengths") or [])
            all_relpos.extend([x for x in (r.get("bp_positions_rel") or []) if x == x])
        out[xml_name] = {
            "n_alignments": len(recs),
            "real_comparison": recs[0].get("real_comparison"),
            "hamming_mean_mean": m("hamming_mean"),
            "hamming_std_mean": m("hamming_std"),
            "6mer_jsd_mean": m("6mer_jsd"),
            "fitness_sites_entropy_mean_mean": m("fitness_sites_entropy_mean"),
            "non_fitness_sites_entropy_mean_mean": m("non_fitness_sites_entropy_mean"),
            "frac_alignments_with_fitness_mwu_sig": frac_fit_sig,
            "hyena_acc_mean": m("hyena_acc"),
            "rdp_min_pvalue_mean": m("rdp_min_pvalue"),
            "frac_alignments_rdp_lt_001": frac_lt("rdp_min_pvalue", 0.001),
            "frac_alignments_rdp_lt_005": frac_lt("rdp_min_pvalue", 0.05),
            "fragment_length_median": float(np.median(all_frag)) if all_frag else None,
            "bp_positions_rel_mean": float(np.mean(all_relpos)) if all_relpos else None,
        }
    return out


def print_headline(summary, real_panel_fp):
    print(f"\n{'='*100}")
    print(f"  M1.4 SANTA realism — per-XML aggregates vs real panel")
    print(f"{'='*100}")
    hdr = ("shard", "real", "Hmean", "rHmean", "JSD", "rJSD?", "Hyena", "rHyena",
           "RDP<.001")
    print(f"  {hdr[0]:<22} {hdr[1]:<15} {hdr[2]:>7} {hdr[3]:>7} {hdr[4]:>6} {hdr[5]:>6} {hdr[6]:>6} {hdr[7]:>7} {hdr[8]:>9}")
    for xml_name, s in summary.items():
        panel = s["real_comparison"]
        rp = real_panel_fp.get(panel, {})
        rh = rp.get("hamming_mean", float("nan"))
        rhy = rp.get("hyena_acc", float("nan"))
        def fmt(x, p=3):
            return ("nan" if x is None or (isinstance(x, float) and np.isnan(x))
                    else f"{x:.{p}f}")
        print(f"  {xml_name:<22} {panel:<15} {fmt(s.get('hamming_mean_mean')):>7} "
              f"{fmt(rh):>7} {fmt(s.get('6mer_jsd_mean')):>6} {'-':>6} "
              f"{fmt(s.get('hyena_acc_mean')):>6} {fmt(rhy):>7} "
              f"{fmt(s.get('frac_alignments_rdp_lt_001'), 2):>9}")


if __name__ == "__main__":
    main()
