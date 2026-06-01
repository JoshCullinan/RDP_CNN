"""Resume M1.4: compute remaining XML-4(10) + XML-5(20) + long_content_30k_001/002/003 records.

Skips RustRDP (set to None / "skipped") because it isn't discriminating on SANTA alignments.
Loads existing partial from m14_realism_measures.json.tmp, appends new records, computes
summary_per_xml, writes final m14_realism_measures.json.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/joshc/Dev/RDP_CNN")
sys.path.insert(0, str(ROOT))

# Import the heavy lifters from the original script
from m14_realism_measures import (  # noqa: E402
    CACHE_ROOT,
    SEED,
    HAMMING_N_PAIRS,
    pairwise_hamming,
    kmer_freq,
    jsd,
    per_position_entropy,
    hyena_next_token_acc,
    collect_santa_bp_data,
    load_fasta_as_int8,
    build_hiv_lanl_panel,
    str_to_int8,
    XML_TO_PANEL,
    XML_CONFIGS,
    REAL_PANELS,
    parse_fitness_sites,
    _rss_gb,
    _watchdog,
)
from cache_v2_reader import CacheV2  # noqa: E402

PARTIAL = ROOT / "m14_realism_measures.json.tmp"
OUT_JSON = ROOT / "m14_realism_measures.json"
PROGRESS_JSON = ROOT / "m14_realism_measures.partial.json"

# Per-shard new record targets the user requested
RESUME_PLAN = [
    ("XML-4", 10),
    ("XML-5", 20),
    ("long_content_30k_001", 7),
    ("long_content_30k_002", 7),
    ("long_content_30k_003", 6),
]
# Original sampling sizes (must match m14_realism_measures.py so seeded sample lines up)
ORIG_SAMPLE_SIZE = {
    "XML-4": 20,
    "XML-5": 50,  # SAMPLE_PER_XML
    "long_content_30k_001": 30,  # SAMPLE_LONG_CONTENT
    "long_content_30k_002": 30,
    "long_content_30k_003": 30,
}


def fingerprint_no_rdp(align, bp_starts, bp_ends, seq_len, real_kmer,
                      fitness_sites, rng):
    out = {}
    out["n_seqs"] = int(align.shape[0])
    out["seq_len"] = int(align.shape[1])
    out.update(pairwise_hamming(align, HAMMING_N_PAIRS, rng))
    k = kmer_freq(align)
    out["6mer_jsd"] = (jsd(k, real_kmer)
                      if real_kmer is not None and real_kmer.sum() > 0
                      else float("nan"))
    out["fragment_lengths"] = [int(e - s) for s, e in zip(bp_starts, bp_ends)]
    out["bp_positions_rel"] = [float(s) / seq_len if seq_len > 0 else float("nan")
                               for s in bp_starts]
    if fitness_sites:
        ent = per_position_entropy(align)
        L = ent.shape[0]
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
            try:
                from scipy.stats import mannwhitneyu
                stat, pval = mannwhitneyu(fs_ent, ns_ent, alternative="less")
                out["fitness_mwu_pvalue"] = float(pval)
            except Exception:
                out["fitness_mwu_pvalue"] = float("nan")
        else:
            out["fitness_sites_entropy_mean"] = float("nan")
            out["non_fitness_sites_entropy_mean"] = float("nan")
    # Measure 5: HyenaDNA
    try:
        if align.shape[0] >= 1:
            out["hyena_acc"] = hyena_next_token_acc(align[0])
    except Exception as e:
        out["hyena_acc"] = float("nan")
        out["hyena_err"] = str(e)[:200]
    # Measure 6: SKIPPED
    out["rdp_min_pvalue"] = None
    out["rdp_status"] = "skipped"
    return out


def save_progress(real_panels, all_records, fitness_sites_by_xml):
    try:
        with open(PROGRESS_JSON, "w") as f:
            json.dump({
                "real_panels": real_panels,
                "santa_alignments": all_records,
                "fitness_sites_by_xml": fitness_sites_by_xml,
            }, f, indent=2)
    except Exception as e:
        print(f"  progress save err: {e}", flush=True)


def summarize(records):
    out = {}
    by_shard = {}
    for r in records:
        by_shard.setdefault(r["xml_shard"], []).append(r)
    for xml_name, recs in by_shard.items():
        def vals_of(field):
            return [r.get(field) for r in recs
                    if r.get(field) is not None
                    and not (isinstance(r.get(field), float) and np.isnan(r[field]))]
        def stat(field):
            v = vals_of(field)
            if not v:
                return {"mean": None, "median": None, "std": None, "n": 0}
            arr = np.array(v, dtype=np.float64)
            return {"mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "std": float(arr.std()),
                    "n": int(len(v))}
        hyena = stat("hyena_acc")
        hamming = stat("hamming_mean")
        jsd_s = stat("6mer_jsd")
        out[xml_name] = {
            "n_alignments": len(recs),
            "real_comparison": recs[0].get("real_comparison"),
            "hamming_mean": hamming,
            "6mer_jsd": jsd_s,
            "hyena_acc": hyena,
        }
    return out


def main():
    t_start = time.time()
    # ---- load partial ----
    print(f"[{time.strftime('%H:%M:%S')}] loading partial: {PARTIAL}", flush=True)
    with open(PARTIAL) as f:
        partial = json.load(f)
    real_panel_fp = partial["real_panels"]
    all_records = list(partial["santa_alignments"])
    fitness_sites_count = partial["fitness_sites_by_xml"]
    print(f"  starting with {len(all_records)} santa records", flush=True)

    done_by_shard = {}
    for r in all_records:
        done_by_shard.setdefault(r["xml_shard"], set()).add(r["file_idx"])

    # ---- need fitness sites lists (not just counts) and real panel kmers ----
    print(f"[{time.strftime('%H:%M:%S')}] re-parsing XML fitness sites...", flush=True)
    genome_lens = {"XML-1": 3822, "XML-2": 9181, "XML-3": 5800,
                   "XML-4": 19025, "XML-5": 10727}
    fitness_sites_by_xml = {}
    for xml_name, xml_path in XML_CONFIGS.items():
        L = genome_lens[xml_name]
        sites = parse_fitness_sites(xml_path, L)
        fitness_sites_by_xml[xml_name] = sites

    # long_content shards: assume no fitness sites declared (not in XML_CONFIGS)
    for sn in ["long_content_30k_001", "long_content_30k_002", "long_content_30k_003"]:
        fitness_sites_by_xml.setdefault(sn, [])

    # ---- rebuild real panel kmers for JSD ----
    print(f"[{time.strftime('%H:%M:%S')}] rebuilding real panel kmers...", flush=True)
    real_panel_kmers = {}
    for name, path in REAL_PANELS.items():
        if not path.exists():
            continue
        align = load_fasta_as_int8(path)
        real_panel_kmers[name] = kmer_freq(align)
        print(f"  {name}: shape={align.shape}", flush=True)
    hiv = build_hiv_lanl_panel()
    real_panel_kmers["hiv_lanl"] = kmer_freq(hiv)
    print(f"  hiv_lanl: shape={hiv.shape}", flush=True)
    del hiv

    # ---- cache ----
    print(f"[{time.strftime('%H:%M:%S')}] loading cache from {CACHE_ROOT}", flush=True)
    cache = CacheV2(CACHE_ROOT)

    rng = random.Random(SEED)

    # ---- per-shard ----
    for xml_name, n_needed in RESUME_PLAN:
        if xml_name not in cache.shards:
            print(f"  skip {xml_name}: shard not loaded", flush=True)
            continue
        shard = cache.shards[xml_name]
        panel_name = XML_TO_PANEL[xml_name]
        real_kmer = real_panel_kmers.get(panel_name)
        fitness_sites = fitness_sites_by_xml.get(xml_name, [])
        avail = shard.n_files

        # Reproduce the original seeded sample so XML-4 already-done indices line up
        orig_n = ORIG_SAMPLE_SIZE.get(xml_name, n_needed)
        local_rng = random.Random(SEED + hash(xml_name) % 1000)
        sample_idxs = local_rng.sample(range(avail), min(orig_n, avail))

        done_set = done_by_shard.get(xml_name, set())
        new_idxs = [fi for fi in sample_idxs if fi not in done_set][:n_needed]
        # If the sample yields too few (e.g. for XML-5 we want 20 new but orig was 50 — easily enough),
        # the slice above handles it
        print(f"\n[{time.strftime('%H:%M:%S')}] {xml_name}: orig_sample={orig_n} avail={avail} "
              f"already_done={len(done_set)} new={len(new_idxs)}", flush=True)
        print(f"  new file_idxs: {new_idxs}", flush=True)

        t0 = time.time()
        for i, fi in enumerate(new_idxs):
            try:
                align = shard.get_alignment(fi)
            except Exception as e:
                print(f"  err getting align {fi}: {e}", flush=True)
                continue
            bp_starts, bp_ends, _triplet_rows = collect_santa_bp_data(shard, fi)
            seq_len = int(align.shape[1])
            try:
                rec = fingerprint_no_rdp(
                    align=align,
                    bp_starts=bp_starts,
                    bp_ends=bp_ends,
                    seq_len=seq_len,
                    real_kmer=real_kmer,
                    fitness_sites=fitness_sites,
                    rng=rng,
                )
            except Exception as e:
                print(f"  err fingerprint shard {xml_name} fi={fi}: {e}", flush=True)
                continue
            rec["xml_shard"] = xml_name
            rec["file_idx"] = int(fi)
            rec["source_file"] = shard.files[fi] if fi < len(shard.files) else ""
            rec["real_comparison"] = panel_name
            all_records.append(rec)
            _watchdog(label=f"{xml_name} {i}/{len(new_idxs)}")
            if (i + 1) % 5 == 0 or i + 1 == len(new_idxs):
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                print(f"    [{i+1}/{len(new_idxs)}] {rate:.2f} files/s  "
                      f"RSS {_rss_gb():.1f} GB  hamming={rec.get('hamming_mean'):.4f} "
                      f"jsd={rec.get('6mer_jsd'):.4f} hyena={rec.get('hyena_acc')}",
                      flush=True)
            # save every 10 records
            if len(all_records) % 10 == 0:
                save_progress(real_panel_fp, all_records,
                              {k: len(v) for k, v in fitness_sites_by_xml.items()})

    # ---- summary + final write ----
    print(f"\n[{time.strftime('%H:%M:%S')}] summarizing and writing final", flush=True)
    summary = summarize(all_records)
    final = {
        "config": {
            "seed": SEED,
            "kmer_k": 6,
            "hamming_n_pairs": HAMMING_N_PAIRS,
            "resume_plan": RESUME_PLAN,
            "rustrdp_status_for_resume": "skipped",
        },
        "real_panels": real_panel_fp,
        "fitness_sites_by_xml": {k: len(v) for k, v in fitness_sites_by_xml.items()},
        "santa_alignments": all_records,
        "summary_per_xml": summary,
        "elapsed_sec": time.time() - t_start,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Saved {OUT_JSON} (elapsed {time.time()-t_start:.0f}s, "
          f"total santa records: {len(all_records)})", flush=True)

    # ---- cleanup tmp + partial ----
    for p in (PARTIAL, PROGRESS_JSON):
        try:
            p.unlink()
            print(f"  removed {p}", flush=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
