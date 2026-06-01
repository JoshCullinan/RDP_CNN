"""M1.4 continuation — finish XML-2..5 + long_content_30k_* given the existing
partial JSON. XML-1 is already complete (50 records) and is NOT recomputed.

Sample sizes:
  XML-2: 20, XML-3: 20, XML-4: 20, XML-5: 20
  long_content_30k_001: 7, _002: 7, _003: 6
Total new alignments: 100.

Reuses functions from m14_realism_measures.py.
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

# Import the existing module — reuses kmer_freq / fingerprint_santa_alignment / etc.
from m14_realism_measures import (
    CACHE_ROOT,
    XML_TO_PANEL,
    XML_CONFIGS,
    REAL_PANELS,
    LANL_TRIPLETS_DIR,
    SEED,
    kmer_freq,
    load_fasta_as_int8,
    build_hiv_lanl_panel,
    parse_fitness_sites,
    fingerprint_santa_alignment,
    collect_santa_bp_data,
    summarize,
    print_headline,
    _watchdog,
    _rss_gb,
)
from cache_v2_reader import CacheV2  # noqa: E402

PARTIAL_JSON = ROOT / "m14_realism_measures.partial.json"
FINAL_JSON = ROOT / "m14_realism_measures.json"
TMP_JSON = ROOT / "m14_realism_measures.json.tmp"

# New sample plan for this continuation run
NEW_SAMPLE_PLAN = [
    # (shard_name, n_files, n_rdp)
    ("XML-2", 20, 20),
    ("XML-3", 20, 20),
    ("XML-4", 20, 20),
    ("XML-5", 20, 20),
    ("long_content_30k_001", 7, 7),
    ("long_content_30k_002", 7, 7),
    ("long_content_30k_003", 6, 6),
]


def _save_intermediate(path: Path, partial_data: dict, new_records: list[dict],
                       summary: dict | None = None):
    payload = dict(partial_data)
    # Combine: existing XML-1 records + new
    combined = list(partial_data.get("santa_alignments", [])) + list(new_records)
    payload["santa_alignments"] = combined
    if summary is not None:
        payload["summary_per_xml"] = summary
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    t_start = time.time()
    rng = random.Random(SEED)

    # ----- load partial -----
    print(f"[{time.strftime('%H:%M:%S')}] loading partial from {PARTIAL_JSON}", flush=True)
    with open(PARTIAL_JSON) as f:
        partial = json.load(f)
    existing_records = partial["santa_alignments"]
    print(f"  existing records: {len(existing_records)} "
          f"(XML-1 done with {sum(1 for r in existing_records if r['xml_shard']=='XML-1')})",
          flush=True)

    real_panel_fp = partial["real_panels"]
    fitness_sites_by_xml_count = partial["fitness_sites_by_xml"]
    # We need to RE-parse fitness sites because partial only has counts not positions
    print(f"\n[{time.strftime('%H:%M:%S')}] re-parsing XML fitness sites...", flush=True)
    genome_lens = {"XML-1": 3822, "XML-2": 9181, "XML-3": 5800, "XML-4": 19025, "XML-5": 10727}
    fitness_sites_by_xml: dict[str, list[int]] = {}
    for xml_name, xml_path in XML_CONFIGS.items():
        L = genome_lens[xml_name]
        sites = parse_fitness_sites(xml_path, L)
        fitness_sites_by_xml[xml_name] = sites
        assert len(sites) == fitness_sites_by_xml_count[xml_name], \
            f"fitness sites mismatch for {xml_name}: got {len(sites)} vs {fitness_sites_by_xml_count[xml_name]}"
        print(f"  {xml_name}: {len(sites)} fitness positions", flush=True)

    # ----- build real_panel_kmers for JSD computation -----
    print(f"\n[{time.strftime('%H:%M:%S')}] rebuilding real-panel k-mer freqs...", flush=True)
    real_panel_aligns: dict[str, np.ndarray] = {}
    for name, path in REAL_PANELS.items():
        if not path.exists():
            print(f"  MISSING: {path}", flush=True)
            continue
        real_panel_aligns[name] = load_fasta_as_int8(path)
    real_panel_aligns["hiv_lanl"] = build_hiv_lanl_panel()
    real_panel_kmers: dict[str, np.ndarray] = {}
    for name, align in real_panel_aligns.items():
        real_panel_kmers[name] = kmer_freq(align)
        print(f"  {name}: shape={align.shape}, kmer sum={real_panel_kmers[name].sum():.3f}",
              flush=True)
    # free the alignments — only kmers needed
    del real_panel_aligns

    # ----- load cache -----
    print(f"\n[{time.strftime('%H:%M:%S')}] loading cache...", flush=True)
    cache = CacheV2(CACHE_ROOT)
    print(f"  shards: {list(cache.shards.keys())}", flush=True)

    # ----- step through each shard -----
    new_records: list[dict] = []

    for xml_name, n_files, n_rdp in NEW_SAMPLE_PLAN:
        if xml_name not in cache.shards:
            print(f"  SKIP {xml_name}: not in cache", flush=True)
            continue
        shard = cache.shards[xml_name]
        panel_name = XML_TO_PANEL[xml_name]
        real_kmer = real_panel_kmers.get(panel_name)
        fitness_sites = fitness_sites_by_xml.get(xml_name, [])
        avail = shard.n_files

        # match original seeding logic so XML-2..5 sampling is reproducible
        local_rng = random.Random(SEED + hash(xml_name) % 1000)
        sample_idxs = local_rng.sample(range(avail), min(n_files, avail))
        rdp_subset = set(sample_idxs[:n_rdp])

        print(f"\n[{time.strftime('%H:%M:%S')}] {xml_name}: "
              f"sampling {len(sample_idxs)} files (RDP on {len(rdp_subset)}), "
              f"panel={panel_name}", flush=True)
        t0 = time.time()
        for i, fi in enumerate(sample_idxs):
            try:
                align = shard.get_alignment(fi)
            except Exception as e:
                print(f"  err getting align fi={fi}: {e}", flush=True)
                continue
            bp_starts, bp_ends, triplet_rows = collect_santa_bp_data(shard, fi)
            seq_len = int(align.shape[1])
            do_rustrdp = fi in rdp_subset
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
                print(f"  err fingerprint {xml_name} fi={fi}: {e}", flush=True)
                continue
            rec["xml_shard"] = xml_name
            rec["file_idx"] = int(fi)
            rec["source_file"] = shard.files[fi] if fi < len(shard.files) else ""
            rec["real_comparison"] = panel_name
            new_records.append(rec)
            _watchdog(label=f"{xml_name} {i}/{len(sample_idxs)}")
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            print(f"    [{i+1}/{len(sample_idxs)}] fi={fi} "
                  f"H={rec.get('hamming_mean'):.4f} "
                  f"JSD={rec.get('6mer_jsd'):.4f} "
                  f"RDPp={rec.get('rdp_min_pvalue')} "
                  f"rate={rate:.2f}/s RSS={_rss_gb():.1f}GB", flush=True)

            # intermediate save every 10 alignments
            if (len(new_records) % 10) == 0:
                _save_intermediate(TMP_JSON, partial, new_records)
                print(f"    [save] tmp written ({len(new_records)} new records)",
                      flush=True)

        # save after each shard too
        _save_intermediate(TMP_JSON, partial, new_records)
        print(f"  {xml_name} done in {time.time()-t0:.0f}s "
              f"({len(new_records)} total new records)", flush=True)

    # ----- compose final JSON -----
    all_records = list(existing_records) + list(new_records)
    summary = summarize(all_records, real_panel_fp)

    final = {
        "config": {
            "sample_per_xml_phase1": 50,    # XML-1 used 50 (from earlier run)
            "sample_per_xml_phase2": 20,    # XML-2..5
            "sample_per_long_content": "7/7/6",
            "kmer_k": 6,
            "hamming_n_pairs": 50,
            "rustrdp_methods": ["maxchi", "rdp", "chimaera", "siscan"],
            "rustrdp_timeout_sec": 60,
            "seed": SEED,
            "notes": "XML-1 was computed in an earlier session with sample_per_xml=50; "
                     "the continuation reduced sample size to 20 (50 for long_content total).",
        },
        "real_panels": real_panel_fp,
        "fitness_sites_by_xml": {k: len(v) for k, v in fitness_sites_by_xml.items()},
        "santa_alignments": all_records,
        "summary_per_xml": summary,
        "elapsed_sec": time.time() - t_start,
    }
    with open(FINAL_JSON, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved {FINAL_JSON} (elapsed {time.time()-t_start:.0f}s, "
          f"{len(all_records)} total alignments)", flush=True)
    try:
        TMP_JSON.unlink()
    except OSError:
        pass
    print_headline(summary, real_panel_fp)


if __name__ == "__main__":
    main()
