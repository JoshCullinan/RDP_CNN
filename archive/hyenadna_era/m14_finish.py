"""Finish the m14 realism-measure computation: XML-5 + long_content.

Loads partial data, samples from the remaining shards, skips RustRDP
(bottleneck + non-discriminating), appends results, writes the final
m14_realism_measures.json with a summary_per_xml block.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import median

import numpy as np

import sys
sys.path.insert(0, "/home/joshc/Dev/RDP_CNN")
from cache_v2_reader import CacheV2

# Import functions from the existing main script
from m14_realism_measures import (
    REAL_PANELS,
    XML_CONFIGS as XML_PATHS,
    XML_TO_PANEL as SHARD_TO_PANEL,
    SAMPLE_PER_XML,
    SEED,
    HAMMING_N_PAIRS,
    str_to_int8,
    load_fasta_as_int8,
    pairwise_hamming,
    kmer_freq,
    jsd,
    parse_fitness_sites,
    per_position_entropy,
    hyena_next_token_acc,
    build_hiv_lanl_panel,
    fingerprint_santa_alignment,
    collect_santa_bp_data,
)


PARTIAL_PATH = Path("/home/joshc/Dev/RDP_CNN/m14_realism_measures.partial.json")
FINAL_PATH = Path("/home/joshc/Dev/RDP_CNN/m14_realism_measures.json")
TMP_PATH = Path("/home/joshc/Dev/RDP_CNN/m14_realism_measures.json.tmp")

# Targets: only the missing shards
TARGETS = [
    # (shard_name, n_samples, panel_name)
    ("XML-5", 20, "zika"),
    ("long_content_30k_001", 7, "sarscov2_full"),
    ("long_content_30k_002", 7, "sarscov2_full"),
    ("long_content_30k_003", 6, "sarscov2_full"),
]


def _aggregates(records: list[dict], xml: str) -> dict:
    h = [r["hamming_mean"] for r in records if r.get("hamming_mean") is not None
         and not (isinstance(r["hamming_mean"], float) and np.isnan(r["hamming_mean"]))]
    j = [r["6mer_jsd"] for r in records if r.get("6mer_jsd") is not None
         and not (isinstance(r["6mer_jsd"], float) and np.isnan(r["6mer_jsd"]))]
    y = [r["hyena_acc"] for r in records if r.get("hyena_acc") is not None
         and not (isinstance(r["hyena_acc"], float) and np.isnan(r["hyena_acc"]))]
    rdp = [r["rdp_min_pvalue"] for r in records
           if r.get("rdp_min_pvalue") not in (None, float("nan"))]
    return {
        "xml": xml,
        "n": len(records),
        "hamming_median": float(median(h)) if h else None,
        "hamming_min": float(min(h)) if h else None,
        "hamming_max": float(max(h)) if h else None,
        "6mer_jsd_median": float(median(j)) if j else None,
        "hyena_acc_median": float(median(y)) if y else None,
        "rdp_strong_fraction": (sum(1 for p in rdp if p is not None and p < 1e-3) /
                                len(rdp) if rdp else None),
        "n_with_rdp": len(rdp),
    }


def main():
    t0 = time.time()
    rng = random.Random(SEED)

    # 1. Load partial
    if TMP_PATH.exists() and TMP_PATH.stat().st_size > PARTIAL_PATH.stat().st_size:
        src = TMP_PATH
    else:
        src = PARTIAL_PATH
    print(f"[{time.strftime('%H:%M:%S')}] loading existing data from {src.name}",
          flush=True)
    state = json.load(src.open())
    print(f"  existing santa_alignments: {len(state['santa_alignments'])}",
          flush=True)
    existing_xmls = {a.get("xml_shard") for a in state["santa_alignments"]}
    print(f"  shards already done: {existing_xmls}", flush=True)

    # 2. Real panels — load aligns we need
    print(f"\n[{time.strftime('%H:%M:%S')}] loading real panels (for k-mer profiles)",
          flush=True)
    real_kmers: dict[str, np.ndarray] = {}
    for name, path in REAL_PANELS.items():
        if path.exists() and name in {"zika", "sarscov2_full"}:
            align = load_fasta_as_int8(path)
            real_kmers[name] = kmer_freq(align)
            print(f"  {name}: align shape {align.shape}", flush=True)

    # 3. v2 cache
    cache = CacheV2()
    rng = random.Random(SEED)  # reset seed so XML-5 sample is reproducible

    # 4. Process targets
    print(f"\n[{time.strftime('%H:%M:%S')}] processing targets...", flush=True)
    new_records: list[dict] = []
    for shard_name, n_samples, panel_name in TARGETS:
        shard = cache.shards.get(shard_name)
        if shard is None or not len(shard):
            print(f"  skip {shard_name} — missing or empty shard", flush=True)
            continue
        print(f"\n=== {shard_name} (n={n_samples}, panel={panel_name}) ===",
              flush=True)
        if shard_name in existing_xmls:
            print(f"  WARN: {shard_name} already in partial — skipping",
                  flush=True)
            continue

        # Parse fitness sites (only for XML-N shards; long_content has no XML)
        xml_path = XML_PATHS.get(shard_name)
        fitness_sites: list[int] = []
        if xml_path and xml_path.exists():
            try:
                genome_lens = {"XML-1": 3822, "XML-2": 9181, "XML-3": 5800,
                               "XML-4": 19025, "XML-5": 10727}
                fitness_sites = parse_fitness_sites(
                    xml_path, genome_lens.get(shard_name, 0))
            except Exception as e:
                print(f"  fitness-site parse error: {e}", flush=True)

        # Sample files
        n_files = shard.n_files
        chosen = rng.sample(range(n_files), min(n_samples, n_files))
        real_kmer = real_kmers.get(panel_name)

        for i, fi in enumerate(chosen):
            try:
                align = shard.get_alignment(fi)
            except Exception as e:
                print(f"  [{i+1}/{len(chosen)}] file_idx={fi}: read error {e}",
                      flush=True)
                continue
            bp_s, bp_e, triplet = collect_santa_bp_data(shard, fi)
            t1 = time.time()
            fp = fingerprint_santa_alignment(
                align, bp_s, bp_e,
                seq_len=int(align.shape[1]),
                real_kmer=real_kmer,
                fitness_sites=fitness_sites,
                do_rustrdp=False,  # skip — non-discriminating + bottleneck
                rng=rng,
                event_triplet_rows=triplet,
            )
            fp["xml_shard"] = shard_name
            fp["file_idx"] = int(fi)
            fp["source_file"] = shard.files[fi]
            fp["real_comparison"] = panel_name
            new_records.append(fp)
            dt = time.time() - t1
            if (i + 1) % 5 == 0 or (i + 1) == len(chosen):
                print(f"  [{i+1}/{len(chosen)}] hamming={fp.get('hamming_mean', 0):.3f} "
                      f"6mer_jsd={fp.get('6mer_jsd', 0):.3f} "
                      f"hyena={fp.get('hyena_acc', 0):.3f} dt={dt:.1f}s",
                      flush=True)

        # Intermediate save
        state["santa_alignments"].extend(
            [r for r in new_records if r.get("xml_shard") == shard_name])
        new_records = [r for r in new_records if r.get("xml_shard") != shard_name]
        TMP_PATH.write_text(json.dumps(state, indent=2))

    # 5. Compose summary_per_xml
    print(f"\n[{time.strftime('%H:%M:%S')}] composing summary_per_xml",
          flush=True)
    by_xml: dict[str, list[dict]] = {}
    for r in state["santa_alignments"]:
        by_xml.setdefault(r.get("xml_shard", "?"), []).append(r)
    state["summary_per_xml"] = {x: _aggregates(records, x)
                                for x, records in sorted(by_xml.items())}

    # 6. Final write
    FINAL_PATH.write_text(json.dumps(state, indent=2))
    print(f"\n[{time.strftime('%H:%M:%S')}] wrote {FINAL_PATH}", flush=True)
    print(f"  total santa_alignments: {len(state['santa_alignments'])}", flush=True)
    print(f"  by_xml: { {k: len(v) for k, v in by_xml.items()} }", flush=True)
    print(f"  elapsed: {time.time() - t0:.0f}s", flush=True)

    # Summary print
    print(f"\n=== summary_per_xml ===")
    for x, s in state["summary_per_xml"].items():
        print(f"  {x}: n={s['n']}  ham_med={s['hamming_median']:.3f} "
              f"jsd_med={s['6mer_jsd_median']:.3f} "
              f"hyena_med={s['hyena_acc_median']:.3f}")


if __name__ == "__main__":
    main()
