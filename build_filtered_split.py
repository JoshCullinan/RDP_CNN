"""Apply realism-based filter to splits/v2_split.json and produce v2_filtered_split.json.

Filter rules — derived from m14_realism_measures.json analysis:

WHOLE-SHARD DROPS (paradigm mismatch — no parameter combo rescues them):
  - XML-1 (Spike, SARS-CoV-2):  every combo >20× too diverse vs real Spike
  - XML-3 (ORF1ab, SARS-CoV-2): same
  - long_content_30k_003:       mutation rates 2E-4 to 5E-4 are 10× XML-1's;
                                produces Hamming 0.16–0.34 vs real 0.002

PARAMETER-LEVEL FILTERS within retained shards:
  - XML-2 (HIV-1): keep all combos
  - XML-4 (Ebola): keep only mut >= 0.01 combos. Lower-mut combos produce
                   Hamming 0.04–0.06, way below real Ebola 0.33.
  - XML-5 (Zika): keep only mut == 0.005 combos. Higher-mut combos produce
                  Hamming 0.20+, far above real Zika 0.065.
  - long_content_30k_001: keep all (closest to real already)
  - long_content_30k_002: drop rp >= 0.10 combos (high-recomb runs that
                          pushed Hamming up to 0.04+)

Output:
  splits/v2_filtered_split.json — same schema as v2_split.json, with filtered
                                  file lists. Adds 'filter_reason' field per dir.
  splits/v2_filter_report.json  — what was dropped and why (counts).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/joshc/Dev/RDP_CNN")
ORIG_SPLIT = ROOT / "splits" / "v2_split.json"
FILTERED_SPLIT = ROOT / "splits" / "v2_filtered_split.json"
FILTER_REPORT = ROOT / "splits" / "v2_filter_report.json"

# Whole-shard drops.
SHARDS_TO_DROP = {"XML-1", "XML-3", "long_content_30k_003"}

# Filename regexes
# format: alignment_XML{n}-{pop}-{mut}-{rec}-{gen}-{rep}.fa
#   where mut is decimal like 0.005, rec is scientific like 8E-5
XML_RE = re.compile(
    r"alignment_XML\d+-(\d+)-(\d+(?:\.\d+)?)-(\d+E-\d+)-(\d+)-(\d+)"
)
# long_content filename: lc30k002_001_gc3300_rp0.05_mr1.14E-4_ss100.fa
LC_RE = re.compile(
    r"(?:lc30k\d+|long30k)_\d+_gc(\d+)_rp([\d.]+)_mr([\d.E+-]+?)_ss(\d+)"
)


def keep_xml_file(shard: str, fname: str) -> tuple[bool, str]:
    """Returns (keep, reason). For XML-N shards. Reason is empty if kept."""
    m = XML_RE.search(fname)
    if not m:
        return True, ""  # unparseable name — keep (safer)
    pop, mut_str, rec_str, gen, rep = m.groups()
    try:
        mut = float(mut_str)
        rec = float(rec_str.replace("E-", "e-"))
    except ValueError:
        return True, ""

    if shard == "XML-4":  # Ebola — need higher divergence; keep mut >= 0.01
        if mut < 0.01:
            return False, f"XML-4 mut={mut} <0.01 (too tight for Ebola)"
    elif shard == "XML-5":  # Zika — need lower divergence; keep mut == 0.005
        if mut > 0.005 + 1e-9:
            return False, f"XML-5 mut={mut} >0.005 (too diverse for Zika)"
    return True, ""


def keep_lc_file(shard: str, fname: str) -> tuple[bool, str]:
    m = LC_RE.search(fname)
    if not m:
        return True, ""
    gc, rp_str, mr_str, ss = m.groups()
    try:
        rp = float(rp_str)
    except ValueError:
        return True, ""
    if shard == "long_content_30k_002":
        if rp >= 0.10 - 1e-9:
            return False, f"lc002 rp={rp} >=0.10 (too aggressive recomb)"
    return True, ""


def keep_file(shard: str, fname: str) -> tuple[bool, str]:
    if shard in SHARDS_TO_DROP:
        return False, f"shard {shard} dropped wholesale (paradigm mismatch)"
    if shard.startswith("XML-"):
        return keep_xml_file(shard, fname)
    if shard.startswith("long_content"):
        return keep_lc_file(shard, fname)
    return True, ""


def filter_split() -> dict:
    doc = json.load(ORIG_SPLIT.open())
    new_doc = {
        "version": doc["version"] + "-filtered",
        "based_on": str(ORIG_SPLIT.name),
        "filter_source": "m14_realism_measures.json",
        "filter_rules_summary": (
            "Whole-shard drops: XML-1 (Spike), XML-3 (ORF1ab), long_content_30k_003. "
            "Per-combo: XML-4 keep mut>=0.01; XML-5 keep mut==0.005; "
            "long_content_30k_002 drop rp>=0.10."
        ),
        "shards_dropped": sorted(SHARDS_TO_DROP),
        "cache_root": doc.get("cache_root"),
        "lanl_dir": doc.get("lanl_dir"),
        "xml6_split_seed": doc.get("xml6_split_seed"),
        "splits": {},
    }
    drop_counts = defaultdict(lambda: defaultdict(int))
    keep_counts = defaultdict(lambda: defaultdict(int))

    for split_name, split in doc["splits"].items():
        new_split = {"dirs": {}, "totals": {"triplets": 0, "events": 0,
                                            "files": 0, "files_dropped": 0}}
        for shard, info in split["dirs"].items():
            new_info = {
                "files": [], "files_dropped": [],
                "n_files": 0, "n_files_dropped": 0,
                "triplets": 0, "events": 0,
            }
            for fname in info["files"]:
                keep, reason = keep_file(shard, fname)
                if keep:
                    new_info["files"].append(fname)
                    keep_counts[split_name][shard] += 1
                else:
                    new_info["files_dropped"].append({"file": fname, "reason": reason})
                    drop_counts[split_name][shard] += 1
            new_info["n_files"] = len(new_info["files"])
            new_info["n_files_dropped"] = len(new_info["files_dropped"])
            # Triplet/event counts will need cache to recompute exactly; for
            # now estimate from kept-fraction × original totals.
            if info["n_files"] > 0:
                kept_frac = new_info["n_files"] / info["n_files"]
                orig_trip = info.get("triplets") or 0
                orig_ev = info.get("events") or 0
                new_info["triplets"] = int(round(orig_trip * kept_frac))
                new_info["events"] = int(round(orig_ev * kept_frac))
            # If a shard is fully dropped, mark with reason
            if shard in SHARDS_TO_DROP:
                new_info["dropped_wholesale"] = True
                new_info["reason"] = (
                    "paradigm mismatch with real-virus reference panel "
                    "(see m14_realism_measures.json)"
                )

            new_split["dirs"][shard] = new_info
            new_split["totals"]["triplets"] += new_info["triplets"]
            new_split["totals"]["events"] += new_info["events"]
            new_split["totals"]["files"] += new_info["n_files"]
            new_split["totals"]["files_dropped"] += new_info["n_files_dropped"]
        new_doc["splits"][split_name] = new_split

    return new_doc, dict(drop_counts), dict(keep_counts)


def main():
    doc, drop_counts, keep_counts = filter_split()
    FILTERED_SPLIT.write_text(json.dumps(doc, indent=2))
    print(f"Wrote {FILTERED_SPLIT}", flush=True)

    # Report
    report = {
        "filter_source": "m14_realism_measures.json",
        "rules": doc["filter_rules_summary"],
        "shards_dropped_wholesale": sorted(SHARDS_TO_DROP),
        "drop_counts": {k: dict(v) for k, v in drop_counts.items()},
        "keep_counts": {k: dict(v) for k, v in keep_counts.items()},
        "totals_by_split": {
            sn: {"files": s["totals"]["files"],
                 "events": s["totals"]["events"],
                 "triplets": s["totals"]["triplets"],
                 "files_dropped": s["totals"]["files_dropped"]}
            for sn, s in doc["splits"].items()
        },
    }
    FILTER_REPORT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {FILTER_REPORT}", flush=True)

    # Summary print
    print()
    print("=== Filter results ===")
    for split_name, split in doc["splits"].items():
        t = split["totals"]
        print(f"  {split_name}: files={t['files']} "
              f"(dropped {t['files_dropped']})  "
              f"events~{t['events']:,}  triplets~{t['triplets']:,}")


if __name__ == "__main__":
    main()
