#!/usr/bin/env python3
"""Parse LANL CRF DB HTML for per-CRF breakpoint tables.

The LANL page at
  https://www.hiv.lanl.gov/components/sequence/HIV/crfdb/crfs.comp?crf=01_AE
is a single document containing all CRFs. For each CRF row we have:
  <td>CRF<NAME></td><td>PROTO (ACCESSION)</td><td>PARENTS</td>...
  <pre>S1 E1 SUBTYPE1<br>S2 E2 SUBTYPE2<br>...</pre>

We extract:
- For each CRF: prototype name, accession, parental subtypes
- For each fragment: HXB2 start, end, parental subtype

Writes data/lanl_crf/truth_breakpoints.csv with one row per fragment, plus
data/lanl_crf/crf_meta.csv with one row per CRF (prototype + accession + parents).

Breakpoint positions (the boundaries between adjacent fragments) are emitted
as a second CSV data/lanl_crf/truth_bps.csv with one row per breakpoint
(crf, position_hxb2). These are what we evaluate the CNN/classical methods
against.
"""
import csv
import re
import sys
from pathlib import Path

HTML = Path('data/lanl_crf/CRF01_AE_raw.html')
OUT_DIR = Path('data/lanl_crf')

# Target the 5 CRFs we picked
TARGETS = ['CRF01_AE', 'CRF02_AG', 'CRF07_BC', 'CRF08_BC', 'CRF12_BF']


def parse():
    text = HTML.read_text()
    # Each CRF row block starts with <tr id=tgl_CRF<NAME> or contains <td>CRF<NAME></td>
    # We'll match on the per-CRF block: from <td>CRF...</td> to closing </tr>.
    # Easier: split on "<tr id=tgl_" sentinel which prefixes each CRF block.

    crf_blocks = {}
    # The very first CRF (01_AE) has a different anchor; search for both patterns.
    for chunk in re.split(r'<tr[^>]*id=[\"\']?tgl_', text)[1:]:
        # chunk starts like "CRF01_AE ...rest of row..."
        m = re.match(r'(CRF\w+)\b', chunk)
        if not m:
            continue
        name = m.group(1)
        # Stop the block at the next sibling row marker (next tgl_ or end-of-table).
        crf_blocks[name] = chunk

    print(f"Found {len(crf_blocks)} CRF blocks: {sorted(crf_blocks)[:10]}...")
    print()

    crf_meta_rows = []
    bp_rows = []        # fragment-level
    bp_only_rows = []   # breakpoint positions (boundaries)

    for crf in TARGETS:
        block = crf_blocks.get(crf)
        if not block:
            print(f"  MISS {crf}: not found in HTML")
            continue
        # Prototype + accession appear in the table preface; pattern:
        # <td>CRF01_AE</td><td>CM240 (U54771)</td><td>A, E</td>
        # That row was BEFORE the <tr id=tgl_...> block; we need to also look
        # at the surrounding context. Easier: search the full HTML for the
        # one-row record.
        pre_match = re.search(
            rf'<td>{re.escape(crf)}</td>\s*<td>([^<]+?)\s*\(([^<)]+?)\)</td>\s*<td>([^<]+)</td>',
            text)
        if pre_match:
            proto = pre_match.group(1).strip()
            acc = pre_match.group(2).strip()
            parents = pre_match.group(3).strip()
        else:
            proto, acc, parents = '?', '?', '?'

        # <pre>...</pre> with fragments
        pre = re.search(r'<pre>(.+?)</pre>', block, re.S)
        if not pre:
            print(f"  MISS {crf}: no <pre> block")
            continue
        frag_text = pre.group(1).replace('<br>', '\n').replace('<br />', '\n').strip()

        fragments = []
        for line in frag_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = int(parts[0]); end = int(parts[1])
            except ValueError:
                continue
            subtype = ' '.join(parts[2:])
            fragments.append((start, end, subtype))

        print(f"  {crf}: proto={proto} ({acc}), parents={parents}, fragments={len(fragments)}")
        crf_meta_rows.append({
            'crf': crf, 'prototype': proto, 'accession': acc,
            'parental_subtypes': parents, 'n_fragments': len(fragments),
        })
        for idx, (s, e, sub) in enumerate(fragments):
            bp_rows.append({
                'crf': crf, 'fragment_idx': idx,
                'hxb2_start': s, 'hxb2_end': e,
                'parental_subtype': sub,
            })
        # Breakpoint positions = inter-fragment boundaries (skip outer edges)
        # We take fragment[i].end as the bp (or fragment[i+1].start - 1) since
        # the LANL convention uses end inclusive.
        for i in range(len(fragments) - 1):
            bp_pos = fragments[i][1]  # end of this fragment
            bp_only_rows.append({
                'crf': crf, 'breakpoint_idx': i, 'hxb2_position': bp_pos,
                'left_subtype': fragments[i][2],
                'right_subtype': fragments[i + 1][2],
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / 'crf_meta.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['crf', 'prototype', 'accession',
                                          'parental_subtypes', 'n_fragments'])
        w.writeheader(); w.writerows(crf_meta_rows)
    with (OUT_DIR / 'truth_fragments.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['crf', 'fragment_idx',
                                          'hxb2_start', 'hxb2_end',
                                          'parental_subtype'])
        w.writeheader(); w.writerows(bp_rows)
    with (OUT_DIR / 'truth_bps.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['crf', 'breakpoint_idx',
                                          'hxb2_position',
                                          'left_subtype', 'right_subtype'])
        w.writeheader(); w.writerows(bp_only_rows)

    print(f"\nWrote {OUT_DIR/'crf_meta.csv'} ({len(crf_meta_rows)} CRFs)")
    print(f"Wrote {OUT_DIR/'truth_fragments.csv'} ({len(bp_rows)} fragments)")
    print(f"Wrote {OUT_DIR/'truth_bps.csv'} ({len(bp_only_rows)} breakpoints)")


if __name__ == '__main__':
    parse()
