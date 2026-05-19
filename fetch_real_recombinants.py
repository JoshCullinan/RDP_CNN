#!/usr/bin/env python3
"""Fetch real-recombinant viral reference panels from NCBI for RDP_CNN.

Downstream: compare SANTA-simulated training data (XML-1..5 + UnseenTestSet)
against these real genomes to filter unrealistic simulations.

Panels:
  - ebola:           XML-4 comparison (~19 kb)
  - zika:            XML-5 comparison (~10.7 kb)
  - sarscov2_full:   Test Set comparison (~29.9 kb, XBB recombinant + parents)
  - sarscov2_spike:  XML-1 comparison (Spike gene 21563-25384 of NC_045512 → 3822 bp)
  - sarscov2_orf1ab: XML-3 comparison (5.8 kb sub-region of ORF1ab 266-21555)

Output: /home/joshc/Dev/RDP_CNN/data/real_recombinants/<panel>/{raw/,aligned.fa,accessions.txt}
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

Entrez.email = 'joshcull1@gmail.com'

ROOT = Path('/home/joshc/Dev/RDP_CNN/data/real_recombinants')
SARSCOV2_REF = 'NC_045512'  # Wuhan-Hu-1 reference
SPIKE_START_1B = 21563      # 1-based incl
SPIKE_END_1B = 25384        # 1-based incl (3822 bp)

# ---------------------------------------------------------------------------
# Panel accession lists.  Each entry: (accession, label)
# Labels go into FASTA record IDs so the panel is human-readable.
# ---------------------------------------------------------------------------
EBOLA = [
    ('NC_002549', 'Zaire_Mayinga_76'),
    ('AF086833', 'Zaire_Kikwit_95'),
    ('KM034562', 'Zaire_Makona_2014'),
    ('AY729654', 'Sudan_Boniface_76'),
    ('FJ217161', 'Bundibugyo_Uganda_2007'),
    ('FJ217162', 'TaiForest_CIv_1994'),
    ('AF522874', 'Reston_1989'),
    ('MK340750', 'Bombali_2016'),
]

ZIKA = [
    ('NC_012532', 'African_MR766'),
    ('KU955591',  'African_Dakar_1968'),
    ('KU312312',  'Asian_PRVABC59_PR_2015'),
    ('KJ776791',  'Asian_FrPolynesia_2013'),
    ('KU321639',  'Asian_Brazil_2015'),
    ('KU501215',  'Asian_Honduras_2016'),
    ('KX087101',  'Asian_Cambodia_2010'),
    ('KU963573',  'Asian_Thailand_2014'),
]

SARSCOV2 = [
    ('NC_045512', 'Wuhan_Hu_1_ref'),
    ('OQ689539',  'XBB_1_5_recombinant'),  # full-length XBB.1.5 (OQ629930 was a 1328 bp fragment)
    ('OP412163',  'BA_2_10_XBB_parent'),       # BA.2.10 — XBB parent A
    ('OP412164',  'BJ_1_XBB_parent'),          # BJ.1   — XBB parent B  (may need search-fallback)
    ('MZ344997',  'Alpha_B_1_1_7'),
    ('MZ359841',  'Delta_B_1_617_2'),
    ('OM617939',  'Omicron_BA_5'),
    ('OM061695',  'Omicron_BA_2'),
]

# ---------------------------------------------------------------------------
SEARCH_FALLBACKS = {
    # If a specific accession is missing, fall back to an Entrez search.
    # term is passed to esearch in db=nucleotide; retmax=1.
    'OP412163': 'BA.2.10 SARS-CoV-2 complete genome',
    'OP412164': 'BJ.1 SARS-CoV-2 complete genome',
    'MK340750': 'Bombali ebolavirus complete genome',
}

# ---------------------------------------------------------------------------

def fetch(acc: str, label: str, cache_dir: Path) -> SeqRecord | None:
    """Fetch one FASTA from NCBI, cached. Returns a SeqRecord with id=label, or None."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{acc}.fa"
    if not (cache.exists() and cache.stat().st_size > 0):
        try:
            print(f"    fetch {acc} ({label})...", flush=True)
            with Entrez.efetch(db='nucleotide', id=acc, rettype='fasta',
                               retmode='text') as h:
                text = h.read()
            if not text.strip().startswith('>'):
                raise RuntimeError('not a FASTA response')
            cache.write_text(text)
            time.sleep(0.4)
        except Exception as e:
            print(f"      direct fetch FAILED: {e}", flush=True)
            # try fallback search
            term = SEARCH_FALLBACKS.get(acc)
            if not term:
                return None
            try:
                print(f"      search fallback: {term!r}", flush=True)
                with Entrez.esearch(db='nucleotide', term=term, retmax=1) as h:
                    res = Entrez.read(h)
                ids = res.get('IdList', [])
                if not ids:
                    print(f"      search returned no hits", flush=True)
                    return None
                gid = ids[0]
                with Entrez.efetch(db='nucleotide', id=gid, rettype='fasta',
                                   retmode='text') as h:
                    text = h.read()
                cache.write_text(text)
                time.sleep(0.4)
                print(f"      got fallback NCBI id={gid}", flush=True)
            except Exception as e2:
                print(f"      search-fallback FAILED: {e2}", flush=True)
                return None
    try:
        rec = next(SeqIO.parse(cache, 'fasta'))
    except StopIteration:
        return None
    rec.id = label
    rec.name = label
    rec.description = f"{acc} {label}"
    return rec


def mafft_align(input_fa: Path, output_fa: Path, threads: int = 4) -> None:
    cmd = ['mafft', '--auto', '--thread', str(threads), '--quiet', str(input_fa)]
    with open(output_fa, 'w') as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE,
                              timeout=600, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {proc.stderr[:500]}")


def aln_stats(aln_path: Path) -> dict:
    recs = list(SeqIO.parse(aln_path, 'fasta'))
    aln_len = len(recs[0].seq) if recs else 0
    n_gaps = sum(str(r.seq).count('-') for r in recs)
    total = aln_len * len(recs)
    return {
        'n_seq': len(recs),
        'aln_len': aln_len,
        'gap_chars': n_gaps,
        'gap_pct': (100.0 * n_gaps / total) if total else 0.0,
        'ids': [r.id for r in recs],
    }


def build_panel(name: str, accession_list, threads: int = 4) -> dict:
    panel_dir = ROOT / name
    raw_dir = panel_dir / 'raw'
    panel_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Panel {name} ===", flush=True)
    recs = []
    used = []
    missing = []
    for acc, label in accession_list:
        r = fetch(acc, label, raw_dir)
        if r is None:
            missing.append((acc, label))
            continue
        recs.append(r)
        used.append((acc, label, len(r.seq)))
        print(f"    OK {acc:<12} {label:<32} len={len(r.seq)}", flush=True)

    if len(recs) < 2:
        print(f"  PANEL {name}: <2 sequences fetched, skipping alignment", flush=True)
        return {'name': name, 'fetched': len(recs), 'missing': missing,
                'aln': None, 'used': used}

    pooled = panel_dir / 'pooled.fa'
    SeqIO.write(recs, pooled, 'fasta')

    aln_out = panel_dir / 'aligned.fa'
    print(f"  MAFFT aligning {len(recs)} sequences ...", flush=True)
    t0 = time.time()
    mafft_align(pooled, aln_out, threads=threads)
    print(f"    aligned in {time.time()-t0:.1f}s -> {aln_out}", flush=True)

    # accessions.txt
    acc_file = panel_dir / 'accessions.txt'
    with open(acc_file, 'w') as f:
        f.write(f"# Panel: {name}\n")
        f.write(f"# Fetched {len(used)} sequences, missing {len(missing)}\n")
        for acc, label, slen in used:
            f.write(f"{acc}\t{label}\t{slen}\n")
        for acc, label in missing:
            f.write(f"# MISSING\t{acc}\t{label}\n")

    s = aln_stats(aln_out)
    print(f"  STATS: n_seq={s['n_seq']} aln_len={s['aln_len']} "
          f"gaps={s['gap_chars']} ({s['gap_pct']:.1f}%)", flush=True)
    return {'name': name, 'fetched': len(recs), 'missing': missing,
            'aln': aln_out, 'used': used, 'stats': s}


def extract_subregion_from_full(full_panel_aln: Path,
                                out_aln: Path,
                                ref_label: str,
                                start_1b: int,
                                end_1b: int) -> dict:
    """Slice every record in `full_panel_aln` to the alignment columns that
    map to ref_label's nucleotide positions [start_1b, end_1b] (1-based incl).
    Then strip columns that are all-gap (resulting from columns where ref_label
    is gap inside the window or all sequences are gaps).

    Writes a multi-record FASTA to out_aln.
    """
    recs = {r.id: r for r in SeqIO.parse(full_panel_aln, 'fasta')}
    if ref_label not in recs:
        raise KeyError(f"reference label {ref_label!r} not in alignment "
                       f"({list(recs)})")
    ref_seq = str(recs[ref_label].seq)

    # 1-based residue index of ref at each alignment column (0 where ref is gap)
    keep_cols = []
    res_pos = 0
    for c, ch in enumerate(ref_seq):
        if ch != '-':
            res_pos += 1
        if start_1b <= res_pos <= end_1b and ch != '-':
            keep_cols.append(c)

    if not keep_cols:
        raise RuntimeError("no columns selected — check coordinates")

    sliced = []
    for rid, rec in recs.items():
        seq = str(rec.seq)
        sub = ''.join(seq[c] for c in keep_cols)
        sliced.append(SeqRecord(Seq(sub), id=rid,
                                description=f"{rec.description} | sliced [{start_1b}-{end_1b}]"))
    # NB: this slices an existing MSA so columns remain co-aligned — no need
    # to re-run MAFFT.
    SeqIO.write(sliced, out_aln, 'fasta')

    s = aln_stats(out_aln)
    return {'aln': out_aln, 'kept_cols': len(keep_cols), 'stats': s,
            'start_1b': start_1b, 'end_1b': end_1b, 'ref_label': ref_label}


def map_seed_to_reference(xml_path: Path, ref_label: str, ref_aln_path: Path):
    """Quick coordinate check: extract the seed sequence from a SANTA XML and
    see whether it's plausibly the Spike (3822) or an ORF1ab sub-region by
    length only. We don't actually BLAST — SANTA seeds are synthetic strings
    drawn from the model's stationary distribution, not real virus, so a
    BLAST is meaningless. Length is the diagnostic.

    We report: XML length, expected ref length, and confirm the length match.
    """
    text = xml_path.read_text()
    # extract <length>NNN</length>
    import re
    m = re.search(r'<length>\s*(\d+)\s*</length>', text)
    xml_len = int(m.group(1)) if m else -1
    return {'xml': str(xml_path), 'xml_length': xml_len}


def main():
    overall_t0 = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    results.append(build_panel('ebola', EBOLA))
    results.append(build_panel('zika', ZIKA))
    cov2 = build_panel('sarscov2_full', SARSCOV2)
    results.append(cov2)

    # --- Spike + ORF1ab sub-panels from the SARS-CoV-2 panel ---
    if cov2.get('aln') is not None:
        wuhan_label = 'Wuhan_Hu_1_ref'
        # Spike: 21563..25384 of NC_045512 → 3822 bp (XML-1's seed length)
        spike_out = ROOT / 'sarscov2_spike' / 'aligned.fa'
        try:
            spike_info = extract_subregion_from_full(
                cov2['aln'], spike_out, wuhan_label,
                SPIKE_START_1B, SPIKE_END_1B)
            print(f"\n[Spike] sliced {spike_info['kept_cols']} cols → "
                  f"{spike_info['stats']['n_seq']} seqs × "
                  f"{spike_info['stats']['aln_len']}", flush=True)
            # mirror accessions.txt
            (ROOT / 'sarscov2_spike' / 'accessions.txt').write_text(
                f"# Sliced from sarscov2_full/aligned.fa columns where "
                f"{wuhan_label} maps to NC_045512[{SPIKE_START_1B}-{SPIKE_END_1B}] "
                f"(Spike, 3822 bp expected, kept={spike_info['kept_cols']})\n"
                + (ROOT / 'sarscov2_full' / 'accessions.txt').read_text()
            )
            results.append({'name': 'sarscov2_spike',
                            'aln': spike_out,
                            'stats': spike_info['stats'],
                            'kept_cols': spike_info['kept_cols']})
        except Exception as e:
            print(f"[Spike] extraction FAILED: {e}", flush=True)

        # ORF1ab sub-region: XML-3 is 5800 bp out of ORF1ab (266..21555, ~21290 bp).
        # We can't know which 5.8 kb sub-region without BLAST, but a sensible
        # default is the polymerase RdRp + helicase + ExoN region, which is
        # the most-conserved chunk and a common simulation target. RdRp starts
        # around position 13442 (ORF1b boundary). We'll take 13442..19241
        # (5800 bp), which spans RdRp + helicase + ExoN.
        # Also report so a downstream BLAST can refine.
        ORF1AB_START = 13442
        ORF1AB_END = ORF1AB_START + 5800 - 1  # 19241
        orf1ab_out = ROOT / 'sarscov2_orf1ab' / 'aligned.fa'
        try:
            orf_info = extract_subregion_from_full(
                cov2['aln'], orf1ab_out, wuhan_label,
                ORF1AB_START, ORF1AB_END)
            print(f"[ORF1ab] sliced {orf_info['kept_cols']} cols → "
                  f"{orf_info['stats']['n_seq']} seqs × "
                  f"{orf_info['stats']['aln_len']}", flush=True)
            (ROOT / 'sarscov2_orf1ab' / 'accessions.txt').write_text(
                f"# Sliced from sarscov2_full/aligned.fa columns where "
                f"{wuhan_label} maps to NC_045512[{ORF1AB_START}-{ORF1AB_END}] "
                f"(ORF1ab RdRp+helicase+ExoN, 5800 bp expected, "
                f"kept={orf_info['kept_cols']})\n"
                + (ROOT / 'sarscov2_full' / 'accessions.txt').read_text()
            )
            results.append({'name': 'sarscov2_orf1ab',
                            'aln': orf1ab_out,
                            'stats': orf_info['stats'],
                            'kept_cols': orf_info['kept_cols'],
                            'start_1b': ORF1AB_START,
                            'end_1b': ORF1AB_END})
        except Exception as e:
            print(f"[ORF1ab] extraction FAILED: {e}", flush=True)

    # XML seed length diagnostics
    print('\n=== XML seed-length diagnostic ===', flush=True)
    for i in (1, 3):
        xml_path = Path(f'/home/joshc/Dev/RDP_CNN/santaSim_RDP/XMLs/{i}.xml')
        info = map_seed_to_reference(xml_path, 'NC_045512',
                                     ROOT / 'sarscov2_full' / 'aligned.fa')
        print(f"  XML-{i} declared <length>={info['xml_length']}", flush=True)

    # ---- Summary ----
    print('\n=== Summary ===', flush=True)
    for r in results:
        if r.get('stats'):
            s = r['stats']
            print(f"  {r['name']:<20} n={s['n_seq']}  aln_len={s['aln_len']}  "
                  f"gaps={s['gap_chars']} ({s['gap_pct']:.1f}%)", flush=True)
        else:
            print(f"  {r['name']:<20} NO ALIGNMENT (fetched={r.get('fetched', 0)})",
                  flush=True)

    # disk usage
    du = subprocess.run(['du', '-sh', str(ROOT)], capture_output=True, text=True)
    print(f"\ndisk usage: {du.stdout.strip()}", flush=True)
    print(f"\ntotal wall time: {time.time()-overall_t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
