#!/usr/bin/env python3
"""Fetch CRF prototype + parental subtype reference sequences from NCBI,
MAFFT-align each triplet alongside HXB2, then drop HXB2 and save the
3-sequence HXB2-coordinate-aligned triplet FASTA per CRF.

We pick subtype representatives that are widely cited:
  A1: U51190 (92UG037)        — Gao 1996
  B:  K03455 (HXB2)           — the canonical reference
  C:  U46016 (ETH2220)        — Salminen 1996
  F1: AF005494 (93BR020)      — Cornelissen 1996
  G:  AF061641 (HH8793)       — Carr 1998

We skip CRF01_AE — pure subtype E does not exist, so "parent E" would
be another CRF01_AE strain (circular). The 4 CRFs we DO build all have
well-defined non-recombinant parental subtypes:

  CRF02_AG → recomb L39106 (IbNG)        | parents U51190 (A) + AF061641 (G)
  CRF07_BC → recomb AF286226 (97CN54)    | parents K03455 (B) + U46016  (C)
  CRF08_BC → recomb AY008715 (97CNGX-6F) | parents K03455 (B) + U46016  (C)
  CRF12_BF → recomb AF385936 (ARMA159)   | parents K03455 (B) + AF005494 (F1)

After MAFFT alignment (recomb + 2 parents + HXB2, --auto), we strip the
HXB2 row but keep the column structure — the remaining 3 sequences are
now in HXB2 coordinates, suitable for direct comparison with the scraped
LANL truth BPs.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

Entrez.email = 'joshcull1@gmail.com'

# Accessions
HXB2 = 'K03455'
REFS = {
    'A1': 'U51190',
    'B':  'K03455',
    'C':  'U46016',
    'F1': 'AF005494',
    'G':  'AF061641',
}

CRFS = {
    'CRF02_AG': {'recomb_acc': 'L39106',   'recomb_name': 'IbNG',     'parents': ('A1', 'G')},
    'CRF07_BC': {'recomb_acc': 'AF286226', 'recomb_name': '97CN54',   'parents': ('B', 'C')},
    'CRF08_BC': {'recomb_acc': 'AY008715', 'recomb_name': '97CNGX_6F','parents': ('B', 'C')},
    'CRF12_BF': {'recomb_acc': 'AF385936', 'recomb_name': 'ARMA159',  'parents': ('B', 'F1')},
}

SEQ_CACHE = Path('data/lanl_crf/genbank_cache')
TRIPLET_DIR = Path('data/lanl_crf/triplets')
TMP = Path('/tmp/lanl_align')


def fetch_genbank(acc, cache_dir=SEQ_CACHE):
    """Fetch one FASTA from NCBI Entrez, cache on disk. Returns a SeqRecord."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{acc}.fa"
    if cache.exists() and cache.stat().st_size > 0:
        return next(SeqIO.parse(cache, 'fasta'))
    print(f"    fetching {acc} from NCBI...", flush=True)
    with Entrez.efetch(db='nucleotide', id=acc, rettype='fasta', retmode='text') as h:
        text = h.read()
    cache.write_text(text)
    time.sleep(0.4)  # be polite to Entrez
    rec = next(SeqIO.parse(cache, 'fasta'))
    return rec


def mafft_align(input_fa, output_fa, threads=4):
    """Run MAFFT --auto."""
    cmd = ['mafft', '--auto', '--thread', str(threads), '--quiet', str(input_fa)]
    with open(output_fa, 'w') as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, timeout=300, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {proc.stderr[:500]}")


def build_triplet(crf, info, hxb2_record):
    print(f"\n  {crf}: recomb {info['recomb_acc']} ({info['recomb_name']}) + parents "
          f"{info['parents']} ({REFS[info['parents'][0]]}, {REFS[info['parents'][1]]})")

    recomb = fetch_genbank(info['recomb_acc'])
    p1 = fetch_genbank(REFS[info['parents'][0]])
    p2 = fetch_genbank(REFS[info['parents'][1]])

    # Rename for clarity (RustRDP labels show these)
    recomb.id = f"recomb_{crf}";  recomb.description = ''
    p1.id = f"parent_{info['parents'][0]}"; p1.description = ''
    p2.id = f"parent_{info['parents'][1]}"; p2.description = ''
    hxb2 = SeqRecord(hxb2_record.seq, id='HXB2_align_ref', description='')

    TMP.mkdir(parents=True, exist_ok=True)
    raw = TMP / f"{crf}_raw.fa"
    aln = TMP / f"{crf}_aln.fa"
    SeqIO.write([recomb, p1, p2, hxb2], raw, 'fasta')
    print(f"    raw seqs: recomb={len(recomb.seq)}bp, "
          f"p1={len(p1.seq)}bp, p2={len(p2.seq)}bp, hxb2={len(hxb2.seq)}bp")

    print(f"    aligning with MAFFT...", flush=True)
    t0 = time.time()
    mafft_align(raw, aln)
    print(f"    aligned in {time.time()-t0:.0f}s")

    # Read alignment, find HXB2 row, identify HXB2-non-gap columns, then for
    # the triplet sequences keep ALL columns (so positions correspond to the
    # full alignment). To map alignment-col -> HXB2-coord, we use the HXB2
    # row's running non-gap count. But for direct RustRDP/eval use, we just
    # need a 3-seq aligned FASTA. The position-to-HXB2 mapping is implicit in
    # whichever sequence we choose as the coord-reference.
    recs = {r.id: r for r in SeqIO.parse(aln, 'fasta')}
    aln_len = len(recs['HXB2_align_ref'].seq)
    print(f"    alignment length: {aln_len} cols")

    # KEY: project columns onto HXB2's non-gap positions. That's what LANL's
    # truth coords assume. For each column c in the MSA, position-in-HXB2 =
    # running count of non-gap residues in HXB2 up to c.
    hxb2_aln = str(recs['HXB2_align_ref'].seq)
    keep_cols = [c for c, ch in enumerate(hxb2_aln) if ch != '-']
    print(f"    HXB2 non-gap cols: {len(keep_cols)} (HXB2 native len ~9719)")

    out_recs = []
    for sid in [recomb.id, p1.id, p2.id]:
        s = str(recs[sid].seq)
        proj = ''.join(s[c] for c in keep_cols)
        # Replace gap-only columns with '-' (already is)
        new = SeqRecord(Seq(proj), id=sid, description='')
        out_recs.append(new)

    TRIPLET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TRIPLET_DIR / f"{crf}.fa"
    SeqIO.write(out_recs, out_path, 'fasta')
    print(f"    wrote {out_path}  (3 seqs × {len(out_recs[0].seq)} HXB2-cols)")


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Fetching HXB2 reference")
    hxb2 = fetch_genbank(HXB2)
    print(f"  HXB2 length: {len(hxb2.seq)}bp")

    print(f"\n[{time.strftime('%H:%M:%S')}] Building 4 CRF triplets")
    for crf, info in CRFS.items():
        if crf == 'CRF07_BC' and (TRIPLET_DIR / 'CRF07_BC.fa').exists():
            # The OpenRDP-sourced CRF07_BC triplet is already on disk and
            # already HXB2-aligned. Overwriting risks breaking the existing
            # baseline; skip unless the user wants a re-build.
            print(f"\n  {crf}: SKIP — using existing OpenRDP triplet")
            continue
        try:
            build_triplet(crf, info, hxb2)
        except Exception as e:
            print(f"  {crf}: FAILED — {e}")

    print(f"\nDone. Triplets in {TRIPLET_DIR}/")


if __name__ == '__main__':
    main()
