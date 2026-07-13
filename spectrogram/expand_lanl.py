"""Grow the real-HIV identification set by pairing each CRF's recombinant with
MULTIPLE reference strains per parent subtype. Same MAFFT+HXB2-strip pipeline
as build_lanl_triplets.py. Independent unit stays the CRF family (4).

Extra subtype reps beyond build_lanl_triplets.py's single-rep set (verified
against NCBI + LANL HIV Sequence Compendium references before committing):
  A1: AF004885 (Q23-17, Kenya)         — Poss et al. 1998, widely-cited A1 ref
  B:  AY423387 (671-00T36, Netherlands) — Geels et al. 2003, subtype B ref
  C:  AF067155 (21068, India)          — GenBank-annotated "subtype C"
  F1: AF077336 (VI850, DRC)            — Carr et al. 2000, F1 ref (ex-"F")
  G:  AF084936 (subtype G, DRC)        — GenBank-annotated "subtype: G"
All 5 confirmed as full-genome, non-recombinant subtype references (not CRFs).
"""
from __future__ import annotations

import subprocess
import time
from itertools import product
from pathlib import Path

from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

Entrez.email = 'joshcull1@gmail.com'

HXB2 = 'K03455'

# >=2 accessions per subtype. Each verified to resolve on NCBI and be a
# full-genome, non-recombinant subtype reference (see module docstring).
SUBTYPE_REPS: dict[str, list[str]] = {
    "A1": ["U51190", "AF004885"],
    "B":  ["K03455", "AY423387"],
    "C":  ["U46016", "AF067155"],
    "F1": ["AF005494", "AF077336"],
    "G":  ["AF061641", "AF084936"],
}

CRF_PARENTS: dict[str, tuple[str, tuple[str, str]]] = {
    "CRF02_AG": ("L39106",   ("A1", "G")),
    "CRF07_BC": ("AF286226", ("B", "C")),
    "CRF08_BC": ("AY008715", ("B", "C")),
    "CRF12_BF": ("AF385936", ("B", "F1")),
}

SEQ_CACHE = Path('data/lanl_crf/genbank_cache')
TRIPLET_EXPANDED_DIR = Path('data/lanl_crf/triplets_expanded')
TMP = Path('/tmp/lanl_align_expanded')


def fetch_genbank(acc: str, cache_dir: Path = SEQ_CACHE):
    """Fetch one FASTA from NCBI Entrez, cache on disk. Returns a SeqRecord.

    Ported from build_lanl_triplets.py's fetch_genbank (not imported) so this
    module is self-contained and doesn't depend on the repo-root script being
    importable."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{acc}.fa"
    if cache.exists() and cache.stat().st_size > 0:
        return next(SeqIO.parse(cache, 'fasta'))
    print(f"    fetching {acc} from NCBI...", flush=True)
    with Entrez.efetch(db='nucleotide', id=acc, rettype='fasta', retmode='text') as h:
        text = h.read()
    cache.write_text(text)
    time.sleep(0.4)  # be polite to Entrez
    return next(SeqIO.parse(cache, 'fasta'))


def mafft_align(input_fa: Path, output_fa: Path, threads: int = 4) -> None:
    """Run MAFFT --auto. Ported from build_lanl_triplets.py's mafft_align."""
    cmd = ['mafft', '--auto', '--thread', str(threads), '--quiet', str(input_fa)]
    with open(output_fa, 'w') as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, timeout=300, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {proc.stderr[:500]}")


def enumerate_pairings(crf: str) -> list[tuple[str, str, str]]:
    """All (recomb_acc, pA_acc, pB_acc) over the cartesian product of the
    two parent subtypes' reps for this CRF."""
    recomb, (sa, sb) = CRF_PARENTS[crf]
    return [(recomb, pa, pb) for pa, pb in product(SUBTYPE_REPS[sa], SUBTYPE_REPS[sb])]


def _build_pairing_triplet(crf: str, recomb_acc: str, pa_acc: str, pb_acc: str,
                            hxb2_record, out_dir: Path) -> Path:
    """Fetch recomb + 2 parent accessions, MAFFT-align alongside HXB2, strip
    HXB2, write the 3-record HXB2-coordinate FASTA. Mirrors build_triplet()
    in build_lanl_triplets.py, generalized to arbitrary parent accessions."""
    recomb = fetch_genbank(recomb_acc, cache_dir=SEQ_CACHE)
    p1 = fetch_genbank(pa_acc, cache_dir=SEQ_CACHE)
    p2 = fetch_genbank(pb_acc, cache_dir=SEQ_CACHE)

    recomb = SeqRecord(recomb.seq, id=f"recomb_{crf}", description='')
    p1 = SeqRecord(p1.seq, id=f"parent_{pa_acc}", description='')
    p2 = SeqRecord(p2.seq, id=f"parent_{pb_acc}", description='')
    hxb2 = SeqRecord(hxb2_record.seq, id='HXB2_align_ref', description='')

    TMP.mkdir(parents=True, exist_ok=True)
    tag = f"{crf}__{pa_acc}__{pb_acc}"
    raw = TMP / f"{tag}_raw.fa"
    aln = TMP / f"{tag}_aln.fa"
    SeqIO.write([recomb, p1, p2, hxb2], raw, 'fasta')

    mafft_align(raw, aln)

    recs = {r.id: r for r in SeqIO.parse(aln, 'fasta')}
    hxb2_aln = str(recs['HXB2_align_ref'].seq)
    keep_cols = [c for c, ch in enumerate(hxb2_aln) if ch != '-']

    out_recs = []
    for sid in [recomb.id, p1.id, p2.id]:
        s = str(recs[sid].seq)
        proj = ''.join(s[c] for c in keep_cols)
        out_recs.append(SeqRecord(Seq(proj), id=sid, description=''))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.fa"
    SeqIO.write(out_recs, out_path, 'fasta')
    return out_path


def build_expanded_triplets(out_dir: Path = TRIPLET_EXPANDED_DIR) -> int:
    """Fetch (cache) each accession via Entrez, then per pairing run MAFFT
    with HXB2, strip HXB2, write a 3-record FASTA (recomb, pA, pB). Returns
    the count of triplets successfully written. Failures (network unreachable,
    MAFFT failure, non-resolving accession) are logged and skipped — the
    caller falls back to the original 4-triplet set."""
    print(f"[{time.strftime('%H:%M:%S')}] Fetching HXB2 reference")
    try:
        hxb2 = fetch_genbank(HXB2, cache_dir=SEQ_CACHE)
    except Exception as e:
        print(f"  FAILED to fetch HXB2 — aborting expanded build: {e}")
        return 0

    count = 0
    for crf in CRF_PARENTS:
        pairings = enumerate_pairings(crf)
        print(f"\n[{time.strftime('%H:%M:%S')}] {crf}: {len(pairings)} pairings")
        for recomb_acc, pa_acc, pb_acc in pairings:
            tag = f"{crf}__{pa_acc}__{pb_acc}"
            try:
                out_path = _build_pairing_triplet(crf, recomb_acc, pa_acc, pb_acc, hxb2, out_dir)
                print(f"  {tag}: wrote {out_path}")
                count += 1
            except Exception as e:
                print(f"  {tag}: FAILED — {e}")

    print(f"\nDone. {count} expanded triplets in {out_dir}/")
    return count


if __name__ == '__main__':
    build_expanded_triplets()
