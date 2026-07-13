import json
import numpy as np
from pathlib import Path
from spectrogram import config
from spectrogram.data import fix_length, load_lanl_triplets, Triplet, split_file_set
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

def test_fix_length_pads_with_gap():
    row = np.array([0, 1, 2], dtype=np.int8)
    out = fix_length(row, seq_len=6)
    assert out.shape == (6,)
    assert out[:3].tolist() == [0, 1, 2]
    assert out[3:].tolist() == [config.GAP_INT] * 3

def test_fix_length_truncates():
    row = np.arange(20, dtype=np.int8)
    assert fix_length(row, seq_len=5).shape == (5,)

def test_lanl_loads_four_crfs():
    """Load the original 4-CRF triplets, pinned to explicit dir."""
    trips = load_lanl_triplets(triplet_dir=config.LANL_TRIPLET_DIR)
    assert len(trips) == 4                      # the 4 built CRF families
    groups = {t.group for t in trips}
    assert groups == {"CRF02_AG", "CRF07_BC", "CRF08_BC", "CRF12_BF"}

    # Verify correct recombinant index per CRF
    recomb_indices = {t.group: t.recomb_idx for t in trips}
    assert recomb_indices == {"CRF02_AG": 0, "CRF07_BC": 2, "CRF08_BC": 0, "CRF12_BF": 0}

    for t in trips:
        assert isinstance(t, Triplet)
        assert t.rows.shape == (3, config.SEQ_LEN)
        assert t.rows.dtype == np.int8
        assert t.source == "lanl"
        assert set(np.unique(t.rows)).issubset({0, 1, 2, 3, 4})


def test_expanded_path_group_and_recomb_idx(tmp_path):
    """Test expanded LANL path: recomb written first, group extracted from stem prefix."""
    # Create two fake expanded FASTA files in tmp_path
    # File 1: CRF07_BC__B__C.fa with recomb first
    fa1_path = tmp_path / "CRF07_BC__B__C.fa"
    recs1 = [
        SeqRecord(Seq("ATGATGATGATG"), id="recomb_CRF07_BC", description=""),
        SeqRecord(Seq("ATGATGATGATG"), id="parent_B", description=""),
        SeqRecord(Seq("ATGATGATGATG"), id="parent_C", description=""),
    ]
    SeqIO.write(recs1, str(fa1_path), "fasta")

    # File 2: CRF02_AG__U51190__AF061641.fa with recomb first
    fa2_path = tmp_path / "CRF02_AG__U51190__AF061641.fa"
    recs2 = [
        SeqRecord(Seq("CGCGCGCGCGCG"), id="recomb_CRF02_AG", description=""),
        SeqRecord(Seq("CGCGCGCGCGCG"), id="parent_U51190", description=""),
        SeqRecord(Seq("CGCGCGCGCGCG"), id="parent_AF061641", description=""),
    ]
    SeqIO.write(recs2, str(fa2_path), "fasta")

    # Load from the expanded directory
    trips = load_lanl_triplets(triplet_dir=tmp_path)

    # Verify 2 triplets loaded
    assert len(trips) == 2

    # Verify group names are extracted from CRF prefix (not full stem)
    groups = {t.group for t in trips}
    assert groups == {"CRF07_BC", "CRF02_AG"}

    # Verify all expanded triplets have recomb at index 0 (written first)
    assert all(t.recomb_idx == 0 for t in trips)

    # Verify basic Triplet properties
    for t in trips:
        assert isinstance(t, Triplet)
        assert t.rows.shape == (3, config.SEQ_LEN)
        assert t.rows.dtype == np.int8
        assert t.source == "lanl"


def test_split_file_set_excludes_dropped_shards():
    fake = {
        "splits": {
            "TRAIN": {
                "dirs": {
                    "XML-2": {"files": ["a.fa", "b.fa"]},
                    "XML-6": {"files": ["c.fa"]},
                }
            }
        }
    }
    result = split_file_set(fake, "TRAIN")
    assert result == {("XML-2", "a.fa"), ("XML-2", "b.fa"), ("XML-6", "c.fa")}
    # A shard not listed at all (e.g. a dropped shard like "XML-1") contributes nothing.
    assert not any(shard == "XML-1" for shard, _ in result)


def test_split_file_set_xml6_train_val_disjoint():
    """XML-6 is split BY FILE between TRAIN and VAL -- the same filename must
    never appear in both arms of the real split."""
    with config.SANTA_SPLIT.open() as f:
        real = json.load(f)
    train = {fname for shard, fname in split_file_set(real, "TRAIN") if shard == "XML-6"}
    val = {fname for shard, fname in split_file_set(real, "VAL") if shard == "XML-6"}
    assert train, "expected XML-6 files in TRAIN"
    assert val, "expected XML-6 files in VAL"
    assert train.isdisjoint(val)
