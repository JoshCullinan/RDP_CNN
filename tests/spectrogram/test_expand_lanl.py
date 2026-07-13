from spectrogram.expand_lanl import SUBTYPE_REPS, enumerate_pairings

def test_multiple_reps_per_subtype():
    # at least the parent subtypes we need must have >1 rep to grow N
    for st in ("A1", "B", "C", "F1", "G"):
        assert st in SUBTYPE_REPS and len(SUBTYPE_REPS[st]) >= 2

def test_enumerate_pairings_is_product():
    p = enumerate_pairings("CRF02_AG")   # parents A1 x G
    assert len(p) == len(SUBTYPE_REPS["A1"]) * len(SUBTYPE_REPS["G"])
    for recomb, pa, pb in p:
        assert recomb == "L39106"        # CRF02_AG recombinant accession
