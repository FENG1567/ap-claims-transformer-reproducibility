import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_hierarchy_map import diagnosis_lookup, encode_hierarchy, procedure_lookup


def test_hierarchy_encoding_reserves_unknown_and_maps_exact_codes():
    frame = pd.DataFrame([["'K850'", "Acute pancreatitis", "'DIG020'", "x", "", "", "", ""]])
    lookup = diagnosis_lookup(frame)
    cat, dom, cat_vocab, dom_vocab, qc = encode_hierarchy(
        {"[PAD]": 0, "[MASK]": 1, "[OOV]": 2, "[MISSING]": 3, "K850": 4, "ZZZ": 5}, lookup
    )
    assert cat[0] == dom[0] == 0
    assert cat[4] == cat_vocab["DIG020"]
    assert dom[4] == dom_vocab["DIG"]
    assert cat[5] == 0
    assert qc["mapped_tokens"] == 1


def test_procedure_domain_is_retained():
    frame = pd.DataFrame([["'0FT44ZZ'", "Resection", "'HEP006'", "x", "Hepatobiliary Procedures"]])
    lookup = procedure_lookup(frame)
    assert lookup["0FT44ZZ"] == ("HEP006", "Hepatobiliary Procedures")


def test_split_single_quoted_procedure_domain_is_rejoined():
    rows = [["'091D070'", "Bypass", "'ENT017'", "ENT procedures, NEC", "'Ear", " Nose", " and Throat Procedures'"]]
    lookup = procedure_lookup(rows)
    assert lookup["091D070"] == ("ENT017", "Ear, Nose, and Throat Procedures")
