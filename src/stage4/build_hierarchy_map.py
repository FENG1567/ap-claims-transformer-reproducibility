#!/usr/bin/env python3
"""Map frozen 2018-2020 claims vocabularies to CCSR hierarchy embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SPECIAL_TOKENS = {"[PAD]", "[MASK]", "[OOV]", "[MISSING]"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().strip("'").strip('"').strip()


def read_zipped_csv(path: Path, member: str) -> list[list[str]]:
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(member).decode("utf-8-sig")
    # The HCUP releases mix single-quoted codes/domains and double-quoted
    # descriptions. Some single-quoted domains contain commas, so ordinary
    # rectangular parsers see 7 rather than 5 fields. csv.reader still protects
    # the double-quoted description; downstream code rejoins domain fragments.
    rows = list(csv.reader(io.StringIO(raw)))
    return rows[1:]


def iter_rows(frame):
    if isinstance(frame, pd.DataFrame):
        return frame.itertuples(index=False, name=None)
    return iter(frame)


def diagnosis_lookup(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    for row in iter_rows(frame):
        code = clean(row[0]).replace(".", "").upper()
        category = clean(row[2])
        if not category and len(row) > 6:
            category = clean(row[6])
        if code and category:
            lookup[code] = (category, category[:3] or "UNK")
    return lookup


def procedure_lookup(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    for row in iter_rows(frame):
        code = clean(row[0]).replace(".", "").upper()
        category = clean(row[2])
        domain = clean(",".join(row[4:])) if len(row) > 4 else ""
        if code and category:
            lookup[code] = (category, domain or "UNK")
    return lookup


def encode_hierarchy(
    token_to_id: dict[str, int], lookup: dict[str, tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int], dict]:
    category_vocab = {"[UNK]": 0}
    domain_vocab = {"[UNK]": 0}
    size = max(map(int, token_to_id.values())) + 1
    category = np.zeros(size, dtype=np.int32)
    domain = np.zeros(size, dtype=np.int32)
    mapped = 0
    unmapped_examples: list[str] = []
    for code, token_id_raw in token_to_id.items():
        token_id = int(token_id_raw)
        if code in SPECIAL_TOKENS:
            continue
        normalized = clean(code).replace(".", "").upper()
        pair = lookup.get(normalized)
        if pair is None:
            if len(unmapped_examples) < 50:
                unmapped_examples.append(normalized)
            continue
        cat, dom = pair
        category_vocab.setdefault(cat, len(category_vocab))
        domain_vocab.setdefault(dom, len(domain_vocab))
        category[token_id] = category_vocab[cat]
        domain[token_id] = domain_vocab[dom]
        mapped += 1
    non_special = len(token_to_id) - sum(k in SPECIAL_TOKENS for k in token_to_id)
    qc = {
        "vocabulary_tokens_non_special": non_special,
        "mapped_tokens": mapped,
        "unmapped_tokens": non_special - mapped,
        "mapping_coverage": mapped / non_special if non_special else 0.0,
        "category_count_including_unknown": len(category_vocab),
        "domain_count_including_unknown": len(domain_vocab),
        "unmapped_examples_first_50": unmapped_examples,
    }
    return category, domain, category_vocab, domain_vocab, qc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    ref = args.reference_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    dx_zip = ref / "DXCCSR_v2021-2.zip"
    pr_zip = ref / "PRCCSR_v2021-1.zip"
    dx_frame = read_zipped_csv(dx_zip, "DXCCSR_v2021-2.csv")
    pr_frame = read_zipped_csv(pr_zip, "PRCCSR_v2021-1.CSV")
    dx_lookup = diagnosis_lookup(dx_frame)
    pr_lookup = procedure_lookup(pr_frame)
    dx_state_path = root / "data" / "nrd" / "diagnosis_vocabulary_state.json"
    pr_state_path = root / "data" / "nrd" / "procedure_vocabulary_state.json"
    dx_state = json.loads(dx_state_path.read_text(encoding="utf-8"))
    pr_state = json.loads(pr_state_path.read_text(encoding="utf-8"))
    dx_cat, dx_dom, dx_cat_vocab, dx_dom_vocab, dx_qc = encode_hierarchy(dx_state["token_to_id"], dx_lookup)
    pr_cat, pr_dom, pr_cat_vocab, pr_dom_vocab, pr_qc = encode_hierarchy(pr_state["token_to_id"], pr_lookup)

    arrays_path = out / "hierarchy_arrays.npz"
    np.savez_compressed(
        arrays_path, dx_category=dx_cat, dx_domain=dx_dom,
        pr_category=pr_cat, pr_domain=pr_dom,
    )
    vocab_path = out / "hierarchy_vocabularies.json"
    vocab_path.write_text(json.dumps({
        "diagnosis_category": dx_cat_vocab, "diagnosis_domain": dx_dom_vocab,
        "procedure_category": pr_cat_vocab, "procedure_domain": pr_dom_vocab,
    }, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "status": "PASS" if dx_qc["mapping_coverage"] >= 0.95 and pr_qc["mapping_coverage"] >= 0.95 else "REVIEW",
        "definition": "exact code embedding -> default inpatient CCSR/PRCCSR category -> diagnosis prefix/procedure clinical domain",
        "diagnosis": dx_qc, "procedure": pr_qc,
        "sources": {
            dx_zip.name: sha256(dx_zip), pr_zip.name: sha256(pr_zip),
            "diagnosis_vocabulary_state.json": sha256(dx_state_path),
            "procedure_vocabulary_state.json": sha256(pr_state_path),
        },
        "artifacts": {
            arrays_path.name: sha256(arrays_path), vocab_path.name: sha256(vocab_path),
        },
        "fitted_years": [2018, 2019, 2020], "year_2021_did_not_expand_mapping": True,
        "year_2022_accessed": False,
    }
    (out / "hierarchy_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
