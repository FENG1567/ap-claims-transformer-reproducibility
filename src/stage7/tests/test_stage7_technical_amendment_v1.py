"""Regression tests for the additive Stage 7 vocabulary amendment."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


STAGE7 = Path(__file__).resolve().parents[1]
ORIGINAL = STAGE7 / "run_locked_2022_etl.py"
AMENDED = STAGE7 / "run_locked_2022_etl_technical_amendment_v1.py"
EXPECTED_ORIGINAL_SHA256 = "b219fd6317702e3637507c2b8fe15ddca6c7a2c478bf764881eca488eac6fcfb"

SPEC = importlib.util.spec_from_file_location("run_locked_2022_etl_technical_amendment_v1", AMENDED)
assert SPEC and SPEC.loader
etl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(etl)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_vocab(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def semantic_tree_without_allowed_changes(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    filtered = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_load_vocab":
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & {"SCHEMA_VERSION", "SPECIAL_TOKEN_IDS"}:
                continue
        filtered.append(node)
    tree.body = filtered
    return ast.dump(tree, include_attributes=False)


def test_original_controller_is_preserved_and_diff_is_semantically_whitelisted() -> None:
    assert digest(ORIGINAL) == EXPECTED_ORIGINAL_SHA256
    assert ORIGINAL.read_bytes() != AMENDED.read_bytes()
    assert semantic_tree_without_allowed_changes(ORIGINAL) == semantic_tree_without_allowed_changes(AMENDED)
    assert etl.SCHEMA_VERSION == "stage7_locked_2022_local_etl_technical_amendment_v1"


def test_real_frozen_layout_accepts_no_growth_2021_use_and_excludes_reserved_tokens(tmp_path: Path) -> None:
    path = write_vocab(tmp_path / "vocab.json", {
        "frozen_after_year": 2020,
        "processed_years": [2018, 2019, 2020, 2021],
        "token_to_id": {
            "[PAD]": 0,
            "[MASK]": 1,
            "[OOV]": 2,
            "[MISSING]": 3,
            "K85.0": 4,
            "K83": 5,
        },
    })
    assert etl._load_vocab(path) == {"K850": 4, "K83": 5}


@pytest.mark.parametrize("processed_years", [
    [2018, 2019, 2021],
    [2018, 2019, 2020, 2021, 2021],
    [2018, 2019, 2020, "2021"],
])
def test_amendment_does_not_relax_frozen_year_integrity(tmp_path: Path, processed_years: list) -> None:
    path = write_vocab(tmp_path / "vocab.json", {
        "frozen_after_year": 2020,
        "processed_years": processed_years,
        "token_to_id": {"K850": 4},
    })
    with pytest.raises(RuntimeError, match="exclusively frozen"):
        etl._load_vocab(path)


def test_reserved_token_mapping_must_be_complete_and_exact(tmp_path: Path) -> None:
    incomplete = write_vocab(tmp_path / "incomplete.json", {
        "frozen_after_year": 2020,
        "processed_years": [2018, 2019, 2020, 2021],
        "token_to_id": {"[PAD]": 0, "K850": 4},
    })
    with pytest.raises(RuntimeError, match="special-token mapping"):
        etl._load_vocab(incomplete)
    changed = write_vocab(tmp_path / "changed.json", {
        "frozen_after_year": 2020,
        "processed_years": [2018, 2019, 2020, 2021],
        "token_to_id": {"[PAD]": 1, "[MASK]": 0, "[OOV]": 2, "[MISSING]": 3, "K850": 4},
    })
    with pytest.raises(RuntimeError, match="special-token mapping"):
        etl._load_vocab(changed)


def test_observed_codes_still_cannot_use_reserved_ids_or_normalize_to_duplicates(tmp_path: Path) -> None:
    reserved_id = write_vocab(tmp_path / "reserved-id.json", {
        "frozen_after_year": 2020,
        "processed_years": [2018, 2019, 2020, 2021],
        "token_to_id": {"K850": 3},
    })
    with pytest.raises(RuntimeError, match="observed code"):
        etl._load_vocab(reserved_id)
    duplicate = write_vocab(tmp_path / "duplicate.json", {
        "frozen_after_year": 2020,
        "processed_years": [2018, 2019, 2020, 2021],
        "token_to_id": {"K85.0": 4, "K850": 5},
    })
    with pytest.raises(RuntimeError, match="observed code"):
        etl._load_vocab(duplicate)
