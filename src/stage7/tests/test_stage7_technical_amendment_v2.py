"""Equivalence tests for the Stage 7 exact-membership performance amendment."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd


STAGE7 = Path(__file__).resolve().parents[1]
V1_PATH = STAGE7 / "run_locked_2022_etl_technical_amendment_v1.py"
V2_PATH = STAGE7 / "run_locked_2022_etl_technical_amendment_v2.py"
EXPECTED_V1_SHA256 = "c279a1af05bc8540a0ca7a014cc67821e1c27475ce443cf47373a3969977b987"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1 = load_module("run_locked_2022_etl_technical_amendment_v1", V1_PATH)
v2 = load_module("run_locked_2022_etl_technical_amendment_v2", V2_PATH)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_tree_without_allowed_changes(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    filtered = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"_exact_code_membership", "_flags_from_raw_codes"}:
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if "SCHEMA_VERSION" in names:
                continue
        filtered.append(node)
    tree.body = filtered
    return ast.dump(tree, include_attributes=False)


def test_v1_is_preserved_and_v2_diff_is_semantically_whitelisted() -> None:
    assert digest(V1_PATH) == EXPECTED_V1_SHA256
    assert semantic_tree_without_allowed_changes(V1_PATH) == semantic_tree_without_allowed_changes(V2_PATH)
    assert v2.SCHEMA_VERSION == "stage7_locked_2022_local_etl_technical_amendment_v2"
    assert "np.isin" not in inspect.getsource(v2._flags_from_raw_codes)


def test_exact_membership_matches_numpy_reference_on_synthetic_codes() -> None:
    rng = np.random.default_rng(20260917)
    universe = np.asarray([f"X{i:05d}" for i in range(3000)] + ["", "K850", "K83", "A41"], dtype=object)
    matrix = rng.choice(universe, size=(400, 48), replace=True)
    frozen = set(universe[::3].tolist())
    expected = np.isin(matrix, list(frozen))
    observed = v2._exact_code_membership(matrix, frozen)
    np.testing.assert_array_equal(observed, expected)


def test_all_flags_are_identical_to_v1_on_synthetic_predeclared_codes() -> None:
    core = pd.DataFrame({
        "DX1": ["K85.0", "K83", "A41", "Z99.9", ""],
        "DX2": ["K83", "", "K85.1", "A41", None],
        "PR1": ["0FC98ZZ", "", "PRA1", "UNKNOWN", None],
        "PR2": ["", "PRA3", "PRA4", "0FC98ZZ", None],
    })
    known_cm = [f"C{i:05d}" for i in range(20000)] + ["K850", "K851", "K83", "A41"]
    known_pcs = [f"P{i:05d}" for i in range(20000)] + ["0FC98ZZ", "PRA1", "PRA3", "PRA4"]
    spec = {
        "diagnosis_columns": ["DX1", "DX2"],
        "procedure_columns": ["PR1", "PR2"],
        "code_sets": {
            "known_cm": known_cm,
            "known_pcs": known_pcs,
            "ap_prefixes": ["K85"],
            "biliary": ["K83"],
            "sepsis_or_organ": ["A41"],
            "PR.1": ["PRA1"],
            "PR.2": ["K83"],
            "PR.3": ["PRA3"],
            "PR.4": ["A41"],
        },
    }
    expected = v1._flags_from_raw_codes(core, spec)
    observed = v2._flags_from_raw_codes(core, spec)
    pd.testing.assert_frame_equal(observed, expected, check_dtype=True, check_exact=True)
