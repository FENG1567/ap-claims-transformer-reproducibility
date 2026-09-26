"""Regression tests for relative artifact identities in the v3 predictor."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest


STAGE7 = Path(__file__).resolve().parents[1]
ORIGINAL = STAGE7 / "predict_all_locked_2022.py"
AMENDED = STAGE7 / "predict_all_locked_2022_technical_amendment_v3.py"
EXPECTED_ORIGINAL_SHA256 = "ce013413cb94be18a66e77c1a62e3225c1d1c4dd0225494d9f635126d90efc95"


SPEC = importlib.util.spec_from_file_location("predict_all_locked_2022_technical_amendment_v3", AMENDED)
assert SPEC and SPEC.loader
predictor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predictor)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_tree_without_allowed_change(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name == "_assert_identity")]
    return ast.dump(tree, include_attributes=False)


def declared(path: Path, file_value: str) -> dict[str, object]:
    return {"file": file_value, "bytes": path.stat().st_size, "sha256": digest(path)}


def test_original_predictor_is_preserved_and_diff_is_whitelisted() -> None:
    assert digest(ORIGINAL) == EXPECTED_ORIGINAL_SHA256
    assert semantic_tree_without_allowed_change(ORIGINAL) == semantic_tree_without_allowed_change(AMENDED)


def test_relative_artifact_file_is_resolved_against_actual_artifact_directory(tmp_path: Path) -> None:
    artifact = tmp_path / "locked_2022_derivative.parquet"
    artifact.write_bytes(b"frozen-derivative")
    assert predictor._assert_identity(artifact, declared(artifact, artifact.name), "derivative") == artifact.resolve()
    assert predictor._assert_identity(artifact, declared(artifact, str(artifact.resolve())), "derivative") == artifact.resolve()


def test_relative_identity_cannot_name_a_different_file(tmp_path: Path) -> None:
    artifact = tmp_path / "locked_2022_derivative.parquet"
    artifact.write_bytes(b"frozen-derivative")
    with pytest.raises(RuntimeError, match="different path"):
        predictor._assert_identity(artifact, declared(artifact, "other.parquet"), "derivative")
