"""Regression tests for frozen fine-tune checkpoint identities in v4."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


STAGE7 = Path(__file__).resolve().parents[1]
V3 = STAGE7 / "predict_all_locked_2022_technical_amendment_v3.py"
V4 = STAGE7 / "predict_all_locked_2022_technical_amendment_v4.py"
EXPECTED_V3_SHA256 = "711ce7684bd3bbea84be27757d30f2a6331e264caec8c0257de545ce6600ba32"


SPEC = importlib.util.spec_from_file_location("predict_all_locked_2022_technical_amendment_v4", V4)
assert SPEC and SPEC.loader
predictor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(predictor)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_tree_without_allowed_change(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name == "_validate_finetune_manifest")]
    return ast.dump(tree, include_attributes=False)


def write_manifest(path: Path, checkpoint: Path, **checkpoint_overrides: object) -> None:
    check = {"file": checkpoint.name, "sha256": digest(checkpoint)}
    check.update(checkpoint_overrides)
    value = {
        "status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022",
        "prediction_partition": "2021A",
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "checkpoint": check,
    }
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_v3_is_preserved_and_v4_diff_is_whitelisted() -> None:
    assert digest(V3) == EXPECTED_V3_SHA256
    assert semantic_tree_without_allowed_change(V3) == semantic_tree_without_allowed_change(V4)


def test_hash_bound_relative_checkpoint_without_redundant_bytes_is_accepted(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_checkpoint.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    manifest = tmp_path / "finetune_manifest.json"
    write_manifest(manifest, checkpoint)
    result = predictor._validate_finetune_manifest(manifest, checkpoint, "synthetic")
    assert result["checkpoint"]["sha256"] == digest(checkpoint)


def test_optional_bytes_and_declared_path_remain_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_checkpoint.pt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    manifest = tmp_path / "finetune_manifest.json"
    write_manifest(manifest, checkpoint, bytes=checkpoint.stat().st_size + 1)
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        predictor._validate_finetune_manifest(manifest, checkpoint, "synthetic")
    write_manifest(manifest, checkpoint, file="other.pt")
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        predictor._validate_finetune_manifest(manifest, checkpoint, "synthetic")
    write_manifest(manifest, checkpoint, sha256="0" * 64)
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        predictor._validate_finetune_manifest(manifest, checkpoint, "synthetic")
