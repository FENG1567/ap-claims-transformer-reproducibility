"""Regression tests for the v5 all-model derivative input contract."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path


STAGE7 = Path(__file__).resolve().parents[1]
BUILDER_PATH = STAGE7 / "build_stage7_technical_amendment_v5.py"
ETL_SPEC = STAGE7.parents[1] / "outputs" / "stage7_pre2022_lock" / "data_specs" / "frozen_2022_etl_spec.json"
DERIVATIVE_SPEC = STAGE7.parents[1] / "outputs" / "stage7_pre2022_lock" / "data_specs" / "frozen_2022_derivative_spec.json"
BASELINE_SUPPORT = STAGE7.parents[0] / "stage4" / "train_baselines.py"


SPEC = importlib.util.spec_from_file_location("build_stage7_technical_amendment_v5", BUILDER_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_specs_are_preserved_and_have_reviewed_identities() -> None:
    etl = json.loads(ETL_SPEC.read_text(encoding="utf-8"))
    derivative = json.loads(DERIVATIVE_SPEC.read_text(encoding="utf-8"))
    assert etl["status"] == "FROZEN_2022_LOCAL_ETL_SPEC" and "technical_amendment" not in etl
    assert derivative["status"] == "FROZEN_2022_DERIVATIVE_SPEC" and "technical_amendment" not in derivative
    assert len(digest(ETL_SPEC)) == 64 and len(digest(DERIVATIVE_SPEC)) == 64


def test_amended_specs_add_exactly_the_frozen_baseline_contract() -> None:
    original_etl = json.loads(ETL_SPEC.read_text(encoding="utf-8"))
    original_derivative = json.loads(DERIVATIVE_SPEC.read_text(encoding="utf-8"))
    etl_before = copy.deepcopy(original_etl)
    derivative_before = copy.deepcopy(original_derivative)
    required = builder.load_baseline_required_fields(BASELINE_SUPPORT)
    etl, derivative = builder.amend_specs(original_etl, original_derivative, required)
    assert original_etl == etl_before and original_derivative == derivative_before
    assert set(builder.CORE_SOURCE_ADDITIONS).issubset(etl["source_columns"]["core"])
    assert set(builder.CORE_SOURCE_ADDITIONS).issubset(etl["core_passthrough_columns"])
    assert set(builder.BASELINE_MISSING_FIELDS).issubset(etl["episode_output_columns"])
    assert set(builder.BASELINE_MISSING_FIELDS).issubset(derivative["episode_columns"])
    assert set(builder.BASELINE_MISSING_FIELDS).issubset(derivative["static_columns"])
    projected = set(derivative["episode_columns"]) | set(derivative["history_columns"])
    assert required.issubset(projected)


def test_only_whitelisted_spec_fields_change() -> None:
    original_etl = json.loads(ETL_SPEC.read_text(encoding="utf-8"))
    original_derivative = json.loads(DERIVATIVE_SPEC.read_text(encoding="utf-8"))
    required = builder.load_baseline_required_fields(BASELINE_SUPPORT)
    etl, derivative = builder.amend_specs(original_etl, original_derivative, required)
    for key in set(original_etl) - {"source_columns", "core_passthrough_columns", "episode_output_columns"}:
        assert etl[key] == original_etl[key]
    for role in set(original_etl["source_columns"]) - {"core"}:
        assert etl["source_columns"][role] == original_etl["source_columns"][role]
    for key in set(original_derivative) - {"episode_columns", "static_columns"}:
        assert derivative[key] == original_derivative[key]
    assert etl["technical_amendment"]["year_2022_patient_values_used"] is False
    assert derivative["technical_amendment"]["year_2022_patient_values_used"] is False
