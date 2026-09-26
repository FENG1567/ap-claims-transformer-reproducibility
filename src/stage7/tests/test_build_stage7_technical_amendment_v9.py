"""Focused fail-closed tests for the v9 coverage-reporting binding lock."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


STAGE7 = Path(__file__).resolve().parents[1]
BUILDER_PATH = STAGE7 / "build_stage7_technical_amendment_v9.py"
SPEC = importlib.util.spec_from_file_location("build_stage7_technical_amendment_v9", BUILDER_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".sha256").write_text(f"{_digest(path)}  {path.name}\n", encoding="ascii")


def _spec(include_minus8: bool = True, coverage_has_minus8: bool = False) -> dict:
    categories = [-9, 1, 2, 3, 4]
    if include_minus8:
        categories.insert(0, -8)
    coverage_values = ["-9", "1", "2", "3", "4"]
    if coverage_has_minus8:
        coverage_values.insert(0, "-8")
    return {
        "status": "FROZEN_2022_EVALUATION_SPEC", "sealed_test_year": 2022,
        "drift_reference": {
            "status": "FROZEN_2022_DRIFT_REFERENCE",
            "covariates": {"zip": {"column": "ZIPINC_QRTL", "kind": "categorical", "categories": categories}},
            "coverage": [{"conformal_set": "global_80", "subgroup_dimension": "ZIPINC_QRTL", "subgroup_value": value,
                          "coverage": 0.8, "mean_set_size": 1.0, "abstention_rate": 0.0} for value in coverage_values],
        },
    }


def _inputs(tmp_path: Path, *, include_minus8: bool = True, coverage_has_minus8: bool = False) -> tuple[Path, Path, Path, Path]:
    analyzer, test = tmp_path / "drift.py", tmp_path / "test_drift.py"
    analyzer.write_text("# frozen analyzer\n", encoding="utf-8")
    test.write_text("# frozen test\n", encoding="utf-8")
    spec = tmp_path / "spec.json"
    _write_json(spec, _spec(include_minus8, coverage_has_minus8))
    previous = tmp_path / "v8.json"
    previous_value = {
        "status": "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION", "sealed_test_year": 2022,
        "2022_access_before_lock": False,
        "frozen_code": {str(spec.resolve()): {"file": str(spec.resolve()), "bytes": spec.stat().st_size, "sha256": _digest(spec)}},
        "technical_amendment": {"status": builder.PREVIOUS_STATUS, "original_unlock_lock": {"file": "original.json", "bytes": 1, "sha256": "0" * 64}},
        "technical_amendment_history": [],
    }
    _write_json(previous, previous_value)
    return previous, analyzer, test, spec


def test_build_adds_report_only_binding_and_preserves_v8_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    previous, analyzer, test, spec = _inputs(tmp_path)
    monkeypatch.setattr(builder, "EXPECTED_PREVIOUS_LOCK_SHA256", _digest(previous))
    result = builder.build(previous, analyzer, test, spec, tmp_path / "v9")
    lock = Path(result["lock"]["file"])
    value = json.loads(lock.read_text(encoding="utf-8"))
    assert value["status"] == "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
    assert value["sealed_test_year"] == 2022 and value["2022_access_before_lock"] is False
    assert value["technical_amendment"]["status"] == builder.STATUS
    assert value["technical_amendment"]["one_corrected_2022_drift_rerun_authorized"] is True
    assert value["technical_amendment"]["corrected_2022_primary_evaluation_reauthorized"] is False
    assert value["technical_amendment_history"][-1]["status"] == builder.PREVIOUS_STATUS
    assert value["frozen_code"][str(analyzer.resolve())]["sha256"] == _digest(analyzer)
    assert lock.with_suffix(lock.suffix + ".sha256").read_text(encoding="ascii").strip() == f"{_digest(lock)}  {lock.name}"


def test_wrong_previous_hash_and_existing_output_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    previous, analyzer, test, spec = _inputs(tmp_path)
    with pytest.raises(RuntimeError, match="reviewed immutable identity"):
        builder.build(previous, analyzer, test, spec, tmp_path / "v9")
    monkeypatch.setattr(builder, "EXPECTED_PREVIOUS_LOCK_SHA256", _digest(previous))
    existing = tmp_path / "existing"; existing.mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        builder.build(previous, analyzer, test, spec, existing)


@pytest.mark.parametrize(("include_minus8", "coverage_has_minus8", "message"), [
    (False, False, "not predeclared"),
    (True, True, "already contains"),
])
def test_inapplicable_or_unpredeclared_coverage_gap_fails_closed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, include_minus8: bool,
        coverage_has_minus8: bool, message: str) -> None:
    previous, analyzer, test, spec = _inputs(tmp_path, include_minus8=include_minus8, coverage_has_minus8=coverage_has_minus8)
    monkeypatch.setattr(builder, "EXPECTED_PREVIOUS_LOCK_SHA256", _digest(previous))
    with pytest.raises(RuntimeError, match=message):
        builder.build(previous, analyzer, test, spec, tmp_path / "v9")
