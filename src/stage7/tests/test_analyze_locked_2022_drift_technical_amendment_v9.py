"""Focused regression tests for the v9 report-only coverage-stratum amendment."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PATH = ROOT / "analyze_locked_2022_drift.py"
MODULE_PATH = ROOT / "analyze_locked_2022_drift_technical_amendment_v9.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drift = load_module(MODULE_PATH, "locked_drift_technical_amendment_v9")


def _coverage_spec() -> dict:
    entries = []
    for name, scope in (("global_80", "global"), ("m_zip_80", "mondrian")):
        item = {
            "name": name,
            "scope": scope,
            "nominal": 0.8,
            "mask_column": f"mask_{name}",
            "size_column": f"size_{name}",
        }
        if scope == "mondrian":
            item["mondrian_dimension"] = "ZIPINC_QRTL"
        entries.append(item)
    coverage = []
    for name in ("global_80", "m_zip_80"):
        coverage.append({
            "conformal_set": name,
            "subgroup_dimension": "ALL",
            "subgroup_value": "ALL",
            "coverage": 0.8,
            "mean_set_size": 1.0,
            "abstention_rate": 0.0,
        })
        for value in ("-9", "1", "2"):
            coverage.append({
                "conformal_set": name,
                "subgroup_dimension": "ZIPINC_QRTL",
                "subgroup_value": value,
                "coverage": 0.8,
                "mean_set_size": 1.0,
                "abstention_rate": 0.0,
            })
    return {
        "conformal_sets": entries,
        "subgroups": {"ZIPINC_QRTL": "ZIPINC_QRTL"},
        "drift_reference": {
            "covariates": {
                "zipinc_qrtl": {
                    "column": "ZIPINC_QRTL",
                    "kind": "categorical",
                    "categories": [-8, -9, 1, 2, 3, 4],
                }
            },
            "coverage": coverage,
        },
    }


def _frame(values: list[int]) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame({
        "primary_leaf": np.array(["none", "ap", "biliary", "none", "ap", "none"][:n], dtype=object),
        "ZIPINC_QRTL": values,
        "mask_global_80": np.full(n, 31, dtype=np.uint8),
        "mask_m_zip_80": np.full(n, 31, dtype=np.uint8),
    })


def test_predeclared_minus8_is_independent_report_only_stratum() -> None:
    spec = _coverage_spec()
    report = drift.coverage_drift(
        _frame([-9, -8, 1, 2, -8, -9]),
        spec,
        spec["drift_reference"],
    )
    new = report.loc[
        (report["subgroup_dimension"] == "ZIPINC_QRTL")
        & (report["subgroup_value"] == "-8")
    ]
    assert len(new) == 2
    assert set(new["status"]) == {"NO_FROZEN_REFERENCE_NEW_STRATUM"}
    assert new["coverage_reference"].isna().all()
    assert new["coverage_change"].isna().all()
    assert new["mean_set_size_reference"].isna().all()
    assert new["mean_set_size_change"].isna().all()
    assert new["abstention_rate_reference"].isna().all()
    assert new["abstention_rate_change"].isna().all()
    assert set(new["subgroup_value"]) == {"-8"}
    assert (report["subgroup_value"] == "-8").any()
    known = report.loc[
        (report["subgroup_dimension"] == "ZIPINC_QRTL")
        & (report["subgroup_value"] == "-9")
    ]
    assert len(known) == 2
    assert known["status"].ne("NO_FROZEN_REFERENCE_NEW_STRATUM").all()
    assert known["coverage_reference"].notna().all()


def test_unpredeclared_minus7_still_fails_closed() -> None:
    spec = _coverage_spec()
    with pytest.raises(RuntimeError, match="Frozen coverage reference lacks exact 2022 reporting stratum"):
        drift.coverage_drift(
            _frame([-7, -8, 1, 2, -8, -9]),
            spec,
            spec["drift_reference"],
        )


def test_noncoverage_functions_are_mechanical_copies_of_original() -> None:
    original = ast.parse(ORIGINAL_PATH.read_text(encoding="utf-8"))
    amended = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))

    def function_map(tree: ast.Module) -> dict[str, str]:
        excluded = {"coverage_drift", "_predeclared_coverage_level", "analyze_locked_2022_drift"}
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name not in excluded
        }

    assert function_map(original) == function_map(amended)
    assert drift.SCHEMA_VERSION == "stage7_locked_2022_drift_technical_amendment_v9"

