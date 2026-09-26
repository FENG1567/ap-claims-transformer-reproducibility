"""Synthetic regression tests for the locked four-domain 2022 drift report."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "analyze_locked_2022_drift.py"
EVALUATOR_PATH = ROOT / "evaluate_locked_2022.py"
SPEC = importlib.util.spec_from_file_location("locked_drift", MODULE_PATH)
assert SPEC and SPEC.loader
drift = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drift)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def full_spec(path: Path) -> dict:
    sets = []
    for label, nominal in (("80", .8), ("90", .9), ("95", .95)):
        sets.append({"name": f"global_{label}", "scope": "global", "nominal": nominal, "mask_column": f"mg{label}", "size_column": f"sg{label}"})
        for dimension in ("sex", "age", "PAY1", "ZIPINC_QRTL"):
            sets.append({"name": f"m_{dimension}_{label}", "scope": "mondrian", "mondrian_dimension": dimension, "nominal": nominal, "mask_column": f"m{dimension}{label}", "size_column": f"s{dimension}{label}"})
    covariates = {"age": {"column": "AGE", "kind": "numeric_binned", "bins": [18, 45, 65, 120], "reference_missing_rate": 0., "reference_weighted_distribution": {"[18,45)": .25, "[45,65)": .25, "[65,120]": .5, "<NONFINITE_OR_OUT_OF_RANGE>": 0.}, "reference_weighted_mean": 58., "reference_weighted_sd": 15.}, "sex": {"column": "FEMALE", "kind": "categorical", "categories": [0, 1], "reference_missing_rate": 0., "reference_weighted_distribution": {"0": .5, "1": .5, "<UNSEEN_OR_OTHER>": 0.}}}
    outcomes = {
        "any_readmission": {"column": "y1", "unweighted_rate": .25, "DISCWT_weighted_rate": .25},
        "ap_specific_readmission": {"column": "y2", "unweighted_rate": .25, "DISCWT_weighted_rate": .25},
        "biliary_readmission": {"column": "y3", "unweighted_rate": .25, "DISCWT_weighted_rate": .25},
        "sepsis_or_organ_readmission": {"column": "y4", "unweighted_rate": .25, "DISCWT_weighted_rate": .25},
        "high_cost": {"column": "high_cost_label", "unweighted_rate": .25, "DISCWT_weighted_rate": .25,
                      "validity": {"policy": "EXCLUDE_NULL_HIGH_COST_LABELS_FROM_ALL_RATES_AND_BOOTSTRAPS", "reference_valid_n": 100,
                                   "reference_missing_n": 11, "reference_valid_weight": 150., "small_cell_counts_suppressed": False}},
        "prolonged_los": {"column": "y6", "unweighted_rate": .25, "DISCWT_weighted_rate": .25},
        "in_hospital_death": {"column": "y7", "unweighted_rate": .25, "DISCWT_weighted_rate": .25},
    }
    calibration = {model: {outcome: {"calibration_intercept": 0., "calibration_slope": 1., "brier": .2, "log_loss": .5} for outcome in ("any_readmission", "ap_specific_readmission")} for model in ("transformer", "lightgbm")}
    coverage = []
    subgroup_values = {"sex": ("0", "1"), "age": ("18-44", "45-64"), "PAY1": ("1", "2"), "ZIPINC_QRTL": ("1", "2")}
    for item in sets:
        coverage.append({"conformal_set": item["name"], "subgroup_dimension": "ALL", "subgroup_value": "ALL", "coverage": .9, "mean_set_size": 1., "abstention_rate": 0.})
        for dimension in ("sex", "age", "PAY1", "ZIPINC_QRTL"):
            for value in subgroup_values[dimension]:
                coverage.append({"conformal_set": item["name"], "subgroup_dimension": dimension, "subgroup_value": value, "coverage": .9, "mean_set_size": 1., "abstention_rate": 0.})
    value = {"status": "FROZEN_2022_EVALUATION_SPEC", "sealed_test_year": 2022, "co_primary_outcomes": {"any_readmission": "y1", "ap_specific_readmission": "y2"},
             "models": {"transformer": {"probabilities": {"any_readmission": "pt1", "ap_specific_readmission": "pt2"}, "thresholds": {"any_readmission": .5, "ap_specific_readmission": .5}}, "lightgbm": {"probabilities": {"any_readmission": "pl1", "ap_specific_readmission": "pl2"}, "thresholds": {"any_readmission": .5, "ap_specific_readmission": .5}}},
             "primary_comparison": {"transformer": "transformer", "lightgbm": "lightgbm"}, "conformal_sets": sets, "subgroups": {"sex": "FEMALE", "age": "age_group", "PAY1": "PAY1", "ZIPINC_QRTL": "ZIPINC_QRTL"},
             "drift_reference": {"status": "FROZEN_2022_DRIFT_REFERENCE", "covariates": covariates, "label_outcomes": outcomes, "calibration": calibration, "coverage": coverage, "bootstrap": {"n_replicates": 100, "seed": 711, "max_threads": 8, "cluster_column": "patient_hash"}}}
    write_json(path, value)
    return value


def unlock(tmp_path: Path, spec_path: Path) -> Path:
    path = tmp_path / "unlock.json"
    lock = {"status": "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION", "sealed_test_year": 2022, "2022_access_before_lock": False, "frozen_code": {str(EVALUATOR_PATH.resolve()): identity(EVALUATOR_PATH), str(MODULE_PATH.resolve()): identity(MODULE_PATH), str(spec_path.resolve()): identity(spec_path)}}
    write_json(path, lock)
    path.with_suffix(".json.sha256").write_text(f"{digest(path)}  {path.name}\n", encoding="ascii")
    return path


def predictions(path: Path) -> None:
    n = 12
    leaves = np.array(["none", "ap", "biliary", "sepsis_or_organ", "other", "none"] * 2)
    y1 = (leaves != "none").astype(int); y2 = (leaves == "ap").astype(int)
    frame = pd.DataFrame({"encounter_hash": [f"e{x}" for x in range(n)], "patient_hash": [f"p{x // 2}" for x in range(n)], "hospital_hash": [f"h{x % 3}" for x in range(n)], "DISCWT": np.linspace(1, 2, n), "analysis_year": 2022, "analysis_partition": "test", "primary_leaf": leaves, "FEMALE": [0, 1] * 6, "age_group": ["18-44", "45-64"] * 6, "PAY1": [1, 2] * 6, "ZIPINC_QRTL": [1, 2] * 6, "AGE": [30, 55, 70, 60, 42, 75] * 2, "y1": y1, "y2": y2, "pt1": np.where(y1, .8, .1), "pl1": np.where(y1, .7, .2), "pt2": np.where(y2, .8, .1), "pl2": np.where(y2, .7, .2)})
    for number in range(3, 8): frame[f"y{number}"] = ((np.arange(n) + number) % 4 == 0).astype(int)
    frame["high_cost_label"] = pd.Series([pd.NA, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0], dtype="Int64")
    for label in ("80", "90", "95"):
        frame[f"mg{label}"] = np.array([1, 2, 4, 8, 16, 1] * 2, dtype=np.uint8); frame[f"sg{label}"] = 1
        for dimension in ("sex", "age", "PAY1", "ZIPINC_QRTL"):
            frame[f"m{dimension}{label}"] = frame[f"mg{label}"]; frame[f"s{dimension}{label}"] = 1
    frame.to_parquet(path, index=False, engine="pyarrow")


def test_full_locked_drift_is_atomic_and_four_domains_are_separate(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"; full_spec(spec_path); lock = unlock(tmp_path, spec_path)
    prediction_path = tmp_path / "predictions.parquet"; predictions(prediction_path)
    result = drift.analyze_locked_2022_drift(prediction_path, lock, spec_path, tmp_path / "drift")
    assert result["status"] == "PASS_LOCKED_2022_DRIFT_ANALYSIS"
    assert set(result["four_separate_domains"]) == {"covariate", "label", "calibration", "coverage"}
    out = Path(result["output"])
    for name in ("covariate_drift", "label_drift", "label_rate_bootstrap", "calibration_drift", "coverage_drift"):
        assert (out / f"{name}.parquet").is_file()
    assert pd.read_parquet(out / "label_rate_bootstrap.parquet").groupby("outcome").size().eq(100).all()
    assert set(pd.read_parquet(out / "calibration_drift.parquet")["model"]) == {"transformer", "lightgbm"}
    assert pd.read_parquet(out / "coverage_drift.parquet")["event_tier"].notna().all()
    with pytest.raises(RuntimeError, match="immutable"):
        drift.analyze_locked_2022_drift(prediction_path, lock, spec_path, out)


def test_self_identity_and_drift_contract_fail_closed_before_data_open(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"; spec = full_spec(spec_path); lock = unlock(tmp_path, spec_path)
    spec["drift_reference"]["bootstrap"]["cluster_column"] = "NRD_STRATUM"; write_json(spec_path, spec)
    # Lock still carries the old spec identity, so tampering is rejected before any parquet operation.
    with pytest.raises(RuntimeError, match="Frozen manifest"):
        drift.analyze_locked_2022_drift(tmp_path / "not-opened.parquet", lock, spec_path, tmp_path / "out")
    full_spec(spec_path); lock = unlock(tmp_path, spec_path)
    lock_data = json.loads(lock.read_text(encoding="utf-8")); lock_data["frozen_code"][str(MODULE_PATH.resolve())]["sha256"] = "0" * 64; write_json(lock, lock_data); lock.with_suffix(".json.sha256").write_text(f"{digest(lock)}  {lock.name}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="Frozen manifest"):
        drift.analyze_locked_2022_drift(tmp_path / "not-opened.parquet", lock, spec_path, tmp_path / "out2")


def test_reference_validation_rejects_nonfrozen_categories_and_stratum_bootstrap(tmp_path: Path) -> None:
    path = tmp_path / "spec.json"; spec = full_spec(path)
    spec["drift_reference"]["covariates"]["sex"]["reference_weighted_distribution"].pop("<UNSEEN_OR_OTHER>")
    with pytest.raises(RuntimeError, match="exactly"):
        drift.validate_drift_reference(spec)
    spec = full_spec(path); spec["drift_reference"]["bootstrap"]["cluster_column"] = "NRD_STRATUM"
    with pytest.raises(RuntimeError, match="NRD_STRATUM"):
        drift.validate_drift_reference(spec)


def test_nullable_high_cost_excludes_missing_rows_from_rates_and_bootstrap(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"; spec = full_spec(spec_path)
    prediction_path = tmp_path / "predictions.parquet"; predictions(prediction_path)
    frame = pd.read_parquet(prediction_path)
    reference = spec["drift_reference"]
    bootstrap = drift._bootstrap_rates(frame, reference["label_outcomes"], reference["bootstrap"])
    report = drift.label_drift(frame, reference, bootstrap)
    high = report.loc[(report["outcome"] == "high_cost") & (report["weighting"] == "unweighted_rate")].iloc[0]
    valid = frame["high_cost_label"].notna().to_numpy()
    expected = float(frame.loc[valid, "high_cost_label"].astype(int).mean())
    assert high["observed_rate"] == pytest.approx(expected)
    assert high["observed_rate"] != pytest.approx(float(frame["high_cost_label"].fillna(0).astype(int).mean()))
    assert high["valid_n"] == 11 and pd.isna(high["missing_n"])
    high_boot = bootstrap.loc[bootstrap["outcome"] == "high_cost"]
    assert high_boot["valid_n"].ge(1).all()
    assert high_boot["valid_weight"].gt(0).all()
