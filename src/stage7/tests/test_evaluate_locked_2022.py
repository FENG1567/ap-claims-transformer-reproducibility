"""Synthetic, no-data regression tests for the locked 2022 evaluation core."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluate_locked_2022.py"
SPEC = importlib.util.spec_from_file_location("evaluate_locked_2022", MODULE_PATH)
assert SPEC and SPEC.loader
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def identity(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def frozen_spec(path: Path) -> dict:
    conformal_sets = []
    for label, nominal in (("80", 0.8), ("90", 0.9), ("95", 0.95)):
        conformal_sets.append({"name": f"global_{label}", "scope": "global", "nominal": nominal,
                               "mask_column": f"mask_global_{label}", "size_column": f"size_global_{label}",
                               **({"abstain_column": "abstain_global_90"} if label == "90" else {})})
        for dimension in ("sex", "age", "PAY1", "ZIPINC_QRTL"):
            conformal_sets.append({"name": f"mondrian_{dimension}_{label}", "scope": "mondrian",
                                   "mondrian_dimension": dimension, "nominal": nominal,
                                   "mask_column": f"mask_{dimension}_{label}", "size_column": f"size_{dimension}_{label}"})
    value = {
        "status": "FROZEN_2022_EVALUATION_SPEC", "sealed_test_year": 2022,
        "co_primary_outcomes": {
            "any_readmission": "outcome_any_readmission",
            "ap_specific_readmission": "outcome_ap_specific_readmission",
        },
        "models": {
            "transformer": {"probabilities": {"any_readmission": "p_transformer_any", "ap_specific_readmission": "p_transformer_ap"},
                            "thresholds": {"any_readmission": 0.50, "ap_specific_readmission": 0.50}},
            "lightgbm": {"probabilities": {"any_readmission": "p_lightgbm_any", "ap_specific_readmission": "p_lightgbm_ap"},
                         "thresholds": {"any_readmission": 0.50, "ap_specific_readmission": 0.50}},
            "elastic_net": {"probabilities": {"any_readmission": "p_elastic_any", "ap_specific_readmission": "p_elastic_ap"},
                            "thresholds": {"any_readmission": 0.50, "ap_specific_readmission": 0.50}},
        },
        "primary_comparison": {"transformer": "transformer", "lightgbm": "lightgbm"},
        "conformal_sets": conformal_sets,
        "subgroups": {"sex": "FEMALE", "age": "age_group", "PAY1": "PAY1", "ZIPINC_QRTL": "ZIPINC_QRTL"},
    }
    write_json(path, value)
    return value


def unlock_with_exact_identities(tmp_path: Path, spec_path: Path) -> Path:
    lock_path = tmp_path / "unlock.json"
    frozen_code = {str(MODULE_PATH.resolve()): identity(MODULE_PATH), str(spec_path.resolve()): identity(spec_path)}
    lock = {"status": "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION", "sealed_test_year": 2022,
            "2022_access_before_lock": False, "frozen_code": frozen_code}
    write_json(lock_path, lock)
    lock_path.with_suffix(".json.sha256").write_text(f"{digest(lock_path)}  {lock_path.name}\n", encoding="ascii")
    return lock_path


def synthetic_predictions(path: Path) -> pd.DataFrame:
    n = 12
    leaves = np.array(["none", "ap", "biliary", "sepsis_or_organ", "other", "none"] * 2)
    any_y = (leaves != "none").astype(int)
    ap_y = (leaves == "ap").astype(int)
    masks = np.array([1, 2, 4, 8, 16, 1] * 2, dtype=np.uint8)
    frame = pd.DataFrame({
        "encounter_hash": [f"e{index}" for index in range(n)], "patient_hash": [f"p{index // 2}" for index in range(n)],
        "hospital_hash": [f"h{index % 3}" for index in range(n)], "DISCWT": np.linspace(1, 2, n),
        "analysis_year": 2022, "analysis_partition": "test", "outcome_any_readmission": any_y,
        "outcome_ap_specific_readmission": ap_y, "primary_leaf": leaves,
        "FEMALE": [0, 1] * 6, "age_group": ["18-44", "65-74", "75+"] * 4,
        "PAY1": [1, 2, 3] * 4, "ZIPINC_QRTL": [1, 2, 3, 4] * 3,
        "p_transformer_any": np.where(any_y == 1, 0.80, 0.15),
        "p_lightgbm_any": np.where(any_y == 1, 0.70, 0.25),
        "p_elastic_any": np.where(any_y == 1, 0.65, 0.30),
        "p_transformer_ap": np.where(ap_y == 1, 0.85, 0.08),
        "p_lightgbm_ap": np.where(ap_y == 1, 0.72, 0.12),
        "p_elastic_ap": np.where(ap_y == 1, 0.67, 0.15),
    })
    for label in ("80", "90", "95"):
        frame[f"mask_global_{label}"] = masks
        frame[f"size_global_{label}"] = 1
        for dimension in ("sex", "age", "PAY1", "ZIPINC_QRTL"):
            frame[f"mask_{dimension}_{label}"] = masks
            frame[f"size_{dimension}_{label}"] = 1
    frame["abstain_global_90"] = False
    frame.to_parquet(path, index=False, engine="pyarrow")
    return frame


def test_unlock_self_hash_gate_fails_closed(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    frozen_spec(spec_path)
    lock_path = unlock_with_exact_identities(tmp_path, spec_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["frozen_code"][str(MODULE_PATH.resolve())]["sha256"] = "0" * 64
    write_json(lock_path, lock)
    lock_path.with_suffix(".json.sha256").write_text(f"{digest(lock_path)}  {lock_path.name}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="Evaluator path/bytes/SHA256"):
        ev.validate_unlock_lock(lock_path, MODULE_PATH, [spec_path])


def test_unlock_manifest_and_spec_tampering_fail_before_prediction_read(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    frozen_spec(spec_path)
    lock_path = unlock_with_exact_identities(tmp_path, spec_path)
    spec_path.write_text(spec_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen manifest"):
        ev.validate_unlock_lock(lock_path, MODULE_PATH, [spec_path])


def test_invalid_mixed_test_input_fails_closed() -> None:
    spec = {"co_primary_outcomes": {"any_readmission": "y1", "ap_specific_readmission": "y2"},
            "models": {"m": {"probabilities": {"any_readmission": "p1", "ap_specific_readmission": "p2"}, "thresholds": {"any_readmission": .5, "ap_specific_readmission": .5}}},
            "conformal_sets": [], "subgroups": {"sex": "sex", "age": "age", "PAY1": "pay", "ZIPINC_QRTL": "zip"}}
    frame = pd.DataFrame({"encounter_hash": ["x", "x"], "patient_hash": ["p1", "p2"], "hospital_hash": ["h1", "h2"],
                          "DISCWT": [1.0, 1.0], "analysis_year": [2022, 2021], "analysis_partition": ["test", "test"],
                          "y1": [0, 1], "y2": [0, 0], "p1": [.2, .8], "p2": [.1, .2],
                          "sex": [0, 1], "age": ["a", "b"], "pay": [1, 2], "zip": [1, 2]})
    with pytest.raises(RuntimeError, match="duplicate encounter_hash"):
        ev.validate_prediction_frame(frame, spec)


def test_missing_conformal_contract_and_undefined_metrics_are_explicit(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    bad = frozen_spec(spec_path)
    bad["conformal_sets"] = [entry for entry in bad["conformal_sets"] if not (
        entry["scope"] == "mondrian" and entry.get("mondrian_dimension") == "age" and entry["nominal"] == 0.9
    )]
    write_json(spec_path, bad)
    lock_path = unlock_with_exact_identities(tmp_path, spec_path)
    with pytest.raises(RuntimeError, match="all prespecified Mondrian"):
        ev.validate_evaluation_spec(spec_path, ev.validate_unlock_lock(lock_path, MODULE_PATH, [spec_path]))
    values = ev.binary_metrics(np.zeros(3, dtype=np.int8), np.array([.1, .2, .3]), np.ones(3))
    assert values["auroc"] is None and values["auprc"] is None
    assert values["discrimination_status"] == "UNDEFINED_SINGLE_OUTCOME_CLASS"


def test_full_synthetic_evaluation_is_atomic_and_reports_contract(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    frozen_spec(spec_path)
    unlock_path = unlock_with_exact_identities(tmp_path, spec_path)
    predictions = tmp_path / "synthetic_2022.parquet"
    synthetic_predictions(predictions)
    output = tmp_path / "primary_2022"
    result = ev.evaluate_locked_2022(predictions, unlock_path, spec_path, output)
    assert result["status"] == "PASS_LOCKED_2022_PRIMARY_EVALUATION"
    assert (output / "metrics.parquet").is_file()
    assert (output / "comparisons_patient_bootstrap.parquet").is_file()
    assert (output / "comparisons_hospital_bootstrap.parquet").is_file()
    assert (output / "patient_bootstrap_replicates.parquet").is_file()
    metrics = pd.read_parquet(output / "metrics.parquet")
    assert len(metrics) == 3 * 2 * 2
    assert metrics["operating_threshold"].eq(0.5).all()
    assert set(pd.read_parquet(output / "decision_curve.parquet")["threshold_probability"]) == set(ev.DECISION_THRESHOLDS)
    boot = pd.read_parquet(output / "patient_bootstrap_replicates.parquet")
    assert boot.groupby(["outcome", "weighting"]).size().eq(1000).all()
    observed = pd.read_parquet(output / "comparisons_patient_bootstrap.parquet")
    assert observed["observed_status"].eq("OK").all()
    assert np.allclose(observed["auprc_difference_observed"], observed["transformer_auprc_observed"] - observed["lightgbm_auprc_observed"])
    hospital = pd.read_parquet(output / "comparisons_hospital_bootstrap.parquet")
    assert hospital["n_replicates"].eq(1000).all()
    assert result["hospital_bootstrap_included"] is True
    conformal = pd.read_parquet(output / "conformal.parquet")
    assert conformal["coverage_ci95_method"].eq("WILSON_95_UNWEIGHTED").all()
    assert conformal["coverage_ci95_lower"].le(conformal["coverage"]).all()
    assert conformal["coverage"].le(conformal["coverage_ci95_upper"]).all()
    subgroup = pd.read_parquet(output / "conformal_subgroups.parquet")
    assert set(subgroup["event_tier"]) == {"DESCRIPTIVE_LT_50_EVENTS"}
    with pytest.raises(RuntimeError, match="immutable"):
        ev.evaluate_locked_2022(predictions, unlock_path, spec_path, output)


def test_conformal_size_and_probability_validation_fail_closed(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec = frozen_spec(spec_path)
    predictions = tmp_path / "synthetic.parquet"
    frame = synthetic_predictions(predictions)
    frame.loc[0, "size_global_80"] = 5
    with pytest.raises(RuntimeError, match="set size does not match"):
        ev.validate_prediction_frame(frame, spec)
    frame.loc[0, "size_global_80"] = 1
    frame.loc[0, "p_transformer_any"] = np.inf
    with pytest.raises(RuntimeError, match="not finite"):
        ev.validate_prediction_frame(frame, spec)


def test_cluster_bootstrap_has_exact_replicates_and_keeps_weighted_option(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"
    spec = frozen_spec(spec_path)
    predictions = tmp_path / "synthetic.parquet"
    frame = synthetic_predictions(predictions)
    summary, details = ev.paired_bootstrap(frame, spec, cluster_column="hospital_hash")
    assert summary["n_replicates"].eq(1000).all()
    assert details.groupby(["outcome", "weighting"]).size().eq(1000).all()
    assert set(summary["cluster_unit"]) == {"hospital_hash"}


def test_calibration_uses_current_sklearn_compatible_unpenalized_form_without_penalty_warning() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        value = ev.binary_metrics(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]), np.ones(4))
    assert value["calibration_status"] == "OK"
    assert not any("penalty" in str(item.message).lower() and "deprecated" in str(item.message).lower() for item in captured)
