#!/usr/bin/env python3
"""Build the immutable, data-free 2022 evaluation contract.

This is a deliberately fail-closed pre-unlock step.  It reads only JSON
manifests and already-frozen model artifacts; it has no parquet reader, no
NRD/MIMIC path argument, and no 2022 input.  The resulting JSON is a complete
wide-prediction schema for the later one-time evaluator and is SHA256-bound to
the exact 2021A operating points, 2021B conformal calibrator, and evaluator
source code from which it was made.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any


TRANSFORMER_ROSTER = (
    "joint_main", "mlm_only", "scratch", "ap_only",
    "no_hierarchy_parameter_matched", "no_prday", "no_prior",
    "no_hospital_socioeconomic", "no_year", "common_variable_only",
)
BASELINE_ROSTER = ("structured_logistic", "elastic_net", "lightgbm")
OUTCOMES = ("any_readmission", "ap_specific_readmission")
NOMINAL_COVERAGES = (("80", 0.80), ("90", 0.90), ("95", 0.95))
SUBGROUPS = {
    "sex": "FEMALE", "age": "age_group", "PAY1": "PAY1", "ZIPINC_QRTL": "ZIPINC_QRTL",
}
MONDRIAN_SOURCE_DIMENSIONS = {
    "sex": "sex", "age": "age_group", "PAY1": "payer", "ZIPINC_QRTL": "zip_income_quartile",
}
EVALUATOR_SCHEMA_VERSION = "stage7_locked_2022_evaluation_v1"
DRIFT_ANALYZER_PATH = Path(__file__).with_name("analyze_locked_2022_drift.py")


def sha256(path: Path) -> str:
    """Return a streaming digest for one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing required artifact: {path}")
    return {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _assert_identity(path: Path, declared: object, label: str) -> None:
    if not isinstance(declared, dict) or not path.is_file():
        raise RuntimeError(f"Missing {label} identity")
    if declared.get("sha256") != sha256(path):
        raise RuntimeError(f"{label} SHA256 mismatch: {path}")
    try:
        expected_bytes = int(declared.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {label} size identity") from exc
    if expected_bytes != path.stat().st_size:
        raise RuntimeError(f"{label} size mismatch: {path}")


def _declared_file_identity(declared: object, label: str) -> tuple[Path, dict[str, Any]]:
    """Resolve and validate an identity record which itself names its file."""
    if not isinstance(declared, dict) or not isinstance(declared.get("file"), str):
        raise RuntimeError(f"Missing {label} file identity")
    path = Path(declared["file"]).resolve()
    _assert_identity(path, declared, label)
    return path, identity(path)


def _regular_child(directory: Path, name: str, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise RuntimeError(f"Invalid {label} filename")
    candidate = directory / name
    if not candidate.is_file() or candidate.resolve().parent != directory.resolve():
        raise RuntimeError(f"Missing or unsafe {label}: {candidate}")
    return candidate


def _numeric_threshold(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise RuntimeError(f"Invalid frozen 2021A threshold: {label}")
    return float(value)


def _validate_operating_manifest(path: Path, run_name: str) -> dict[str, Any]:
    manifest = read_json(path)
    if (
        manifest.get("status") != "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A"
        or manifest.get("selection_partition") != "2021A"
        or manifest.get("2021B_accessed") is not False
        or manifest.get("year_2022_accessed") is not False
        or manifest.get("capacity_fraction") != 0.20
    ):
        raise RuntimeError(f"Transformer operating-point gate failed: {run_name}")
    artifacts = manifest.get("artifacts")
    required = {
        "transformer_binary_calibrators.joblib",
        "transformer_operating_thresholds_2021A.json",
        "transformer_calibration_selection_2021A.json",
        "predictions_2021A_calibrated.parquet",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise RuntimeError(f"Transformer operating-point artifact roster is incomplete: {run_name}")
    directory = path.resolve().parent
    files: dict[str, Path] = {}
    for name in sorted(required):
        artifact = _regular_child(directory, name, "transformer operating-point artifact")
        _assert_identity(artifact, artifacts[name], f"{run_name}/{name}")
        files[name] = artifact
    thresholds = read_json(files["transformer_operating_thresholds_2021A.json"])
    if set(thresholds) != set(OUTCOMES):
        raise RuntimeError(f"Transformer threshold outcome roster is incomplete: {run_name}")
    endpoints = manifest.get("endpoints")
    if not isinstance(endpoints, list) or tuple(endpoints) != OUTCOMES:
        raise RuntimeError(f"Transformer operating-point endpoint order is unstable: {run_name}")
    for outcome in OUTCOMES:
        record = thresholds.get(outcome)
        if not isinstance(record, dict):
            raise RuntimeError(f"Missing transformer threshold outcome: {run_name}/{outcome}")
        if record.get("probability_column") != f"p_{outcome}_calibrated":
            raise RuntimeError(f"Unexpected transformer calibrated probability: {run_name}/{outcome}")
        if record.get("rule") != "fixed 20% capacity on 2021A; threshold transported unchanged":
            raise RuntimeError(f"Transformer threshold is not the frozen 20% 2021A rule: {run_name}/{outcome}")
        _numeric_threshold(record.get("threshold"), f"{run_name}/{outcome}")
    return {"manifest": manifest, "files": files, "thresholds": thresholds}


def validate_all_operating_points(path: Path) -> dict[str, dict[str, Any]]:
    """Validate the exact ten-run registry and return source-bound details."""
    path = path.resolve()
    registry = read_json(path)
    if (
        registry.get("status") != "PASS_ALL_TRANSFORMER_CALIBRATIONS_AND_THRESHOLDS_LOCKED_ON_2021A"
        or registry.get("selection_partition") != "2021A"
        or registry.get("2021B_accessed") is not False
        or registry.get("year_2022_accessed") is not False
        or registry.get("exact_roster") != list(TRANSFORMER_ROSTER)
    ):
        raise RuntimeError("All-transformer registry is not an exact pre-2021B/pre-2022 seal")
    runs = registry.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(TRANSFORMER_ROSTER):
        missing = sorted(set(TRANSFORMER_ROSTER) - set(runs or {}))
        extra = sorted(set(runs or {}) - set(TRANSFORMER_ROSTER))
        raise RuntimeError(f"Incomplete or mixed transformer roster; missing={missing}, extra={extra}")
    details: dict[str, dict[str, Any]] = {}
    seen_directories: set[Path] = set()
    for run_name in TRANSFORMER_ROSTER:
        entry = runs[run_name]
        if not isinstance(entry, dict) or not isinstance(entry.get("operating_point_directory"), str):
            raise RuntimeError(f"Missing transformer operating-point directory: {run_name}")
        directory = Path(entry["operating_point_directory"]).resolve()
        if not directory.is_dir() or directory in seen_directories:
            raise RuntimeError(f"Unsafe or duplicate transformer operating-point directory: {run_name}")
        seen_directories.add(directory)
        manifest_path = _regular_child(directory, "transformer_operating_point_manifest.json", "operating-point manifest")
        _assert_identity(manifest_path, entry.get("operating_point_manifest"), f"{run_name} operating-point manifest")
        detail = _validate_operating_manifest(manifest_path, run_name)
        source = entry.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("prediction"), dict):
            raise RuntimeError(f"Missing immutable fine-tune source provenance: {run_name}")
        source_prediction = source["prediction"]
        source_from_manifest = detail["manifest"].get("source", {})
        if (
            source_from_manifest.get("sha256") != source_prediction.get("sha256")
            or source_from_manifest.get("bytes") != source_prediction.get("bytes")
        ):
            raise RuntimeError(f"Transformer 2021A source provenance mismatch: {run_name}")
        checkpoint_path, checkpoint_identity = _declared_file_identity(source.get("checkpoint"), f"{run_name} fine-tune checkpoint")
        finetune_path, finetune_identity = _declared_file_identity(source.get("finetune_manifest"), f"{run_name} fine-tune manifest")
        details[run_name] = {"directory": directory, "manifest_path": manifest_path,
                             "checkpoint_path": checkpoint_path, "checkpoint_identity": checkpoint_identity,
                             "finetune_path": finetune_path, "finetune_identity": finetune_identity, **detail}
    return details


def _validate_registered_artifacts(lock: dict[str, Any], directory: Path, label: str) -> None:
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError(f"Missing {label} artifact registry")
    for relative, declared in artifacts.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError(f"Unsafe {label} artifact path")
        file = directory / relative
        if not file.is_file() or file.resolve().is_relative_to(directory.resolve()) is False:
            raise RuntimeError(f"Missing registered {label} artifact: {relative}")
        _assert_identity(file, declared, f"{label}/{relative}")


def validate_baseline_lock(path: Path) -> dict[str, Any]:
    path = path.resolve()
    lock = read_json(path)
    if (
        lock.get("status") != "PASS_BASELINES_FROZEN_PRE_2021B_PRE_2022"
        or lock.get("2021B_outcomes_accessed") is not False
        or lock.get("year_2022_accessed") is not False
    ):
        raise RuntimeError("Baseline lock is not sealed before 2021B/2022")
    directory = path.parent.resolve()
    _validate_registered_artifacts(lock, directory, "baseline")
    threshold_path = _regular_child(directory, "operating_thresholds_2021A.json", "baseline threshold file")
    calibration_path = _regular_child(directory, "calibration_selection_2021A.json", "baseline calibration file")
    selection_path = _regular_child(directory, "model_selection_2021A.json", "baseline selection file")
    for required_path in (threshold_path, calibration_path, selection_path):
        registered = lock["artifacts"].get(str(required_path.relative_to(directory)))
        _assert_identity(required_path, registered, f"baseline/{required_path.name}")
    thresholds, calibration, selection = read_json(threshold_path), read_json(calibration_path), read_json(selection_path)
    if set(thresholds) != set(OUTCOMES) or set(calibration) != set(OUTCOMES):
        raise RuntimeError("Baseline outcome roster is incomplete")
    for outcome in OUTCOMES:
        if set(thresholds[outcome]) != set(BASELINE_ROSTER) or set(calibration[outcome]) != set(BASELINE_ROSTER):
            raise RuntimeError(f"Baseline model roster is incomplete: {outcome}")
        if not isinstance(selection.get(outcome), dict) or set(selection[outcome]) != set(BASELINE_ROSTER):
            raise RuntimeError(f"Baseline model-selection roster is incomplete: {outcome}")
        for model in BASELINE_ROSTER:
            record = thresholds[outcome][model]
            if not isinstance(record, dict):
                raise RuntimeError(f"Missing baseline threshold: {outcome}/{model}")
            if not isinstance(record.get("rule"), str) or "fixed 20% capacity" not in record["rule"]:
                raise RuntimeError(f"Baseline threshold is not the frozen 20% rule: {outcome}/{model}")
            _numeric_threshold(record.get("threshold"), f"{outcome}/{model}")
            if not isinstance(calibration[outcome][model], dict):
                raise RuntimeError(f"Missing baseline calibration provenance: {outcome}/{model}")
            model_path = _regular_child(directory / "models", f"{outcome}__{model}.joblib", "baseline inference model")
            registered = lock["artifacts"].get(str(model_path.relative_to(directory)))
            _assert_identity(model_path, registered, f"baseline/{model_path.relative_to(directory)}")
    return {
        "lock": lock, "directory": directory, "threshold_path": threshold_path,
        "calibration_path": calibration_path, "selection_path": selection_path,
        "thresholds": thresholds,
    }


def validate_stage6_conformal(path: Path, joint_main_manifest: Path) -> dict[str, Any]:
    path = path.resolve()
    registry = read_json(path)
    if (
        registry.get("status") != "PASS_2021B_CONFORMAL_LOCKED_PRE_2022"
        or registry.get("partition") != "2021B only"
        or registry.get("model_or_threshold_selection_on_2021B") is not False
        or registry.get("year_2022_accessed") is not False
        or registry.get("nominal_coverages") != [0.90, 0.80, 0.95]
    ):
        raise RuntimeError("2021B conformal registry is not a sealed calibration-only registry")
    conformal = registry.get("conformal")
    if not isinstance(conformal, dict) or not isinstance(conformal.get("manifest"), str):
        raise RuntimeError("Missing 2021B hierarchical conformal manifest")
    manifest_path = Path(conformal["manifest"]).resolve()
    if not manifest_path.is_file() or conformal.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError("2021B conformal manifest identity mismatch")
    calibrator = read_json(manifest_path)
    if (
        calibrator.get("status") != "PASS_LOCKED_PRE_2022"
        or calibrator.get("calibration_partition") != "2021B only"
        or calibrator.get("model_or_threshold_selection_on_2021B") is not False
        or calibrator.get("year_2022_accessed") is not False
        or calibrator.get("nominal_coverages") != [0.90, 0.80, 0.95]
        or calibrator.get("ancestor_closure") is not True
    ):
        raise RuntimeError("2021B hierarchical conformal manifest gate failed")
    artifact = conformal.get("artifact")
    if not isinstance(artifact, str):
        raise RuntimeError("Missing 2021B conformal set artifact")
    artifact_path = Path(artifact).resolve()
    if not artifact_path.is_file() or conformal.get("artifact_sha256") != sha256(artifact_path):
        raise RuntimeError("2021B conformal set artifact identity mismatch")
    calibrator_artifact = calibrator.get("artifact", {})
    if (calibrator_artifact.get("sha256") != sha256(artifact_path)
            or int(calibrator_artifact.get("bytes", -1)) != artifact_path.stat().st_size):
        raise RuntimeError("2021B conformal artifact is not source-bound to calibrator")
    operating = calibrator.get("operating_point_source", {})
    if operating.get("manifest_sha256") != sha256(joint_main_manifest):
        raise RuntimeError("2021B conformal calibrator is not bound to joint_main 2021A operating point")
    if calibrator.get("any_readmission_probability_column") != "p_any_readmission_calibrated":
        raise RuntimeError("2021B conformal binary source is not the frozen calibrated probability")
    _numeric_threshold(calibrator.get("any_readmission_threshold_from_2021A"), "2021B conformal binary source")
    calibration, outputs = calibrator.get("calibration"), calibrator.get("risk_set_outputs")
    if not isinstance(calibration, dict) or not isinstance(outputs, dict):
        raise RuntimeError("Missing 2021B conformal q/output registry")
    if set(calibration.get("global", {})) != {name for name, _ in NOMINAL_COVERAGES}:
        raise RuntimeError("Incomplete global 2021B conformal q roster")
    for label, _ in NOMINAL_COVERAGES:
        item = calibration["global"][label]
        if not isinstance(item, dict) or not isinstance(item.get("q"), (int, float)) or not 0 <= float(item["q"]) <= 1:
            raise RuntimeError(f"Invalid global 2021B conformal q: {label}")
        if not isinstance(outputs.get("global", {}).get(label), dict):
            raise RuntimeError(f"Missing global conformal output contract: {label}")
    if set(calibration.get("mondrian", {})) != set(MONDRIAN_SOURCE_DIMENSIONS.values()):
        raise RuntimeError("Incomplete Mondrian 2021B conformal q roster")
    for display, source_dimension in MONDRIAN_SOURCE_DIMENSIONS.items():
        levels = calibration["mondrian"].get(source_dimension)
        source_outputs = outputs.get("mondrian", {}).get(source_dimension, {})
        if not isinstance(levels, dict) or set(levels) != {name for name, _ in NOMINAL_COVERAGES}:
            raise RuntimeError(f"Incomplete Mondrian conformal q roster: {display}")
        if not isinstance(source_outputs.get("levels"), dict) or set(source_outputs["levels"]) != {name for name, _ in NOMINAL_COVERAGES}:
            raise RuntimeError(f"Incomplete Mondrian conformal output contract: {display}")
        for label, _ in NOMINAL_COVERAGES:
            if not isinstance(levels[label], dict) or not isinstance(source_outputs["levels"][label], dict):
                raise RuntimeError(f"Invalid Mondrian conformal q/output: {display}/{label}")
    return {"registry": registry, "manifest_path": manifest_path, "calibrator": calibrator, "artifact_path": artifact_path}


def _validate_reference_sidecar(path: Path) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError("Frozen drift reference or SHA256 sidecar is missing")
    if sidecar.read_text(encoding="ascii").strip() != f"{sha256(path)}  {path.name}":
        raise RuntimeError("Frozen drift reference SHA256 sidecar mismatch")


def _load_drift_validator() -> Any:
    path = DRIFT_ANALYZER_PATH.resolve()
    if not path.is_file():
        raise RuntimeError("Missing locked drift analyzer required to validate drift reference")
    import_spec = importlib.util.spec_from_file_location("stage7_drift_reference_validator", path)
    if import_spec is None or import_spec.loader is None:
        raise RuntimeError("Unable to load locked drift analyzer for reference validation")
    module = importlib.util.module_from_spec(import_spec)
    import_spec.loader.exec_module(module)
    return module


def validate_drift_reference_input(reference_path: Path, models: dict[str, dict[str, Any]],
                                   conformal_sets: list[dict[str, Any]]) -> dict[str, Any]:
    """Authenticate a pre-2022 drift reference and its consumer-facing schema."""
    reference_path = reference_path.resolve()
    _validate_reference_sidecar(reference_path)
    reference = read_json(reference_path)
    if reference.get("status") != "FROZEN_2022_DRIFT_REFERENCE" or reference.get("year_2022_accessed") is not False:
        raise RuntimeError("Drift reference is not sealed before 2022 access")
    source_partitions = reference.get("source_partitions")
    required_source_partitions = {
        "development_years": [2018, 2019, 2020], "selection_partition": "2021A",
        "conformal_partition": "2021B only", "year_2022_accessed": False,
        "model_or_threshold_selection_on_2021B": False,
    }
    if source_partitions != required_source_partitions:
        raise RuntimeError("Drift reference source partitions are not restricted to 2018-2020/2021A/2021B")
    # Use the downstream consumer's own validator, not a reimplementation.
    candidate = {"models": models, "conformal_sets": conformal_sets, "drift_reference": reference}
    _load_drift_validator().validate_drift_reference(candidate)
    coverage = reference["coverage"]
    for entry in conformal_sets:
        name = entry["name"]
        rows = [row for row in coverage if row.get("conformal_set") == name]
        if not any(row.get("subgroup_dimension") == "ALL" and row.get("subgroup_value") == "ALL" for row in rows):
            raise RuntimeError(f"Drift reference lacks all-patient coverage reference: {name}")
        for subgroup in SUBGROUPS:
            if not any(row.get("subgroup_dimension") == subgroup and str(row.get("subgroup_value")) != "ALL" for row in rows):
                raise RuntimeError(f"Drift reference lacks prespecified subgroup coverage: {name}/{subgroup}")
    bootstrap = reference["bootstrap"]
    if bootstrap.get("n_replicates") != 1000 or bootstrap.get("cluster_column") != "patient_hash" or bootstrap.get("max_threads") not in range(1, 9):
        raise RuntimeError("Drift reference bootstrap must be fixed at 1000 patient-clustered replicates with <=8 threads")
    return reference


def _conformal_entries(calibrator: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = calibrator["risk_set_outputs"]
    entries: list[dict[str, Any]] = []
    for label, nominal in NOMINAL_COVERAGES:
        global_output = outputs["global"][label]
        entry: dict[str, Any] = {
            "name": f"global_{label}", "scope": "global", "nominal": nominal,
            "mask_column": global_output["leaf_set_mask"], "size_column": global_output["leaf_set_size"],
            "q_source": {"partition": "2021B only", "scope": "global", "nominal": nominal,
                         "q": float(calibrator["calibration"]["global"][label]["q"])},
        }
        if label == "90":
            entry["abstain_column"] = "abstain_90"
        entries.append(entry)
        for display, source_dimension in MONDRIAN_SOURCE_DIMENSIONS.items():
            output = outputs["mondrian"][source_dimension]["levels"][label]
            entries.append({
                "name": f"mondrian_{display}_{label}", "scope": "mondrian",
                "mondrian_dimension": display, "nominal": nominal,
                "mask_column": output["leaf_set_mask"], "size_column": output["leaf_set_size"],
                "q_source": {"partition": "2021B only", "scope": "mondrian", "source_dimension": source_dimension,
                             "nominal": nominal, "rule": output.get("q_source")},
            })
    return entries


def _transformer_models(details: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for run_name in TRANSFORMER_ROSTER:
        detail = details[run_name]
        thresholds = detail["thresholds"]
        models[run_name] = {
            "family": "claims_transformer",
            "probabilities": {outcome: f"p_cal__{outcome}__{run_name}" for outcome in OUTCOMES},
            "thresholds": {outcome: _numeric_threshold(thresholds[outcome]["threshold"], f"{run_name}/{outcome}")
                           for outcome in OUTCOMES},
            "2021A_operating_point_source": {
                "manifest": identity(detail["manifest_path"]),
                "thresholds": identity(detail["files"]["transformer_operating_thresholds_2021A.json"]),
                "calibrator": identity(detail["files"]["transformer_binary_calibrators.joblib"]),
                "calibration_selection": identity(detail["files"]["transformer_calibration_selection_2021A.json"]),
                "rule": "2021A-only calibration selection and fixed 20% capacity threshold transported unchanged",
            },
            "inference_artifacts": {"checkpoint": detail["checkpoint_identity"], "finetune_manifest": detail["finetune_identity"]},
        }
    return models


def _baseline_models(detail: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for model in BASELINE_ROSTER:
        models[model] = {
            "family": "baseline",
            "probabilities": {outcome: f"p_cal__{outcome}__{model}" for outcome in OUTCOMES},
            "thresholds": {outcome: _numeric_threshold(detail["thresholds"][outcome][model]["threshold"], f"{model}/{outcome}")
                           for outcome in OUTCOMES},
            "2021A_operating_point_source": {
                "baseline_lock": identity(detail["directory"] / "baseline_lock.json"),
                "thresholds": identity(detail["threshold_path"]),
                "calibration_selection": identity(detail["calibration_path"]),
                "model_selection": identity(detail["selection_path"]),
                "rule": "2021A-only calibration selection and fixed 20% capacity threshold transported unchanged",
            },
            "inference_artifacts": {
                outcome: identity(detail["directory"] / "models" / f"{outcome}__{model}.joblib") for outcome in OUTCOMES
            },
        }
    return models


def _assert_unique_prediction_columns(models: dict[str, dict[str, Any]]) -> None:
    columns = [model["probabilities"][outcome] for model in models.values() for outcome in OUTCOMES]
    if len(columns) != len(set(columns)):
        raise RuntimeError("Frozen standardized prediction schema has duplicate probability columns")


def _protocol() -> dict[str, Any]:
    return {
        "weighted_and_unweighted_metrics": ["unweighted", "DISCWT"],
        "binary_metrics": ["AUROC", "AUPRC", "Brier", "log_loss", "calibration_intercept", "calibration_slope", "fixed_20pct_operating_point"],
        "paired_bootstrap": {"primary_comparison": "joint_main_minus_lightgbm", "cluster_units": ["patient_hash", "hospital_hash"], "replicates_each": 1000, "ci": "95% percentile", "paired": True},
        "decision_curve": {"threshold_probability_grid": [round(index / 100, 2) for index in range(1, 51)], "weightings": ["unweighted", "DISCWT"]},
        "conformal": {"coverage": [0.80, 0.90, 0.95], "scope": ["global", "Mondrian"], "abstention": "non-singleton prediction set or frozen 90% binary/hierarchy conflict", "interval": "unweighted Wilson 95%"},
        "subgroup_event_tiers": {"confirmatory": ">=100 events", "exploratory": "50-99 events", "descriptive": "<50 events"},
        "drift": {
            "covariate": "2018-2020 development versus 2022 test standardized covariate summaries without refitting",
            "label": "2021A/2021B reference versus 2022 outcome prevalence and cause-leaf distribution",
            "calibration": "frozen-model 2022 calibration intercept, slope, Brier and log loss; no recalibration",
            "coverage": "2021B-frozen global/Mondrian 80/90/95% conformal coverage, set size and abstention on 2022",
        },
        "prohibitions": ["no model selection on 2021B", "no threshold selection on 2021B", "no 2022 access before unlock", "no 2022 recalibration or tuning"],
    }


def _atomic_publish(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = path.resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise RuntimeError("Frozen evaluation specification already exists and is immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(stable_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError("Frozen evaluation specification appeared concurrently") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    digest = sha256(path)
    sidecar_text = f"{digest}  {path.name}\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{sidecar.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(sidecar_text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, sidecar)
        except FileExistsError as exc:
            raise RuntimeError("Frozen evaluation specification SHA256 sidecar appeared concurrently") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"path": str(path), "sha256": digest, "sidecar": str(sidecar)}


def validate_frozen_evaluation_spec(path: Path, evaluator: Path, drift_reference_path: Path) -> dict[str, Any]:
    """Verify a completed spec, its immutable sidecar, and evaluator identity."""
    path, evaluator = path.resolve(), evaluator.resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError("Frozen evaluation specification or SHA256 sidecar is missing")
    if sidecar.read_text(encoding="ascii").strip() != f"{sha256(path)}  {path.name}":
        raise RuntimeError("Frozen evaluation specification SHA256 sidecar mismatch")
    spec = read_json(path)
    if (spec.get("status") != "FROZEN_2022_EVALUATION_SPEC" or spec.get("sealed_test_year") != 2022
            or spec.get("2021B_model_or_threshold_selection") is not False or spec.get("year_2022_accessed_before_unlock") is not False):
        raise RuntimeError("Frozen evaluation specification seal/access gate failed")
    if spec.get("transformer_roster") != list(TRANSFORMER_ROSTER) or spec.get("baseline_roster") != list(BASELINE_ROSTER):
        raise RuntimeError("Frozen evaluation specification model roster changed")
    models = spec.get("models")
    expected_models = set(TRANSFORMER_ROSTER) | set(BASELINE_ROSTER)
    if not isinstance(models, dict) or set(models) != expected_models:
        raise RuntimeError("Frozen evaluation specification model roster is incomplete or mixed")
    outcomes = spec.get("co_primary_outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != set(OUTCOMES):
        raise RuntimeError("Frozen evaluation specification outcome roster is incomplete")
    for model_name, model in models.items():
        if not isinstance(model, dict) or set(model.get("probabilities", {})) != set(OUTCOMES) or set(model.get("thresholds", {})) != set(OUTCOMES):
            raise RuntimeError("Frozen evaluation specification model endpoint roster is incomplete")
        inference = model.get("inference_artifacts")
        if model_name in TRANSFORMER_ROSTER:
            if not isinstance(inference, dict) or set(inference) != {"checkpoint", "finetune_manifest"}:
                raise RuntimeError(f"Frozen Transformer inference artifacts are incomplete: {model_name}")
            for label in ("checkpoint", "finetune_manifest"):
                _declared_file_identity(inference[label], f"{model_name} {label}")
        elif not isinstance(inference, dict) or set(inference) != set(OUTCOMES):
            raise RuntimeError(f"Frozen baseline inference artifacts are incomplete: {model_name}")
        else:
            for outcome in OUTCOMES:
                _declared_file_identity(inference[outcome], f"{model_name}/{outcome} baseline model")
    _assert_unique_prediction_columns(models)
    protocol = spec.get("evaluation_protocol")
    if not isinstance(protocol, dict) or protocol.get("paired_bootstrap", {}).get("replicates_each") != 1000:
        raise RuntimeError("Frozen evaluation specification bootstrap contract changed")
    if protocol.get("decision_curve", {}).get("threshold_probability_grid") != [round(index / 100, 2) for index in range(1, 51)]:
        raise RuntimeError("Frozen evaluation specification decision-curve grid changed")
    conformal = spec.get("conformal_sets")
    if not isinstance(conformal, list) or len(conformal) != 15:
        raise RuntimeError("Frozen evaluation specification conformal roster is incomplete")
    observed = {(entry.get("scope"), entry.get("nominal"), entry.get("mondrian_dimension")) for entry in conformal if isinstance(entry, dict)}
    for _, nominal in NOMINAL_COVERAGES:
        if ("global", nominal, None) not in observed:
            raise RuntimeError("Frozen evaluation specification lacks global conformal coverage")
        for subgroup in SUBGROUPS:
            if ("mondrian", nominal, subgroup) not in observed:
                raise RuntimeError("Frozen evaluation specification lacks Mondrian conformal coverage")
    declared_evaluator = spec.get("evaluator")
    _assert_identity(evaluator, declared_evaluator, "evaluation code")
    provenance = spec.get("drift_reference_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {"reference", "sidecar", "consumer"}:
        raise RuntimeError("Frozen evaluation specification drift-reference provenance is incomplete")
    drift_reference_path = drift_reference_path.resolve()
    _assert_identity(drift_reference_path, provenance["reference"], "drift reference")
    sidecar = drift_reference_path.with_suffix(drift_reference_path.suffix + ".sha256")
    _assert_identity(sidecar, provenance["sidecar"], "drift reference sidecar")
    if provenance["consumer"] != identity(DRIFT_ANALYZER_PATH.resolve()):
        raise RuntimeError("Frozen evaluation specification drift consumer code identity changed")
    reference = validate_drift_reference_input(drift_reference_path, models, conformal)
    if spec.get("drift_reference") != reference:
        raise RuntimeError("Frozen evaluation specification drift reference differs from its hash-bound source")
    return spec


def build_frozen_evaluation_spec(all_operating_points: Path, baseline_lock: Path, stage6_registry: Path,
                                 evaluator: Path, drift_reference_path: Path, output: Path) -> dict[str, Any]:
    """Create the once-only 2022 evaluation specification without opening patient data."""
    details = validate_all_operating_points(all_operating_points)
    baseline = validate_baseline_lock(baseline_lock)
    evaluator_identity = identity(evaluator)
    stage6 = validate_stage6_conformal(stage6_registry, details["joint_main"]["manifest_path"])
    models = {**_transformer_models(details), **_baseline_models(baseline)}
    if tuple(models) != TRANSFORMER_ROSTER + BASELINE_ROSTER:
        raise RuntimeError("Internal model roster construction changed")
    _assert_unique_prediction_columns(models)
    conformal_sets = _conformal_entries(stage6["calibrator"])
    drift_reference = validate_drift_reference_input(drift_reference_path, models, conformal_sets)
    drift_reference_path = drift_reference_path.resolve()
    spec = {
        "status": "FROZEN_2022_EVALUATION_SPEC",
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "sealed_test_year": 2022,
        "2021B_model_or_threshold_selection": False,
        "year_2022_accessed_before_unlock": False,
        "transformer_roster": list(TRANSFORMER_ROSTER),
        "baseline_roster": list(BASELINE_ROSTER),
        "co_primary_outcomes": {"any_readmission": "outcome_any_readmission", "ap_specific_readmission": "outcome_ap_specific_readmission"},
        "prediction_table_contract": {
            "one_row_per": "encounter_hash",
            "required_common_columns": ["encounter_hash", "patient_hash", "hospital_hash", "DISCWT", "analysis_year", "analysis_partition", "primary_leaf", *SUBGROUPS.values()],
            "required_analysis_year": 2022,
            "required_analysis_partition": "test",
            "no_duplicate_probability_columns": True,
        },
        "models": models,
        "primary_comparison": {"transformer": "joint_main", "lightgbm": "lightgbm"},
        "conformal_sets": conformal_sets,
        "subgroups": SUBGROUPS,
        "conformal_2021B_source": {"stage6_registry": identity(stage6_registry), "calibrator": identity(stage6["manifest_path"]), "sets_artifact": identity(stage6["artifact_path"]), "partition": "2021B only", "selection_performed": False},
        "evaluation_protocol": _protocol(),
        "evaluator": evaluator_identity,
        "drift_reference": drift_reference,
        "drift_reference_provenance": {"reference": identity(drift_reference_path),
                                        "sidecar": identity(drift_reference_path.with_suffix(drift_reference_path.suffix + ".sha256")),
                                        "consumer": identity(DRIFT_ANALYZER_PATH.resolve())},
    }
    published = _atomic_publish(output, spec)
    # Re-open after publication to bind the returned result to the actual immutable bytes.
    validate_frozen_evaluation_spec(output, evaluator, drift_reference_path)
    return {**spec, "spec_sha256": published["sha256"], "sidecar": published["sidecar"], "output": published["path"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-operating-points", type=Path, required=True)
    parser.add_argument("--baseline-lock", type=Path, required=True)
    parser.add_argument("--stage6-registry", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--drift-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(stable_json(build_frozen_evaluation_spec(args.all_operating_points, args.baseline_lock, args.stage6_registry,
                                                   args.evaluator, args.drift_reference, args.output)))


if __name__ == "__main__":
    main()
