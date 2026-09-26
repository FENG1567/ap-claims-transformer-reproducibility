#!/usr/bin/env python3
"""Produce the sole standardized prediction table for the locked 2022 test.

This is intentionally a *producer*, not an analysis script.  It validates the
pre-2022 unlock and every frozen inference artifact before it opens the 2022
derivative.  It then applies, without fitting, the 2021A binary calibrators
and the 2021B split-conformal q values.  The only published object is an
atomic wide parquet suitable for :mod:`evaluate_locked_2022` and the drift
report; a failed or repeated run is rejected.

The production path accepts real Torch/joblib artifacts only.  ``predictors``
in the Python API is deliberately an in-memory test seam used by the synthetic
unit tests; it is not exposed by the CLI and still passes all artifact gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


TRANSFORMERS = (
    "joint_main", "mlm_only", "scratch", "ap_only",
    "no_hierarchy_parameter_matched", "no_prday", "no_prior",
    "no_hospital_socioeconomic", "no_year", "common_variable_only",
)
BASELINES = ("structured_logistic", "elastic_net", "lightgbm")
OUTCOMES = ("any_readmission", "ap_specific_readmission")
LEAVES = ("none", "ap", "biliary", "sepsis_or_organ", "other")
LEAF_NAMES = dict(enumerate(LEAVES))
MONDRIAN = {"sex": "sex", "age": "age_group", "PAY1": "payer", "ZIPINC_QRTL": "zip_income_quartile"}
OUTPUT_OUTCOMES = {
    "any_readmission": "outcome_any_readmission",
    "ap_specific_readmission": "outcome_ap_specific_readmission",
    "biliary_event": "outcome_biliary_event",
    "sepsis_or_organ_complication": "outcome_sepsis_or_organ_complication",
    "high_cost": "outcome_high_cost",
    "prolonged_los": "outcome_prolonged_los",
    "in_hospital_death": "outcome_in_hospital_death",
}
SMALL_CELL_THRESHOLD = 10
DERIVATIVE_OUTCOME_SOURCES = {
    "any_readmission": ("any_unplanned_readmission_30d", "outcome_any_readmission"),
    "ap_specific_readmission": ("ap_specific_readmission_30d", "outcome_ap_specific_readmission"),
    "biliary_event": ("biliary_readmission_30d", "biliary_event_label", "outcome_biliary_event"),
    "sepsis_or_organ_complication": ("sepsis_or_organ_readmission_30d", "sepsis_or_organ_complication_label", "outcome_sepsis_or_organ_complication"),
    "high_cost": ("high_cost_label", "outcome_high_cost"),
    "prolonged_los": ("prolonged_los_label", "outcome_prolonged_los"),
    "in_hospital_death": ("in_hospital_death_label", "outcome_in_hospital_death"),
}


def sha256(path: Path) -> str:
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


def _assert_identity(path: Path, declared: object, label: str) -> Path:
    path = path.resolve()
    if not isinstance(declared, dict) or not path.is_file():
        raise RuntimeError(f"Missing {label} identity")
    try:
        expected_bytes = int(declared.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {label} byte identity") from exc
    if declared.get("sha256") != sha256(path) or expected_bytes != path.stat().st_size:
        raise RuntimeError(f"{label} path/bytes/SHA256 mismatch: {path}")
    declared_file = declared.get("file", declared.get("path"))
    if declared_file is not None:
        declared_path = Path(str(declared_file))
        if not declared_path.is_absolute():
            declared_path = path.parent / declared_path
        if path != declared_path.resolve():
            raise RuntimeError(f"{label} identity names a different path")
    return path


def _sidecar(path: Path, label: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != f"{sha256(path)}  {path.name}":
        raise RuntimeError(f"{label} SHA256 sidecar mismatch")


def _listed(lock: dict[str, Any], path: Path, label: str) -> None:
    path = path.resolve()
    entry = lock.get("frozen_code", {}).get(str(path))
    if not isinstance(entry, dict):
        raise RuntimeError(f"{label} is not listed in pre-2022 unlock lock")
    _assert_identity(path, entry, label)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen support module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_unlock(unlock_path: Path, script: Path, evaluation_spec: Path) -> dict[str, Any]:
    unlock_path = unlock_path.resolve()
    if not unlock_path.is_file():
        raise RuntimeError("Missing pre-2022 unlock lock")
    _sidecar(unlock_path, "Pre-2022 unlock lock")
    lock = read_json(unlock_path)
    if (lock.get("status") != "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
            or lock.get("sealed_test_year") != 2022 or lock.get("2022_access_before_lock") is not False):
        raise RuntimeError("Pre-2022 unlock lock is not authorized")
    _listed(lock, script, "Prediction producer")
    _listed(lock, evaluation_spec.resolve(), "Frozen evaluation specification")
    return lock


def _validate_spec(spec_path: Path, lock: dict[str, Any]) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    _sidecar(spec_path, "Frozen evaluation specification")
    _listed(lock, spec_path, "Frozen evaluation specification")
    spec = read_json(spec_path)
    expected = set(TRANSFORMERS) | set(BASELINES)
    if (spec.get("status") != "FROZEN_2022_EVALUATION_SPEC" or spec.get("sealed_test_year") != 2022
            or tuple(spec.get("transformer_roster", ())) != TRANSFORMERS
            or tuple(spec.get("baseline_roster", ())) != BASELINES
            or not isinstance(spec.get("models"), dict) or set(spec["models"]) != expected
            or set(spec.get("co_primary_outcomes", {})) != set(OUTCOMES)):
        raise RuntimeError("Frozen evaluation specification does not have the exact 13-model/two-outcome contract")
    columns = []
    for name in (*TRANSFORMERS, *BASELINES):
        model = spec["models"][name]
        if not isinstance(model, dict) or set(model.get("probabilities", {})) != set(OUTCOMES) or set(model.get("thresholds", {})) != set(OUTCOMES):
            raise RuntimeError(f"Incomplete frozen model contract: {name}")
        columns.extend(model["probabilities"].values())
        for outcome in OUTCOMES:
            threshold = model["thresholds"][outcome]
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
                raise RuntimeError(f"Invalid frozen threshold: {name}/{outcome}")
    if len(columns) != 26 or len(set(columns)) != 26:
        raise RuntimeError("Frozen specification must name exactly 26 unique probability columns")
    conformal = spec.get("conformal_sets")
    expected_sets = {(scope, nominal, dimension) for nominal in (0.8, 0.9, 0.95)
                     for scope, dimension in [("global", None), *( ("mondrian", d) for d in MONDRIAN )]}
    actual_sets = {(x.get("scope"), x.get("nominal"), x.get("mondrian_dimension")) for x in conformal or [] if isinstance(x, dict)}
    if not isinstance(conformal, list) or len(conformal) != 15 or actual_sets != expected_sets:
        raise RuntimeError("Frozen specification does not have exactly the 15 global/Mondrian conformal sets")
    return spec


def _validate_derivative(derivative_dir: Path, lock: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Authenticate, but do not open, the 2022 parquet derivative."""
    directory = derivative_dir.resolve()
    manifest_path = directory / "manifest.json"
    artifact_path = directory / "locked_2022_derivative.parquet"
    if not directory.is_dir() or not manifest_path.is_file() or not artifact_path.is_file():
        raise RuntimeError("Locked 2022 derivative directory is incomplete")
    _sidecar(manifest_path, "Locked 2022 derivative manifest")
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS_LOCKED_2022_MINIMUM_DERIVATIVE" or manifest.get("2022_accessed") is not True
            or manifest.get("one_shot") is not True or manifest.get("data_dependent_adaptation") is not False):
        raise RuntimeError("2022 derivative manifest is not a one-shot locked derivative")
    _assert_identity(artifact_path, manifest.get("derivative"), "2022 derivative artifact")
    # The actual lock binding is definitive and does not depend on a path naming convention.
    declared = manifest.get("unlock_lock", {})
    if not isinstance(declared, dict):
        raise RuntimeError("2022 derivative is not bound to the supplied pre-2022 unlock lock")
    try:
        _assert_identity(Path(lock["__path"]), declared, "2022 derivative unlock lock")
    except RuntimeError as exc:
        raise RuntimeError("2022 derivative is not bound to the supplied pre-2022 unlock lock") from exc
    return artifact_path, manifest


def _validate_finetune_manifest(path: Path, checkpoint: Path, run: str) -> dict[str, Any]:
    path = path.resolve()
    checkpoint = checkpoint.resolve()
    manifest = read_json(path)
    if (manifest.get("status") != "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022"
            or manifest.get("prediction_partition") != "2021A" or manifest.get("2021B_accessed") is not False
            or manifest.get("year_2022_accessed") is not False):
        raise RuntimeError(f"Fine-tune manifest is not sealed: {run}")
    check = manifest.get("checkpoint", {})
    if not isinstance(check, dict) or check.get("sha256") != sha256(checkpoint):
        raise RuntimeError(f"Fine-tune manifest/checkpoint mismatch: {run}")
    if check.get("bytes") is not None:
        try:
            if int(check["bytes"]) != checkpoint.stat().st_size:
                raise RuntimeError(f"Fine-tune manifest/checkpoint mismatch: {run}")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Fine-tune manifest/checkpoint mismatch: {run}") from exc
    declared_file = check.get("file", check.get("path"))
    if declared_file is not None:
        declared_path = Path(str(declared_file))
        if not declared_path.is_absolute():
            declared_path = path.parent / declared_path
        if declared_path.resolve() != checkpoint:
            raise RuntimeError(f"Fine-tune manifest/checkpoint mismatch: {run}")
    return manifest


def _validate_baseline_support(lock_path: Path) -> tuple[Path, Path]:
    baseline_lock = read_json(lock_path)
    if (baseline_lock.get("status") != "PASS_BASELINES_FROZEN_PRE_2021B_PRE_2022"
            or baseline_lock.get("2021B_outcomes_accessed") is not False or baseline_lock.get("year_2022_accessed") is not False):
        raise RuntimeError("Baseline lock is not pre-2022 sealed")
    directory = lock_path.parent
    artifacts = baseline_lock.get("artifacts", {})
    preprocessor = directory / "models" / "static_preprocessor.joblib"
    maps = directory / "models" / "development_token_maps.joblib"
    for path in (preprocessor, maps):
        relative = str(path.relative_to(directory))
        _assert_identity(path, artifacts.get(relative), f"Baseline frozen support {relative}")
    return preprocessor, maps


def validate_preflight(unlock_path: Path, evaluation_spec: Path, derivative_dir: Path) -> dict[str, Any]:
    """Run every lock/artifact identity gate without reading the derivative rows."""
    script = Path(__file__).resolve()
    lock = _validate_unlock(unlock_path, script, evaluation_spec)
    lock["__path"] = str(unlock_path.resolve())
    spec = _validate_spec(evaluation_spec, lock)
    derivative_path, derivative_manifest = _validate_derivative(derivative_dir, lock)
    transformer_sources: dict[str, dict[str, Any]] = {}
    for run in TRANSFORMERS:
        model = spec["models"][run]
        inference = model.get("inference_artifacts", {})
        if set(inference) != {"checkpoint", "finetune_manifest"}:
            raise RuntimeError(f"Transformer inference roster is incomplete: {run}")
        checkpoint = _assert_identity(Path(inference["checkpoint"].get("file", "")), inference["checkpoint"], f"{run} checkpoint")
        finetune = _assert_identity(Path(inference["finetune_manifest"].get("file", "")), inference["finetune_manifest"], f"{run} fine-tune manifest")
        operating = model.get("2021A_operating_point_source", {})
        for name in ("manifest", "thresholds", "calibrator", "calibration_selection"):
            record = operating.get(name)
            if not isinstance(record, dict) or not isinstance(record.get("file"), str):
                raise RuntimeError(f"Missing 2021A {name} identity: {run}")
            _assert_identity(Path(record["file"]), record, f"{run} 2021A {name}")
        transformer_sources[run] = {"checkpoint": checkpoint, "finetune": finetune,
                                    "manifest": _validate_finetune_manifest(finetune, checkpoint, run),
                                    "calibrator": Path(operating["calibrator"]["file"]).resolve()}
    baseline_sources: dict[str, dict[str, Any]] = {}
    shared_support: tuple[Path, Path] | None = None
    for name in BASELINES:
        model = spec["models"][name]
        inference = model.get("inference_artifacts", {})
        if set(inference) != set(OUTCOMES):
            raise RuntimeError(f"Baseline inference roster is incomplete: {name}")
        operating = model.get("2021A_operating_point_source", {})
        baseline_record = operating.get("baseline_lock")
        if not isinstance(baseline_record, dict) or not isinstance(baseline_record.get("file"), str):
            raise RuntimeError(f"Missing baseline lock identity: {name}")
        baseline_lock = _assert_identity(Path(baseline_record["file"]), baseline_record, f"{name} baseline lock")
        for label in ("thresholds", "calibration_selection", "model_selection"):
            record = operating.get(label)
            if not isinstance(record, dict) or not isinstance(record.get("file"), str):
                raise RuntimeError(f"Missing 2021A baseline {label} identity: {name}")
            _assert_identity(Path(record["file"]), record, f"{name} 2021A baseline {label}")
        support = _validate_baseline_support(baseline_lock)
        if shared_support is None:
            shared_support = support
        elif support != shared_support:
            raise RuntimeError("Baseline models must share one frozen preprocessing state")
        artifacts: dict[str, Path] = {}
        for outcome in OUTCOMES:
            record = inference[outcome]
            if not isinstance(record, dict) or not isinstance(record.get("file"), str):
                raise RuntimeError(f"Missing baseline model identity: {name}/{outcome}")
            artifacts[outcome] = _assert_identity(Path(record["file"]), record, f"{name}/{outcome} baseline model")
        baseline_sources[name] = {"models": artifacts, "support": support}
    conformal_source = spec.get("conformal_2021B_source", {})
    if not isinstance(conformal_source, dict):
        raise RuntimeError("Missing frozen 2021B conformal provenance")
    for name in ("stage6_registry", "calibrator", "sets_artifact"):
        record = conformal_source.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise RuntimeError(f"Missing 2021B conformal {name} identity")
        _assert_identity(Path(record["file"]), record, f"2021B conformal {name}")
    calibrator = read_json(Path(conformal_source["calibrator"]["file"]))
    if (calibrator.get("status") != "PASS_LOCKED_PRE_2022" or calibrator.get("calibration_partition") != "2021B only"
            or calibrator.get("model_or_threshold_selection_on_2021B") is not False or calibrator.get("year_2022_accessed") is not False):
        raise RuntimeError("2021B conformal calibrator is not sealed")
    joint_manifest = spec["models"]["joint_main"]["2021A_operating_point_source"]["manifest"]
    operating_source = calibrator.get("operating_point_source", {})
    if not isinstance(operating_source, dict) or operating_source.get("manifest_sha256") != joint_manifest.get("sha256"):
        raise RuntimeError("2021B conformal calibrator is not bound to joint_main 2021A operating point")
    return {"lock": lock, "spec": spec, "derivative_path": derivative_path, "derivative_manifest": derivative_manifest,
            "transformers": transformer_sources, "baselines": baseline_sources, "conformal": calibrator}


def _age_group(age: pd.Series) -> pd.Series:
    values = pd.to_numeric(age, errors="coerce")
    if values.isna().any() or (values < 18).any():
        raise RuntimeError("2022 derivative has invalid AGE for frozen age group")
    return pd.cut(values, bins=[17, 44, 64, 74, np.inf], labels=["18-44", "45-64", "65-74", "75+"]).astype(str)


def _validate_derivative_frame(frame: pd.DataFrame) -> None:
    needed = {"encounter_hash", "patient_hash", "hospital_hash", "NRD_STRATUM", "DISCWT", "AGE", "FEMALE", "PAY1", "ZIPINC_QRTL", "analysis_year", "analysis_partition", "readmission_leaf", "dx_tokens", "pr_tokens", "prday", "prior_dx_tokens_180d", "prior_pr_tokens_180d"}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise RuntimeError(f"2022 derivative lacks frozen prediction/evaluation fields: {missing}")
    if frame.empty or frame["encounter_hash"].duplicated().any():
        raise RuntimeError("2022 derivative must have one non-empty row per encounter_hash")
    if not pd.to_numeric(frame["analysis_year"], errors="coerce").eq(2022).all() or not frame["analysis_partition"].astype(str).eq("test").all():
        raise RuntimeError("2022 derivative is not exclusively the test partition")
    for column in ("encounter_hash", "patient_hash", "hospital_hash"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"2022 derivative has missing {column}")
    weights = pd.to_numeric(frame["DISCWT"], errors="coerce")
    if weights.isna().any() or (~np.isfinite(weights)).any() or (weights <= 0).any():
        raise RuntimeError("2022 derivative DISCWT must be finite and positive")


def _import_transformer_components() -> tuple[Any, Any]:
    stage5 = Path(__file__).resolve().parents[1] / "stage5"
    if str(stage5) not in sys.path:
        sys.path.insert(0, str(stage5))
    from claims_transformer import ClaimsTransformer, ClaimsTransformerConfig
    from finetune_ap_transformer_v2 import ExampleDataset, StaticPreprocessor, collate_examples, load_unified_hierarchy, make_examples, predict
    return (ClaimsTransformer, ClaimsTransformerConfig, ExampleDataset, StaticPreprocessor, collate_examples, load_unified_hierarchy, make_examples, predict)


def _transformer_predict(frame: pd.DataFrame, source: dict[str, Any], root: Path, hierarchy_dir: Path,
                         batch_size: int, workers: int, device_name: str) -> tuple[np.ndarray, np.ndarray]:
    import torch
    (ClaimsTransformer, ClaimsTransformerConfig, ExampleDataset, StaticPreprocessor, collate_examples,
     load_unified_hierarchy, make_examples, predict) = _import_transformer_components()
    checkpoint = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not all(k in checkpoint for k in ("model_state", "model_config", "static_preprocessor", "training_contract")):
        raise RuntimeError("Fine-tune checkpoint lacks the frozen inference interface")
    if checkpoint["training_contract"] != source["manifest"].get("training_contract"):
        raise RuntimeError("Fine-tune checkpoint training contract differs from its manifest")
    static = StaticPreprocessor.from_dict(checkpoint["static_preprocessor"])
    needed_static = set(static.numeric_columns) | set(static.categorical_columns)
    missing_static = sorted(needed_static - set(frame.columns))
    if missing_static:
        raise RuntimeError(f"2022 derivative lacks frozen static model inputs: {missing_static}")
    static_values = static.transform(frame)
    bundle = load_unified_hierarchy(root.resolve(), hierarchy_dir.resolve())
    config = ClaimsTransformerConfig(**checkpoint["model_config"])
    if config.static_dim != static.dimension:
        raise RuntimeError("Checkpoint static preprocessor dimension does not match model configuration")
    model = ClaimsTransformer(config, bundle["token_to_category"], bundle["token_to_domain"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device(device_name)
    model = model.to(device)
    max_tokens = int(checkpoint["training_contract"].get("max_tokens", 0))
    if max_tokens < 2:
        raise RuntimeError("Checkpoint has invalid frozen max_tokens")
    # The reserved fifth year embedding is a frozen inference-only OOV bucket.
    # It is deliberately opt-in here, after the pre-2022 lock and only for the
    # sealed 2022 producer; development/2021A callers retain the default false.
    examples = make_examples(frame, bundle["dx_size"], max_tokens, static_values,
                             allow_future_oov_year=True)
    loader = torch.utils.data.DataLoader(ExampleDataset(examples), batch_size=batch_size, shuffle=False,
                                         num_workers=workers, collate_fn=collate_examples)
    prediction = predict(model, loader, device, device.type == "cuda")
    if len(prediction) != len(frame) or not prediction["encounter_hash"].astype(str).eq(frame["encounter_hash"].astype(str)).all():
        raise RuntimeError("Transformer prediction order/key contract failed")
    leaves = prediction[[f"p_leaf_{leaf}" for leaf in LEAVES]].to_numpy(dtype=float)
    return prediction["p_any_readmission"].to_numpy(dtype=float), leaves


def _apply_transformer_calibrator(path: Path, raw_any: np.ndarray, leaf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("version") != 1 or set(bundle.get("endpoints", {})) != set(OUTCOMES):
        raise RuntimeError("Frozen Transformer calibrator bundle has an invalid contract")
    stage5 = Path(__file__).resolve().parents[1] / "stage5" / "freeze_transformer_calibration.py"
    calibrate = _load_module("stage7_frozen_transformer_calibrator", stage5).apply_calibrator
    output: list[np.ndarray] = []
    for outcome, raw in (("any_readmission", raw_any), ("ap_specific_readmission", leaf[:, 1])):
        item = bundle["endpoints"][outcome]
        if not isinstance(item, dict) or not isinstance(item.get("method"), str):
            raise RuntimeError("Frozen Transformer calibration endpoint is malformed")
        output.append(np.asarray(calibrate(item["method"], item.get("model"), raw), dtype=float))
    return output[0], output[1]


def _baseline_predict(frame: pd.DataFrame, source: dict[str, Any], name: str) -> dict[str, np.ndarray]:
    stage4 = Path(__file__).resolve().parents[1] / "stage4" / "train_baselines.py"
    base = _load_module("stage7_frozen_baseline_support", stage4)
    required = set(base.BASE_COLUMNS) | set(base.HISTORY_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"2022 derivative lacks frozen baseline model inputs: {missing}")
    preprocessor_path, maps_path = source["support"]
    preprocessor, maps = joblib.load(preprocessor_path), joblib.load(maps_path)
    engineered = base.engineer_static(frame)
    static = preprocessor.transform(engineered).astype(np.float32)
    full = base.combine_features(static, engineered, maps)
    result: dict[str, np.ndarray] = {}
    for outcome, path in source["models"].items():
        item = joblib.load(path)
        if not isinstance(item, dict) or item.get("target") != outcome or not all(k in item for k in ("model", "calibration_method", "calibrator")):
            raise RuntimeError(f"Frozen baseline artifact interface is invalid: {name}/{outcome}")
        raw = base.raw_prediction(name, item["model"], static, full)
        values = np.asarray(base.apply_calibrator(item["calibration_method"], item["calibrator"], raw), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise RuntimeError(f"Frozen baseline calibration produced invalid probability: {name}/{outcome}")
        result[outcome] = values
    return result


def _risk_set(probabilities: np.ndarray, q: float) -> np.ndarray:
    sets = probabilities >= (1.0 - float(q))
    empty = ~sets.any(axis=1)
    if empty.any():
        sets[np.flatnonzero(empty), probabilities[empty].argmax(axis=1)] = True
    return sets


def _mask(sets: np.ndarray) -> np.ndarray:
    return (sets.astype(np.uint8) * (1 << np.arange(len(LEAVES), dtype=np.uint8))).sum(axis=1).astype(np.uint8)


def _groups(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {"sex": frame["FEMALE"].astype("Int64").astype(str), "age_group": frame["age_group"].astype(str),
            "payer": frame["PAY1"].astype("Int64").astype(str), "zip_income_quartile": frame["ZIPINC_QRTL"].astype("Int64").astype(str)}


def _materialize_conformal(frame: pd.DataFrame, leaves: np.ndarray, calibrated_any: np.ndarray,
                           threshold: float, calibrator: dict[str, Any], spec: dict[str, Any]) -> None:
    if leaves.shape != (len(frame), len(LEAVES)) or not np.isfinite(leaves).all() or (leaves < 0).any():
        raise RuntimeError("joint_main leaf probability interface is invalid")
    totals = leaves.sum(axis=1)
    if (totals <= 0).any():
        raise RuntimeError("joint_main leaf probabilities have nonpositive mass")
    leaves = leaves / totals[:, None]
    calibration = calibrator.get("calibration", {})
    global_q = calibration.get("global", {})
    mondrian_q = calibration.get("mondrian", {})
    groups = _groups(frame)
    for entry in spec["conformal_sets"]:
        label = str(int(round(float(entry["nominal"]) * 100)))
        if entry["scope"] == "global":
            record = global_q.get(label, {})
            q = record.get("q")
            if not isinstance(q, (int, float)) or not 0 <= q <= 1:
                raise RuntimeError(f"Missing frozen global conformal q: {label}")
            sets = _risk_set(leaves, float(q))
        else:
            source = MONDRIAN[entry["mondrian_dimension"]]
            levels = mondrian_q.get(source, {}).get(label)
            if not isinstance(levels, dict):
                raise RuntimeError(f"Missing frozen Mondrian q registry: {source}/{label}")
            fallback = global_q.get(label, {}).get("q")
            if not isinstance(fallback, (int, float)) or not 0 <= fallback <= 1:
                raise RuntimeError(f"Missing frozen global fallback q: {label}")
            sets = np.empty_like(leaves, dtype=bool)
            values = groups[source]
            for value in pd.unique(values):
                chosen = levels.get(str(value), {})
                q = chosen.get("q", fallback)
                if not isinstance(q, (int, float)) or not 0 <= q <= 1:
                    raise RuntimeError(f"Invalid frozen Mondrian q: {source}/{value}/{label}")
                mask = values.eq(value).to_numpy()
                sets[mask] = _risk_set(leaves[mask], float(q))
        mask_values = _mask(sets)
        frame[entry["mask_column"]] = mask_values
        frame[entry["size_column"]] = sets.sum(axis=1).astype(np.int8)
        if entry.get("abstain_column"):
            singleton = sets.sum(axis=1) == 1
            predicted = sets.argmax(axis=1)
            conflict = ((predicted == 0) & (calibrated_any >= threshold)) | ((predicted > 0) & (calibrated_any < threshold))
            frame[entry["abstain_column"]] = (~(singleton & ~conflict)).astype(bool)
            frame["binary_hierarchy_conflict_90"] = conflict.astype(bool)
    if "abstain_90" not in frame or "binary_hierarchy_conflict_90" not in frame:
        raise RuntimeError("Frozen 90% conformal abstention/conflict contract was not materialized")


def _output_columns(spec: dict[str, Any]) -> list[str]:
    columns = ["encounter_hash", "patient_hash", "hospital_hash", "NRD_STRATUM", "DISCWT", "analysis_year", "analysis_partition", "AGE", "FEMALE", "age_group", "PAY1", "ZIPINC_QRTL", "primary_leaf", *OUTPUT_OUTCOMES.values()]
    for name in (*TRANSFORMERS, *BASELINES):
        columns.extend(spec["models"][name]["probabilities"][outcome] for outcome in OUTCOMES)
    for entry in spec["conformal_sets"]:
        columns.extend((entry["mask_column"], entry["size_column"]))
    columns.extend(("abstain_90", "binary_hierarchy_conflict_90"))
    if len(columns) != len(set(columns)):
        raise RuntimeError("Frozen standardized output schema has duplicate fields")
    return columns


def _prepare_output(output: Path) -> Path:
    output = output.resolve()
    partials = list(output.parent.glob(output.name + ".partial*")) if output.parent.exists() else []
    if output.exists() or partials:
        raise RuntimeError("2022 standardized prediction output or partial output already exists and is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))


def _suppress_small_count(value: int) -> tuple[int | None, bool]:
    """Apply the project-wide HCUP disclosure rule to a manifest count."""
    return (None, True) if value <= SMALL_CELL_THRESHOLD else (int(value), False)


Predictor = Callable[[str, pd.DataFrame], Mapping[str, np.ndarray] | tuple[np.ndarray, np.ndarray]]


def produce_locked_2022(derivative_dir: Path, unlock_lock: Path, evaluation_spec: Path, output_dir: Path,
                        *, root: Path | None = None, hierarchy_dir: Path | None = None, batch_size: int = 128,
                        workers: int = 4, threads: int = 8, device: str | None = None,
                        predictors: Predictor | None = None) -> dict[str, Any]:
    """Create the irreversible all-model prediction wide table.

    ``predictors`` is test-only: it cannot be passed from the command line and
    never bypasses identity validation.  The real path remains strictly serial
    (one Torch model at a time) and bounded to eight CPU threads.
    """
    if not (1 <= threads <= 8 and 0 <= workers <= 8 and batch_size > 0):
        raise RuntimeError("Resource contract requires 1-8 threads, 0-8 workers, and positive batch size")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(threads)
    # Absolutely no derivative parquet read precedes this complete validation.
    gate = validate_preflight(unlock_lock, evaluation_spec, derivative_dir)
    temporary = _prepare_output(output_dir)
    try:
        pa.set_cpu_count(threads)
        frame = pq.read_table(gate["derivative_path"]).to_pandas()
        _validate_derivative_frame(frame)
        frame = frame.copy()
        frame["age_group"] = _age_group(frame["AGE"])
        leaf = pd.to_numeric(frame["readmission_leaf"], errors="coerce")
        if leaf.isna().any() or not leaf.isin(range(len(LEAVES))).all():
            raise RuntimeError("2022 derivative primary leaf is invalid")
        frame["primary_leaf"] = leaf.astype(int).map(LEAF_NAMES)
        high_cost_valid_n = 0
        high_cost_missing_n = 0
        for source, target in OUTPUT_OUTCOMES.items():
            found = next((column for column in DERIVATIVE_OUTCOME_SOURCES[source] if column in frame), None)
            if found is None:
                raise RuntimeError(f"2022 derivative lacks frozen outcome source for {source}")
            values = frame[found]
            if source == "high_cost":
                observed = values.notna()
                if not values.loc[observed].isin((0, 1, False, True)).all():
                    raise RuntimeError("2022 derivative has invalid binary outcome source for high_cost")
                # Cost validity is an upstream measurement property, not an outcome.
                # Preserve it exactly: converting unavailable costs to zero would
                # spuriously create low-cost labels and contaminate auxiliary
                # descriptive reporting.  It never affects primary prediction,
                # 2021A thresholds, or the 2021B conformal sets.
                frame[target] = values.astype("Int8")
                high_cost_valid_n = int(observed.sum())
                high_cost_missing_n = int((~observed).sum())
                continue
            if values.isna().any() or not values.isin((0, 1, False, True)).all():
                raise RuntimeError(f"2022 derivative has invalid binary outcome source for {source}")
            frame[target] = values.astype(np.int8)
        progress: dict[str, dict[str, str]] = {}
        joint_leaf: np.ndarray | None = None
        for name in TRANSFORMERS:
            if predictors is not None:
                value = predictors(name, frame)
                if not isinstance(value, tuple) or len(value) != 2:
                    raise RuntimeError("Synthetic Transformer predictor must return (any_probability, leaf_probabilities)")
                raw_any, leaves = value
            else:
                if root is None or hierarchy_dir is None:
                    raise RuntimeError("Production Transformer inference requires --root and --hierarchy-dir")
                if device is None:
                    import torch
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                raw_any, leaves = _transformer_predict(frame, gate["transformers"][name], root, hierarchy_dir, batch_size, workers, device)
            raw_any = np.asarray(raw_any, dtype=float); leaves = np.asarray(leaves, dtype=float)
            if raw_any.shape != (len(frame),) or leaves.shape != (len(frame), len(LEAVES)):
                raise RuntimeError(f"Transformer prediction shape mismatch: {name}")
            if not np.isfinite(raw_any).all() or (raw_any < 0).any() or (raw_any > 1).any():
                raise RuntimeError(f"Transformer raw probability invalid: {name}")
            if predictors is None:
                any_cal, ap_cal = _apply_transformer_calibrator(gate["transformers"][name]["calibrator"], raw_any, leaves)
            else:
                # Synthetic fixtures use identity calibration but production artifacts were still authenticated above.
                any_cal, ap_cal = raw_any, leaves[:, 1]
            model = gate["spec"]["models"][name]
            frame[model["probabilities"]["any_readmission"]] = any_cal
            frame[model["probabilities"]["ap_specific_readmission"]] = ap_cal
            if name == "joint_main":
                joint_leaf = leaves
            progress[name] = {"checkpoint_sha256": sha256(gate["transformers"][name]["checkpoint"])}
            (temporary / "verified_model_progress.json").write_text(stable_json(progress), encoding="utf-8")
        for name in BASELINES:
            if predictors is not None:
                result = predictors(name, frame)
                if not isinstance(result, Mapping) or set(result) != set(OUTCOMES):
                    raise RuntimeError("Synthetic baseline predictor must return both endpoint probabilities")
                probabilities = {k: np.asarray(v, dtype=float) for k, v in result.items()}
            else:
                probabilities = _baseline_predict(frame, gate["baselines"][name], name)
            for outcome in OUTCOMES:
                values = probabilities[outcome]
                if values.shape != (len(frame),) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                    raise RuntimeError(f"Baseline probability invalid: {name}/{outcome}")
                frame[gate["spec"]["models"][name]["probabilities"][outcome]] = values
            progress[name] = {"artifact_sha256": sha256(gate["baselines"][name]["models"]["any_readmission"])}
            (temporary / "verified_model_progress.json").write_text(stable_json(progress), encoding="utf-8")
        if joint_leaf is None:
            raise RuntimeError("joint_main predictions are unavailable for frozen conformal materialization")
        _materialize_conformal(frame, joint_leaf,
                               frame[gate["spec"]["models"]["joint_main"]["probabilities"]["any_readmission"]].to_numpy(dtype=float),
                               float(gate["spec"]["models"]["joint_main"]["thresholds"]["any_readmission"]),
                               gate["conformal"], gate["spec"])
        columns = _output_columns(gate["spec"])
        wide = frame.loc[:, columns].copy()
        path = temporary / "predictions_2022.parquet"
        pq.write_table(pa.Table.from_pandas(wide, preserve_index=False), path, compression="zstd", compression_level=6,
                       row_group_size=131072, use_dictionary=True, write_statistics=True)
        artifact = identity(path)
        high_cost_valid_report, high_cost_valid_suppressed = _suppress_small_count(high_cost_valid_n)
        high_cost_missing_report, high_cost_missing_suppressed = _suppress_small_count(high_cost_missing_n)
        manifest = {
            "status": "PASS_LOCKED_2022_STANDARDIZED_PREDICTIONS", "schema_version": "stage7_locked_2022_predictions_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "one_shot": True, "no_2022_recalibration_or_selection": True,
            "rows": int(len(wide)), "models": list((*TRANSFORMERS, *BASELINES)), "co_primary_outcomes": list(OUTCOMES),
            "unlock_lock": identity(Path(gate["lock"]["__path"])), "evaluation_spec": identity(evaluation_spec),
            "derivative_manifest": identity(derivative_dir.resolve() / "manifest.json"), "derivative": identity(gate["derivative_path"]),
            "artifact": {"file": path.name, "bytes": artifact["bytes"], "sha256": artifact["sha256"]},
            "high_cost_label_completeness": {
                "valid_n": high_cost_valid_report, "valid_n_suppressed": high_cost_valid_suppressed,
                "missing_n": high_cost_missing_report, "missing_n_suppressed": high_cost_missing_suppressed,
                "small_cell_threshold": "n<=10", "missing_is_not_reclassified_as_low_cost": True,
            },
            "conformal": {"source": "2021B only", "sets": 15, "joint_main_only": True,
                          "no_q_recalibration_on_2022": True},
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(stable_json(manifest), encoding="utf-8")
        digest = sha256(manifest_path)
        (temporary / "manifest.json.sha256").write_text(f"{digest}  manifest.json\n", encoding="ascii")
        (temporary / "verified_model_progress.json").unlink(missing_ok=True)
        os.replace(temporary, output_dir.resolve())
        return {**manifest, "manifest_sha256": digest, "output": str(output_dir.resolve())}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derivative-dir", type=Path, required=True)
    parser.add_argument("--unlock-lock", type=Path, required=True)
    parser.add_argument("--evaluation-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hierarchy-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    print(stable_json(produce_locked_2022(args.derivative_dir, args.unlock_lock, args.evaluation_spec, args.output_dir,
                                          root=args.root, hierarchy_dir=args.hierarchy_dir, batch_size=args.batch_size,
                                          workers=args.workers, threads=args.threads, device=args.device)))


if __name__ == "__main__":
    main()
