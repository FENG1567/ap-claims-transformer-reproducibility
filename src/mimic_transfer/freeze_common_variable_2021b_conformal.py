#!/usr/bin/env python3
"""Freeze the pre-specified common-variable Transformer before any MIMIC use.

This controller has a deliberately narrow responsibility.  It binds the
already-run ``common_variable_only`` 2021A ablation to a dedicated model lock,
runs the existing 2021B-only Stage 6 controller, and publishes a *projection*
of its conformal output for MIMIC.  The projection deliberately contains only
global, sex, and age-group conformal information.  It cannot select a model,
threshold, feature, or calibration setting and it has no MIMIC input argument.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


ALL_OPERATING_STATUS = "PASS_ALL_TRANSFORMER_CALIBRATIONS_AND_THRESHOLDS_LOCKED_ON_2021A"
MODEL_LOCK_STATUS = "LOCKED_ON_2021A_PRE_2021B_PRE_2022"
FINAL_STATUS = "PASS_COMMON_VARIABLE_ONLY_2021B_CONFORMAL_PRE_MIMIC"
TRANSFER_BINDING_STATUS = "PASS_MIMIC_CONFORMAL_PROJECTION_PRE_DATA"
COMMON_MODEL_ID = "common_variable_only"
ALLOWED_TRANSFER_DIMENSIONS = ("sex", "age_group")
PROHIBITED_TRANSFER_DIMENSIONS = frozenset({"payer", "zip_income_quartile"})
CONFORMAL_COVERAGES = ("80", "90", "95")
NOMINAL_COVERAGES = (0.8, 0.9, 0.95)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def artifact_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing required artifact: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def assert_identity(path: Path, declared: object, label: str) -> dict[str, Any]:
    if not isinstance(declared, Mapping):
        raise RuntimeError(f"Missing {label} identity")
    actual = artifact_identity(path)
    if (declared.get("sha256") != actual["sha256"]
            or int(declared.get("bytes", -1)) != actual["bytes"]):
        raise RuntimeError(f"{label} identity mismatch")
    declared_path = declared.get("path", declared.get("file"))
    if isinstance(declared_path, str) and Path(declared_path).is_absolute():
        if Path(declared_path).resolve() != path.resolve():
            raise RuntimeError(f"{label} path mismatch")
    return actual


def assert_declared_identity_without_open(path: Path, declared: object, label: str) -> dict[str, Any]:
    """Bind a registered row artifact without opening it during no-data preflight."""
    if not isinstance(declared, Mapping):
        raise RuntimeError(f"Missing {label} identity")
    path = path.resolve()
    digest = declared.get("sha256")
    if not path.is_file() or not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError(f"Malformed {label} identity")
    if int(declared.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"{label} size mismatch")
    declared_path = declared.get("path", declared.get("file"))
    if isinstance(declared_path, str) and Path(declared_path).is_absolute():
        if Path(declared_path).resolve() != path:
            raise RuntimeError(f"{label} path mismatch")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def identity_path(declared: Mapping[str, Any], label: str) -> Path:
    raw = declared.get("path", declared.get("file"))
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"Missing {label} path")
    return Path(raw).resolve()


def atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    """Create once, atomically; any old or partial output is a hard failure."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"Immutable output already exists: {path}")
    descriptor, raw_temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(stable_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"Concurrent immutable publication: {path}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _require_false(value: Mapping[str, Any], key: str, label: str) -> None:
    if value.get(key) is not False:
        raise RuntimeError(f"{label}.{key} must be exactly false")


def _run_source(run: Mapping[str, Any]) -> tuple[Path, Path, Path, dict[str, Any]]:
    source = run.get("source")
    if not isinstance(source, Mapping):
        raise RuntimeError("common-variable registry run lacks source identities")
    checkpoint_declared = source.get("checkpoint")
    finetune_declared = source.get("finetune_manifest")
    prediction_declared = source.get("prediction")
    if not all(isinstance(item, Mapping) and isinstance(item.get("path", item.get("file")), str)
               for item in (checkpoint_declared, finetune_declared, prediction_declared)):
        raise RuntimeError("common-variable source artifacts require path identities")
    checkpoint = identity_path(checkpoint_declared, "checkpoint")
    finetune = identity_path(finetune_declared, "finetune manifest")
    prediction = identity_path(prediction_declared, "2021A prediction")
    assert_identity(checkpoint, checkpoint_declared, "checkpoint")
    assert_identity(finetune, finetune_declared, "finetune manifest")
    assert_declared_identity_without_open(prediction, prediction_declared, "2021A prediction")
    return checkpoint, finetune, prediction, dict(source)


def validate_all_operating_points_registry(path: Path) -> dict[str, Any]:
    """Return the sealed common-variable source after binding every identity."""
    path = path.resolve()
    registry = read_json(path)
    if registry.get("status") != ALL_OPERATING_STATUS or registry.get("selection_partition") != "2021A":
        raise RuntimeError("All-operating-points registry is not a sealed 2021A registry")
    _require_false(registry, "2021B_accessed", "all-operating-points registry")
    _require_false(registry, "year_2022_accessed", "all-operating-points registry")
    if registry.get("exact_roster") is not None and COMMON_MODEL_ID not in registry["exact_roster"]:
        raise RuntimeError("All-operating-points registry omits common_variable_only")
    runs = registry.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != set(registry.get("exact_roster", runs)):
        raise RuntimeError("All-operating-points registry roster is incomplete or mixed")
    run = runs.get(COMMON_MODEL_ID)
    if not isinstance(run, Mapping):
        raise RuntimeError("All-operating-points registry has no common-variable run")
    source = run.get("source")
    if not isinstance(source, Mapping) or source.get("active_ablations") != [COMMON_MODEL_ID]:
        raise RuntimeError("common-variable run configuration is not the pre-specified projection")
    checkpoint, finetune_path, prediction, source = _run_source(run)
    finetune = read_json(finetune_path)
    if (finetune.get("status") != "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022"
            or finetune.get("prediction_partition") != "2021A"):
        raise RuntimeError("common-variable fine-tuning manifest is not sealed on 2021A")
    _require_false(finetune, "2021B_accessed", "common-variable fine-tuning manifest")
    _require_false(finetune, "year_2022_accessed", "common-variable fine-tuning manifest")
    ablations = finetune.get("ablations")
    if not isinstance(ablations, Mapping) or {key for key, enabled in ablations.items() if enabled is True} != {COMMON_MODEL_ID}:
        raise RuntimeError("common-variable fine-tuning configuration mismatch")
    if finetune.get("checkpoint", {}).get("sha256") != sha256(checkpoint):
        raise RuntimeError("common-variable checkpoint does not match finetune manifest")
    if finetune.get("prediction", {}).get("sha256") != source["prediction"].get("sha256"):
        raise RuntimeError("common-variable 2021A prediction does not match finetune manifest")

    operating_declared = run.get("operating_point_manifest")
    operating_dir_name = run.get("operating_point_directory")
    if not isinstance(operating_declared, Mapping) or not isinstance(operating_declared.get("path", operating_declared.get("file")), str):
        raise RuntimeError("common-variable operating-point manifest identity missing")
    operating_path = identity_path(operating_declared, "operating-point manifest")
    operating_dir = Path(str(operating_dir_name)).resolve() if isinstance(operating_dir_name, str) else operating_path.parent
    if operating_path.parent != operating_dir:
        raise RuntimeError("common-variable operating-point directory mismatch")
    assert_identity(operating_path, operating_declared, "operating-point manifest")
    operating = read_json(operating_path)
    if (operating.get("status") != "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A"
            or operating.get("selection_partition") != "2021A"):
        raise RuntimeError("common-variable operating-point manifest is not sealed on 2021A")
    _require_false(operating, "2021B_accessed", "common-variable operating-point manifest")
    _require_false(operating, "year_2022_accessed", "common-variable operating-point manifest")
    operating_source = operating.get("source")
    if not isinstance(operating_source, Mapping) or operating_source.get("sha256") != source["prediction"].get("sha256"):
        raise RuntimeError("operating-point source does not bind to common-variable 2021A prediction")
    if operating_source.get("manifest_sha256") != sha256(finetune_path):
        raise RuntimeError("operating-point source does not bind to common-variable finetune manifest")
    required_artifacts = {"transformer_binary_calibrators.joblib", "transformer_operating_thresholds_2021A.json"}
    artifacts = operating.get("artifacts")
    if not isinstance(artifacts, Mapping) or not required_artifacts.issubset(artifacts):
        raise RuntimeError("common-variable operating-point artifact roster is incomplete")
    for name in sorted(required_artifacts):
        assert_identity(operating_dir / name, artifacts[name], f"operating-point {name}")
    return {
        "all_registry": artifact_identity(path),
        "model_id": COMMON_MODEL_ID,
        "selected_model_dir": str(checkpoint.parent),
        "checkpoint": artifact_identity(checkpoint),
        "finetune_manifest": artifact_identity(finetune_path),
        "prediction_2021A": assert_declared_identity_without_open(prediction, source["prediction"], "2021A prediction"),
        "operating_point_dir": str(operating_dir),
        "operating_point_manifest": artifact_identity(operating_path),
        "operating_thresholds": artifact_identity(operating_dir / "transformer_operating_thresholds_2021A.json"),
        "binary_calibrators": artifact_identity(operating_dir / "transformer_binary_calibrators.joblib"),
    }


def make_model_lock(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Create the compatibility lock required by the existing Stage 6 predictor."""
    return {
        "status": MODEL_LOCK_STATUS,
        "selection_partition": "2021A",
        "selected_id": COMMON_MODEL_ID,
        "selection_mode": "pre_specified_common_variable_only_no_metric_selection",
        "model_or_threshold_selection_on_2021A": False,
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "all_operating_points_registry": binding["all_registry"],
        "selected_model": {
            "directory": binding["selected_model_dir"], "checkpoint": binding["checkpoint"],
            "finetune_manifest": binding["finetune_manifest"], "prediction_2021A": binding["prediction_2021A"],
            "configuration": {"common_variable_only": True},
        },
        "operating_point": {
            "directory": binding["operating_point_dir"], "manifest": binding["operating_point_manifest"],
            "thresholds": binding["operating_thresholds"], "binary_calibrators": binding["binary_calibrators"],
        },
    }


def validate_model_lock(path: Path) -> dict[str, Any]:
    lock = read_json(path)
    if (lock.get("status") != MODEL_LOCK_STATUS or lock.get("selection_partition") != "2021A"
            or lock.get("selected_id") != COMMON_MODEL_ID
            or lock.get("selection_mode") != "pre_specified_common_variable_only_no_metric_selection"):
        raise RuntimeError("common-variable model lock has an invalid selection contract")
    _require_false(lock, "2021B_accessed", "common-variable model lock")
    _require_false(lock, "year_2022_accessed", "common-variable model lock")
    selected = lock.get("selected_model")
    operating = lock.get("operating_point")
    if not isinstance(selected, Mapping) or not isinstance(operating, Mapping):
        raise RuntimeError("common-variable model lock lacks required bindings")
    if selected.get("configuration") != {"common_variable_only": True}:
        raise RuntimeError("common-variable model lock configuration mismatch")
    checkpoint = selected.get("checkpoint")
    finetune = selected.get("finetune_manifest")
    prediction = selected.get("prediction_2021A")
    operating_manifest = operating.get("manifest")
    thresholds = operating.get("thresholds")
    calibrators = operating.get("binary_calibrators")
    registry = lock.get("all_operating_points_registry")
    if not all(isinstance(item, Mapping) and isinstance(item.get("path"), str)
               for item in (checkpoint, finetune, prediction, operating_manifest, thresholds, calibrators, registry)):
        raise RuntimeError("common-variable model lock artifact bindings are malformed")
    checkpoint_path = Path(str(checkpoint["path"])).resolve()
    finetune_path = Path(str(finetune["path"])).resolve()
    prediction_path = Path(str(prediction["path"])).resolve()
    operating_path = Path(str(operating_manifest["path"])).resolve()
    assert_identity(checkpoint_path, checkpoint, "locked checkpoint")
    assert_identity(finetune_path, finetune, "locked finetune manifest")
    assert_declared_identity_without_open(prediction_path, prediction, "locked 2021A prediction")
    assert_identity(operating_path, operating_manifest, "locked operating-point manifest")
    assert_identity(Path(str(thresholds["path"])).resolve(), thresholds, "locked operating thresholds")
    assert_identity(Path(str(calibrators["path"])).resolve(), calibrators, "locked binary calibrators")
    assert_identity(Path(str(registry["path"])).resolve(), registry, "locked all-operating-points registry")
    if selected.get("directory") != str(checkpoint_path.parent) or checkpoint_path.parent.name != COMMON_MODEL_ID:
        raise RuntimeError("common-variable selected model directory does not match the locked model id")
    finetune_payload = read_json(finetune_path)
    operating_payload = read_json(operating_path)
    if (finetune_payload.get("prediction_partition") != "2021A"
            or operating_payload.get("selection_partition") != "2021A"):
        raise RuntimeError("common-variable lock no longer binds 2021A artifacts")
    _require_false(finetune_payload, "2021B_accessed", "locked finetune manifest")
    _require_false(finetune_payload, "year_2022_accessed", "locked finetune manifest")
    _require_false(operating_payload, "2021B_accessed", "locked operating-point manifest")
    _require_false(operating_payload, "year_2022_accessed", "locked operating-point manifest")
    return lock


def validate_stage6_output(output_root: Path, model_lock: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Reject partial, cross-model, future-year, or non-2021B Stage 6 output."""
    output_root = output_root.resolve()
    prediction_dir = output_root / "predictions_2021B"
    conformal_dir = output_root / "conformal_2021B"
    prediction_manifest_path = prediction_dir / "prediction_manifest.json"
    prediction_path = prediction_dir / "predictions_2021B.parquet"
    conformal_manifest_path = conformal_dir / "conformal_calibrator.json"
    conformal_sets = conformal_dir / "conformal_sets_2021B.parquet"
    registry_path = output_root / "stage6_registry.json"
    expected = (prediction_manifest_path, prediction_path, conformal_manifest_path, conformal_sets, registry_path)
    if not all(item.is_file() for item in expected):
        raise RuntimeError("Partial Stage 6 output exists")
    prediction_manifest = read_json(prediction_manifest_path)
    if (prediction_manifest.get("status") != "PASS_PREDICTIONS_2021B_ONLY"
            or prediction_manifest.get("partition") != "2021B"
            or prediction_manifest.get("model_or_threshold_selection_on_2021B") is not False
            or prediction_manifest.get("year_2022_accessed") is not False):
        raise RuntimeError("Stage 6 prediction partition or access seal failed")
    if (prediction_manifest.get("model", {}).get("sha256") != binding["checkpoint"]["sha256"]
            or prediction_manifest.get("operating_point", {}).get("manifest_sha256") != binding["operating_point_manifest"]["sha256"]
            or prediction_manifest.get("artifact", {}).get("sha256") != sha256(prediction_path)):
        raise RuntimeError("Stage 6 prediction identity mismatch")
    conformal_manifest = read_json(conformal_manifest_path)
    if (conformal_manifest.get("status") != "PASS_LOCKED_PRE_2022"
            or conformal_manifest.get("calibration_partition") != "2021B only"
            or conformal_manifest.get("model_or_threshold_selection_on_2021B") is not False
            or conformal_manifest.get("year_2022_accessed") is not False
            or conformal_manifest.get("source", {}).get("sha256") != sha256(prediction_path)
            or conformal_manifest.get("artifact", {}).get("sha256") != sha256(conformal_sets)
            or conformal_manifest.get("operating_point_source", {}).get("manifest_sha256") != binding["operating_point_manifest"]["sha256"]):
        raise RuntimeError("Stage 6 conformal identity or access seal failed")
    stage6_registry = read_json(registry_path)
    if (stage6_registry.get("status") != "PASS_2021B_CONFORMAL_LOCKED_PRE_2022"
            or stage6_registry.get("partition") != "2021B only"
            or stage6_registry.get("model_or_threshold_selection_on_2021B") is not False
            or stage6_registry.get("year_2022_accessed") is not False):
        raise RuntimeError("Stage 6 final registry partition or access seal failed")
    if (stage6_registry.get("prediction", {}).get("manifest_sha256") != sha256(prediction_manifest_path)
            or stage6_registry.get("prediction", {}).get("artifact_sha256") != sha256(prediction_path)
            or stage6_registry.get("conformal", {}).get("manifest_sha256") != sha256(conformal_manifest_path)
            or stage6_registry.get("conformal", {}).get("artifact_sha256") != sha256(conformal_sets)):
        raise RuntimeError("Stage 6 final registry identity mismatch")
    validate_model_lock(model_lock)
    return {
        "registry": artifact_identity(registry_path),
        "prediction_manifest": artifact_identity(prediction_manifest_path),
        "prediction": artifact_identity(prediction_path),
        "conformal_manifest": artifact_identity(conformal_manifest_path),
        "conformal_sets": artifact_identity(conformal_sets),
        "conformal_payload": conformal_manifest,
    }


def make_transfer_binding(stage6: Mapping[str, Any]) -> dict[str, Any]:
    """Project actual frozen q values to MIMIC's permitted conformal contract."""
    manifest = stage6["conformal_payload"]
    outputs = manifest.get("risk_set_outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get("global"), Mapping):
        raise RuntimeError("Conformal manifest lacks global risk-set metadata")
    output_mondrian = outputs.get("mondrian")
    generic_dimensions = set(ALLOWED_TRANSFER_DIMENSIONS) | PROHIBITED_TRANSFER_DIMENSIONS
    if not isinstance(output_mondrian, Mapping) or set(output_mondrian) != generic_dimensions:
        raise RuntimeError("Generic Stage 6 Mondrian dimension contract drifted")
    calibration = manifest.get("calibration")
    if not isinstance(calibration, Mapping) or not isinstance(calibration.get("global"), Mapping):
        raise RuntimeError("Conformal manifest lacks frozen global q calibration")
    calibration_mondrian = calibration.get("mondrian")
    if not isinstance(calibration_mondrian, Mapping) or set(calibration_mondrian) != generic_dimensions:
        raise RuntimeError("Conformal manifest has an invalid Mondrian q roster")
    if tuple(manifest.get("nominal_coverages", ())) != (0.90, 0.80, 0.95):
        raise RuntimeError("Generic conformal manifest nominal coverages drifted")

    global_q: dict[str, dict[str, float | int]] = {}
    for coverage in CONFORMAL_COVERAGES:
        entry = calibration["global"].get(coverage)
        if not isinstance(entry, Mapping):
            raise RuntimeError("Global conformal q roster is incomplete")
        q = entry.get("q")
        n = entry.get("n")
        alpha = entry.get("alpha")
        if (isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(float(q)) or not 0 <= float(q) <= 1
                or isinstance(n, bool) or not isinstance(n, int) or n < 1
                or isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha))):
            raise RuntimeError("Global conformal q entry is malformed")
        global_q[coverage] = {"alpha": float(alpha), "n": n, "q": float(q)}

    mondrian_q: dict[str, dict[str, dict[str, dict[str, float | int | str]]]] = {}
    for dimension in ALLOWED_TRANSFER_DIMENSIONS:
        source = calibration_mondrian[dimension]
        if not isinstance(source, Mapping):
            raise RuntimeError(f"Conformal {dimension} q roster is malformed")
        projected_dimension: dict[str, dict[str, dict[str, float | int | str]]] = {}
        for coverage in CONFORMAL_COVERAGES:
            groups = source.get(coverage)
            if not isinstance(groups, Mapping) or not groups:
                raise RuntimeError(f"Conformal {dimension} q roster is incomplete")
            projected_groups: dict[str, dict[str, float | int | str]] = {}
            for group, entry in sorted(groups.items(), key=lambda item: str(item[0])):
                if not isinstance(group, str) or not group or not isinstance(entry, Mapping):
                    raise RuntimeError(f"Conformal {dimension} group q entry is malformed")
                q, n, status, events = entry.get("q"), entry.get("n"), entry.get("status"), entry.get("events_any_readmission")
                if (isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(float(q)) or not 0 <= float(q) <= 1
                        or isinstance(n, bool) or not isinstance(n, int) or n < 1
                        or not isinstance(status, str) or not status
                        or isinstance(events, bool) or not isinstance(events, int) or events < 0 or events > n):
                    raise RuntimeError(f"Conformal {dimension} group q entry is invalid")
                projected_groups[group] = {"n": n, "q": float(q), "status": status, "events_any_readmission": events}
            projected_dimension[coverage] = projected_groups
        mondrian_q[dimension] = projected_dimension
    binding = {
        "status": TRANSFER_BINDING_STATUS,
        "calibration_partition": "2021B only",
        "model_or_threshold_selection_on_2021B": False,
        "year_2022_accessed": False,
        "conformal_manifest": stage6["conformal_manifest"],
        "conformal_sets": stage6["conformal_sets"],
        "nominal_coverages": list(NOMINAL_COVERAGES),
        "global": global_q,
        "mondrian": mondrian_q,
        "risk_set_fields": {"global": outputs["global"],
                            "mondrian": {name: output_mondrian[name] for name in ALLOWED_TRANSFER_DIMENSIONS}},
        "allowed_mondrian_dimensions": list(ALLOWED_TRANSFER_DIMENSIONS),
    }
    serialized = stable_json(binding).lower()
    if any(item in serialized for item in PROHIBITED_TRANSFER_DIMENSIONS):
        raise RuntimeError("Exported MIMIC conformal binding contains prohibited payer/ZIP dimensions")
    return binding


def validate_transfer_binding(path: Path) -> dict[str, Any]:
    binding = read_json(path)
    if (binding.get("status") != TRANSFER_BINDING_STATUS
            or binding.get("calibration_partition") != "2021B only"
            or binding.get("model_or_threshold_selection_on_2021B") is not False
            or binding.get("year_2022_accessed") is not False
            or binding.get("allowed_mondrian_dimensions") != list(ALLOWED_TRANSFER_DIMENSIONS)
            or tuple(binding.get("nominal_coverages", ())) != NOMINAL_COVERAGES):
        raise RuntimeError("MIMIC conformal projection contract is invalid")
    mondrian = binding.get("mondrian")
    if not isinstance(mondrian, Mapping) or set(mondrian) != set(ALLOWED_TRANSFER_DIMENSIONS):
        raise RuntimeError("MIMIC conformal projection has an invalid Mondrian dimension set")
    serialized = stable_json(binding).lower()
    if any(item in serialized for item in PROHIBITED_TRANSFER_DIMENSIONS):
        raise RuntimeError("MIMIC conformal projection contains prohibited payer/ZIP data")
    global_q = binding.get("global")
    if not isinstance(global_q, Mapping) or set(global_q) != set(CONFORMAL_COVERAGES):
        raise RuntimeError("MIMIC conformal projection global q roster is incomplete")
    for coverage in CONFORMAL_COVERAGES:
        q = global_q[coverage].get("q") if isinstance(global_q[coverage], Mapping) else None
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(float(q)) or not 0 <= float(q) <= 1:
            raise RuntimeError("MIMIC conformal projection has an invalid global q")
    for dimension in ALLOWED_TRANSFER_DIMENSIONS:
        source = mondrian[dimension]
        if not isinstance(source, Mapping) or set(source) != set(CONFORMAL_COVERAGES):
            raise RuntimeError(f"MIMIC conformal projection {dimension} q roster is incomplete")
        for coverage in CONFORMAL_COVERAGES:
            groups = source[coverage]
            if not isinstance(groups, Mapping) or not groups:
                raise RuntimeError(f"MIMIC conformal projection {dimension} group q roster is empty")
            for group, entry in groups.items():
                if not isinstance(group, str) or not group or not isinstance(entry, Mapping):
                    raise RuntimeError("MIMIC conformal projection group entry is malformed")
                q, n, status, events = entry.get("q"), entry.get("n"), entry.get("status"), entry.get("events_any_readmission")
                if (isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(float(q)) or not 0 <= float(q) <= 1
                        or isinstance(n, bool) or not isinstance(n, int) or n < 1
                        or not isinstance(status, str) or not status
                        or isinstance(events, bool) or not isinstance(events, int) or events < 0 or events > n):
                    raise RuntimeError("MIMIC conformal projection group q entry is invalid")
    for name in ("conformal_manifest", "conformal_sets"):
        declared = binding.get(name)
        if not isinstance(declared, Mapping) or not isinstance(declared.get("path"), str):
            raise RuntimeError(f"MIMIC conformal projection lacks {name} identity")
        assert_identity(Path(str(declared["path"])), declared, name)
    return binding


def preflight(all_registry: Path) -> dict[str, Any]:
    """No-data validation: this function never loads an NRD or MIMIC row."""
    binding = validate_all_operating_points_registry(all_registry)
    lock = make_model_lock(binding)
    if lock["2021B_accessed"] is not False or lock["year_2022_accessed"] is not False:
        raise RuntimeError("Pre-2021B common-variable lock access gate failed")
    return {"status": "PASS_PREFLIGHT_PRE_2021B_PRE_2022_NO_DATA_ACCESS", "binding": binding,
            "model_lock": lock, "mimic_archive_opened": False, "nrd_patient_rows_opened": False}


def run_controller(all_registry: Path, root: Path, history_dir: Path, hierarchy_dir: Path,
                   output_root: Path, *, batch_size: int = 128, workers: int = 4,
                   threads: int = 8, min_mondrian_rows: int = 200,
                   stage6_controller: Path | None = None) -> dict[str, Any]:
    if not 1 <= threads <= 8 or not 0 <= workers <= 8:
        raise RuntimeError("Resource arguments require 1..8 threads and 0..8 workers")
    output_root = output_root.resolve()
    if output_root.exists():
        raise RuntimeError("Output root already exists; refuse partial or replacement publication")
    pre = preflight(all_registry)
    output_root.mkdir(parents=True)
    model_lock_path = output_root / "common_variable_model_lock_2021A.json"
    atomic_create_json(model_lock_path, pre["model_lock"])
    validate_model_lock(model_lock_path)
    stage6 = (stage6_controller or Path(__file__).resolve().parents[1] / "stage6" / "run_2021b_conformal.py").resolve()
    if not stage6.is_file():
        raise RuntimeError(f"Missing existing Stage 6 controller: {stage6}")
    stage6_output = output_root / "stage6_2021B"
    command = [sys.executable, str(stage6), "--root", str(root.resolve()), "--history-dir", str(history_dir.resolve()),
               "--hierarchy-dir", str(hierarchy_dir.resolve()), "--model-selection-lock", str(model_lock_path),
               "--selected-model-dir", str(pre["binding"]["selected_model_dir"]),
               "--operating-point-dir", str(pre["binding"]["operating_point_dir"]), "--output-root", str(stage6_output),
               "--batch-size", str(batch_size), "--workers", str(workers), "--threads", str(threads),
               "--min-mondrian-rows", str(min_mondrian_rows)]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Existing Stage 6 controller failed (exit {completed.returncode}): {detail}")
    stage6_bound = validate_stage6_output(stage6_output, model_lock_path, pre["binding"])
    transfer = make_transfer_binding(stage6_bound)
    transfer_path = output_root / "mimic_conformal_binding.json"
    atomic_create_json(transfer_path, transfer)
    validate_transfer_binding(transfer_path)
    result = {
        "status": FINAL_STATUS, "model_id": COMMON_MODEL_ID, "selection_partition": "2021A",
        "calibration_partition": "2021B only", "model_or_threshold_selection_on_2021B": False,
        "2021B_accessed_before_run": False, "year_2022_accessed": False,
        "model_lock": artifact_identity(model_lock_path), "stage6": {key: value for key, value in stage6_bound.items() if key != "conformal_payload"},
        "mimic_transfer_conformal_binding": artifact_identity(transfer_path),
    }
    final_path = output_root / "common_variable_2021b_conformal_registry.json"
    atomic_create_json(final_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-operating-points-registry", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--history-dir", type=Path)
    parser.add_argument("--hierarchy-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--min-mondrian-rows", type=int, default=200)
    parser.add_argument("--stage6-controller", type=Path)
    args = parser.parse_args()
    if args.validate_only:
        if any(value is not None for value in (args.root, args.history_dir, args.hierarchy_dir, args.output_root, args.stage6_controller)):
            parser.error("--validate-only accepts only the all-operating-points registry and resource options")
        print(stable_json(preflight(args.all_operating_points_registry)))
        return
    if not all((args.root, args.history_dir, args.hierarchy_dir, args.output_root)):
        parser.error("execution requires --root, --history-dir, --hierarchy-dir, and --output-root")
    print(stable_json(run_controller(args.all_operating_points_registry, args.root, args.history_dir,
                                     args.hierarchy_dir, args.output_root, batch_size=args.batch_size,
                                     workers=args.workers, threads=args.threads,
                                     min_mondrian_rows=args.min_mondrian_rows,
                                     stage6_controller=args.stage6_controller)))


if __name__ == "__main__":
    main()
