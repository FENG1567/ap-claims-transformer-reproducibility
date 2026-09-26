#!/usr/bin/env python3
"""Build the pre-2022 reference contract consumed by locked drift analysis.

The builder is intentionally a pre-unlock operation.  Every input is an
explicit, hash-bound 2018--2021 artifact; there is no discovery/globbing and
no 2022/MIMIC/server argument.  It only summarizes fixed data/model outputs;
it never selects a model, adjusts a threshold, or fits a conformal quantile.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BUILDER_PATH = ROOT / "build_frozen_evaluation_spec.py"
EVALUATOR_PATH = ROOT / "evaluate_locked_2022.py"
DRIFT_PATH = ROOT / "analyze_locked_2022_drift.py"
TRANSFORMER_ROSTER = ("joint_main", "mlm_only", "scratch", "ap_only", "no_hierarchy_parameter_matched", "no_prday", "no_prior", "no_hospital_socioeconomic", "no_year", "common_variable_only")
BASELINE_ROSTER = ("structured_logistic", "elastic_net", "lightgbm")
OUTCOMES = ("any_readmission", "ap_specific_readmission")
LEAVES = ("none", "ap", "biliary", "sepsis_or_organ", "other")
NOMINAL = (("80", .80), ("90", .90), ("95", .95))
SUBGROUPS = {"sex": "FEMALE", "age": "age_group", "PAY1": "PAY1", "ZIPINC_QRTL": "ZIPINC_QRTL"}
MONDRIAN = {"sex": "sex", "age": "age_group", "PAY1": "payer", "ZIPINC_QRTL": "zip_income_quartile"}
AGE_BINS = [18.0, 45.0, 65.0, 75.0, 120.0]
OPTIONAL_CATEGORICAL_COVARIATES = ("HOSP_BEDSIZE", "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH", "PL_NCHS", "APRDRG_Severity", "APRDRG_Risk_Mortality")


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load required sealed code: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


builder = _load(BUILDER_PATH, "stage7_evaluation_contract")
ev = _load(EVALUATOR_PATH, "stage7_evaluator")
drift = _load(DRIFT_PATH, "stage7_drift_consumer")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing required pre-2022 artifact: {path}")
    return {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable pre-2022 JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _binary(series: pd.Series, name: str) -> np.ndarray:
    if series.isna().any() or not series.isin([0, 1, False, True]).all():
        raise RuntimeError(f"Required outcome {name} is not complete binary 0/1")
    return series.astype(np.int8).to_numpy()


def _weights(frame: pd.DataFrame) -> np.ndarray:
    if "DISCWT" not in frame:
        raise RuntimeError("Pre-2022 artifact lacks DISCWT")
    values = pd.to_numeric(frame["DISCWT"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise RuntimeError("DISCWT must be finite and strictly positive")
    return values


def _partition(frame: pd.DataFrame, years: set[int], partition: str, label: str) -> None:
    required = {"year", "analysis_partition", "encounter_hash", "patient_hash"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} lacks required columns: {missing}")
    year = pd.to_numeric(frame["year"], errors="coerce")
    if year.isna().any() or set(year.astype(int)) != years or not frame["analysis_partition"].astype(str).eq(partition).all():
        raise RuntimeError(f"{label} is not restricted to the authorized {partition}/{sorted(years)} partition")
    if frame["encounter_hash"].isna().any() or frame["encounter_hash"].duplicated().any():
        raise RuntimeError(f"{label} has missing or duplicate encounter_hash")


def _scored_2021a_partition(frame: pd.DataFrame, label: str, *, allow_legacy_baseline: bool = False) -> None:
    """Validate a frozen 2021A scored cohort.

    The historical baseline prediction artifact predates the explicit
    ``analysis_partition`` column.  It is still safe to consume only after its
    baseline-lock identity has been verified and its encounter, patient, and
    weight columns have been proven exactly equal to the strict 2021A label
    source.  Transformer artifacts never receive this compatibility path.
    """
    if "analysis_partition" in frame.columns:
        _partition(frame, {2021}, "2021A", label)
        return
    if not allow_legacy_baseline:
        raise RuntimeError(f"{label} lacks required columns: ['analysis_partition']")
    required = {"year", "encounter_hash", "patient_hash"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{label} lacks required columns: {missing}")
    year = pd.to_numeric(frame["year"], errors="coerce")
    if year.isna().any() or set(year.astype(int)) != {2021}:
        raise RuntimeError(f"{label} is not restricted to the authorized 2021A/[2021] cohort")
    if frame["encounter_hash"].isna().any() or frame["encounter_hash"].duplicated().any():
        raise RuntimeError(f"{label} has missing or duplicate encounter_hash")
    if frame["patient_hash"].isna().any():
        raise RuntimeError(f"{label} has missing patient_hash")


def _bin_labels(edges: list[float]) -> list[str]:
    return [f"[{left:g},{right:g}{']' if index == len(edges) - 2 else ')'}" for index, (left, right) in enumerate(zip(edges, edges[1:]))]


def _distribution(labels: np.ndarray, weights: np.ndarray, ordered: list[str]) -> dict[str, float]:
    total = float(weights.sum())
    result = {label: 0.0 for label in ordered}
    for label, weight in zip(labels, weights): result[str(label)] += float(weight)
    return {label: float(result[label] / total) for label in ordered}


def _numeric_age(frame: pd.DataFrame, weights: np.ndarray) -> dict[str, Any]:
    if "AGE" not in frame: raise RuntimeError("Development covariates lack AGE")
    age = pd.to_numeric(frame["AGE"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(age) & (age >= AGE_BINS[0]) & (age <= AGE_BINS[-1])
    labels = _bin_labels(AGE_BINS) + ["<NONFINITE_OR_OUT_OF_RANGE>"]
    index = np.minimum(np.searchsorted(np.asarray(AGE_BINS), age, side="right") - 1, len(AGE_BINS) - 2)
    assigned = np.asarray([labels[value] if okay else "<NONFINITE_OR_OUT_OF_RANGE>" for value, okay in zip(index, valid)], dtype=object)
    observed = np.isfinite(age)
    if not observed.any(): raise RuntimeError("Development AGE has no finite value")
    mean = float(np.dot(age[observed], weights[observed]) / weights[observed].sum())
    sd = float(np.sqrt(np.dot(weights[observed], (age[observed] - mean) ** 2) / weights[observed].sum()))
    if not math.isfinite(sd) or sd <= 0: raise RuntimeError("Development AGE has nonpositive weighted SD")
    return {"column": "AGE", "kind": "numeric_binned", "bins": AGE_BINS,
            "reference_missing_rate": float((~observed).mean()), "reference_weighted_distribution": _distribution(assigned, weights, labels),
            "reference_weighted_mean": mean, "reference_weighted_sd": sd}


def _categorical(frame: pd.DataFrame, column: str, weights: np.ndarray) -> dict[str, Any]:
    raw = frame[column]
    nonmissing = raw.dropna()
    if nonmissing.empty: raise RuntimeError(f"Development covariate {column} is entirely missing")
    values = sorted({str(value) for value in nonmissing.tolist()})
    if len(values) > 50: raise RuntimeError(f"Development covariate {column} exceeds the prespecified categorical cardinality cap")
    output_categories: list[Any] = []
    for item in values:
        # JSON-native categories retain their unambiguous printed value; the
        # consumer canonicalizes to str before matching 2022 observations.
        output_categories.append(item)
    labels = values + ["<UNSEEN_OR_OTHER>"]
    observed = raw.astype("string").fillna("<UNSEEN_OR_OTHER>").astype(str).to_numpy()
    assigned = np.where(np.isin(observed, values), observed, "<UNSEEN_OR_OTHER>")
    return {"column": column, "kind": "categorical", "categories": output_categories,
            "reference_missing_rate": float(raw.isna().mean()), "reference_weighted_distribution": _distribution(assigned, weights, labels)}


def build_covariate_reference(development: pd.DataFrame) -> dict[str, Any]:
    _partition(development, {2018, 2019, 2020}, "development", "development covariate artifact")
    weights = _weights(development)
    result = {"age": _numeric_age(development, weights)}
    for column in ("FEMALE", "PAY1", "ZIPINC_QRTL", *OPTIONAL_CATEGORICAL_COVARIATES):
        if column in development.columns:
            result[column.lower()] = _categorical(development, column, weights)
    required = {"FEMALE", "PAY1", "ZIPINC_QRTL"}
    absent = required - set(item["column"] for item in result.values())
    if absent: raise RuntimeError(f"Development covariates lack required demographic/insurance/income fields: {sorted(absent)}")
    return result


def _labels(frame: pd.DataFrame, high_cost_threshold: float) -> dict[str, np.ndarray]:
    required = {"any_unplanned_readmission_30d", "readmission_leaf", "cost_2021_usd", "LOS", "DIED"}
    missing = sorted(required - set(frame.columns))
    if missing: raise RuntimeError(f"2021A label artifact lacks exact seven-outcome ingredients: {missing}")
    any_readmission = _binary(frame["any_unplanned_readmission_30d"], "any_readmission")
    leaf = pd.to_numeric(frame["readmission_leaf"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(leaf).all() or not np.equal(leaf, np.floor(leaf)).all() or not np.isin(leaf, range(len(LEAVES))).all():
        raise RuntimeError("2021A readmission_leaf is invalid")
    leaf = leaf.astype(np.int8)
    if not np.array_equal(any_readmission, (leaf > 0).astype(np.int8)):
        raise RuntimeError("2021A parent/leaf outcome hierarchy is inconsistent")
    cost = pd.to_numeric(frame["cost_2021_usd"], errors="coerce").to_numpy(dtype=float)
    los = pd.to_numeric(frame["LOS"], errors="coerce").to_numpy(dtype=float)
    # Costs can be absent in an otherwise valid AP episode.  Such an episode
    # has no high-cost label; it must never be converted to a negative label.
    # A nonpositive/non-finite standardized cost is likewise invalid rather
    # than a valid value below the threshold.  This exactly matches the
    # Stage-5 auxiliary-label contract (finite standardized cost > 0).
    # LOS remains a complete outcome.
    high_cost_valid = np.isfinite(cost) & (cost > 0)
    if not np.isfinite(los).all(): raise RuntimeError("2021A LOS outcome is incomplete")
    died = _binary(frame["DIED"], "in_hospital_death")
    return {"any_readmission": any_readmission, "ap_specific_readmission": (leaf == 1).astype(np.int8),
            "biliary_readmission": (leaf == 2).astype(np.int8), "sepsis_or_organ_readmission": (leaf == 3).astype(np.int8),
            "high_cost": np.where(high_cost_valid, (cost > high_cost_threshold).astype(np.float64), np.nan),
            "prolonged_los": (los > 7).astype(np.int8), "in_hospital_death": died}


def _suppressed_count(value: int) -> int | None:
    """Do not expose HCUP small-cell counts in a portable/public JSON."""
    return None if value <= 10 else int(value)


def build_label_reference(labels_2021a: pd.DataFrame, high_cost_threshold: float) -> dict[str, Any]:
    _partition(labels_2021a, {2021}, "2021A", "2021A labels")
    weights = _weights(labels_2021a); labels = _labels(labels_2021a, high_cost_threshold)
    result: dict[str, Any] = {}
    for name, value in labels.items():
        if name == "high_cost":
            valid = np.isfinite(value)
            if not valid.any():
                raise RuntimeError("2021A high-cost outcome has no valid standardized-cost labels")
            y, w = value[valid].astype(np.int8), weights[valid]
            valid_n, missing_n = int(valid.sum()), int((~valid).sum())
            result[name] = {
                "column": "outcome_high_cost", "unweighted_rate": float(y.mean()),
                "DISCWT_weighted_rate": float(np.dot(y, w) / w.sum()),
                "validity": {"policy": "EXCLUDE_NULL_HIGH_COST_LABELS_FROM_ALL_RATES_AND_BOOTSTRAPS",
                             "reference_valid_n": _suppressed_count(valid_n), "reference_missing_n": _suppressed_count(missing_n),
                             "reference_valid_weight": None if valid_n <= 10 else float(w.sum()),
                             "small_cell_counts_suppressed": bool(valid_n <= 10 or missing_n <= 10)},
            }
        else:
            result[name] = {"column": f"outcome_{name}", "unweighted_rate": float(value.mean()),
                            "DISCWT_weighted_rate": float(np.dot(value, weights) / weights.sum())}
    return result


def _assert_label_alignment(labels: pd.DataFrame, details: dict[str, dict[str, Any]], baseline: dict[str, Any]) -> None:
    """Require the seven-outcome source to be the exact 2021A cohort scored by every model.

    The auxiliary cost/LOS/death labels are not present in binary prediction
    files, but encounter, patient, partition, and DISCWT must be identical.
    This prevents silently attaching an otherwise plausible 2021A cohort.
    """
    _partition(labels, {2021}, "2021A", "seven-outcome 2021A label source")
    _weights(labels)
    expected = labels[["encounter_hash", "patient_hash", "DISCWT"]].sort_values("encounter_hash").reset_index(drop=True)
    paths = [Path(details[name]["manifest"]["source"]["file"]).resolve() for name in TRANSFORMER_ROSTER]
    paths.append(baseline["directory"] / "predictions_2021A.parquet")
    for index, path in enumerate(paths):
        columns = ["year", "encounter_hash", "patient_hash", "DISCWT"]
        frame = pd.read_parquet(path)
        available = set(frame.columns)
        if "analysis_partition" in available:
            columns.insert(1, "analysis_partition")
        frame = frame[columns]
        _scored_2021a_partition(
            frame,
            f"2021A scored cohort {index + 1}",
            allow_legacy_baseline=index == len(paths) - 1,
        )
        _weights(frame)
        actual = frame[["encounter_hash", "patient_hash", "DISCWT"]].sort_values("encounter_hash").reset_index(drop=True)
        if len(actual) != len(expected) or not actual["encounter_hash"].equals(expected["encounter_hash"]) or not actual["patient_hash"].equals(expected["patient_hash"]) or not np.allclose(actual["DISCWT"].to_numpy(dtype=float), expected["DISCWT"].to_numpy(dtype=float), rtol=0.0, atol=0.0):
            raise RuntimeError("Seven-outcome label source is not one-to-one aligned with every frozen 2021A model prediction")


def _assert_declared(path: Path, declared: dict[str, Any], label: str) -> None:
    if not isinstance(declared, dict) or declared.get("sha256") != sha256(path) or int(declared.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"{label} identity mismatch")


def _transformer_calibration(details: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    reference: dict[str, Any] = {}; sources: dict[str, dict[str, Any]] = {}
    for model in TRANSFORMER_ROSTER:
        detail = details[model]
        source = Path(detail["manifest"]["source"]["file"]).resolve()
        _assert_declared(source, detail["manifest"]["source"], f"{model} raw 2021A prediction")
        raw = pd.read_parquet(source, columns=["year", "analysis_partition", "encounter_hash", "patient_hash", "DISCWT", "any_unplanned_readmission_30d", "readmission_leaf"])
        _partition(raw, {2021}, "2021A", f"{model} raw 2021A prediction"); weights = _weights(raw)
        calibrated_path = detail["files"]["predictions_2021A_calibrated.parquet"]
        cal = pd.read_parquet(calibrated_path)
        _partition(cal, {2021}, "2021A", f"{model} calibrated 2021A prediction")
        required = {"encounter_hash", "p_any_readmission_calibrated", "p_ap_specific_readmission_calibrated"}
        if not required.issubset(cal.columns): raise RuntimeError(f"{model} calibrated 2021A prediction lacks frozen endpoint columns")
        merged = raw.merge(cal[["encounter_hash", "p_any_readmission_calibrated", "p_ap_specific_readmission_calibrated"]], on="encounter_hash", how="inner", validate="one_to_one")
        if len(merged) != len(raw) or len(merged) != len(cal): raise RuntimeError(f"{model} calibrated/raw 2021A predictions are not one-to-one")
        y = {"any_readmission": _binary(merged["any_unplanned_readmission_30d"], "any_readmission"),
             "ap_specific_readmission": (pd.to_numeric(merged["readmission_leaf"], errors="coerce").to_numpy(dtype=int) == 1).astype(np.int8)}
        w = _weights(merged); reference[model] = {}
        for outcome in OUTCOMES:
            p = pd.to_numeric(merged[f"p_{outcome}_calibrated"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any(): raise RuntimeError(f"{model}/{outcome} calibrated probabilities invalid")
            metrics = ev.binary_metrics(y[outcome], p, w)
            if metrics["calibration_status"] != "OK": raise RuntimeError(f"{model}/{outcome} calibration is undefined")
            reference[model][outcome] = {key: float(metrics[key]) for key in ("calibration_intercept", "calibration_slope", "brier", "log_loss")}
        sources[model] = {"raw_prediction": identity(source), "calibrated_prediction": identity(calibrated_path)}
    return reference, sources


def _baseline_calibration(baseline: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    directory, lock = baseline["directory"], baseline["lock"]
    path = directory / "predictions_2021A.parquet"
    _assert_declared(path, lock.get("artifacts", {}).get(path.name), "baseline 2021A prediction")
    frame = pd.read_parquet(path)
    _scored_2021a_partition(frame, "baseline 2021A prediction", allow_legacy_baseline=True)
    weights = _weights(frame)
    y = {"any_readmission": _binary(frame["any_unplanned_readmission_30d"], "any_readmission"), "ap_specific_readmission": _binary(frame["ap_specific_readmission_30d"], "ap_specific_readmission")}
    result: dict[str, Any] = {}
    for model in BASELINE_ROSTER:
        result[model] = {}
        for outcome in OUTCOMES:
            column = f"p_cal__{outcome}__{model}"
            if column not in frame: raise RuntimeError(f"Baseline 2021A prediction lacks {column}")
            p = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any(): raise RuntimeError(f"Baseline probability invalid: {column}")
            metrics = ev.binary_metrics(y[outcome], p, weights)
            if metrics["calibration_status"] != "OK": raise RuntimeError(f"Baseline calibration undefined: {model}/{outcome}")
            result[model][outcome] = {key: float(metrics[key]) for key in ("calibration_intercept", "calibration_slope", "brier", "log_loss")}
    return result, {"baseline_lock": identity(directory / "baseline_lock.json"), "prediction": identity(path)}


def _stage6_prediction(stage6_registry: Path, stage6: dict[str, Any]) -> Path:
    registry = read_json(stage6_registry); item = registry.get("prediction")
    if not isinstance(item, dict) or not isinstance(item.get("manifest"), str) or not isinstance(item.get("artifact"), str):
        raise RuntimeError("2021B predictor/conformal registry lacks hash-bound prediction artifacts")
    manifest_path, prediction = Path(item["manifest"]).resolve(), Path(item["artifact"]).resolve()
    if item.get("manifest_sha256") != sha256(manifest_path) or item.get("artifact_sha256") != sha256(prediction):
        raise RuntimeError("2021B predictor registry identity mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS_PREDICTIONS_2021B_ONLY" or manifest.get("partition") != "2021B" or manifest.get("year_2022_accessed") is not False or manifest.get("model_or_threshold_selection_on_2021B") is not False:
        raise RuntimeError("2021B predictor registry is not sealed calibration-only")
    artifact = manifest.get("artifact", {})
    if artifact.get("sha256") != sha256(prediction) or int(artifact.get("bytes", -1)) != prediction.stat().st_size:
        raise RuntimeError("2021B prediction manifest does not bind its artifact")
    return prediction


def build_coverage_reference(stage6_registry: Path, stage6: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction_path = _stage6_prediction(stage6_registry, stage6)
    prediction = pd.read_parquet(prediction_path)
    _partition(prediction, {2021}, "2021B", "2021B prediction")
    if "readmission_leaf" not in prediction: raise RuntimeError("2021B prediction lacks truth readmission_leaf")
    leaf = pd.to_numeric(prediction["readmission_leaf"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(leaf).all() or not np.isin(leaf, range(len(LEAVES))).all(): raise RuntimeError("2021B truth leaf invalid")
    leaf = leaf.astype(np.int8)
    sets_path = stage6["artifact_path"]; sets = pd.read_parquet(sets_path)
    required = {"encounter_hash", *[f"group_{value}" for value in MONDRIAN.values()]}
    if not required.issubset(sets.columns): raise RuntimeError("2021B conformal sets lack frozen subgroup columns")
    merged = prediction[["encounter_hash", "readmission_leaf"]].merge(sets, on="encounter_hash", how="inner", validate="one_to_one")
    if len(merged) != len(prediction) or len(merged) != len(sets): raise RuntimeError("2021B conformal truth/set artifact is not one-to-one")
    calibrator = stage6["calibrator"]; outputs = calibrator["risk_set_outputs"]
    rows: list[dict[str, Any]] = []
    for label, nominal in NOMINAL:
        entries = [(f"global_{label}", "global", None, outputs["global"][label])]
        entries.extend((f"mondrian_{display}_{label}", "mondrian", display, outputs["mondrian"][source]["levels"][label]) for display, source in MONDRIAN.items())
        for name, scope, dimension, artifact in entries:
            mask_col, size_col = artifact["leaf_set_mask"], artifact["leaf_set_size"]
            if mask_col not in merged or size_col not in merged: raise RuntimeError(f"2021B conformal sets lack {name} columns")
            masks = pd.to_numeric(merged[mask_col], errors="coerce").to_numpy(dtype=float)
            sizes = pd.to_numeric(merged[size_col], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(masks).all() or not np.equal(masks, np.floor(masks)).all() or (masks < 0).any() or (masks > 31).any() or not np.array_equal(sizes.astype(int), ev._bitcount(masks.astype(np.uint8))):
                raise RuntimeError(f"2021B conformal masks/sizes invalid: {name}")
            abstain = merged["abstain_90"].to_numpy(dtype=bool) if label == "90" and "abstain_90" in merged else None
            def summary(selected: np.ndarray) -> dict[str, Any]: return ev._conformal_summary(masks.astype(np.uint8)[selected], np.asarray([LEAVES[x] for x in leaf])[selected], None if abstain is None else abstain[selected])
            all_summary = summary(np.ones(len(merged), dtype=bool))
            rows.append({"conformal_set": name, "subgroup_dimension": "ALL", "subgroup_value": "ALL", "coverage": all_summary["coverage"], "mean_set_size": all_summary["mean_set_size"], "abstention_rate": all_summary["abstention_rate"]})
            for display, source in MONDRIAN.items():
                group = merged[f"group_{source}"].astype("string").fillna("<MISSING>").astype(str)
                for value in sorted(group.unique().tolist()):
                    detail = summary(group.eq(value).to_numpy())
                    rows.append({"conformal_set": name, "subgroup_dimension": display, "subgroup_value": value, "coverage": detail["coverage"], "mean_set_size": detail["mean_set_size"], "abstention_rate": detail["abstention_rate"]})
    return rows, {"stage6_registry": identity(stage6_registry), "prediction": identity(prediction_path), "conformal_sets": identity(sets_path), "conformal_manifest": identity(stage6["manifest_path"])}


def _high_cost_threshold(path: Path) -> float:
    spec = read_json(path)
    try: value = float(spec["high_cost"]["threshold_2021_usd"])
    except (KeyError, TypeError, ValueError) as exc: raise RuntimeError("Auxiliary specification lacks high_cost threshold_2021_usd") from exc
    if not math.isfinite(value) or value <= 0: raise RuntimeError("Auxiliary high-cost threshold is invalid")
    return value


def _atomic_publish(output: Path, payload: dict[str, Any]) -> dict[str, Any]:
    output = output.resolve(); sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists(): raise RuntimeError("Frozen drift reference or sidecar already exists and is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent); temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle: handle.write(stable_json(payload)); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary.exists(): temporary.unlink()
    digest = sha256(output)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{sidecar.name}.", suffix=".tmp", dir=output.parent); temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{digest}  {output.name}\n"); handle.flush(); os.fsync(handle.fileno())
        # A concurrent writer is a hard failure; do not overwrite its sidecar.
        os.link(temporary, sidecar)
    finally:
        if temporary.exists(): temporary.unlink()
    return {"path": str(output), "sha256": digest, "sidecar": str(sidecar)}


def build_frozen_drift_reference(all_operating_points: Path, baseline_lock: Path, stage6_registry: Path,
                                 development_covariates: Path, labels_2021a: Path, auxiliary_spec: Path,
                                 output: Path) -> dict[str, Any]:
    """Generate the consumer-valid frozen drift reference from authorised inputs."""
    details = builder.validate_all_operating_points(all_operating_points)
    baseline = builder.validate_baseline_lock(baseline_lock)
    stage6 = builder.validate_stage6_conformal(stage6_registry, details["joint_main"]["manifest_path"])
    development = pd.read_parquet(development_covariates.resolve())
    labels = pd.read_parquet(labels_2021a.resolve())
    high_cost = _high_cost_threshold(auxiliary_spec.resolve())
    _assert_label_alignment(labels, details, baseline)
    transformer, transformer_sources = _transformer_calibration(details)
    baselines, baseline_sources = _baseline_calibration(baseline)
    coverage, coverage_sources = build_coverage_reference(stage6_registry.resolve(), stage6)
    result = {"status": "FROZEN_2022_DRIFT_REFERENCE", "schema_version": "stage7_frozen_drift_reference_v1", "year_2022_accessed": False,
              "source_partitions": {"development_years": [2018, 2019, 2020], "selection_partition": "2021A", "conformal_partition": "2021B only", "year_2022_accessed": False, "model_or_threshold_selection_on_2021B": False},
              "covariates": build_covariate_reference(development), "label_outcomes": build_label_reference(labels, high_cost),
              "calibration": {**transformer, **baselines}, "coverage": coverage,
              "bootstrap": {"n_replicates": 1000, "seed": 20220914, "max_threads": 8, "cluster_column": "patient_hash"},
              "input_identities": {"all_operating_points": identity(all_operating_points), "baseline_lock": identity(baseline_lock), "stage6_registry": identity(stage6_registry), "development_covariates": identity(development_covariates), "labels_2021a": identity(labels_2021a), "auxiliary_spec": identity(auxiliary_spec), "transformer_2021a": transformer_sources, "baselines_2021a": baseline_sources, "conformal_2021b": coverage_sources},
              "publication_note": "HCUP small cells (n<=10) must be suppressed in public reports; this sealed internal reference is a required pre-2022 evaluation artifact.", "model_or_threshold_selection_on_2021B": False}
    # Prove the emitted JSON will be accepted by the future locked consumer.
    contract_models = {name: {} for name in (*TRANSFORMER_ROSTER, *BASELINE_ROSTER)}
    conformal_contract = [{"name": f"global_{label}", "scope": "global", "nominal": nominal} for label, nominal in NOMINAL]
    conformal_contract.extend({"name": f"mondrian_{dimension}_{label}", "scope": "mondrian", "nominal": nominal, "mondrian_dimension": dimension} for label, nominal in NOMINAL for dimension in SUBGROUPS)
    drift.validate_drift_reference({"models": contract_models, "conformal_sets": conformal_contract, "drift_reference": result})
    published = _atomic_publish(output, result)
    return {**result, "reference_sha256": published["sha256"], "output": published["path"], "sidecar": published["sidecar"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-operating-points", type=Path, required=True)
    parser.add_argument("--baseline-lock", type=Path, required=True)
    parser.add_argument("--stage6-registry", type=Path, required=True)
    parser.add_argument("--development-covariates", type=Path, required=True)
    parser.add_argument("--labels-2021a", type=Path, required=True)
    parser.add_argument("--auxiliary-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(stable_json(build_frozen_drift_reference(args.all_operating_points, args.baseline_lock, args.stage6_registry, args.development_covariates, args.labels_2021a, args.auxiliary_spec, args.output)))


if __name__ == "__main__": main()
