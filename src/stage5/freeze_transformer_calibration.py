#!/usr/bin/env python3
"""Freeze 2021A-only binary calibration and operating points for the main Transformer.

This program is deliberately downstream of model selection and upstream of
2021B conformal calibration.  It never accepts 2021B or 2022 rows.  Calibration
method selection is performed on a patient-disjoint deterministic split inside
2021A, after which the selected method is refit on all of 2021A.  The fixed 20%
resource-capacity threshold is then frozen for later untouched-cohort use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


ENDPOINTS = {
    "any_readmission": ("any_unplanned_readmission_30d", "p_any_readmission"),
    "ap_specific_readmission": ("readmission_leaf", "p_leaf_ap"),
}
METHOD_ORDER = ("none", "platt", "isotonic")
CAPACITY_FRACTION = 0.20
BRIER_SIMPLICITY_TOLERANCE = 0.0005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def validate_selected_model(manifest_path: Path, prediction_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022"
            or manifest.get("prediction_partition") != "2021A"
            or manifest.get("2021B_accessed") is not False
            or manifest.get("year_2022_accessed") is not False):
        raise RuntimeError("Selected Transformer is not frozen before 2021B/2022")
    artifact = manifest.get("prediction", {})
    if (artifact.get("sha256") != sha256(prediction_path)
            or int(artifact.get("bytes", -1)) != prediction_path.stat().st_size):
        raise RuntimeError("2021A prediction identity does not match the fine-tuning manifest")
    return manifest


def patient_calibration_split(patient_hash: np.ndarray) -> np.ndarray:
    values = np.asarray(patient_hash)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("patient_hash must be a non-empty one-dimensional array")
    try:
        hashes = values.astype(np.uint64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("patient_hash must be representable as uint64") from exc
    fit = ((hashes >> np.uint64(1)) & np.uint64(1)) == 0
    if fit.all() or (~fit).all():
        raise RuntimeError("Degenerate patient-level calibration split")
    return fit


def fit_calibrator(method: str, probability: np.ndarray, outcome: np.ndarray) -> Any:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(outcome, dtype=np.int8)
    if method == "none":
        return None
    if method == "platt":
        logit = np.log(p / (1 - p)).reshape(-1, 1)
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        return model.fit(logit, y)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(p, y)
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(method: str, calibrator: Any, probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    if method == "none":
        result = p
    elif method == "platt":
        result = calibrator.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        result = np.asarray(calibrator.predict(p), dtype=np.float64)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise RuntimeError("Calibrator produced invalid probabilities")
    return result


def binary_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(outcome, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if not set(np.unique(y)).issubset({0, 1}) or not np.isfinite(p).all():
        raise ValueError("Invalid binary outcome or probability")
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(np.clip(p, 1e-7, 1 - 1e-7))
                                      + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1 - 1e-7)))),
    }


def choose_calibrator(probability: np.ndarray, outcome: np.ndarray,
                      fit_mask: np.ndarray) -> tuple[str, Any, dict[str, Any]]:
    trials: dict[str, dict[str, float | int]] = {}
    for method in METHOD_ORDER:
        fitted = fit_calibrator(method, probability[fit_mask], outcome[fit_mask])
        selected = apply_calibrator(method, fitted, probability[~fit_mask])
        trials[method] = binary_metrics(outcome[~fit_mask], selected)
    best_brier = min(float(item["brier"]) for item in trials.values())
    eligible = [method for method in METHOD_ORDER
                if float(trials[method]["brier"]) <= best_brier + BRIER_SIMPLICITY_TOLERANCE]
    method = eligible[0]
    fitted_full = fit_calibrator(method, probability, outcome)
    report = {
        "nested_split_rule": "((patient_hash >> 1) & 1): 0=calibrator-fit, 1=method-selection",
        "fit_rows": int(fit_mask.sum()),
        "selection_rows": int((~fit_mask).sum()),
        "fit_events": int(outcome[fit_mask].sum()),
        "selection_events": int(outcome[~fit_mask].sum()),
        "selection_metric": "lowest Brier; prefer none then Platt then isotonic within 0.0005 absolute Brier",
        "trials": trials,
        "selected_method": method,
        "refit_on_full_2021A_after_selection": True,
    }
    return method, fitted_full, report


def capacity_constrained_flags(probability: np.ndarray, encounter_hash: np.ndarray,
                               fraction: float = CAPACITY_FRACTION) -> tuple[np.ndarray, dict[str, Any]]:
    p = np.asarray(probability, dtype=np.float64)
    hashes = np.asarray(encounter_hash)
    if p.ndim != 1 or hashes.ndim != 1 or len(p) != len(hashes) or len(p) == 0:
        raise ValueError("probability and encounter_hash must be non-empty, one-dimensional and equal-length")
    if not np.isfinite(p).all() or not 0 < fraction <= 1:
        raise ValueError("probability must be finite and fraction must lie in (0,1]")
    if len(np.unique(hashes)) != len(hashes):
        raise ValueError("encounter_hash must be unique")
    target = int(math.ceil(fraction * len(p)))
    ranked = np.lexsort((hashes, -p))
    flagged = np.zeros(len(p), dtype=bool)
    flagged[ranked[:target]] = True
    boundary = float(p[ranked[target - 1]])
    ties = p == boundary
    return flagged, {
        "capacity_fraction_requested": float(fraction),
        "capacity_rows_requested": target,
        "capacity_fraction_realized": float(flagged.mean()),
        "boundary_risk": boundary,
        "boundary_tie_rows": int(ties.sum()),
        "boundary_tie_rows_selected": int((flagged & ties).sum()),
        "tie_break_rule": "descending calibrated risk; ascending encounter_hash at an equal-risk boundary",
    }


def endpoint_outcome(frame: pd.DataFrame, endpoint: str, label_column: str) -> np.ndarray:
    if endpoint == "ap_specific_readmission":
        return (frame[label_column].astype(np.int64).to_numpy() == 1).astype(np.int8)
    return frame[label_column].astype(np.int8).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-2021a", type=Path, required=True)
    parser.add_argument("--finetune-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    predictions_path = args.predictions_2021a.resolve()
    manifest_path = args.finetune_manifest.resolve()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Calibration output directory is not empty; frozen artifacts are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    selected_manifest = validate_selected_model(manifest_path, predictions_path)

    required = ["year", "analysis_partition", "encounter_hash", "patient_hash",
                "any_unplanned_readmission_30d", "readmission_leaf",
                "p_any_readmission", "p_leaf_ap"]
    frame = pq.read_table(predictions_path, columns=required).to_pandas()
    if set(frame["year"].astype(int)) != {2021} or set(frame["analysis_partition"]) != {"2021A"}:
        raise RuntimeError("Calibration may use 2021A only")
    if frame["encounter_hash"].duplicated().any():
        raise RuntimeError("Duplicate 2021A encounters")
    fit_mask = patient_calibration_split(frame["patient_hash"].to_numpy())
    result = frame[["year", "analysis_partition", "encounter_hash", "patient_hash"]].copy()
    calibration_report: dict[str, Any] = {}
    threshold_report: dict[str, Any] = {}
    calibrators: dict[str, Any] = {}

    for endpoint, (label_column, probability_column) in ENDPOINTS.items():
        outcome = endpoint_outcome(frame, endpoint, label_column)
        probability = frame[probability_column].to_numpy(dtype=np.float64)
        method, calibrator, report = choose_calibrator(probability, outcome, fit_mask)
        calibrated = apply_calibrator(method, calibrator, probability)
        flags, capacity = capacity_constrained_flags(calibrated, frame["encounter_hash"].to_numpy())
        calibration_report[endpoint] = report
        threshold_report[endpoint] = {
            "probability_column": f"p_{endpoint}_calibrated",
            "rule": "fixed 20% capacity on 2021A; threshold transported unchanged",
            "threshold": capacity["boundary_risk"],
            "flagged_rows": int(flags.sum()),
            "flagged_fraction": float(flags.mean()),
            "sensitivity": float(((outcome == 1) & flags).sum() / max(1, outcome.sum())),
            "positive_predictive_value": float(((outcome == 1) & flags).sum() / max(1, flags.sum())),
            "capacity_selection": capacity,
        }
        result[f"p_{endpoint}_calibrated"] = calibrated
        result[f"flag_{endpoint}_capacity20"] = flags
        calibrators[endpoint] = {"method": method, "model": calibrator}

    prediction_out = out / "predictions_2021A_calibrated.parquet"
    calibrator_out = out / "transformer_binary_calibrators.joblib"
    threshold_out = out / "transformer_operating_thresholds_2021A.json"
    calibration_out = out / "transformer_calibration_selection_2021A.json"
    pq.write_table(pa.Table.from_pandas(result, preserve_index=False), prediction_out,
                   compression="zstd", compression_level=6)
    joblib.dump({"version": 1, "source_prediction_sha256": sha256(predictions_path),
                 "endpoints": calibrators}, calibrator_out, compress=3)
    threshold_out.write_text(stable_json(threshold_report), encoding="utf-8")
    calibration_out.write_text(stable_json(calibration_report), encoding="utf-8")
    manifest = {
        "status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A",
        "selection_partition": "2021A",
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "capacity_fraction": CAPACITY_FRACTION,
        "endpoints": list(ENDPOINTS),
        "fine_tuning_model_selection": selected_manifest["model_selection"],
        "source": {"file": str(predictions_path), "bytes": predictions_path.stat().st_size,
                   "sha256": sha256(predictions_path), "manifest": str(manifest_path),
                   "manifest_sha256": sha256(manifest_path)},
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (prediction_out, calibrator_out, threshold_out, calibration_out)
        },
    }
    manifest_out = out / "transformer_operating_point_manifest.json"
    manifest_out.write_text(stable_json(manifest), encoding="utf-8")
    print(stable_json(manifest))


if __name__ == "__main__":
    main()
