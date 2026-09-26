#!/usr/bin/env python3
"""Fail-closed audit of the repaired Stage 4 baseline artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


TARGETS = ("any_readmission", "ap_specific_readmission")
MODELS = ("structured_logistic", "elastic_net", "lightgbm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, output = args.baseline_dir.resolve(), args.output.resolve()
    failures: list[str] = []

    lock = load_json(baseline / "baseline_lock.json")
    selection = load_json(baseline / "model_selection_2021A.json")
    thresholds = load_json(baseline / "operating_thresholds_2021A.json")
    calibration = load_json(baseline / "calibration_selection_2021A.json")
    feature_spec = load_json(baseline / "baseline_feature_spec.json")

    expected_rows = 63_368
    if lock.get("validation_2021A_rows") != expected_rows:
        failures.append("lock validation row count is not 63,368")
    if lock.get("2021B_outcomes_accessed") is not False or lock.get("year_2022_accessed") is not False:
        failures.append("sealed-partition flags are not false")
    if lock.get("development_rows") != 404_528:
        failures.append("development row count is not 404,528")
    if "supersed" not in json.dumps(lock, ensure_ascii=False).lower():
        failures.append("lock does not identify the superseded SGD implementation")

    artifact_results = {}
    for relative, registered in lock.get("artifacts", {}).items():
        path = baseline / relative
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha = sha256(path) if exists else None
        match = bool(exists and actual_bytes == registered.get("bytes") and actual_sha == registered.get("sha256"))
        artifact_results[relative] = {"exists": exists, "bytes": actual_bytes, "sha256": actual_sha, "match": match}
        if not match:
            failures.append(f"artifact identity mismatch: {relative}")

    prediction_path = baseline / "predictions_2021A.parquet"
    prediction = pq.read_table(prediction_path).to_pandas()
    if len(prediction) != expected_rows:
        failures.append("prediction table row count mismatch")
    if prediction["encounter_hash"].duplicated().any():
        failures.append("duplicate encounter_hash in predictions")
    if set(prediction["year"].astype(int)) != {2021}:
        failures.append("predictions are not exclusively year 2021")
    probability_columns = [name for name in prediction if name.startswith(("p_raw__", "p_cal__"))]
    expected_probability_columns = 2 * len(TARGETS) * len(MODELS)
    if len(probability_columns) != expected_probability_columns:
        failures.append("unexpected number of prediction probability columns")
    probability_diagnostics = {}
    for name in probability_columns:
        values = prediction[name].to_numpy(dtype=np.float64)
        finite = bool(np.isfinite(values).all())
        in_range = bool(((values >= 0) & (values <= 1)).all()) if finite else False
        standard_deviation = float(np.std(values)) if finite else None
        nondegenerate = bool(finite and standard_deviation > 1e-8)
        probability_diagnostics[name] = {
            "finite": finite, "in_range": in_range, "standard_deviation": standard_deviation,
            "nondegenerate": nondegenerate,
        }
        if not (finite and in_range and nondegenerate):
            failures.append(f"invalid or degenerate predictions: {name}")

    model_diagnostics = {}
    for target in TARGETS:
        model_diagnostics[target] = {}
        for model in MODELS:
            report = selection[target][model]
            metrics = report["metrics"]
            prevalence = float(metrics["prevalence"])
            auprc, auroc = float(metrics["auprc"]), float(metrics["auroc"])
            selected_validity = report.get(
                "selected_numerical_validity", report.get("numerical_validity", {"valid_for_selection": True})
            )
            valid = bool(selected_validity.get("valid_for_selection", False))
            if model == "elastic_net":
                convergence = report.get("selected_convergence", {})
                valid = valid and bool(convergence.get("converged", False))
                if report.get("implementation") != ("LogisticRegression(solver='saga', penalty='elasticnet'); "
                                                     "sparse direct objective optimization; no resampling"):
                    failures.append(f"unexpected elastic-net implementation for {target}")
            useful_direction = bool(auroc > 0.5 and auprc > prevalence)
            model_diagnostics[target][model] = {
                "auprc": auprc, "auroc": auroc, "prevalence": prevalence,
                "valid_for_selection": valid, "useful_direction": useful_direction,
            }
            if not valid:
                failures.append(f"selected candidate is numerically invalid: {target}/{model}")
            if not useful_direction:
                failures.append(f"selected candidate fails direction gate: {target}/{model}")

            threshold = thresholds[target][model]
            expected_flagged = math.ceil(0.20 * expected_rows)
            if threshold.get("flagged_rows") != expected_flagged:
                failures.append(f"capacity is not exact for {target}/{model}")
            if not math.isclose(float(threshold.get("flagged_fraction")), expected_flagged / expected_rows,
                                rel_tol=0.0, abs_tol=1e-12):
                failures.append(f"capacity fraction mismatch for {target}/{model}")
            if model not in calibration.get(target, {}):
                failures.append(f"missing calibration report for {target}/{model}")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "baseline_directory": str(baseline),
        "sealed_scope": {"development_years": [2018, 2019, 2020], "selection_partition": "2021A",
                         "2021B_outcomes_accessed": False, "year_2022_accessed": False},
        "rows": {"development": lock.get("development_rows"), "validation_2021A": len(prediction),
                 "unique_encounters": int(prediction["encounter_hash"].nunique())},
        "feature_spec_status": feature_spec.get("status"),
        "implementation_revision": selection.get("implementation_revision"),
        "artifact_results": artifact_results,
        "probability_diagnostics": probability_diagnostics,
        "model_diagnostics": model_diagnostics,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
