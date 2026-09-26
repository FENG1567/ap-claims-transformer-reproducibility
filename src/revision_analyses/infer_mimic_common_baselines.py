#!/usr/bin/env python3
"""Aggregate-only frozen common-variable baseline comparison on MIMIC.

This program is deliberately an *inference-only* bridge.  It reconstructs the
post-hoc comparator design matrix mechanically from the feature object already
emitted by the locked MIMIC v7 runtime, applies registered baseline artifacts,
and evaluates them alongside the frozen common-variable Transformer.  It never
fits a model, estimates calibration parameters, selects a threshold, or writes
row-level MIMIC records or predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


SEED = 20260921
SPECIAL_MAX = 3
EXPECTED_BASELINE_STATUS = "PASS_POSTHOC_COMMON_VARIABLE_BASELINES_FROZEN_PRE_MIMIC"
EXPECTED_MODEL_ID = "common_variable_only"
OUTCOMES = {
    "any_readmission": {
        "label": "any_readmission",
        "transformer_probability": "p_any_readmission_calibrated",
    },
    "ap_specific_readmission": {
        "label": "AP_specific",
        "transformer_probability": "p_ap_specific_readmission_calibrated",
    },
}
BASELINE_MODELS = ("structured_logistic", "elastic_net", "lightgbm")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_concatenated_json(path: Path) -> list[dict[str, Any]]:
    """Read the locked pretty-printed object stream without assuming JSONL."""
    text = path.read_text(encoding="utf-8")
    decoder, pos, rows = json.JSONDecoder(), 0, []
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return rows
        value, pos = decoder.raw_decode(text, pos)
        if not isinstance(value, dict):
            raise RuntimeError("Expected an object stream in locked predictions")
        rows.append(value)


def tokens(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("Token input is not a list")
    result = []
    for item in value:
        if item is not None:
            result.append(int(item))
    return result


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def derive_timing_counts(pr_tokens: Any, prday: Any, los: float) -> dict[str, float]:
    names = ("prday_preadmission_count", "prday_day0_count", "prday_day1_2_count",
             "prday_day3plus_count", "prday_missing_count",
             "prday_invalid_after_discharge_count")
    result = Counter({name: 0 for name in names})
    day_values = tokens(prday)
    los = max(0, int(los)) if math.isfinite(los) else 0
    for token, day in zip(tokens(pr_tokens), day_values):
        if token <= SPECIAL_MAX:
            continue
        if day <= -90 or day == -66:
            result["prday_missing_count"] += 1
        elif day < 0:
            result["prday_preadmission_count"] += 1
        elif day > los:
            result["prday_invalid_after_discharge_count"] += 1
        elif day == 0:
            result["prday_day0_count"] += 1
        elif day <= 2:
            result["prday_day1_2_count"] += 1
        else:
            result["prday_day3plus_count"] += 1
    return {name: float(result[name]) for name in names}


def feature_row(record: dict[str, Any], numeric_features: list[str]) -> dict[str, float]:
    feature = record.get("features")
    if not isinstance(feature, dict):
        raise RuntimeError("Missing MIMIC feature object")
    required = {"AGE", "FEMALE", "LOS", "I10_NDX", "I10_NPR", "dx_tokens",
                "pr_tokens", "prday", "prior_dx_tokens_180d", "prior_pr_tokens_180d",
                "oov_audit"}
    if not required.issubset(feature):
        raise RuntimeError("Locked MIMIC feature object is incomplete")
    audit = feature["oov_audit"]
    if not isinstance(audit, dict) or set(("diagnosis", "procedure", "total")) - set(audit):
        raise RuntimeError("Locked MIMIC OOV audit is incomplete")
    dx, pr = tokens(feature["dx_tokens"]), tokens(feature["pr_tokens"])
    values = {name: finite_or_nan(feature.get(name)) for name in numeric_features}
    values.update({
        "dx_observed_count": float(sum(item > SPECIAL_MAX for item in dx)),
        "pr_observed_count": float(sum(item > SPECIAL_MAX for item in pr)),
        "dx_oov_count": float(sum(item == 2 for item in dx)),
        "pr_oov_count": float(sum(item == 2 for item in pr)),
    })
    values.update(derive_timing_counts(pr, feature["prday"], values["LOS"]))
    absent = set(numeric_features) - set(values)
    if absent:
        raise RuntimeError(f"Feature derivation omitted registered inputs: {sorted(absent)}")
    return values


def numeric_matrix(rows: list[dict[str, float]], numeric_features: list[str],
                   numeric_spec: dict[str, dict[str, float]]) -> tuple[np.ndarray, dict[str, int]]:
    columns, missing = [], {}
    for name in numeric_features:
        spec = numeric_spec.get(name)
        if not isinstance(spec, dict) or not {"mean", "median", "sd"}.issubset(spec):
            raise RuntimeError(f"Registered numeric specification is invalid: {name}")
        values = np.asarray([row[name] for row in rows], dtype=float)
        missing[name] = int((~np.isfinite(values)).sum())
        values = np.where(np.isfinite(values), values, float(spec["median"]))
        sd = float(spec["sd"])
        if not math.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"Registered numeric scale is invalid: {name}")
        columns.append((values - float(spec["mean"])) / sd)
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite comparator design matrix")
    return matrix, missing


def token_matrix(records: list[dict[str, Any]], field: str, map_values: list[int]) -> sparse.csr_matrix:
    mapping = {int(token): position for position, token in enumerate(map_values)}
    rr: list[int] = []
    cc: list[int] = []
    for row_number, record in enumerate(records):
        feature = record["features"]
        for token in set(tokens(feature[field])):
            if token in mapping:
                rr.append(row_number)
                cc.append(mapping[token])
    return sparse.csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)),
                             shape=(len(records), len(mapping)), dtype=np.float32)


def full_matrix(records: list[dict[str, Any]], numeric: np.ndarray,
                token_fields: list[str], token_maps: dict[str, list[int]]) -> sparse.csr_matrix:
    blocks: list[sparse.spmatrix] = [sparse.csr_matrix(numeric)]
    for field in token_fields:
        if field not in token_maps or not isinstance(token_maps[field], list):
            raise RuntimeError(f"Missing registered token map: {field}")
        blocks.append(token_matrix(records, field, token_maps[field]))
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def apply_platt(raw: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    intercept, slope = float(parameters["intercept"]), float(parameters["slope"])
    raw = np.clip(np.asarray(raw, dtype=float), 1e-7, 1 - 1e-7)
    result = expit(intercept + slope * np.log(raw / (1 - raw)))
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite frozen calibrated probability")
    return result


def predict_linear(path: Path, matrix: sparse.csr_matrix) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    coefficients, intercept = np.asarray(payload["coef"], float), np.asarray(payload["intercept"], float)
    if coefficients.ndim != 2 or coefficients.shape[0] != 1 or coefficients.shape[1] != matrix.shape[1]:
        raise RuntimeError(f"Linear model dimensionality mismatch: {path.name}")
    raw_logit = np.asarray(matrix @ coefficients[0]).ravel() + float(intercept.ravel()[0])
    return expit(raw_logit)


def calibration_fit(y: np.ndarray, p: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """Weighted logistic calibration fit used solely as an evaluation metric."""
    x = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    b0, b1 = 0.0, 1.0
    for _ in range(30):
        mean = expit(b0 + b1 * x)
        variance = np.clip(mean * (1 - mean), 1e-9, None)
        g0, g1 = np.sum(weight * (y - mean)), np.sum(weight * (y - mean) * x)
        h00, h01, h11 = (np.sum(weight * variance), np.sum(weight * variance * x),
                         np.sum(weight * variance * x * x))
        determinant = h00 * h11 - h01 * h01
        if determinant <= 1e-12:
            break
        d0 = (h11 * g0 - h01 * g1) / determinant
        d1 = (-h01 * g0 + h00 * g1) / determinant
        b0, b1 = b0 + d0, b1 + d1
        if max(abs(d0), abs(d1)) < 1e-9:
            break
    return float(b0), float(b1)


def metrics(y: np.ndarray, p: np.ndarray, weight: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    weight = np.ones(len(y), dtype=float) if weight is None else np.asarray(weight, dtype=float)
    intercept, slope = calibration_fit(y, p, weight)
    observed, expected = np.average(y, weights=weight), np.average(p, weights=weight)
    return {
        "n": int(len(y)), "events": int(y.sum()), "prevalence": float(observed),
        "auroc": float(roc_auc_score(y, p, sample_weight=weight)),
        "auprc": float(average_precision_score(y, p, sample_weight=weight)),
        "brier": float(np.average((y - p) ** 2, weights=weight)),
        "log_loss": float(log_loss(y, p, sample_weight=weight, labels=[0, 1])),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "observed_expected_ratio": float(observed / expected),
    }


def operating(y: np.ndarray, p: np.ndarray, threshold: float,
              weight: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    weight = np.ones(len(y), dtype=float) if weight is None else np.asarray(weight, dtype=float)
    flagged = np.asarray(p, dtype=float) >= threshold
    tp = float(np.sum(weight * (flagged & (y == 1))))
    fp = float(np.sum(weight * (flagged & (y == 0))))
    tn = float(np.sum(weight * (~flagged & (y == 0))))
    fn = float(np.sum(weight * (~flagged & (y == 1))))
    total = tp + fp + tn + fn
    return {
        "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp), "npv": tn / (tn + fn),
        "alerts_per_1000": 1000 * (tp + fp) / total,
    }


def bootstrap_weights(subject_ids: np.ndarray, repetitions: int) -> np.ndarray:
    unique_subjects, inverse = np.unique(subject_ids, return_inverse=True)
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(len(unique_subjects), np.repeat(1 / len(unique_subjects), len(unique_subjects)),
                           size=repetitions).astype(np.int16)
    return counts[:, inverse].astype(np.float64, copy=False)


def confidence_interval(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(array, [0.025, 0.975]))


def bootstrap_metric_frame(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for replicate_weight in weights:
        if replicate_weight[y == 1].sum() and replicate_weight[y == 0].sum():
            rows.append(metrics(y, p, replicate_weight))
    return pd.DataFrame(rows)


def add_ci(point: dict[str, Any], bootstrap: pd.DataFrame, names: Iterable[str]) -> dict[str, Any]:
    result = dict(point)
    for name in names:
        low, high = confidence_interval(bootstrap[name] if name in bootstrap else [])
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = low, high
    return result


def require_baseline_bundle(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    needed = ("manifest.json", "common_feature_spec.json", "calibration_parameters.json", "operating_thresholds.json")
    if not path.is_dir() or any(not (path / name).is_file() for name in needed):
        raise RuntimeError("Frozen baseline bundle is incomplete")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != EXPECTED_BASELINE_STATUS or manifest.get("mimic_outcomes_accessed") is not False:
        raise RuntimeError("Baseline bundle is not registered as pre-MIMIC frozen")
    expected = {entry["name"]: entry["sha256"] for entry in manifest.get("files", [])}
    for name, digest in expected.items():
        candidate = path / name
        if not candidate.is_file() or sha256(candidate) != digest:
            raise RuntimeError(f"Baseline bundle hash mismatch: {name}")
    return (manifest,
            json.loads((path / "common_feature_spec.json").read_text(encoding="utf-8")),
            json.loads((path / "calibration_parameters.json").read_text(encoding="utf-8")),
            json.loads((path / "operating_thresholds.json").read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reps", type=int, default=5000)
    args = parser.parse_args()
    if args.reps != 5000:
        raise SystemExit("Formal comparison requires exactly 5000 subject bootstrap replicates")
    output = args.output_dir.resolve()
    if output.exists() or list(output.parent.glob(output.name + ".partial-*")):
        raise SystemExit("Refusing to overwrite output or a previous partial output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        baseline_dir = args.baseline_dir.resolve()
        bundle_manifest, feature_spec, frozen_calibration, frozen_thresholds = require_baseline_bundle(baseline_dir)
        numeric_features = list(feature_spec.get("numeric_features", []))
        token_fields = list(feature_spec.get("token_fields", []))
        numeric_spec, token_maps = feature_spec.get("numeric_spec", {}), feature_spec.get("token_maps", {})
        if len(numeric_features) != 29 or token_fields != ["dx_tokens", "pr_tokens", "prior_dx_tokens_180d", "prior_pr_tokens_180d"]:
            raise RuntimeError("Frozen common-variable baseline feature identity mismatch")
        runtime_manifest = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
        records_all = read_concatenated_json(args.predictions)
        records = [record for record in records_all if record.get("label", {}).get("any_readmission") is not None]
        if len(records_all) != 1810 or len(records) != 1786:
            raise RuntimeError("Unexpected locked MIMIC endpoint gate count")
        if sum(record["label"]["any_readmission"] is None for record in records_all) != 24:
            raise RuntimeError("Unexpected algorithm-unknown exclusion count")
        if any(record.get("prediction", {}).get("model_id") != EXPECTED_MODEL_ID for record in records):
            raise RuntimeError("MIMIC Transformer model identity is not common_variable_only")
        if sum(int(record["label"]["any_readmission"]) for record in records) != 341:
            raise RuntimeError("Unexpected any-readmission event count")
        if sum(int(record["label"]["AP_specific"]) for record in records) != 163:
            raise RuntimeError("Unexpected AP-specific event count")
        if any(int(record["label"]["AP_specific"]) != int(record["label"]["leaf"] == "AP_specific") for record in records):
            raise RuntimeError("AP-specific leaf/label binding mismatch")

        feature_rows = [feature_row(record, numeric_features) for record in records]
        numeric, missing_by_feature = numeric_matrix(feature_rows, numeric_features, numeric_spec)
        combined = full_matrix(records, numeric, token_fields, token_maps)
        if combined.shape[1] != 29 + sum(len(token_maps[field]) for field in token_fields):
            raise RuntimeError("Frozen comparator matrix dimension mismatch")
        transformer: dict[str, np.ndarray] = {}
        predictions: dict[str, dict[str, np.ndarray]] = {}
        thresholds: dict[str, dict[str, float]] = {}
        for outcome, descriptor in OUTCOMES.items():
            transformer[outcome] = np.asarray([record["prediction"][descriptor["transformer_probability"]]
                                                for record in records], dtype=float)
            if not np.isfinite(transformer[outcome]).all():
                raise RuntimeError("Non-finite frozen Transformer probability")
            predictions[outcome] = {"common_variable_only Transformer": transformer[outcome]}
            thresholds[outcome] = {"common_variable_only Transformer": float(records[0]["prediction"]["thresholds"][outcome])}
            for model_name in BASELINE_MODELS:
                artifact = baseline_dir / (f"{outcome}__{model_name}.txt" if model_name == "lightgbm"
                                           else f"{outcome}__{model_name}.npz")
                if not artifact.is_file():
                    raise RuntimeError(f"Registered frozen model is missing: {artifact.name}")
                if model_name == "structured_logistic":
                    raw = predict_linear(artifact, sparse.csr_matrix(numeric))
                elif model_name == "elastic_net":
                    raw = predict_linear(artifact, combined)
                else:
                    raw = np.asarray(lgb.Booster(model_file=str(artifact)).predict(combined), dtype=float)
                calibrated = apply_platt(raw, frozen_calibration[outcome][model_name])
                predictions[outcome][model_name] = calibrated
                thresholds[outcome][model_name] = float(frozen_thresholds[outcome][model_name])

        subject_ids = np.asarray([str(record["subject_id"]) for record in records])
        weights = bootstrap_weights(subject_ids, args.reps)
        metric_names = ("auroc", "auprc", "brier", "log_loss", "calibration_intercept",
                        "calibration_slope", "observed_expected_ratio")
        discrimination_names = ("auroc", "auprc", "brier", "log_loss")
        overall_rows, difference_rows, operating_rows, oov_rows = [], [], [], []
        oov_mask = np.asarray([int(record["features"]["oov_audit"]["total"]) > 0 for record in records])
        for outcome, descriptor in OUTCOMES.items():
            y = np.asarray([int(record["label"][descriptor["label"]]) for record in records], dtype=np.int8)
            boot_by_model: dict[str, pd.DataFrame] = {}
            for model_name, probability in predictions[outcome].items():
                point = metrics(y, probability)
                boot = bootstrap_metric_frame(y, probability, weights)
                if len(boot) != args.reps:
                    raise RuntimeError("A formal subject bootstrap replicate was invalid")
                boot_by_model[model_name] = boot
                overall_rows.append(add_ci({"outcome": outcome, "model": model_name,
                                            "bootstrap_replicates": int(len(boot)),
                                            "bootstrap_unit": "subject", **point}, boot, metric_names))
                operation_point = operating(y, probability, thresholds[outcome][model_name])
                operation_boot = pd.DataFrame([operating(y, probability, thresholds[outcome][model_name], w)
                                               for w in weights])
                operating_rows.append(add_ci({"outcome": outcome, "model": model_name,
                                               "threshold": thresholds[outcome][model_name], **operation_point},
                                             operation_boot, operation_point.keys()))
                for oov_group, mask in (("OOV=0", ~oov_mask), ("OOV>0", oov_mask)):
                    ys, ps = y[mask], probability[mask]
                    if len(np.unique(ys)) != 2:
                        continue
                    subgroup_boot = bootstrap_metric_frame(ys, ps, weights[:, mask])
                    oov_rows.append(add_ci({"outcome": outcome, "model": model_name,
                                            "oov_group": oov_group, **metrics(ys, ps),
                                            "bootstrap_replicates": int(len(subgroup_boot)),
                                            "bootstrap_unit": "subject"}, subgroup_boot, metric_names))
            transformer_boot = boot_by_model["common_variable_only Transformer"]
            for model_name in BASELINE_MODELS:
                difference = boot_by_model[model_name][list(discrimination_names)].to_numpy() - transformer_boot[list(discrimination_names)].to_numpy()
                row: dict[str, Any] = {"outcome": outcome, "contrast": f"{model_name} minus common_variable_only Transformer",
                                        "bootstrap_replicates": args.reps, "bootstrap_unit": "paired subject"}
                point_base, point_transformer = metrics(y, predictions[outcome][model_name]), metrics(y, transformer[outcome])
                for index, name in enumerate(discrimination_names):
                    row[name] = point_base[name] - point_transformer[name]
                    row[f"{name}_ci_low"], row[f"{name}_ci_high"] = confidence_interval(difference[:, index])
                difference_rows.append(row)

        pd.DataFrame(overall_rows).to_csv(temporary / "mimic_common_baseline_overall_metrics_ci.csv", index=False)
        pd.DataFrame(difference_rows).to_csv(temporary / "mimic_common_baseline_paired_differences_ci.csv", index=False)
        pd.DataFrame(operating_rows).to_csv(temporary / "mimic_common_baseline_frozen_threshold_metrics_ci.csv", index=False)
        pd.DataFrame(oov_rows).to_csv(temporary / "mimic_common_baseline_oov_stratified_metrics_ci.csv", index=False)
        audit = {
            "status": "PASS_MIMIC_COMMON_VARIABLE_BASELINE_INFERENCE_AUDIT",
            "row_level_released": False,
            "inference_only": True,
            "mimic_fit_calibration_selection_or_threshold_updates": False,
            "formal_endpoint_gate": {"pre_gate_episodes": len(records_all), "algorithm_unknown_excluded": 24,
                                      "analyzed_episodes": len(records), "any_events": 341, "ap_specific_events": 163},
            "model_identity": EXPECTED_MODEL_ID,
            "baseline_feature_contract": {"numeric_feature_count": len(numeric_features),
                                           "token_fields": token_fields,
                                           "combined_matrix_columns": int(combined.shape[1]),
                                           "numeric_missing_before_frozen_imputation": missing_by_feature},
            "oov": {"episodes_oov_gt_zero": int(oov_mask.sum()), "episodes_oov_zero": int((~oov_mask).sum())},
            "input_hashes": {
                "locked_predictions": sha256(args.predictions),
                "locked_runtime_manifest": sha256(args.runtime_manifest),
                "baseline_manifest": sha256(baseline_dir / "manifest.json"),
                "baseline_common_feature_spec": sha256(baseline_dir / "common_feature_spec.json"),
                "baseline_calibration_parameters": sha256(baseline_dir / "calibration_parameters.json"),
                "baseline_operating_thresholds": sha256(baseline_dir / "operating_thresholds.json"),
            },
            "runtime_manifest_status": runtime_manifest.get("status"),
            "baseline_manifest_status": bundle_manifest.get("status"),
            "bootstrap": {"repetitions_requested": args.reps, "repetitions_valid": args.reps,
                          "unit": "subject", "seed": SEED},
        }
        (temporary / "mimic_common_baseline_feature_binding_audit.json").write_text(stable_json(audit), encoding="utf-8")
        release = {"status": "PASS_RESTRICTED_MIMIC_COMMON_VARIABLE_BASELINE_COMPARISON",
                   "row_level_released": False, "inference_only": True, "files": []}
        for path in sorted(temporary.glob("*")):
            if path.is_file():
                release["files"].append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        (temporary / "manifest.json").write_text(stable_json(release), encoding="utf-8")
        os.replace(temporary, output)
        print(stable_json({"status": release["status"], "output": str(output)}))
    except Exception:
        # Preserve the atomic partial as a non-overwritable audit trail.
        raise


if __name__ == "__main__":
    main()
