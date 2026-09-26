#!/usr/bin/env python3
"""Fail-closed evaluator for the single, locked 2022 temporal test.

This module is deliberately *data agnostic*: it accepts only an already
materialized standardized prediction parquet and JSON manifests.  It never
constructs a cohort, opens an NRD extract, generates predictions, calibrates a
probability, selects a model, or searches a threshold.

Required prediction-table schema
--------------------------------
The table must contain one row per ``encounter_hash`` and exactly one 2022 test
partition (``analysis_year == 2022`` and ``analysis_partition == "test"``).
The required common columns are ``encounter_hash``, ``patient_hash``,
``hospital_hash``, ``DISCWT``, ``analysis_year``, and ``analysis_partition``.
The frozen evaluation specification names two binary outcome columns and, for
every frozen model, one already-calibrated probability column per outcome.  It
also supplies the previously frozen 2021A threshold for each such pair.

For hierarchical conformal reporting, ``primary_leaf`` is one of
``none, ap, biliary, sepsis_or_organ, other``.  Each frozen conformal entry
names an already materialized uint bit-mask column (bit order follows that
leaf order) and, optionally, a size column.  The implementation never derives
or changes a conformal threshold.  The four prespecified subgroup source
columns are supplied by the frozen specification; missing values are retained
as the literal subgroup ``<MISSING>`` rather than dropped.

Frozen evaluation-spec JSON schema (minimal):
```
{
  "status": "FROZEN_2022_EVALUATION_SPEC", "sealed_test_year": 2022,
  "co_primary_outcomes": {"any_readmission": "outcome_any_readmission",
                           "ap_specific_readmission": "outcome_ap_specific_readmission"},
  "models": {"transformer": {"probabilities": {"any_readmission": "...", ...},
                              "thresholds": {"any_readmission": 0.2, ...}}, ...},
  "primary_comparison": {"transformer": "transformer", "lightgbm": "lightgbm"},
  "conformal_sets": [{"name": "global_90", "scope": "global", "nominal": 0.90,
                       "mask_column": "leaf_set_mask_90",
                       "size_column": "leaf_set_size_90", "abstain_column": "abstain_90"},
                      {"name": "sex_90", "scope": "mondrian", "mondrian_dimension": "sex", ...}, ...],
  "subgroups": {"sex": "FEMALE", "age": "age_group", "PAY1": "PAY1",
                "ZIPINC_QRTL": "ZIPINC_QRTL"}
}
```
The specification itself is treated as a frozen manifest and must be SHA256
listed in the pre-2022 unlock lock's ``frozen_code`` map.  This makes a changed
threshold/model roster fail before the prediction parquet is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


LEAVES = ("none", "ap", "biliary", "sepsis_or_organ", "other")
LEAF_TO_BIT = {leaf: bit for bit, leaf in enumerate(LEAVES)}
OUTCOME_ORDER = ("any_readmission", "ap_specific_readmission")
DECISION_THRESHOLDS = tuple(round(i / 100, 2) for i in range(1, 51))
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20220913
SCHEMA_VERSION = "stage7_locked_2022_evaluation_v1"


def sha256(path: Path) -> str:
    """Return a streaming SHA256 digest for a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _listed_identity(lock: dict[str, Any], path: Path) -> bool:
    """Return true only when exact resolved path, bytes, and digest are locked."""
    resolved = path.resolve()
    entry = lock.get("frozen_code", {}).get(str(resolved))
    return bool(
        isinstance(entry, dict)
        and path.is_file()
        and entry.get("sha256") == sha256(resolved)
        and int(entry.get("bytes", -1)) == resolved.stat().st_size
    )


def validate_unlock_lock(unlock_path: Path, evaluator_path: Path,
                         frozen_manifest_paths: Iterable[Path] = ()) -> dict[str, Any]:
    """Validate unlock status, sidecar, and exact frozen evaluator/manifests.

    This must run before reading a test prediction file.  In particular, a
    matching filename without matching content is not sufficient.
    """
    unlock_path = unlock_path.resolve()
    evaluator_path = evaluator_path.resolve()
    if not unlock_path.is_file():
        raise RuntimeError("Missing pre-2022 unlock lock")
    sidecar = unlock_path.with_suffix(unlock_path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError("Missing immutable pre-2022 unlock SHA256 sidecar")
    expected = f"{sha256(unlock_path)}  {unlock_path.name}"
    if sidecar.read_text(encoding="ascii").strip() != expected:
        raise RuntimeError("Pre-2022 unlock lock SHA256 sidecar mismatch")
    lock = read_json_object(unlock_path)
    if (
        lock.get("status") != "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
        or lock.get("sealed_test_year") != 2022
        or lock.get("2022_access_before_lock") is not False
    ):
        raise RuntimeError("Pre-2022 unlock lock is not authorized for one 2022 primary evaluation")
    if not _listed_identity(lock, evaluator_path):
        raise RuntimeError("Evaluator path/bytes/SHA256 are not frozen in the unlock lock")
    for path in frozen_manifest_paths:
        if not _listed_identity(lock, path.resolve()):
            raise RuntimeError(f"Frozen manifest is not hash-locked: {path}")
    return lock


def validate_evaluation_spec(spec_path: Path, unlock_lock: dict[str, Any]) -> dict[str, Any]:
    """Validate the evaluator's immutable model/threshold/conformal contract."""
    spec_path = spec_path.resolve()
    if not _listed_identity(unlock_lock, spec_path):
        raise RuntimeError("Frozen evaluation specification is not hash-locked in unlock lock")
    spec = read_json_object(spec_path)
    if spec.get("status") != "FROZEN_2022_EVALUATION_SPEC" or spec.get("sealed_test_year") != 2022:
        raise RuntimeError("Evaluation specification is not frozen for the 2022 test")
    outcomes = spec.get("co_primary_outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != set(OUTCOME_ORDER) or not all(
        isinstance(outcomes[name], str) and outcomes[name] for name in OUTCOME_ORDER
    ):
        raise RuntimeError("Specification must name exactly both co-primary outcome columns in fixed order")
    models = spec.get("models")
    if not isinstance(models, dict) or not models:
        raise RuntimeError("Specification has no frozen models")
    for model_name, model in models.items():
        if not isinstance(model_name, str) or not model_name or not isinstance(model, dict):
            raise RuntimeError("Invalid frozen model entry")
        probabilities, thresholds = model.get("probabilities"), model.get("thresholds")
        if not isinstance(probabilities, dict) or not isinstance(thresholds, dict):
            raise RuntimeError(f"Model {model_name} is missing frozen probabilities or thresholds")
        for outcome in OUTCOME_ORDER:
            probability, threshold = probabilities.get(outcome), thresholds.get(outcome)
            if not isinstance(probability, str) or not probability:
                raise RuntimeError(f"Model {model_name} lacks probability for {outcome}")
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise RuntimeError(f"Model {model_name} lacks valid frozen threshold for {outcome}")
    comparison = spec.get("primary_comparison")
    if not isinstance(comparison, dict) or comparison.get("transformer") not in models or comparison.get("lightgbm") not in models:
        raise RuntimeError("Specification primary Transformer-vs-LightGBM comparison is invalid")
    conformal = spec.get("conformal_sets")
    if not isinstance(conformal, list) or not conformal:
        raise RuntimeError("Specification must include frozen global and Mondrian conformal sets")
    observed_conformal: set[tuple[str, float, str | None]] = set()
    observed_names: set[str] = set()
    for entry in conformal:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise RuntimeError("Invalid conformal entry")
        if entry["name"] in observed_names:
            raise RuntimeError("Duplicate frozen conformal entry name")
        observed_names.add(entry["name"])
        nominal = entry.get("nominal")
        if nominal not in (0.8, 0.9, 0.95):
            raise RuntimeError("Only prespecified 80/90/95% conformal coverages are permitted")
        scope = entry.get("scope")
        dimension = entry.get("mondrian_dimension")
        if scope not in ("global", "mondrian"):
            raise RuntimeError("Conformal entry must declare global or Mondrian scope")
        if scope == "global" and dimension is not None:
            raise RuntimeError("Global conformal entries may not declare a Mondrian dimension")
        if scope == "mondrian" and dimension not in ("sex", "age", "PAY1", "ZIPINC_QRTL"):
            raise RuntimeError("Mondrian conformal entry has invalid dimension")
        if not isinstance(entry.get("mask_column"), str):
            raise RuntimeError("Conformal entry missing materialized mask column")
        if entry.get("size_column") is not None and not isinstance(entry.get("size_column"), str):
            raise RuntimeError("Invalid conformal size column")
        if entry.get("abstain_column") is not None and not isinstance(entry.get("abstain_column"), str):
            raise RuntimeError("Invalid conformal abstention column")
        key = (scope, float(nominal), dimension)
        if key in observed_conformal:
            raise RuntimeError("Duplicate frozen conformal evaluation contract")
        observed_conformal.add(key)
    for nominal in (0.8, 0.9, 0.95):
        if ("global", nominal, None) not in observed_conformal:
            raise RuntimeError("Every prespecified nominal coverage requires a global conformal entry")
        for dimension in ("sex", "age", "PAY1", "ZIPINC_QRTL"):
            if ("mondrian", nominal, dimension) not in observed_conformal:
                raise RuntimeError("Every nominal coverage requires all prespecified Mondrian entries")
    subgroups = spec.get("subgroups", {})
    if set(subgroups) != {"sex", "age", "PAY1", "ZIPINC_QRTL"} or not all(
        isinstance(value, str) and value for value in subgroups.values()
    ):
        raise RuntimeError("Specification must include sex, age, PAY1, and ZIPINC_QRTL subgroup columns")
    return spec


def _required_columns(spec: dict[str, Any]) -> set[str]:
    columns = {"encounter_hash", "patient_hash", "hospital_hash", "DISCWT", "analysis_year", "analysis_partition"}
    columns.update(spec["co_primary_outcomes"].values())
    columns.update(spec["subgroups"].values())
    for model in spec["models"].values():
        columns.update(model["probabilities"].values())
    if spec["conformal_sets"]:
        columns.add("primary_leaf")
        for entry in spec["conformal_sets"]:
            columns.add(entry["mask_column"])
            if entry.get("size_column"):
                columns.add(entry["size_column"])
            if entry.get("abstain_column"):
                columns.add(entry["abstain_column"])
    return columns


def _binary(values: pd.Series, name: str) -> np.ndarray:
    if values.isna().any() or not values.isin([0, 1, False, True]).all():
        raise RuntimeError(f"Outcome {name} must be fully observed binary 0/1")
    return values.astype(np.int8).to_numpy()


def _bitcount(values: np.ndarray) -> np.ndarray:
    return np.asarray([int(value).bit_count() for value in values], dtype=np.int8)


def validate_prediction_frame(frame: pd.DataFrame, spec: dict[str, Any]) -> None:
    """Fail closed on an invalid or mixed standardized 2022 prediction table."""
    missing = sorted(_required_columns(spec) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction table missing required columns: {missing}")
    if frame.empty:
        raise RuntimeError("Prediction table is empty")
    for column in ("encounter_hash", "patient_hash", "hospital_hash"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"Prediction table has missing {column}")
    if frame["encounter_hash"].duplicated().any():
        raise RuntimeError("Prediction table has duplicate encounter_hash values")
    years = pd.to_numeric(frame["analysis_year"], errors="coerce")
    if years.isna().any() or not years.eq(2022).all():
        raise RuntimeError("Prediction table contains non-2022 or mixed test years")
    if frame["analysis_partition"].isna().any() or not frame["analysis_partition"].astype(str).eq("test").all():
        raise RuntimeError("Prediction table contains non-test or mixed partitions")
    weights = pd.to_numeric(frame["DISCWT"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise RuntimeError("DISCWT must be finite and strictly positive")
    for outcome, column in spec["co_primary_outcomes"].items():
        _binary(frame[column], outcome)
    for model_name, model in spec["models"].items():
        for outcome, column in model["probabilities"].items():
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise RuntimeError(f"Probability {column} for {model_name}/{outcome} is not finite in [0, 1]")
    if spec["conformal_sets"]:
        if frame["primary_leaf"].isna().any() or not frame["primary_leaf"].astype(str).isin(LEAVES).all():
            raise RuntimeError("primary_leaf must be fully observed and belong to the frozen hierarchy")
        for entry in spec["conformal_sets"]:
            masks = pd.to_numeric(frame[entry["mask_column"]], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(masks).all() or not np.equal(masks, np.floor(masks)).all() or (masks < 0).any() or (masks > 31).any():
                raise RuntimeError(f"Invalid conformal mask column {entry['mask_column']}")
            computed = _bitcount(masks.astype(np.uint8))
            size_column = entry.get("size_column")
            if size_column:
                sizes = pd.to_numeric(frame[size_column], errors="coerce").to_numpy(dtype=float)
                if not np.isfinite(sizes).all() or not np.equal(sizes, computed).all():
                    raise RuntimeError(f"Conformal set size does not match mask: {size_column}")
            abstain_column = entry.get("abstain_column")
            if abstain_column:
                if frame[abstain_column].isna().any() or not frame[abstain_column].isin([0, 1, False, True]).all():
                    raise RuntimeError(f"Invalid frozen conformal abstention column {abstain_column}")
                # A materialized conflict rule may abstain more often, but it
                # may never override the pre-specified non-singleton abstain.
                if ((computed != 1) & ~frame[abstain_column].astype(bool).to_numpy()).any():
                    raise RuntimeError(f"Frozen abstention column violates non-singleton abstention: {abstain_column}")


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    return None if total <= 0 else float(np.dot(values, weights) / total)


def _calibration(y: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> tuple[float | None, float | None, str]:
    """Intercept/slope from the fixed calibration model; never refits predictions."""
    if len(np.unique(y)) < 2:
        return None, None, "UNDEFINED_SINGLE_OUTCOME_CLASS"
    from sklearn.linear_model import LogisticRegression

    logit = np.log(np.clip(probabilities, 1e-15, 1 - 1e-15) / np.clip(1 - probabilities, 1e-15, 1))
    try:
        # sklearn >=1.8 deprecates an explicit ``penalty`` argument.  Its
        # documented compatibility form is the default L2 formulation with
        # infinite C, which is effectively unpenalized without a warning.
        model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)
        model.fit(logit.reshape(-1, 1), y, sample_weight=weights)
    except (TypeError, ValueError, OverflowError):
        try:
            # Fallback for older sklearn releases that reject infinite C.
            model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
            model.fit(logit.reshape(-1, 1), y, sample_weight=weights)
        except (TypeError, ValueError, OverflowError):
            try:
                model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000)
                model.fit(logit.reshape(-1, 1), y, sample_weight=weights)
            except (TypeError, ValueError, OverflowError) as exc:
                return None, None, f"UNDEFINED_CALIBRATION_FIT:{type(exc).__name__}"
    return float(model.intercept_[0]), float(model.coef_[0, 0]), "OK"


def binary_metrics(y: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Return fixed-model performance estimates or explicit undefined statuses."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probabilities, dtype=float)
    w = np.asarray(weights, dtype=float)
    prevalence = _weighted_mean(y.astype(float), w)
    brier = _weighted_mean((p - y) ** 2, w)
    logloss = _weighted_mean(-(y * np.log(np.clip(p, 1e-15, 1 - 1e-15)) + (1 - y) * np.log(np.clip(1 - p, 1e-15, 1 - 1e-15))), w)
    if len(np.unique(y)) < 2:
        auroc = auprc = None
        discrimination_status = "UNDEFINED_SINGLE_OUTCOME_CLASS"
    else:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auroc = float(roc_auc_score(y, p, sample_weight=w))
        auprc = float(average_precision_score(y, p, sample_weight=w))
        discrimination_status = "OK"
    intercept, slope, calibration_status = _calibration(y, p, w)
    return {
        "n_rows": int(len(y)), "weight_sum": float(w.sum()), "event_count": int(y.sum()),
        "weighted_event_count": float(np.dot(y, w)), "prevalence": prevalence,
        "auroc": auroc, "auprc": auprc, "brier": brier, "log_loss": logloss,
        "calibration_intercept": intercept, "calibration_slope": slope,
        "discrimination_status": discrimination_status, "calibration_status": calibration_status,
    }


def threshold_metrics(y: np.ndarray, probabilities: np.ndarray, weights: np.ndarray, threshold: float) -> dict[str, Any]:
    flagged = np.asarray(probabilities >= threshold, dtype=bool)
    y = np.asarray(y, dtype=bool)
    w = np.asarray(weights, dtype=float)
    tp = float(w[flagged & y].sum()); fp = float(w[flagged & ~y].sum())
    tn = float(w[~flagged & ~y].sum()); fn = float(w[~flagged & y].sum())
    return {
        "threshold": float(threshold), "flagged_fraction": _safe_divide(float(w[flagged].sum()), float(w.sum())),
        "sensitivity": _safe_divide(tp, tp + fn), "specificity": _safe_divide(tn, tn + fp),
        "ppv": _safe_divide(tp, tp + fp), "npv": _safe_divide(tn, tn + fn),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "status": "OK",
    }


def decision_curve(y: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> list[dict[str, Any]]:
    y = np.asarray(y, dtype=bool); p = np.asarray(probabilities, dtype=float); w = np.asarray(weights, dtype=float)
    denominator = float(w.sum())
    event_fraction = _safe_divide(float(w[y].sum()), denominator)
    rows: list[dict[str, Any]] = []
    for threshold in DECISION_THRESHOLDS:
        flagged = p >= threshold
        tp = float(w[flagged & y].sum()); fp = float(w[flagged & ~y].sum())
        odds = threshold / (1 - threshold)
        net_benefit = _safe_divide(tp - fp * odds, denominator)
        treat_all = None if event_fraction is None else float(event_fraction - (1 - event_fraction) * odds)
        rows.append({"threshold_probability": threshold, "net_benefit": net_benefit,
                     "treat_all_net_benefit": treat_all, "treat_none_net_benefit": 0.0,
                     "status": "OK"})
    return rows


def _weight_schemes(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"unweighted": np.ones(len(frame), dtype=float), "DISCWT_weighted": frame["DISCWT"].to_numpy(dtype=float)}


def evaluate_binary_models(frame: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute every prespecified model/outcome metric and fixed threshold report."""
    metric_rows: list[dict[str, Any]] = []
    dca_rows: list[dict[str, Any]] = []
    schemes = _weight_schemes(frame)
    for outcome in OUTCOME_ORDER:
        y = frame[spec["co_primary_outcomes"][outcome]].to_numpy(dtype=np.int8)
        for model_name, model in spec["models"].items():
            p = frame[model["probabilities"][outcome]].to_numpy(dtype=float)
            threshold = float(model["thresholds"][outcome])
            for weight_name, weights in schemes.items():
                row = {"outcome": outcome, "model": model_name, "weighting": weight_name,
                       **binary_metrics(y, p, weights), **{f"operating_{k}": v for k, v in threshold_metrics(y, p, weights, threshold).items()}}
                metric_rows.append(row)
                for dca in decision_curve(y, p, weights):
                    dca_rows.append({"outcome": outcome, "model": model_name, "weighting": weight_name, **dca})
    return pd.DataFrame(metric_rows), pd.DataFrame(dca_rows)


def _cluster_bootstrap_difference(frame: pd.DataFrame, y: np.ndarray, p_transformer: np.ndarray,
                                  p_lightgbm: np.ndarray, weights: np.ndarray, cluster_column: str,
                                  seed: int, n_replicates: int = BOOTSTRAP_REPLICATES) -> np.ndarray:
    """Paired cluster bootstrap retaining every row of selected clusters and multiplicity."""
    from sklearn.metrics import average_precision_score

    clusters = frame[cluster_column].astype(str).to_numpy()
    unique, inverse = np.unique(clusters, return_inverse=True)
    membership = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    estimates = np.full(n_replicates, np.nan, dtype=float)
    for replicate in range(n_replicates):
        sampled_clusters = rng.integers(0, len(unique), size=len(unique))
        positions = np.concatenate([membership[index] for index in sampled_clusters])
        sampled_y = y[positions]
        if np.unique(sampled_y).size < 2:
            continue
        estimates[replicate] = float(average_precision_score(sampled_y, p_transformer[positions], sample_weight=weights[positions]) -
                                     average_precision_score(sampled_y, p_lightgbm[positions], sample_weight=weights[positions]))
    return estimates


def _observed_auprc_difference(y: np.ndarray, p_transformer: np.ndarray,
                               p_lightgbm: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """The frozen 2022 effect estimate, distinct from bootstrap uncertainty."""
    if np.unique(y).size < 2:
        return {"transformer_auprc_observed": None, "lightgbm_auprc_observed": None,
                "auprc_difference_observed": None, "observed_status": "UNDEFINED_SINGLE_OUTCOME_CLASS"}
    from sklearn.metrics import average_precision_score
    transformer = float(average_precision_score(y, p_transformer, sample_weight=weights))
    lightgbm = float(average_precision_score(y, p_lightgbm, sample_weight=weights))
    return {"transformer_auprc_observed": transformer, "lightgbm_auprc_observed": lightgbm,
            "auprc_difference_observed": float(transformer - lightgbm), "observed_status": "OK"}


def paired_bootstrap(frame: pd.DataFrame, spec: dict[str, Any], cluster_column: str = "patient_hash",
                     n_replicates: int = BOOTSTRAP_REPLICATES, seed: int = BOOTSTRAP_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired AUPRC differences for both co-primary targets and both weightings."""
    if n_replicates != BOOTSTRAP_REPLICATES:
        raise RuntimeError(f"Primary evaluation requires exactly {BOOTSTRAP_REPLICATES} bootstrap replicates")
    comparison = spec["primary_comparison"]
    transformer = spec["models"][comparison["transformer"]]
    lightgbm = spec["models"][comparison["lightgbm"]]
    results: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(OUTCOME_ORDER):
        y = frame[spec["co_primary_outcomes"][outcome]].to_numpy(dtype=np.int8)
        p_t = frame[transformer["probabilities"][outcome]].to_numpy(dtype=float)
        p_l = frame[lightgbm["probabilities"][outcome]].to_numpy(dtype=float)
        for weighting_index, (weighting, weights) in enumerate(_weight_schemes(frame).items()):
            estimates = _cluster_bootstrap_difference(
                frame, y, p_t, p_l, weights, cluster_column,
                seed + outcome_index * 100 + weighting_index, n_replicates,
            )
            defined = estimates[np.isfinite(estimates)]
            status = "OK" if len(defined) == n_replicates else "UNDEFINED_SOME_BOOTSTRAP_REPLICATES_SINGLE_CLASS"
            result = {"comparison": f"{comparison['transformer']}_minus_{comparison['lightgbm']}",
                      "outcome": outcome, "weighting": weighting, "cluster_unit": cluster_column,
                      "seed": seed + outcome_index * 100 + weighting_index, "n_replicates": n_replicates,
                      "n_defined": int(len(defined)), **_observed_auprc_difference(y, p_t, p_l, weights),
                      "bootstrap_auprc_difference_mean": None if not len(defined) else float(np.mean(defined)),
                      "ci95_percentile_lower": None if not len(defined) else float(np.quantile(defined, 0.025)),
                      "ci95_percentile_upper": None if not len(defined) else float(np.quantile(defined, 0.975)), "status": status}
            results.append(result)
            replicate_rows.extend({"comparison": result["comparison"], "outcome": outcome, "weighting": weighting,
                                   "cluster_unit": cluster_column, "replicate": int(index + 1),
                                   "auprc_difference": None if not math.isfinite(value) else float(value),
                                   "status": "OK" if math.isfinite(value) else "UNDEFINED_SINGLE_OUTCOME_CLASS"}
                                  for index, value in enumerate(estimates))
    return pd.DataFrame(results), pd.DataFrame(replicate_rows)


def _event_tier(events: int) -> str:
    if events >= 100:
        return "CONFIRMATORY_GE_100_EVENTS"
    if events >= 50:
        return "EXPLORATORY_50_TO_99_EVENTS"
    return "DESCRIPTIVE_LT_50_EVENTS"


def wilson_95_interval(successes: int, total: int) -> tuple[float | None, float | None, str, str]:
    """Deterministic unweighted Wilson interval for a binomial coverage rate."""
    if total <= 0:
        return None, None, "WILSON_95_UNWEIGHTED", "UNDEFINED_EMPTY_SUBGROUP"
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return float(max(0.0, centre - radius)), float(min(1.0, centre + radius)), "WILSON_95_UNWEIGHTED", "OK"


def _conformal_summary(mask: np.ndarray, leaves: np.ndarray, abstain: np.ndarray | None = None) -> dict[str, Any]:
    size = _bitcount(mask)
    truth_bits = np.asarray([1 << LEAF_TO_BIT[str(leaf)] for leaf in leaves], dtype=np.uint8)
    covered = (mask & truth_bits) != 0
    abstained = size != 1 if abstain is None else np.asarray(abstain, dtype=bool)
    event = leaves != "none"
    singleton_prediction = np.asarray([LEAVES[int(value).bit_length() - 1] if value and int(value).bit_count() == 1 else None for value in mask], dtype=object)
    retained = ~abstained
    ci_lower, ci_upper, ci_method, ci_status = wilson_95_interval(int(covered.sum()), int(len(mask)))
    return {
        "n_rows": int(len(mask)), "event_count": int(event.sum()), "coverage": float(covered.mean()) if len(mask) else None,
        "coverage_ci95_lower": ci_lower, "coverage_ci95_upper": ci_upper,
        "coverage_ci95_method": ci_method, "coverage_ci95_status": ci_status,
        "mean_set_size": float(size.mean()) if len(size) else None, "median_set_size": float(np.median(size)) if len(size) else None,
        "singleton_rate": float((size == 1).mean()) if len(size) else None, "empty_set_rate": float((size == 0).mean()) if len(size) else None,
        "full_set_rate": float((size == len(LEAVES)).mean()) if len(size) else None, "abstention_rate": float(abstained.mean()) if len(size) else None,
        "retained_event_rate": _safe_divide(float(event[retained].sum()), float(retained.sum())),
        "abstained_event_rate": _safe_divide(float(event[abstained].sum()), float(abstained.sum())),
        "retained_error_rate": _safe_divide(float((singleton_prediction[retained] != leaves[retained]).sum()), float(retained.sum())),
        "abstained_set_error_rate": _safe_divide(float((~covered[abstained]).sum()), float(abstained.sum())),
        "status": "OK" if len(mask) else "UNDEFINED_EMPTY_SUBGROUP",
    }


def evaluate_conformal(frame: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen materialized global/Mondrian sets without re-calibration."""
    rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    if not spec["conformal_sets"]:
        return pd.DataFrame(columns=["conformal_set", "nominal_coverage"]), pd.DataFrame()
    leaves = frame["primary_leaf"].astype(str).to_numpy()
    for entry in spec["conformal_sets"]:
        masks = frame[entry["mask_column"]].to_numpy(dtype=np.uint8)
        abstain = frame[entry["abstain_column"]].to_numpy(dtype=bool) if entry.get("abstain_column") else None
        rows.append({"conformal_set": entry["name"], "nominal_coverage": float(entry["nominal"]),
                     "scope": entry["scope"], "mondrian_dimension": entry.get("mondrian_dimension"),
                     "mask_column": entry["mask_column"], "abstention_rule": "FROZEN_MATERIALIZED_COLUMN" if abstain is not None else "NON_SINGLETON_SET",
                     **_conformal_summary(masks, leaves, abstain)})
        for dimension, column in spec["subgroups"].items():
            groups = frame[column].astype("string").fillna("<MISSING>").astype(str)
            for value in sorted(groups.unique().tolist()):
                selected = groups.eq(value).to_numpy()
                detail = _conformal_summary(masks[selected], leaves[selected], None if abstain is None else abstain[selected])
                subgroup_rows.append({"conformal_set": entry["name"], "nominal_coverage": float(entry["nominal"]),
                                      "subgroup_dimension": dimension, "subgroup_value": value,
                                      "event_tier": _event_tier(int(detail["event_count"])), **detail})
    return pd.DataFrame(rows), pd.DataFrame(subgroup_rows)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def _prepare_output(output: Path) -> Path:
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    partials = list(output.parent.glob(output.name + ".partial*")) if output.parent.exists() else []
    if output.exists() or partial.exists() or partials:
        raise RuntimeError("Primary 2022 output or partial output already exists and is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))


def evaluate_locked_2022(predictions_path: Path, unlock_path: Path, spec_path: Path, output: Path,
                         ) -> dict[str, Any]:
    """Run the once-only evaluator after lock checks; this is the CLI's core."""
    # No prediction path operation is allowed before all lock/self identity gates succeed.
    evaluator_path = Path(__file__).resolve()
    unlock = validate_unlock_lock(unlock_path, evaluator_path, [spec_path])
    spec = validate_evaluation_spec(spec_path, unlock)
    temporary = _prepare_output(output)
    try:
        predictions_path = predictions_path.resolve()
        if not predictions_path.is_file():
            raise RuntimeError("Missing standardized 2022 prediction parquet")
        frame = pd.read_parquet(predictions_path, engine="pyarrow")
        validate_prediction_frame(frame, spec)
        metrics, decision = evaluate_binary_models(frame, spec)
        comparisons, replicates = paired_bootstrap(frame, spec, "patient_hash")
        tables: dict[str, pd.DataFrame] = {"metrics": metrics, "decision_curve": decision,
                                            "comparisons_patient_bootstrap": comparisons,
                                            "patient_bootstrap_replicates": replicates}
        # This prespecified sensitivity is mandatory: hospital_hash is already
        # a required standardized-prediction column, so there is no valid
        # production reason to omit it.
        hospital, hospital_replicates = paired_bootstrap(frame, spec, "hospital_hash")
        tables["comparisons_hospital_bootstrap"] = hospital
        tables["hospital_bootstrap_replicates"] = hospital_replicates
        conformal, subgroups = evaluate_conformal(frame, spec)
        tables["conformal"] = conformal
        tables["conformal_subgroups"] = subgroups
        artifacts: dict[str, dict[str, Any]] = {}
        for name, table in tables.items():
            path = temporary / f"{name}.parquet"
            _write_parquet(table, path)
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest = {
            "status": "PASS_LOCKED_2022_PRIMARY_EVALUATION", "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(), "prediction": {"path": str(predictions_path), "bytes": predictions_path.stat().st_size, "sha256": sha256(predictions_path)},
            "unlock_lock": {"path": str(unlock_path.resolve()), "sha256": sha256(unlock_path.resolve())},
            "evaluation_spec": {"path": str(spec_path.resolve()), "sha256": sha256(spec_path.resolve())},
            "evaluator": {"path": str(evaluator_path), "sha256": sha256(evaluator_path)},
            "n_rows": int(len(frame)), "models": list(spec["models"]), "co_primary_outcomes": list(OUTCOME_ORDER),
            "patient_bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
            "hospital_bootstrap_included": True, "decision_curve_thresholds": list(DECISION_THRESHOLDS),
            "no_tuning_or_recalibration": True, "artifacts": artifacts,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(stable_json(manifest), encoding="utf-8")
        # The manifest's immutable SHA256 is carried by its sidecar.  Avoid a
        # mathematically impossible self-referential digest in ``artifacts``.
        digest = sha256(manifest_path)
        (temporary / "manifest.json.sha256").write_text(f"{digest}  manifest.json\n", encoding="ascii")
        os.replace(temporary, output.resolve())
        return {**manifest, "manifest_sha256": digest, "output": str(output.resolve())}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-2022", type=Path, required=True)
    parser.add_argument("--unlock-lock", type=Path, required=True)
    parser.add_argument("--evaluation-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_locked_2022(args.predictions_2022, args.unlock_lock, args.evaluation_spec, args.output_dir)
    print(stable_json(result))


if __name__ == "__main__":
    main()
