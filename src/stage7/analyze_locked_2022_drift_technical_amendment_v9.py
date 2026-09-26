#!/usr/bin/env python3
"""Fail-closed, descriptive four-domain drift analysis for the sealed 2022 test.

This is deliberately separate from :mod:`evaluate_locked_2022`.  It reads an
already materialized, standardized 2022 prediction table only after the
pre-2022 unlock lock has authenticated this module, the evaluation module, and
the frozen evaluation specification.  It cannot recalibrate a probability,
change a decision threshold, select a model, or refit a conformal quantile.

The evaluation specification must contain a hash-locked ``drift_reference``
object.  Its reference distributions, outcome rates, calibration statistics,
coverage summaries and bootstrap configuration must have been generated from
development/2021A/2021B before the 2022 lock.  Reference bins and categories
are never estimated from 2022 observations.

The resulting immutable directory contains five distinct reports: covariate,
label, calibration, and coverage drift plus fixed-seed label-rate bootstrap
replicates.  A combined p-value is intentionally not calculated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVALUATOR_PATH = Path(__file__).with_name("evaluate_locked_2022.py").resolve()
SCHEMA_VERSION = "stage7_locked_2022_drift_technical_amendment_v9"
REQUIRED_DRIFT_DOMAINS = ("covariates", "label_outcomes", "calibration", "coverage", "bootstrap")
CALIBRATION_METRICS = ("calibration_intercept", "calibration_slope", "brier", "log_loss")
RATE_METRICS = ("unweighted_rate", "DISCWT_weighted_rate")


def _load_evaluator() -> Any:
    spec = importlib.util.spec_from_file_location("stage7_locked_evaluator", EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load locked 2022 evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ev = _load_evaluator()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"Frozen drift reference {label} must be a finite number")
    return float(value)


def _fraction(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if not 0 <= result <= 1:
        raise RuntimeError(f"Frozen drift reference {label} must be in [0, 1]")
    return result


def _reference_distribution(value: Any, labels: list[str], name: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(labels):
        raise RuntimeError(f"Frozen reference weighted_distribution for {name} must name exactly its frozen bins/categories")
    result = {label: _fraction(value[label], f"{name}/{label}") for label in labels}
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"Frozen reference weighted_distribution for {name} must sum to one")
    return result


def _coverage_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (str(entry["conformal_set"]), str(entry["subgroup_dimension"]), str(entry["subgroup_value"]))


def validate_drift_reference(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate the entirely pre-specified reference contract.

    ``label_outcomes`` are deliberately separate from the evaluator's two
    co-primary outcomes: exactly seven named, fully observed binary outcomes
    are required for temporal label-shift reporting.
    """
    reference = spec.get("drift_reference")
    if not isinstance(reference, dict) or reference.get("status") != "FROZEN_2022_DRIFT_REFERENCE":
        raise RuntimeError("Evaluation specification lacks FROZEN_2022_DRIFT_REFERENCE")
    if any(name not in reference for name in REQUIRED_DRIFT_DOMAINS):
        raise RuntimeError("Frozen drift reference is missing a required domain")

    covariates = reference["covariates"]
    if not isinstance(covariates, dict) or not covariates:
        raise RuntimeError("Frozen drift reference has no covariates")
    for name, item in covariates.items():
        if not isinstance(name, str) or not name or not isinstance(item, dict):
            raise RuntimeError("Invalid frozen covariate entry")
        if not isinstance(item.get("column"), str) or not item["column"]:
            raise RuntimeError(f"Frozen covariate {name} lacks a source column")
        kind = item.get("kind")
        if kind not in ("categorical", "numeric_binned"):
            raise RuntimeError(f"Frozen covariate {name} must be categorical or numeric_binned")
        if kind == "categorical":
            categories = item.get("categories")
            if not isinstance(categories, list) or not categories or any(not isinstance(x, (str, int, float)) for x in categories):
                raise RuntimeError(f"Frozen categorical covariate {name} has invalid categories")
            labels = [str(x) for x in categories] + ["<UNSEEN_OR_OTHER>"]
        else:
            bins = item.get("bins")
            if not isinstance(bins, list) or len(bins) < 2:
                raise RuntimeError(f"Frozen numeric covariate {name} needs at least two bin edges")
            numbers = [_finite_number(edge, f"{name}/bin") for edge in bins]
            if any(right <= left for left, right in zip(numbers, numbers[1:])):
                raise RuntimeError(f"Frozen numeric covariate {name} bin edges must be strictly increasing")
            labels = _bin_labels(numbers) + ["<NONFINITE_OR_OUT_OF_RANGE>"]
        _fraction(item.get("reference_missing_rate"), f"{name}/reference_missing_rate")
        _reference_distribution(item.get("reference_weighted_distribution"), labels, name)
        if "reference_weighted_mean" in item or "reference_weighted_sd" in item:
            mean = _finite_number(item.get("reference_weighted_mean"), f"{name}/reference_weighted_mean")
            sd = _finite_number(item.get("reference_weighted_sd"), f"{name}/reference_weighted_sd")
            if sd <= 0:
                raise RuntimeError(f"Frozen covariate {name} reference_weighted_sd must be positive")
            del mean  # Explicitly inspected; avoids an accidental partial SMD contract.

    outcomes = reference["label_outcomes"]
    if not isinstance(outcomes, dict) or len(outcomes) != 7:
        raise RuntimeError("Frozen drift reference must name exactly seven label outcomes")
    for outcome, item in outcomes.items():
        if not isinstance(outcome, str) or not outcome or not isinstance(item, dict) or not isinstance(item.get("column"), str):
            raise RuntimeError("Invalid frozen label outcome entry")
        for metric in RATE_METRICS:
            _fraction(item.get(metric), f"label/{outcome}/{metric}")
        validity = item.get("validity")
        if outcome == "high_cost":
            # Older synthetic/fully-observed reference fixtures may omit this
            # field.  They are accepted only as a *complete-label* contract;
            # nullable 2022 high-cost labels require the explicit new policy.
            if validity is not None:
                if not isinstance(validity, dict) or validity.get("policy") != "EXCLUDE_NULL_HIGH_COST_LABELS_FROM_ALL_RATES_AND_BOOTSTRAPS":
                    raise RuntimeError("Frozen high-cost reference has invalid null-label policy")
                if not isinstance(validity.get("small_cell_counts_suppressed"), bool):
                    raise RuntimeError("Frozen high-cost reference lacks small-cell suppression declaration")
                for field in ("reference_valid_n", "reference_missing_n", "reference_valid_weight"):
                    value = validity.get(field)
                    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0 or not math.isfinite(float(value))):
                        raise RuntimeError(f"Frozen high-cost reference has invalid {field}")
        elif validity is not None:
            raise RuntimeError("Only high_cost may have nullable-label validity metadata")

    calibration = reference["calibration"]
    if not isinstance(calibration, dict) or set(calibration) != set(spec["models"]):
        raise RuntimeError("Frozen calibration reference must contain exactly every frozen model")
    for model_name, by_outcome in calibration.items():
        if not isinstance(by_outcome, dict) or set(by_outcome) != set(ev.OUTCOME_ORDER):
            raise RuntimeError(f"Frozen calibration reference for {model_name} must contain both co-primary outcomes")
        for outcome, metrics in by_outcome.items():
            if not isinstance(metrics, dict):
                raise RuntimeError(f"Invalid frozen calibration reference for {model_name}/{outcome}")
            for metric in CALIBRATION_METRICS:
                _finite_number(metrics.get(metric), f"calibration/{model_name}/{outcome}/{metric}")

    coverage = reference["coverage"]
    if not isinstance(coverage, list) or not coverage:
        raise RuntimeError("Frozen drift reference has no conformal coverage rows")
    seen: set[tuple[str, str, str]] = set()
    for item in coverage:
        if not isinstance(item, dict) or not {"conformal_set", "subgroup_dimension", "subgroup_value", "coverage", "mean_set_size", "abstention_rate"}.issubset(item):
            raise RuntimeError("Invalid frozen coverage reference row")
        key = _coverage_key(item)
        if key in seen:
            raise RuntimeError("Duplicate frozen coverage reference row")
        seen.add(key)
        _fraction(item["coverage"], "coverage/coverage")
        _finite_number(item["mean_set_size"], "coverage/mean_set_size")
        _fraction(item["abstention_rate"], "coverage/abstention_rate")
    required_sets = {entry["name"] for entry in spec["conformal_sets"]}
    if not required_sets.issubset({key[0] for key in seen}):
        raise RuntimeError("Frozen coverage reference omits a required global or Mondrian conformal set")

    bootstrap = reference["bootstrap"]
    if not isinstance(bootstrap, dict):
        raise RuntimeError("Frozen drift bootstrap configuration is invalid")
    replicates = bootstrap.get("n_replicates")
    seed = bootstrap.get("seed")
    max_threads = bootstrap.get("max_threads")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise RuntimeError("Frozen drift bootstrap requires at least 100 exact replicates")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RuntimeError("Frozen drift bootstrap seed must be an integer")
    if isinstance(max_threads, bool) or not isinstance(max_threads, int) or not 1 <= max_threads <= 8:
        raise RuntimeError("Frozen drift maximum thread count must be 1..8")
    if bootstrap.get("cluster_column", "patient_hash") != "patient_hash":
        raise RuntimeError("Drift bootstrap must be patient_hash clustered; NRD_STRATUM is not a hospital")
    return reference


def _bin_labels(edges: list[float]) -> list[str]:
    return [f"[{left:g},{right:g}{']' if index == len(edges) - 2 else ')'}" for index, (left, right) in enumerate(zip(edges, edges[1:]))]


def _weighted_distribution(labels: np.ndarray, weights: np.ndarray, expected: list[str]) -> dict[str, float]:
    totals = {label: 0.0 for label in expected}
    for label, weight in zip(labels, weights):
        totals[str(label)] += float(weight)
    denominator = float(weights.sum())
    return {label: totals[label] / denominator for label in expected}


def _psi(observed: dict[str, float], reference: dict[str, float]) -> float:
    epsilon = 1e-12
    return float(sum((observed[key] - reference[key]) * math.log((observed[key] + epsilon) / (reference[key] + epsilon)) for key in reference))


def covariate_drift(frame: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weights = frame["DISCWT"].to_numpy(dtype=float)
    for name, item in reference["covariates"].items():
        raw = frame[item["column"]]
        missing = raw.isna().to_numpy()
        if item["kind"] == "categorical":
            frozen = [str(value) for value in item["categories"]]
            labels = frozen + ["<UNSEEN_OR_OTHER>"]
            values = raw.astype("string").fillna("<UNSEEN_OR_OTHER>").astype(str).to_numpy()
            labels_array = np.where(np.isin(values, frozen), values, "<UNSEEN_OR_OTHER>")
            weighted_mean = weighted_sd = smd = None
        else:
            edges = [_finite_number(value, f"{name}/bin") for value in item["bins"]]
            labels = _bin_labels(edges) + ["<NONFINITE_OR_OUT_OF_RANGE>"]
            numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
            index = np.searchsorted(np.asarray(edges), numeric, side="right") - 1
            valid = np.isfinite(numeric) & (numeric >= edges[0]) & (numeric <= edges[-1])
            index = np.minimum(index, len(edges) - 2)
            labels_array = np.asarray([labels[value] if good else "<NONFINITE_OR_OUT_OF_RANGE>" for value, good in zip(index, valid)], dtype=object)
            good = np.isfinite(numeric)
            weighted_mean = float(np.dot(numeric[good], weights[good]) / weights[good].sum()) if good.any() else None
            weighted_sd = float(np.sqrt(np.dot(weights[good], (numeric[good] - weighted_mean) ** 2) / weights[good].sum())) if good.any() else None
            if weighted_mean is not None and "reference_weighted_mean" in item:
                smd = float((weighted_mean - float(item["reference_weighted_mean"])) / float(item["reference_weighted_sd"]))
            else:
                smd = None
        observed = _weighted_distribution(labels_array, weights, labels)
        ref = {key: float(value) for key, value in item["reference_weighted_distribution"].items()}
        for label in labels:
            rows.append({"domain": "covariate", "covariate": name, "source_column": item["column"], "kind": item["kind"],
                         "frozen_level": label, "reference_weighted_proportion": ref[label], "observed_DISCWT_weighted_proportion": observed[label],
                         "proportion_change": observed[label] - ref[label], "population_stability_index": _psi(observed, ref),
                         "reference_missing_rate": float(item["reference_missing_rate"]), "observed_missing_rate": float(missing.mean()),
                         "missing_rate_change": float(missing.mean()) - float(item["reference_missing_rate"]),
                         "observed_weighted_mean": weighted_mean, "observed_weighted_sd": weighted_sd,
                         "standardized_mean_difference": smd, "analysis": "DESCRIPTIVE_NO_RETUNING"})
    return pd.DataFrame(rows)


def _weighted_wilson(rate: float, weights: np.ndarray) -> tuple[float | None, float | None, str]:
    effective_n = float(weights.sum() ** 2 / np.dot(weights, weights))
    if effective_n <= 0:
        return None, None, "UNDEFINED_EFFECTIVE_N"
    lower, upper, _, status = ev.wilson_95_interval(int(round(rate * effective_n)), int(round(effective_n)))
    return lower, upper, f"KISH_EFFECTIVE_N_WILSON_95:{status}"


def _label_validity(frame: pd.DataFrame, outcome: str, item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return label values and the frozen eligible set, without imputing nulls."""
    numeric = pd.to_numeric(frame[item["column"]], errors="coerce").to_numpy(dtype=float)
    if outcome == "high_cost":
        valid = np.isfinite(numeric)
        if valid.any() and not np.isin(numeric[valid], (0.0, 1.0)).all():
            raise RuntimeError("High-cost labels must be null or binary 0/1")
        if not valid.any():
            raise RuntimeError("2022 high-cost labels have no valid rows")
        if item.get("validity") is None and not valid.all():
            raise RuntimeError("High-cost labels may be nullable only under the frozen null-label exclusion policy")
        return numeric, valid
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise RuntimeError(f"Outcome {outcome} must be fully observed binary 0/1")
    return numeric, np.ones(len(numeric), dtype=bool)


def _public_count(value: int) -> tuple[int | None, bool]:
    return (None, True) if value <= 10 else (int(value), False)


def _bootstrap_rates(frame: pd.DataFrame, outcomes: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    all_clusters = frame["patient_hash"].astype(str).to_numpy()
    weights = frame["DISCWT"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for outcome_index, (outcome, details) in enumerate(outcomes.items()):
        numeric, valid = _label_validity(frame, outcome, details)
        y = numeric[valid].astype(np.int8); outcome_weights = weights[valid]; clusters = all_clusters[valid]
        unique, inverse = np.unique(clusters, return_inverse=True)
        memberships = [np.flatnonzero(inverse == item) for item in range(len(unique))]
        rng = np.random.default_rng(int(config["seed"]) + outcome_index * 1009)
        for replicate in range(1, int(config["n_replicates"]) + 1):
            sampled = rng.integers(0, len(unique), size=len(unique))
            positions = np.concatenate([memberships[index] for index in sampled])
            sampled_y, sampled_w = y[positions], outcome_weights[positions]
            rows.append({"outcome": outcome, "cluster_unit": "patient_hash", "seed": int(config["seed"]) + outcome_index * 1009,
                         "replicate": replicate, "unweighted_rate": float(sampled_y.mean()),
                         "DISCWT_weighted_rate": float(np.dot(sampled_y, sampled_w) / sampled_w.sum()),
                         "valid_n": int(len(sampled_y)), "valid_weight": float(sampled_w.sum()), "status": "OK"})
    return pd.DataFrame(rows)


def label_drift(frame: pd.DataFrame, reference: dict[str, Any], bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weights = frame["DISCWT"].to_numpy(dtype=float)
    for outcome, item in reference["label_outcomes"].items():
        numeric, valid = _label_validity(frame, outcome, item)
        y, outcome_weights = numeric[valid].astype(np.int8), weights[valid]
        valid_n, missing_n = int(valid.sum()), int((~valid).sum())
        reported_valid_n, valid_n_suppressed = _public_count(valid_n)
        reported_missing_n, missing_n_suppressed = _public_count(missing_n)
        reported_events, event_suppressed = _public_count(int(y.sum()))
        for metric, observed, low, high, method in (
            ("unweighted_rate", float(y.mean()), *ev.wilson_95_interval(int(y.sum()), int(len(y)))[:2], "WILSON_95_UNWEIGHTED"),
            ("DISCWT_weighted_rate", float(np.dot(y, outcome_weights) / outcome_weights.sum()), *_weighted_wilson(float(np.dot(y, outcome_weights) / outcome_weights.sum()), outcome_weights)[:2], "KISH_EFFECTIVE_N_WILSON_95"),
        ):
            reps = bootstrap.loc[bootstrap["outcome"].eq(outcome), metric].to_numpy(dtype=float)
            # Only high-cost may be nullable.  Its descriptive output never
            # publishes raw n/event counts when they are an HCUP small cell.
            nullable = outcome == "high_cost" and item.get("validity") is not None
            rows.append({"domain": "label", "outcome": outcome, "source_column": item["column"], "weighting": metric,
                         "n_rows": int(len(y)) if not nullable else reported_valid_n, "event_count": int(y.sum()) if not nullable else reported_events, "event_tier": ev._event_tier(int(y.sum())),
                         "valid_n": int(len(y)) if not nullable else reported_valid_n, "missing_n": 0 if not nullable else reported_missing_n,
                         "valid_weight": float(outcome_weights.sum()), "valid_n_suppressed": bool(nullable and valid_n_suppressed),
                         "missing_n_suppressed": bool(nullable and missing_n_suppressed), "event_count_suppressed": bool(nullable and event_suppressed),
                         "high_cost_null_label_policy": item.get("validity", {}).get("policy") if nullable else "NOT_APPLICABLE",
                         "reference_rate": float(item[metric]), "observed_rate": observed, "rate_change": observed - float(item[metric]),
                         "ci95_lower": low, "ci95_upper": high, "ci95_method": method,
                         "bootstrap_ci95_lower": float(np.quantile(reps, .025)), "bootstrap_ci95_upper": float(np.quantile(reps, .975)),
                         "analysis": "DESCRIPTIVE_NO_MODEL_CHANGE"})
    return pd.DataFrame(rows)


def calibration_drift(frame: pd.DataFrame, spec: dict[str, Any], reference: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    weights = frame["DISCWT"].to_numpy(dtype=float)
    for model_name, model in spec["models"].items():
        for outcome in ev.OUTCOME_ORDER:
            y = frame[spec["co_primary_outcomes"][outcome]].to_numpy(dtype=np.int8)
            p = frame[model["probabilities"][outcome]].to_numpy(dtype=float)
            observed = ev.binary_metrics(y, p, weights)
            for metric in CALIBRATION_METRICS:
                value = observed[metric]
                ref = float(reference["calibration"][model_name][outcome][metric])
                rows.append({"domain": "calibration", "model": model_name, "outcome": outcome, "weighting": "DISCWT_weighted",
                             "metric": metric, "reference_value": ref, "observed_value": value,
                             "change_from_reference": None if value is None else float(value - ref), "calibration_status": observed["calibration_status"],
                             "analysis": "DESCRIPTIVE_NO_RECALIBRATION"})
    return pd.DataFrame(rows)


def _predeclared_coverage_level(spec: dict[str, Any], reference: dict[str, Any], dimension: str, value: str) -> bool:
    """Return whether a newly observed coverage level was frozen in advance.

    A coverage stratum is reportable without a matching reference row only when
    its source column is a prespecified categorical covariate and the exact
    level (after the evaluator's string representation) is in that covariate's
    frozen category list.  This deliberately does not coalesce HCUP sentinels
    (for example -8 and -9) and it fails closed for any level not present in
    the frozen category contract.
    """
    source_column = spec.get("subgroups", {}).get(dimension)
    if not isinstance(source_column, str) or not source_column:
        return False
    for item in reference.get("covariates", {}).values():
        if item.get("column") != source_column or item.get("kind") != "categorical":
            continue
        categories = item.get("categories")
        if not isinstance(categories, list):
            return False
        return str(value) in {str(category) for category in categories}
    return False


def coverage_drift(frame: pd.DataFrame, spec: dict[str, Any], reference: dict[str, Any]) -> pd.DataFrame:
    overall, subgroups = ev.evaluate_conformal(frame, spec)
    observed_rows: list[dict[str, Any]] = []
    for _, item in overall.iterrows():
        observed_rows.append({"conformal_set": item["conformal_set"], "subgroup_dimension": "ALL", "subgroup_value": "ALL", **item.to_dict()})
    observed_rows.extend(item.to_dict() for _, item in subgroups.iterrows())
    ref = {_coverage_key(item): item for item in reference["coverage"]}
    observed = {_coverage_key(item): item for item in observed_rows}
    unexpected = set(observed) - set(ref)
    if unexpected:
        # A missing reference row is reportable only for an exact category
        # explicitly declared in the frozen covariate contract.  Completely
        # unpredeclared levels remain fail-closed.
        undeclared = [key for key in sorted(unexpected)
                      if not _predeclared_coverage_level(spec, reference, key[1], key[2])]
        if undeclared:
            raise RuntimeError(f"Frozen coverage reference lacks exact 2022 reporting stratum: {undeclared[0]}")
    rows: list[dict[str, Any]] = []
    # Retain reference strata absent from the 2022 table as explicit undefined
    # rows rather than silently dropping a prespecified subgroup.
    for key, baseline in ref.items():
        item = observed.get(key)
        is_empty = item is None
        rows.append({"domain": "coverage", "conformal_set": key[0], "subgroup_dimension": key[1], "subgroup_value": key[2],
                     "nominal_coverage": None if is_empty else item.get("nominal_coverage"), "scope": None if is_empty else item.get("scope"), "mondrian_dimension": None if is_empty else item.get("mondrian_dimension"),
                     "n_rows": 0 if is_empty else item.get("n_rows"), "event_count": 0 if is_empty else item.get("event_count"), "event_tier": ev._event_tier(0 if is_empty else int(item.get("event_count", 0))),
                     "coverage_reference": float(baseline["coverage"]), "coverage_observed": None if is_empty else item.get("coverage"), "coverage_change": None if is_empty or item.get("coverage") is None else float(item["coverage"] - baseline["coverage"]),
                     "mean_set_size_reference": float(baseline["mean_set_size"]), "mean_set_size_observed": None if is_empty else item.get("mean_set_size"),
                     "mean_set_size_change": None if is_empty or item.get("mean_set_size") is None else float(item["mean_set_size"] - baseline["mean_set_size"]),
                     "abstention_rate_reference": float(baseline["abstention_rate"]), "abstention_rate_observed": None if is_empty else item.get("abstention_rate"),
                     "abstention_rate_change": None if is_empty or item.get("abstention_rate") is None else float(item["abstention_rate"] - baseline["abstention_rate"]),
                     "coverage_ci95_lower": None if is_empty else item.get("coverage_ci95_lower"), "coverage_ci95_upper": None if is_empty else item.get("coverage_ci95_upper"),
                     "coverage_ci95_method": "UNDEFINED_EMPTY_SUBGROUP" if is_empty else item.get("coverage_ci95_method"), "status": "UNDEFINED_EMPTY_SUBGROUP" if is_empty else item.get("status"), "analysis": "DESCRIPTIVE_NO_CONFORMAL_RECALIBRATION"})
    # Frozen categories that occur in 2022 without a corresponding pre-2022
    # coverage row are retained as independent report-only strata.  Reference
    # and change fields remain null; -8 and -9, for example, stay distinct.
    for key in sorted(unexpected):
        item = observed[key]
        rows.append({"domain": "coverage", "conformal_set": key[0], "subgroup_dimension": key[1], "subgroup_value": key[2],
                     "nominal_coverage": item.get("nominal_coverage"), "scope": item.get("scope"), "mondrian_dimension": item.get("mondrian_dimension"),
                     "n_rows": item.get("n_rows"), "event_count": item.get("event_count"), "event_tier": ev._event_tier(int(item.get("event_count", 0))),
                     "coverage_reference": None, "coverage_observed": item.get("coverage"), "coverage_change": None,
                     "mean_set_size_reference": None, "mean_set_size_observed": item.get("mean_set_size"), "mean_set_size_change": None,
                     "abstention_rate_reference": None, "abstention_rate_observed": item.get("abstention_rate"), "abstention_rate_change": None,
                     "coverage_ci95_lower": item.get("coverage_ci95_lower"), "coverage_ci95_upper": item.get("coverage_ci95_upper"),
                     "coverage_ci95_method": item.get("coverage_ci95_method"), "status": "NO_FROZEN_REFERENCE_NEW_STRATUM",
                     "analysis": "DESCRIPTIVE_NO_CONFORMAL_RECALIBRATION"})
    return pd.DataFrame(rows)

def _validate_drift_prediction_frame(frame: pd.DataFrame, spec: dict[str, Any], reference: dict[str, Any]) -> None:
    ev.validate_prediction_frame(frame, spec)
    required = {item["column"] for item in reference["covariates"].values()} | {item["column"] for item in reference["label_outcomes"].values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction table lacks frozen drift columns: {missing}")
    for outcome, item in reference["label_outcomes"].items():
        _label_validity(frame, outcome, item)
    # Explicitly assert the protected hospital identifier is available.  It is
    # never replaced by NRD_STRATUM in this drift module.
    if "hospital_hash" not in frame or frame["hospital_hash"].isna().any():
        raise RuntimeError("Standardized prediction table must preserve hospital_hash; NRD_STRATUM is not a substitute")


def _prepare_output(output: Path) -> Path:
    output = output.resolve()
    partials = list(output.parent.glob(output.name + ".partial*")) if output.parent.exists() else []
    if output.exists() or partials:
        raise RuntimeError("Locked 2022 drift output or partial output already exists and is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def analyze_locked_2022_drift(predictions_path: Path, unlock_path: Path, spec_path: Path, output: Path) -> dict[str, Any]:
    """Run only the locked, descriptive drift protocol once.

    All identity and schema gates execute before the prediction parquet is
    opened.  This function has no parameters capable of retuning a model.
    """
    own_path = Path(__file__).resolve()
    unlock = ev.validate_unlock_lock(unlock_path, EVALUATOR_PATH, [spec_path, own_path])
    spec = ev.validate_evaluation_spec(spec_path, unlock)
    reference = validate_drift_reference(spec)
    temporary = _prepare_output(output)
    try:
        predictions_path = predictions_path.resolve()
        if not predictions_path.is_file():
            raise RuntimeError("Missing standardized 2022 prediction parquet")
        frame = pd.read_parquet(predictions_path, engine="pyarrow")
        _validate_drift_prediction_frame(frame, spec, reference)
        bootstrap = _bootstrap_rates(frame, reference["label_outcomes"], reference["bootstrap"])
        tables = {
            "covariate_drift": covariate_drift(frame, reference),
            "label_drift": label_drift(frame, reference, bootstrap),
            "label_rate_bootstrap": bootstrap,
            "calibration_drift": calibration_drift(frame, spec, reference),
            "coverage_drift": coverage_drift(frame, spec, reference),
        }
        artifacts: dict[str, dict[str, Any]] = {}
        for name, table in tables.items():
            path = temporary / f"{name}.parquet"
            _write_parquet(table, path)
            artifacts[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest = {"status": "PASS_LOCKED_2022_DRIFT_ANALYSIS", "schema_version": SCHEMA_VERSION,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "prediction": {"path": str(predictions_path), "bytes": predictions_path.stat().st_size, "sha256": sha256(predictions_path)},
                    "unlock_lock": {"path": str(unlock_path.resolve()), "sha256": sha256(unlock_path.resolve())},
                    "evaluation_spec": {"path": str(spec_path.resolve()), "sha256": sha256(spec_path.resolve())},
                    "drift_analyzer": {"path": str(own_path), "bytes": own_path.stat().st_size, "sha256": sha256(own_path)},
                    "evaluator": {"path": str(EVALUATOR_PATH), "bytes": EVALUATOR_PATH.stat().st_size, "sha256": sha256(EVALUATOR_PATH)},
                    "n_rows": int(len(frame)), "four_separate_domains": ["covariate", "label", "calibration", "coverage"],
                    "bootstrap": reference["bootstrap"], "no_posthoc_recalibration_threshold_adjustment_or_model_selection": True,
                    "posthoc_changes_require_independent_registry": True, "technical_amendment": {"id": "V9_COVERAGE_PREDECLARED_NEW_STRATA_REPORT_ONLY", "predeclared_categories_only": True, "do_not_merge_categories": True, "model_probability_threshold_conformal_unchanged": True}, "artifacts": artifacts}
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(stable_json(manifest), encoding="utf-8")
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
    print(stable_json(analyze_locked_2022_drift(args.predictions_2022, args.unlock_lock, args.evaluation_spec, args.output_dir)))


if __name__ == "__main__":
    main()


