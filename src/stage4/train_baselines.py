#!/usr/bin/env python3
"""Train and freeze leakage-safe structured, elastic-net, and LightGBM baselines.

Development is restricted to 2018-2020. 2021A is used for model selection,
calibration-method selection, and the fixed-capacity operating threshold.
2021B outcomes and all 2022 files are intentionally never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260912
DEV_YEARS = (2018, 2019, 2020)
VALIDATION_YEAR = 2021
SPECIAL_TOKEN_MAX = 3
TARGETS = {
    "any_readmission": "any_unplanned_readmission_30d",
    "ap_specific_readmission": "ap_specific_readmission_30d",
}

BASE_COLUMNS = [
    "year", "encounter_hash", "patient_hash", "DISCWT", "analysis_partition",
    "primary_analysis_eligible", "any_unplanned_readmission_30d",
    "ap_specific_readmission_30d", "AGE", "AWEEKEND", "DISPUNIFORM", "DMONTH",
    "DRG", "DRGVER", "ELECTIVE", "FEMALE", "HCUP_ED", "LOS", "MDC",
    "I10_NDX", "I10_NPR", "PAY1", "PL_NCHS", "RESIDENT", "TOTCHG",
    "ZIPINC_QRTL", "dx_tokens", "pr_tokens", "prday", "APRDRG",
    "APRDRG_Risk_Mortality", "APRDRG_Severity", "HOSP_BEDSIZE", "H_CONTRL",
    "HOSP_URCAT4", "HOSP_UR_TEACH", "N_DISC_U", "N_HOSP_U", "S_DISC_U",
    "S_HOSP_U", "TOTAL_DISC", "CCR_NRD", "WAGEINDEX", "cost_2021_usd",
]

HISTORY_COLUMNS = [
    "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d",
    "prior_ed_count_180d", "prior_nonelective_count_180d", "prior_ap_count_180d",
    "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d",
    "prior_los_sum_180d", "prior_max_severity_180d",
    "prior_max_mortality_risk_180d", "days_since_prior_discharge",
    "history_30d_fully_observable", "history_90d_fully_observable",
    "history_180d_fully_observable", "prior_dx_tokens_180d", "prior_pr_tokens_180d",
]

NUMERIC_COLUMNS = [
    "AGE", "LOS", "I10_NDX", "I10_NPR", "log_cost_2021_usd", "log_total_disc",
    "log_n_disc_u", "log_n_hosp_u", "log_s_disc_u", "log_s_hosp_u",
    "CCR_NRD", "WAGEINDEX", "dx_observed_count", "pr_observed_count",
    "dx_oov_count", "pr_oov_count", "prday_preadmission_count",
    "prday_day0_count", "prday_day1_2_count", "prday_day3plus_count",
    "prday_missing_count", "prday_invalid_after_discharge_count",
    "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d",
    "prior_ed_count_180d", "prior_nonelective_count_180d", "prior_ap_count_180d",
    "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d",
    "prior_los_sum_180d", "prior_max_severity_180d",
    "prior_max_mortality_risk_180d", "days_since_prior_discharge",
    "history_30d_fully_observable", "history_90d_fully_observable",
    "history_180d_fully_observable",
]

CATEGORICAL_COLUMNS = [
    "year", "AWEEKEND", "DISPUNIFORM", "DMONTH", "DRG", "DRGVER", "ELECTIVE",
    "FEMALE", "HCUP_ED", "MDC", "PAY1", "PL_NCHS", "RESIDENT", "ZIPINC_QRTL",
    "APRDRG", "APRDRG_Risk_Mortality", "APRDRG_Severity", "HOSP_BEDSIZE",
    "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH",
]

ELASTIC_GRID = [
    {"C": 0.005, "l1_ratio": 0.15},
    {"C": 0.010, "l1_ratio": 0.15},
    {"C": 0.010, "l1_ratio": 0.50},
    {"C": 0.050, "l1_ratio": 0.50},
    {"C": 0.050, "l1_ratio": 0.85},
]
ELASTIC_MAX_ITER = 1500
ELASTIC_TOL = 1e-3

LIGHTGBM_GRID = [
    {"num_leaves": 15, "min_child_samples": 100, "learning_rate": 0.05,
     "feature_fraction": 1.0, "bagging_fraction": 0.8, "lambda_l2": 5.0},
    {"num_leaves": 31, "min_child_samples": 100, "learning_rate": 0.05,
     "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l2": 1.0},
    {"num_leaves": 63, "min_child_samples": 200, "learning_rate": 0.03,
     "feature_fraction": 0.8, "bagging_fraction": 0.8, "lambda_l2": 5.0},
]

BASELINE_IMPLEMENTATION_REVISION = {
    "revision_id": "elastic_net_saga_stability_and_exact_capacity_v1",
    "supersedes": (
        "The pre-revision SGDClassifier elastic-net configuration, identified on 2021A as "
        "numerically unstable or degenerate, before any 2021B or 2022 access."
    ),
    "replacement": (
        "LogisticRegression(solver='saga', penalty='elasticnet') with a finite pre-specified "
        "C/l1_ratio grid, direct sparse logistic-objective optimization, convergence capture, "
        "and numerical-validity gates."
    ),
    "revision_data_scope": "2018-2020 development and 2021A model-selection only",
    "2021B_outcomes_accessed_for_revision": False,
    "year_2022_accessed_for_revision": False,
    "unchanged_components": [
        "LightGBM grid", "primary endpoints", "development years", "2021A model-selection partition",
        "2021B conformal-calibration partition", "2022 sealed temporal test set", "no-resampling policy",
    ],
    "capacity_policy_revision": (
        "A fixed 20% capacity set is selected by descending calibrated risk; ties at the boundary "
        "are resolved by ascending encounter_hash, rather than flagging every tied row."
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def read_partition(root: Path, history_dir: Path, year: int, partition: str | None) -> pd.DataFrame:
    path = root / "data" / "nrd" / f"year={year}" / "ap_episodes.parquet"
    filters: list[tuple[str, str, object]] = [("primary_analysis_eligible", "=", True)]
    if partition is not None:
        filters.append(("analysis_partition", "=", partition))
    table = pq.read_table(path, columns=BASE_COLUMNS, filters=filters)
    frame = table.to_pandas()
    hist_cols = ["encounter_hash", "patient_hash", "analysis_partition"] + HISTORY_COLUMNS
    history = pq.read_table(history_dir / f"ap_history_{year}.parquet", columns=hist_cols)
    if partition is not None:
        history = history.filter(pc.equal(history["analysis_partition"], partition))
    history_df = history.drop(["patient_hash", "analysis_partition"]).to_pandas()
    if history_df["encounter_hash"].duplicated().any():
        raise RuntimeError(f"Duplicate history keys for {year}")
    merged = frame.merge(history_df, on="encounter_hash", how="left", validate="one_to_one")
    if merged[HISTORY_COLUMNS].isna().all(axis=1).any():
        raise RuntimeError(f"Missing history join for {year}")
    return merged


def _tokens(values) -> list[int]:
    if values is None:
        return []
    return [int(x) for x in values if x is not None]


def engineer_static(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    for source, target in [
        ("cost_2021_usd", "log_cost_2021_usd"), ("TOTAL_DISC", "log_total_disc"),
        ("N_DISC_U", "log_n_disc_u"), ("N_HOSP_U", "log_n_hosp_u"),
        ("S_DISC_U", "log_s_disc_u"), ("S_HOSP_U", "log_s_hosp_u"),
    ]:
        x[target] = np.log1p(pd.to_numeric(x[source], errors="coerce").clip(lower=0))
    x["dx_observed_count"] = x["dx_tokens"].map(lambda z: sum(v > SPECIAL_TOKEN_MAX for v in _tokens(z)))
    x["pr_observed_count"] = x["pr_tokens"].map(lambda z: sum(v > SPECIAL_TOKEN_MAX for v in _tokens(z)))
    x["dx_oov_count"] = x["dx_tokens"].map(lambda z: sum(v == 2 for v in _tokens(z)))
    x["pr_oov_count"] = x["pr_tokens"].map(lambda z: sum(v == 2 for v in _tokens(z)))

    timing = {name: [] for name in [
        "prday_preadmission_count", "prday_day0_count", "prday_day1_2_count",
        "prday_day3plus_count", "prday_missing_count", "prday_invalid_after_discharge_count",
    ]}
    for toks, days, los in zip(x["pr_tokens"], x["prday"], x["LOS"]):
        los_i = max(0, int(los)) if pd.notna(los) else 0
        counts = Counter()
        for tok, day in zip(_tokens(toks), _tokens(days)):
            if tok <= SPECIAL_TOKEN_MAX:
                continue
            if day <= -90 or day == -66:
                counts["prday_missing_count"] += 1
            elif day < 0:
                counts["prday_preadmission_count"] += 1
            elif day > los_i:
                counts["prday_invalid_after_discharge_count"] += 1
            elif day == 0:
                counts["prday_day0_count"] += 1
            elif day <= 2:
                counts["prday_day1_2_count"] += 1
            else:
                counts["prday_day3plus_count"] += 1
        for name in timing:
            timing[name].append(counts[name])
    for name, values in timing.items():
        x[name] = values
    for name in CATEGORICAL_COLUMNS:
        x[name] = x[name].astype("Int64").astype("string").fillna("<MISSING>")
    for name in NUMERIC_COLUMNS:
        x[name] = pd.to_numeric(x[name], errors="coerce")
    return x


def make_static_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler(with_mean=False)),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_COLUMNS), ("categorical", categorical, CATEGORICAL_COLUMNS)],
        sparse_threshold=1.0,
    )


def fit_token_map(series: pd.Series, min_frequency: int) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for values in series:
        counts.update({v for v in _tokens(values) if v > SPECIAL_TOKEN_MAX})
    kept = sorted(v for v, n in counts.items() if n >= min_frequency)
    return {token: i for i, token in enumerate(kept)}


def token_matrix(series: pd.Series, mapping: dict[int, int]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for i, values in enumerate(series):
        for token in {v for v in _tokens(values) if v in mapping}:
            rows.append(i)
            cols.append(mapping[token])
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(series), len(mapping)), dtype=np.float32)


def combine_features(static_x, frame: pd.DataFrame, maps: dict[str, dict[int, int]]) -> sparse.csr_matrix:
    blocks = [sparse.csr_matrix(static_x, dtype=np.float32)]
    for name in ("dx_tokens", "pr_tokens", "prior_dx_tokens_180d", "prior_pr_tokens_180d"):
        blocks.append(token_matrix(frame[name], maps[name]))
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def metric_dict(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.int8)
    result = {
        "n": int(len(y)), "events": int(y.sum()), "prevalence": float(np.average(y, weights=weights)),
        "auprc": float(average_precision_score(y, p, sample_weight=weights)),
        "auroc": float(roc_auc_score(y, p, sample_weight=weights)),
        "brier": float(brier_score_loss(y, p, sample_weight=weights)),
        "log_loss": float(log_loss(y, p, sample_weight=weights, labels=[0, 1])),
    }
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    cal = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    cal.fit(logits, y, sample_weight=weights)
    result["calibration_intercept"] = float(cal.intercept_[0])
    result["calibration_slope"] = float(cal.coef_[0, 0])
    return result


def assess_prediction_validity(
    y: np.ndarray,
    p: np.ndarray,
    *,
    converged: bool | None,
    coefficients: np.ndarray | None = None,
) -> dict[str, Any]:
    """Apply pre-2022 numerical and ranking gates to a candidate prediction.

    These gates deliberately do not use a performance target from 2022.  A
    candidate must be numerically usable, non-constant, and not rank the
    2021A outcome in the inverse direction before it can enter the finite
    development-grid selection.  Convergence is mandatory for optimizers that
    expose it; tree models pass ``None`` because that notion is inapplicable.
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=np.int8)
    finite_predictions = bool(p.ndim == 1 and len(p) == len(y) and np.isfinite(p).all())
    finite_coefficients = bool(coefficients is None or np.isfinite(np.asarray(coefficients)).all())
    probability_range = float(np.ptp(p)) if finite_predictions and len(p) else 0.0
    prediction_std = float(np.std(p)) if finite_predictions and len(p) else 0.0
    nondegenerate_predictions = bool(
        finite_predictions and probability_range > 1e-7 and prediction_std > 1e-8
    )
    if finite_predictions and len(np.unique(y)) == 2:
        auroc = float(roc_auc_score(y, p))
        ranking_direction_plausible = bool(auroc >= 0.5)
    else:
        auroc = None
        ranking_direction_plausible = False

    reasons: list[str] = []
    if not finite_predictions:
        reasons.append("nonfinite_or_shape_mismatched_predictions")
    if not finite_coefficients:
        reasons.append("nonfinite_coefficients")
    if not nondegenerate_predictions:
        reasons.append("degenerate_prediction_distribution")
    if converged is False:
        reasons.append("optimizer_not_converged")
    if not ranking_direction_plausible:
        reasons.append("inverse_or_undefined_2021A_ranking")
    return {
        "finite_predictions": finite_predictions,
        "finite_coefficients": finite_coefficients,
        "probability_range": probability_range,
        "prediction_std": prediction_std,
        "nondegenerate_predictions": nondegenerate_predictions,
        "validation_auroc_for_ranking_sanity": auroc,
        "ranking_direction_plausible": ranking_direction_plausible,
        "optimizer_converged": converged,
        "valid_for_selection": not reasons,
        "rejection_reasons": reasons,
    }


def capacity_constrained_flags(
    calibrated_risk: np.ndarray, encounter_hash: np.ndarray, capacity_fraction: float = 0.20,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Flag an exact, deterministic risk-capacity set even when scores tie.

    Boundary ties are resolved by ascending encounter hash, which is an
    outcome-independent pre-existing identifier.  This prevents a constant
    calibrated score from silently converting a 20% capacity policy into a
    treat-all policy.
    """
    p = np.asarray(calibrated_risk, dtype=float)
    hashes = np.asarray(encounter_hash)
    if p.ndim != 1 or hashes.ndim != 1 or len(p) != len(hashes) or len(p) == 0:
        raise ValueError("calibrated_risk and encounter_hash must be non-empty one-dimensional equal-length arrays")
    if not np.isfinite(p).all() or not 0 < capacity_fraction <= 1:
        raise ValueError("risk must be finite and capacity_fraction must be in (0, 1]")
    if len(np.unique(hashes)) != len(hashes):
        raise ValueError("encounter_hash must be unique for deterministic tie resolution")
    target_rows = int(math.ceil(capacity_fraction * len(p)))
    # lexsort's final key is primary: descending risk, then ascending hash.
    ranked = np.lexsort((hashes, -p))
    flagged = np.zeros(len(p), dtype=bool)
    flagged[ranked[:target_rows]] = True
    boundary = float(p[ranked[target_rows - 1]])
    boundary_mask = p == boundary
    return flagged, {
        "capacity_fraction_requested": float(capacity_fraction),
        "capacity_rows_requested": target_rows,
        "capacity_fraction_realized": float(flagged.mean()),
        "boundary_risk": boundary,
        "boundary_tie_rows": int(boundary_mask.sum()),
        "boundary_tie_rows_selected": int((flagged & boundary_mask).sum()),
        "tie_break_rule": "descending calibrated risk; ascending encounter_hash at an equal-risk boundary",
    }


def fit_candidate_models(
    target: str, y_train: np.ndarray, y_val: np.ndarray, x_static_train,
    x_static_val, x_full_train, x_full_val, threads: int,
) -> tuple[dict[str, Any], dict[str, dict]]:
    import lightgbm as lgb

    models: dict[str, Any] = {}
    reports: dict[str, dict] = {}

    structured = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        structured.fit(x_static_train, y_train)
    p = structured.predict_proba(x_static_val)[:, 1]
    structured_warnings = [
        str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
    ]
    structured_convergence = {
        "solver": "lbfgs", "max_iter": 2000, "n_iter": int(np.max(np.asarray(structured.n_iter_))),
        "convergence_warning": bool(structured_warnings), "warning_messages": structured_warnings,
        "converged": bool(not structured_warnings and np.max(np.asarray(structured.n_iter_)) < 2000),
    }
    structured_validity = assess_prediction_validity(
        y_val, p, converged=structured_convergence["converged"], coefficients=structured.coef_
    )
    if not structured_validity["valid_for_selection"]:
        raise RuntimeError(
            f"Structured logistic failed numerical validity for {target}: "
            f"{structured_validity['rejection_reasons']}"
        )
    models["structured_logistic"] = structured
    reports["structured_logistic"] = {
        "selected_config": {"penalty": "l2", "C": 1.0},
        "metrics": metric_dict(y_val, p),
        "selected_convergence": structured_convergence,
        "numerical_validity": structured_validity,
    }

    en_trials = []
    best_en = None
    best_en_score = -math.inf
    for config in ELASTIC_GRID:
        # saga is the standard sparse elastic-net logistic optimizer.  Unlike
        # the former single-pass SGD setup, it optimizes the regularized
        # logistic objective directly and exposes an explicit convergence
        # diagnostic that is carried into the candidate report.
        model = LogisticRegression(
            penalty="elasticnet", solver="saga", C=config["C"],
            l1_ratio=config["l1_ratio"], max_iter=ELASTIC_MAX_ITER,
            tol=ELASTIC_TOL, random_state=SEED, class_weight=None,
            fit_intercept=True, warm_start=False,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_full_train, y_train)
        p = model.predict_proba(x_full_val)[:, 1]
        metrics = metric_dict(y_val, p)
        convergence_warnings = [
            str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)
        ]
        n_iter = int(np.max(np.asarray(model.n_iter_)))
        convergence = {
            "solver": "saga", "max_iter": ELASTIC_MAX_ITER, "tol": ELASTIC_TOL,
            "n_iter": n_iter, "convergence_warning": bool(convergence_warnings),
            "warning_messages": convergence_warnings,
            "converged": bool(not convergence_warnings and n_iter < ELASTIC_MAX_ITER),
        }
        validity = assess_prediction_validity(
            y_val, p, converged=convergence["converged"], coefficients=model.coef_
        )
        trial = {"config": config, "metrics": metrics, "convergence": convergence,
                 "numerical_validity": validity}
        en_trials.append(trial)
        if validity["valid_for_selection"] and metrics["auprc"] > best_en_score:
            best_en_score, best_en = metrics["auprc"], (model, config, metrics, convergence, validity)
    if best_en is None:
        rejected = [
            {"config": trial["config"], "rejection_reasons": trial["numerical_validity"]["rejection_reasons"]}
            for trial in en_trials
        ]
        raise RuntimeError(
            f"No numerically valid elastic-net candidate for {target}; refusing to select a failed model: {rejected}"
        )
    models["elastic_net"] = best_en[0]
    reports["elastic_net"] = {"selected_config": best_en[1], "metrics": best_en[2],
                                "selected_convergence": best_en[3],
                                "selected_numerical_validity": best_en[4], "all_trials": en_trials,
                                "implementation": "LogisticRegression(solver='saga', penalty='elasticnet'); sparse direct objective optimization; no resampling"}

    lgb_trials = []
    best_lgb = None
    best_lgb_score = -math.inf
    for config in LIGHTGBM_GRID:
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=1600, max_depth=-1, max_bin=127,
            subsample_freq=1, verbosity=-1, n_jobs=threads, deterministic=True,
            force_col_wise=True, random_state=SEED, bagging_seed=SEED,
            feature_fraction_seed=SEED, data_random_seed=SEED, **config,
        )
        model.fit(
            x_full_train, y_train, eval_set=[(x_full_val, y_val)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(stopping_rounds=75, verbose=False)],
        )
        p = model.predict_proba(x_full_val, num_iteration=model.best_iteration_)[:, 1]
        metrics = metric_dict(y_val, p)
        trial = {
            "config": config, "best_iteration": int(model.best_iteration_), "metrics": metrics,
            "numerical_validity": assess_prediction_validity(y_val, p, converged=None),
        }
        lgb_trials.append(trial)
        if trial["numerical_validity"]["valid_for_selection"] and metrics["auprc"] > best_lgb_score:
            best_lgb_score, best_lgb = metrics["auprc"], (model, config, metrics, trial["numerical_validity"])
    if best_lgb is None:
        rejected = [
            {"config": trial["config"], "rejection_reasons": trial["numerical_validity"]["rejection_reasons"]}
            for trial in lgb_trials
        ]
        raise RuntimeError(
            f"No numerically valid LightGBM candidate for {target}; refusing to select a failed model: {rejected}"
        )
    models["lightgbm"] = best_lgb[0]
    reports["lightgbm"] = {"selected_config": best_lgb[1], "best_iteration": int(best_lgb[0].best_iteration_),
                            "metrics": best_lgb[2], "selected_numerical_validity": best_lgb[3], "all_trials": lgb_trials,
                            "implementation": "LGBMClassifier with early stopping; optimizer convergence not applicable"}
    return models, reports


def raw_prediction(name: str, model: Any, x_static, x_full) -> np.ndarray:
    x = x_static if name == "structured_logistic" else x_full
    kwargs = {"num_iteration": model.best_iteration_} if name == "lightgbm" else {}
    return model.predict_proba(x, **kwargs)[:, 1]


def fit_one_calibrator(method: str, p: np.ndarray, y: np.ndarray):
    p = np.clip(np.asarray(p), 1e-7, 1 - 1e-7)
    if method == "none":
        return None
    if method == "platt":
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, y)
        return model
    raise ValueError(method)


def apply_calibrator(method: str, model: Any, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p), 1e-7, 1 - 1e-7)
    if method == "none":
        return p
    if method == "platt":
        return model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
    return np.asarray(model.predict(p))


def choose_calibration(p: np.ndarray, y: np.ndarray, patient_hash: np.ndarray) -> tuple[str, Any, dict]:
    # A deterministic patient-level nested split inside 2021A avoids evaluating a
    # calibrator on the exact observations used to fit it.
    hashes = np.asarray(patient_hash, dtype=np.uint64)
    fit_mask = ((hashes >> np.uint64(1)) & np.uint64(1)) == 0
    if fit_mask.all() or (~fit_mask).all():
        raise RuntimeError("Degenerate nested 2021A calibration split")
    trials = {}
    for method in ("none", "platt", "isotonic"):
        fitted = fit_one_calibrator(method, p[fit_mask], y[fit_mask])
        pred = apply_calibrator(method, fitted, p[~fit_mask])
        trials[method] = metric_dict(y[~fit_mask], pred)
    best_brier = min(v["brier"] for v in trials.values())
    tolerance = 0.0005
    eligible = [m for m in ("none", "platt", "isotonic") if trials[m]["brier"] <= best_brier + tolerance]
    method = eligible[0]
    fitted_full = fit_one_calibrator(method, p, y)
    report = {
        "nested_split_rule": "((patient_hash >> 1) & 1): 0=calibrator-fit, 1=method-selection",
        "fit_rows": int(fit_mask.sum()), "selection_rows": int((~fit_mask).sum()),
        "fit_events": int(y[fit_mask].sum()), "selection_events": int(y[~fit_mask].sum()),
        "selection_metric": "lowest Brier; prefer simpler method within 0.0005 absolute Brier",
        "trials": trials, "selected_method": method,
        "refit_on_full_2021A_after_selection": True,
    }
    return method, fitted_full, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--min-code-frequency", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        raise SystemExit("--threads must be between 1 and 8")
    if args.min_code_frequency < 1:
        raise SystemExit("--min-code-frequency must be positive")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    random.seed(SEED)
    np.random.seed(SEED)

    root = args.root.resolve()
    history_dir = args.history_dir.resolve()
    out = args.output_dir.resolve()
    models_dir = out / "models"
    out.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dev_raw = pd.concat([read_partition(root, history_dir, y, None) for y in DEV_YEARS], ignore_index=True)
    val_raw = read_partition(root, history_dir, VALIDATION_YEAR, "2021A")
    if set(dev_raw["year"].unique()) != set(DEV_YEARS) or set(val_raw["analysis_partition"].unique()) != {"2021A"}:
        raise RuntimeError("Partition gate failed")
    if set(dev_raw["patient_hash"]) & set(val_raw["patient_hash"]):
        # Hashes are year-salted, so overlap would signal a construction bug rather than identity.
        raise RuntimeError("Unexpected cross-year salted patient-hash overlap")

    dev = engineer_static(dev_raw)
    val = engineer_static(val_raw)
    preprocessor = make_static_preprocessor()
    x_static_dev = preprocessor.fit_transform(dev).astype(np.float32).tocsr()
    x_static_val = preprocessor.transform(val).astype(np.float32).tocsr()
    token_maps = {
        name: fit_token_map(dev[name], args.min_code_frequency)
        for name in ("dx_tokens", "pr_tokens", "prior_dx_tokens_180d", "prior_pr_tokens_180d")
    }
    x_full_dev = combine_features(x_static_dev, dev, token_maps)
    x_full_val = combine_features(x_static_val, val, token_maps)

    feature_spec = {
        "status": "FROZEN_BEFORE_2021A_MODEL_RESULTS",
        "prediction_time": "index discharge",
        "development_years": list(DEV_YEARS), "model_selection_partition": "2021A",
        "conformal_partition_not_accessed": "2021B", "sealed_test_year_not_accessed": 2022,
        "numeric_features": NUMERIC_COLUMNS, "categorical_features": CATEGORICAL_COLUMNS,
        "code_bag_features": list(token_maps), "special_token_ids_excluded": [0, 1, 2, 3],
        "code_minimum_development_episode_frequency": args.min_code_frequency,
        "token_feature_counts": {k: len(v) for k, v in token_maps.items()},
        "static_matrix_shape": list(x_static_dev.shape), "full_matrix_shape": list(x_full_dev.shape),
        "history_definition": "same-year nonoverlapping prior admissions; 30/90/180d plus left-truncation flags",
        "forbidden_predictors": [
            "future admission data", "readmission label or gap", "patient/hospital identifiers",
            "DISCWT as predictor", "2021B/2022-fitted preprocessing",
        ],
    }
    feature_spec_path = out / "baseline_feature_spec.json"
    feature_spec_path.write_text(stable_json(feature_spec), encoding="utf-8")
    joblib.dump(preprocessor, models_dir / "static_preprocessor.joblib", compress=3)
    joblib.dump(token_maps, models_dir / "development_token_maps.joblib", compress=3)

    predictions = val_raw[["year", "encounter_hash", "patient_hash", "DISCWT"] + list(TARGETS.values())].copy()
    selection_report: dict[str, Any] = {}
    calibration_report: dict[str, Any] = {}
    threshold_report: dict[str, Any] = {}

    for target_name, label_column in TARGETS.items():
        y_dev = dev_raw[label_column].astype(np.int8).to_numpy()
        y_val = val_raw[label_column].astype(np.int8).to_numpy()
        models, reports = fit_candidate_models(
            target_name, y_dev, y_val, x_static_dev, x_static_val, x_full_dev, x_full_val, args.threads
        )
        selection_report[target_name] = reports
        calibration_report[target_name] = {}
        threshold_report[target_name] = {}
        for model_name, model in models.items():
            raw = raw_prediction(model_name, model, x_static_val, x_full_val)
            method, calibrator, cal_report = choose_calibration(
                raw, y_val, val_raw["patient_hash"].to_numpy(dtype=np.uint64)
            )
            calibrated = apply_calibrator(method, calibrator, raw)
            flagged, capacity_report = capacity_constrained_flags(
                calibrated, val_raw["encounter_hash"].to_numpy(), capacity_fraction=0.20
            )
            threshold_report[target_name][model_name] = {
                "rule": "fixed 20% capacity: descending 2021A calibrated risk; ascending encounter_hash resolves boundary ties",
                "threshold": capacity_report["boundary_risk"], "flagged_rows": int(flagged.sum()),
                "flagged_fraction": float(flagged.mean()),
                "sensitivity": float(((flagged) & (y_val == 1)).sum() / max(1, y_val.sum())),
                "positive_predictive_value": float(y_val[flagged].mean()) if flagged.any() else 0.0,
                "capacity_selection": capacity_report,
            }
            calibration_report[target_name][model_name] = cal_report
            predictions[f"p_raw__{target_name}__{model_name}"] = raw.astype(np.float32)
            predictions[f"p_cal__{target_name}__{model_name}"] = calibrated.astype(np.float32)
            joblib.dump(
                {"model": model, "calibration_method": method, "calibrator": calibrator,
                 "target": target_name, "label_column": label_column},
                models_dir / f"{target_name}__{model_name}.joblib", compress=3,
            )
        print(f"completed target={target_name}", flush=True)

    pq.write_table(pa.Table.from_pandas(predictions, preserve_index=False), out / "predictions_2021A.parquet",
                   compression="zstd", compression_level=6)
    selection_report["implementation_revision"] = BASELINE_IMPLEMENTATION_REVISION
    (out / "model_selection_2021A.json").write_text(stable_json(selection_report), encoding="utf-8")
    (out / "calibration_selection_2021A.json").write_text(stable_json(calibration_report), encoding="utf-8")
    (out / "operating_thresholds_2021A.json").write_text(stable_json(threshold_report), encoding="utf-8")

    input_lock = root / "project" / "outputs" / "stage2_lock" / "analysis_lock.json"
    artifacts = {}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            artifacts[str(path.relative_to(out))] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    lock = {
        "status": "PASS_BASELINES_FROZEN_PRE_2021B_PRE_2022",
        "development_rows": int(len(dev_raw)), "validation_2021A_rows": int(len(val_raw)),
        "development_events": {k: int(dev_raw[v].sum()) for k, v in TARGETS.items()},
        "validation_2021A_events": {k: int(val_raw[v].sum()) for k, v in TARGETS.items()},
        "analysis_lock_sha256": sha256(input_lock), "feature_spec_sha256": sha256(feature_spec_path),
        "script_sha256": sha256(Path(__file__).resolve()),
        "2021B_outcomes_accessed": False, "year_2022_accessed": False,
        "threads": args.threads, "random_seed": SEED, "artifacts": artifacts,
        "implementation_revision": BASELINE_IMPLEMENTATION_REVISION,
    }
    (out / "baseline_lock.json").write_text(stable_json(lock), encoding="utf-8")
    print(stable_json(lock), flush=True)


if __name__ == "__main__":
    main()
