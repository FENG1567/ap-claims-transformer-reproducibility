#!/usr/bin/env python3
"""Train post-hoc NRD-only common-variable baselines for MIMIC transport.

Model construction and selection use only 2018-2020 development and 2021A.
2021B, 2022 outcomes, and MIMIC are never used for fitting, calibration,
threshold selection, or feature selection.  The resulting comparators are
explicitly post-hoc and are not replacements for the frozen primary models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


SEED = 20260921
SPECIAL_MAX = 3
EXPECTED_COMMON_VARIABLE_FINETUNE_MANIFEST_SHA256 = "13c5271a70df4761764e2a681d5e1cd0db1d66d82cc4e544268c72dc39e52018"
EXPECTED_COMMON_VARIABLE_CHECKPOINT_SHA256 = "e4680d697f303574ab946e6f7e6fcdc32409a46ca8e603d1f264ed8aa5c62efe"
TARGETS = {
    "any_readmission": "any_unplanned_readmission_30d",
    "ap_specific_readmission": "ap_specific_readmission_30d",
}
RAW_COLUMNS = [
    "encounter_hash", "patient_hash", "DISCWT", "analysis_partition",
    "primary_analysis_eligible", "any_unplanned_readmission_30d",
    "ap_specific_readmission_30d", "AGE", "FEMALE", "LOS", "I10_NDX",
    "I10_NPR", "dx_tokens", "pr_tokens", "prday",
]
# This tuple is deliberately duplicated from the immutable common-variable
# checkpoint manifest rather than inferred from an NRD convenience table.  The
# baseline may derive counts from the frozen token/timing inputs below, but it
# may not introduce any information absent from this projection.
FORMAL_STATIC_NUMERIC_FEATURES = (
    "AGE", "LOS", "I10_NDX", "I10_NPR", "prior_count_ytd", "prior_count_30d",
    "prior_count_90d", "prior_count_180d", "prior_ed_count_180d",
    "prior_nonelective_count_180d", "prior_ap_count_180d",
    "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d",
    "prior_los_sum_180d", "days_since_prior_discharge",
)
FORMAL_STATIC_CATEGORICAL_FEATURES = (
    "FEMALE", "history_30d_fully_observable", "history_90d_fully_observable",
    "history_180d_fully_observable",
)
TOKEN_FIELDS = ("dx_tokens", "pr_tokens", "prior_dx_tokens_180d", "prior_pr_tokens_180d")
DERIVED_COMMON_INPUT_NUMERIC = (
    "dx_observed_count", "pr_observed_count", "dx_oov_count", "pr_oov_count",
    "prday_preadmission_count", "prday_day0_count", "prday_day1_2_count",
    "prday_day3plus_count", "prday_missing_count",
    "prday_invalid_after_discharge_count",
)
# The categorical variables are binary frozen static inputs.  They are encoded
# numerically for the comparators, but retained as categorical in the audit.
NUMERIC = [*FORMAL_STATIC_NUMERIC_FEATURES, *FORMAL_STATIC_CATEGORICAL_FEATURES,
           *DERIVED_COMMON_INPUT_NUMERIC]
HISTORY_COLUMNS = [
    *FORMAL_STATIC_NUMERIC_FEATURES[4:], *FORMAL_STATIC_CATEGORICAL_FEATURES[1:],
    *TOKEN_FIELDS[2:],
]
EXCLUDED_ABSENT_APR_DRG_HISTORY_FEATURES = (
    "prior_max_severity_180d", "prior_max_mortality_risk_180d",
)


def stable_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_formal_common_variable_contract(path: Path) -> dict[str, Any]:
    """Fail closed unless ``path`` is the registered frozen projection.

    This makes the post-hoc comparators auditable against the exact static
    feature schema accepted by the frozen MIMIC adapter.  It intentionally
    does not accept a merely similar manifest or a user-supplied schema.
    """
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError("Formal common-variable finetune manifest is missing")
    digest = sha256(path)
    if digest != EXPECTED_COMMON_VARIABLE_FINETUNE_MANIFEST_SHA256:
        raise RuntimeError("Formal common-variable finetune manifest SHA-256 mismatch")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Formal common-variable finetune manifest is unreadable") from exc
    expected_static = {
        "numeric": list(FORMAL_STATIC_NUMERIC_FEATURES),
        "categorical": list(FORMAL_STATIC_CATEGORICAL_FEATURES),
        "dimension": 46,
    }
    if (manifest.get("status") != "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022"
            or manifest.get("checkpoint", {}).get("sha256") != EXPECTED_COMMON_VARIABLE_CHECKPOINT_SHA256
            or manifest.get("development_years") != [2018, 2019, 2020]
            or manifest.get("prediction_partition") != "2021A"
            or manifest.get("2021B_accessed") is not False
            or manifest.get("year_2022_accessed") is not False
            or manifest.get("static_feature_set") != expected_static
            or manifest.get("static_preprocessor", {}).get("numeric_columns") != expected_static["numeric"]
            or manifest.get("static_preprocessor", {}).get("categorical_columns") != expected_static["categorical"]):
        raise RuntimeError("Formal common-variable checkpoint feature contract differs from the registered schema")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "checkpoint_sha256": EXPECTED_COMMON_VARIABLE_CHECKPOINT_SHA256,
        "static_feature_set": expected_static,
    }


def tokens(v: Any) -> list[int]:
    if v is None:
        return []
    return [int(x) for x in v if x is not None]


def read_partition(project_root: Path, year: int, partition: str | None) -> pd.DataFrame:
    eps = project_root / "data" / "nrd" / f"year={year}" / "ap_episodes.parquet"
    filters = [("primary_analysis_eligible", "=", True)]
    if partition:
        filters.append(("analysis_partition", "=", partition))
    frame = pq.read_table(eps, columns=RAW_COLUMNS, filters=filters).to_pandas()
    hist = pq.read_table(project_root / "project" / "outputs" / "stage4_modeling" / "history" /
                         f"ap_history_{year}.parquet",
                         columns=["encounter_hash", "analysis_partition", *HISTORY_COLUMNS]).to_pandas()
    if partition:
        hist = hist[hist.analysis_partition.eq(partition)]
    hist = hist.drop(columns="analysis_partition")
    return frame.merge(hist, on="encounter_hash", how="left", validate="one_to_one")


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x["dx_observed_count"] = x.dx_tokens.map(lambda z: sum(v > SPECIAL_MAX for v in tokens(z)))
    x["pr_observed_count"] = x.pr_tokens.map(lambda z: sum(v > SPECIAL_MAX for v in tokens(z)))
    x["dx_oov_count"] = x.dx_tokens.map(lambda z: sum(v == 2 for v in tokens(z)))
    x["pr_oov_count"] = x.pr_tokens.map(lambda z: sum(v == 2 for v in tokens(z)))
    timing_names = ["prday_preadmission_count", "prday_day0_count", "prday_day1_2_count",
                    "prday_day3plus_count", "prday_missing_count",
                    "prday_invalid_after_discharge_count"]
    timing = {k: [] for k in timing_names}
    for toks, days, los in zip(x.pr_tokens, x.prday, x.LOS):
        c = Counter(); los = max(0, int(los)) if pd.notna(los) else 0
        for tok, day in zip(tokens(toks), tokens(days)):
            if tok <= SPECIAL_MAX:
                continue
            if day <= -90 or day == -66: c["prday_missing_count"] += 1
            elif day < 0: c["prday_preadmission_count"] += 1
            elif day > los: c["prday_invalid_after_discharge_count"] += 1
            elif day == 0: c["prday_day0_count"] += 1
            elif day <= 2: c["prday_day1_2_count"] += 1
            else: c["prday_day3plus_count"] += 1
        for k in timing_names: timing[k].append(c[k])
    for k, v in timing.items(): x[k] = v
    for k in NUMERIC: x[k] = pd.to_numeric(x[k], errors="coerce")
    return x


def fit_numeric_spec(dev: pd.DataFrame) -> dict[str, Any]:
    spec = {}
    for name in NUMERIC:
        v = pd.to_numeric(dev[name], errors="coerce").to_numpy(float)
        med = float(np.nanmedian(v)); v = np.where(np.isfinite(v), v, med)
        mean, sd = float(v.mean()), float(v.std())
        spec[name] = {"median": med, "mean": mean, "sd": sd if sd > 1e-8 else 1.0}
    return spec


def numeric_matrix(frame: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    cols = []
    for name in NUMERIC:
        v = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        s = spec[name]; v = np.where(np.isfinite(v), v, s["median"])
        cols.append((v - s["mean"]) / s["sd"])
    return np.column_stack(cols).astype(np.float32)


def fit_token_maps(dev: pd.DataFrame, minimum: int = 100) -> dict[str, dict[int, int]]:
    maps = {}
    for name in TOKEN_FIELDS:
        c = Counter()
        for values in dev[name]:
            c.update({v for v in tokens(values) if v > SPECIAL_MAX})
        kept = sorted(v for v, n in c.items() if n >= minimum)
        maps[name] = {v: i for i, v in enumerate(kept)}
    return maps


def token_matrix(series: pd.Series, mapping: dict[int, int]) -> sparse.csr_matrix:
    rr, cc = [], []
    for i, values in enumerate(series):
        for v in {v for v in tokens(values) if v in mapping}:
            rr.append(i); cc.append(mapping[v])
    return sparse.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)),
                             shape=(len(series), len(mapping)))


def full_matrix(frame: pd.DataFrame, num_spec: dict[str, Any], maps: dict[str, dict[int, int]]) -> sparse.csr_matrix:
    blocks = [sparse.csr_matrix(numeric_matrix(frame, num_spec))]
    blocks.extend(token_matrix(frame[name], maps[name]) for name in TOKEN_FIELDS)
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def fit_platt(raw: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    x = np.log(np.clip(raw, 1e-7, 1 - 1e-7) / np.clip(1 - raw, 1e-7, 1)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(x, y, sample_weight=w)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def apply_platt(raw: np.ndarray, pars: tuple[float, float]) -> np.ndarray:
    x = np.log(np.clip(raw, 1e-7, 1 - 1e-7) / np.clip(1 - raw, 1e-7, 1))
    return expit(pars[0] + pars[1] * x)


def metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    return {"auroc": float(roc_auc_score(y, p, sample_weight=w)),
            "auprc": float(average_precision_score(y, p, sample_weight=w)),
            "brier": float(brier_score_loss(y, p, sample_weight=w)),
            "prevalence": float(np.average(y, weights=w))}


def export_linear(path: Path, model: LogisticRegression) -> None:
    np.savez_compressed(path, coef=model.coef_.astype(np.float64), intercept=model.intercept_.astype(np.float64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--finetune-manifest", type=Path, required=True,
                    help="registered frozen common-variable finetune manifest")
    args = ap.parse_args()
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    if output.exists() or list(output.parent.glob(output.name + ".partial-*")):
        raise SystemExit("Refusing to overwrite output/partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))

    formal_contract = validate_formal_common_variable_contract(args.finetune_manifest)
    dev = pd.concat([read_partition(root, y, None) for y in (2018, 2019, 2020)], ignore_index=True)
    val = read_partition(root, 2021, "2021A")
    dev, val = engineer(dev), engineer(val)
    num_spec, maps = fit_numeric_spec(dev), fit_token_maps(dev)
    xdev_num, xval_num = numeric_matrix(dev, num_spec), numeric_matrix(val, num_spec)
    xdev, xval = full_matrix(dev, num_spec, maps), full_matrix(val, num_spec, maps)
    wdev, wval = dev.DISCWT.to_numpy(float), val.DISCWT.to_numpy(float)

    feature_spec = {"status": "FROZEN_POSTHOC_COMMON_VARIABLE_BASELINE_SPEC",
                    "post_hoc": True, "development_years": [2018, 2019, 2020],
                    "selection_calibration_partition": "2021A",
                    "mimic_outcomes_accessed": False, "year_2022_outcomes_accessed": False,
                    "formal_common_variable_contract": formal_contract,
                    "formal_static_numeric_features": list(FORMAL_STATIC_NUMERIC_FEATURES),
                    "formal_static_categorical_features": list(FORMAL_STATIC_CATEGORICAL_FEATURES),
                    "derived_from_frozen_token_or_timing_inputs": list(DERIVED_COMMON_INPUT_NUMERIC),
                    "numeric_features": NUMERIC, "numeric_spec": num_spec,
                    "token_fields": list(TOKEN_FIELDS),
                    "token_maps": {k: [int(v) for v in sorted(m, key=m.get)] for k, m in maps.items()},
                    "token_minimum_development_frequency": 100,
                    "prediction_time": "index discharge",
                    "excluded_nrd_only_variables": ["payer", "ZIP income", "hospital structure", "survey design", "cost"],
                    "excluded_absent_apr_drg_history_features": list(EXCLUDED_ABSENT_APR_DRG_HISTORY_FEATURES)}
    (temp / "common_feature_spec.json").write_text(stable_json(feature_spec), encoding="utf-8")

    summary = []
    calibration = {}
    thresholds = {}
    for target, ycol in TARGETS.items():
        ydev, yval = dev[ycol].to_numpy(np.int8), val[ycol].to_numpy(np.int8)

        candidates: list[tuple[str, Any, np.ndarray, np.ndarray]] = []
        structured = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
                                        random_state=SEED)
        structured.fit(xdev_num, ydev, sample_weight=wdev)
        candidates.append(("structured_logistic", structured,
                           structured.predict_proba(xdev_num)[:, 1], structured.predict_proba(xval_num)[:, 1]))

        best_elastic = None
        for C, ratio in ((0.005, .15), (.01, .15), (.01, .5), (.05, .5), (.05, .85)):
            model = LogisticRegression(C=C, l1_ratio=ratio, penalty="elasticnet", solver="saga",
                                       max_iter=1500, tol=1e-3, random_state=SEED, n_jobs=8)
            model.fit(xdev, ydev, sample_weight=wdev)
            pv = model.predict_proba(xval)[:, 1]
            score = average_precision_score(yval, pv, sample_weight=wval)
            if best_elastic is None or score > best_elastic[0]:
                best_elastic = (score, model, pv, C, ratio)
        _, elastic, elastic_val, eC, er = best_elastic
        candidates.append(("elastic_net", elastic, elastic.predict_proba(xdev)[:, 1], elastic_val))

        best_lgb = None
        for leaves, child, lr, frac, l2 in ((15,100,.05,1.0,5.0),(31,100,.05,.8,1.0),(63,200,.03,.8,5.0)):
            model = lgb.LGBMClassifier(objective="binary", n_estimators=500, num_leaves=leaves,
                min_child_samples=child, learning_rate=lr, colsample_bytree=frac,
                subsample=.8, reg_lambda=l2, random_state=SEED, n_jobs=8, verbosity=-1)
            model.fit(xdev, ydev, sample_weight=wdev,
                      eval_set=[(xval, yval)], eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(40, verbose=False)])
            pv = model.predict_proba(xval)[:, 1]
            score = average_precision_score(yval, pv, sample_weight=wval)
            if best_lgb is None or score > best_lgb[0]:
                best_lgb = (score, model, pv, leaves, child, lr, frac, l2)
        _, lgbm, lgb_val, *_ = best_lgb
        candidates.append(("lightgbm", lgbm, lgbm.predict_proba(xdev)[:, 1], lgb_val))

        calibration[target], thresholds[target] = {}, {}
        for name, model, raw_dev, raw_val in candidates:
            pars = fit_platt(raw_val, yval, wval)
            pval = apply_platt(raw_val, pars)
            threshold = float(np.quantile(pval, .80, method="higher"))
            calibration[target][name] = {"intercept": pars[0], "slope": pars[1]}
            thresholds[target][name] = threshold
            row = {"outcome": target, "model": name, "threshold": threshold,
                   **metrics(yval, pval, wval)}
            if name == "elastic_net": row.update({"selected_C": eC, "selected_l1_ratio": er})
            summary.append(row)
            if name in ("structured_logistic", "elastic_net"):
                export_linear(temp / f"{target}__{name}.npz", model)
            else:
                model.booster_.save_model(str(temp / f"{target}__lightgbm.txt"))

    (temp / "calibration_parameters.json").write_text(stable_json(calibration), encoding="utf-8")
    (temp / "operating_thresholds.json").write_text(stable_json(thresholds), encoding="utf-8")
    pd.DataFrame(summary).to_csv(temp / "validation_2021A_metrics.csv", index=False)
    release = {"status": "PASS_POSTHOC_COMMON_VARIABLE_BASELINES_FROZEN_PRE_MIMIC",
               "post_hoc": True, "mimic_outcomes_accessed": False,
               "year_2022_outcomes_accessed": False, "files": []}
    for p in sorted(temp.glob("*")):
        if p.is_file(): release["files"].append({"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    (temp / "manifest.json").write_text(stable_json(release), encoding="utf-8")
    os.replace(temp, output)
    print(stable_json({"status": release["status"], "output": str(output),
                       "dev_n": len(dev), "validation_n": len(val)}))


if __name__ == "__main__":
    main()
