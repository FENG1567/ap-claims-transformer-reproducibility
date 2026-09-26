"""Aggregate-only upgrade analysis for the locked MIMIC-IV transfer.

This module reads the already-published restricted prediction archive but writes
only aggregate summaries.  It adds an explicit AP_specific label binding,
subject-cluster bootstrap intervals, full decision-curve tables, conformal
selective-prediction summaries, and prespecified subgroup/OOV audits.  It never
fits or recalibrates on MIMIC and never emits row-level predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


BOOTSTRAPS = 1000
RNG_SEED = 20260920
LEVELS = ("80", "90", "95")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_concatenated_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        obj, end = decoder.raw_decode(text, pos)
        if not isinstance(obj, dict):
            raise ValueError("prediction archive contains a non-object record")
        rows.append(obj)
        pos = end
    return rows


def endpoint_arrays(rows: list[dict[str, Any]], endpoint: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if endpoint == "any_readmission":
        label_key = "any_readmission"
        prob_key = "p_any_readmission_calibrated"
    elif endpoint == "ap_specific_readmission":
        # This explicit binding is the correction that the prior dispatcher lacked.
        label_key = "AP_specific"
        prob_key = "p_ap_specific_readmission_calibrated"
    else:
        raise ValueError(endpoint)
    y, p, keep = [], [], []
    for row in rows:
        status = row.get("label", {}).get("label_status")
        prob = row.get("prediction", {}).get(prob_key)
        value = row.get("label", {}).get(label_key)
        valid = status != "algorithm_unknown" and value is not None and prob is not None
        keep.append(bool(valid))
        if valid:
            y.append(int(value))
            p.append(float(prob))
        else:
            y.append(np.nan)
            p.append(np.nan)
    return np.asarray(y, dtype=float), np.asarray(p, dtype=float), np.asarray(keep, dtype=bool)


def calibration_params(y: np.ndarray, p: np.ndarray) -> tuple[float | None, float | None]:
    if len(np.unique(y)) < 2:
        return None, None
    x = logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(x, y.astype(int))
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    out: dict[str, Any] = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "non_events": int(len(y) - y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "auroc": None,
        "auprc": None,
        "brier": None,
        "log_loss": None,
        "calibration_intercept": None,
        "calibration_slope": None,
    }
    if len(y) and len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
        out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
        out["calibration_intercept"], out["calibration_slope"] = calibration_params(y, p)
    if threshold is not None and len(y):
        pred = p >= threshold
        out["threshold"] = float(threshold)
        out["sensitivity"] = float(((pred == 1) & (y == 1)).sum() / max(1, int((y == 1).sum())))
        out["specificity"] = float(((pred == 0) & (y == 0)).sum() / max(1, int((y == 0).sum())))
        out["ppv"] = float(((pred == 1) & (y == 1)).sum() / max(1, int((pred == 1).sum())))
        out["npv"] = float(((pred == 0) & (y == 0)).sum() / max(1, int((pred == 0).sum())))
    return out


def bootstrap_metrics(rows: list[dict[str, Any]], endpoint: str, bootstraps: int = BOOTSTRAPS) -> dict[str, Any]:
    y_all, p_all, keep = endpoint_arrays(rows, endpoint)
    indices = np.flatnonzero(keep)
    y, p = y_all[indices].astype(int), p_all[indices]
    point = metrics(y, p, float(rows[0]["prediction"]["thresholds"][endpoint]))
    groups: dict[str, list[int]] = defaultdict(list)
    for local, original in enumerate(indices):
        groups[str(rows[int(original)].get("subject_id"))].append(local)
    group_values = list(groups.values())
    rng = np.random.default_rng(RNG_SEED + (1 if endpoint == "ap_specific_readmission" else 0))
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstraps):
        chosen = rng.integers(0, len(group_values), size=len(group_values))
        take = np.concatenate([np.asarray(group_values[i], dtype=int) for i in chosen])
        m = metrics(y[take], p[take], point["threshold"])
        for key in ("auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope"):
            if m.get(key) is not None and math.isfinite(float(m[key])):
                samples[key].append(float(m[key]))
    ci: dict[str, list[float] | None] = {}
    for key in samples:
        ci[key] = [float(np.percentile(samples[key], 2.5)), float(np.percentile(samples[key], 97.5))]
    return {"endpoint": endpoint, "point": point, "bootstrap_replicates": bootstraps, "unit": "subject_id", "ci95": ci}


def decision_curve(rows: list[dict[str, Any]], endpoint: str) -> list[dict[str, Any]]:
    y_all, p_all, keep = endpoint_arrays(rows, endpoint)
    y, p = y_all[keep].astype(int), p_all[keep]
    n = len(y)
    prev = float(y.mean())
    output = []
    for threshold in np.linspace(0.01, 0.50, 100):
        pred = p >= threshold
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        model_nb = tp / n - fp / n * threshold / (1 - threshold)
        output.append({
            "endpoint": endpoint,
            "threshold": float(threshold),
            "model_net_benefit": float(model_nb),
            "treat_all_net_benefit": float(prev - (1 - prev) * threshold / (1 - threshold)),
            "treat_none_net_benefit": 0.0,
            "n": n,
        })
    return output


def subgroup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions: list[tuple[str, Any]] = [
        ("all", lambda r: True),
        ("sex_0", lambda r: str(r.get("sex")) == "0"),
        ("sex_1", lambda r: str(r.get("sex")) == "1"),
        ("oov_none", lambda r: int(r.get("features", {}).get("oov_audit", {}).get("total", 0)) == 0),
        ("oov_any", lambda r: int(r.get("features", {}).get("oov_audit", {}).get("total", 0)) > 0),
    ]
    age_values = sorted({str(r.get("age_group")) for r in rows})
    definitions.extend((f"age_{age}", lambda r, age=age: str(r.get("age_group")) == age) for age in age_values)
    output: list[dict[str, Any]] = []
    for endpoint in ("any_readmission", "ap_specific_readmission"):
        y_all, p_all, valid = endpoint_arrays(rows, endpoint)
        threshold = float(rows[0]["prediction"]["thresholds"][endpoint])
        for name, predicate in definitions:
            mask = np.asarray([predicate(r) for r in rows], dtype=bool) & valid
            m = metrics(y_all[mask].astype(int), p_all[mask], threshold)
            output.append({"endpoint": endpoint, "subgroup": name, **m})
    return output


def selective_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    y_all, p_all, valid = endpoint_arrays(rows, "any_readmission")
    for level in LEVELS:
        abstain = np.asarray([
            bool(r.get("prediction", {}).get("conformal", {}).get(level, {}).get("global", {}).get("abstain", True))
            for r in rows
        ])
        evaluated = valid & ~abstain
        m = metrics(y_all[evaluated].astype(int), p_all[evaluated], float(rows[0]["prediction"]["thresholds"]["any_readmission"]))
        # The locked conformal registry uses short leaves.  Map the row-level
        # endpoint label to the same vocabulary before checking coverage.
        observed_leaf = np.asarray([
            "none" if (r.get("label", {}).get("any_readmission") or 0) == 0
            else ("ap" if (r.get("label", {}).get("AP_specific") or 0) == 1 else "other")
            for r in rows
        ], dtype=object)
        sets = [set(r.get("prediction", {}).get("conformal", {}).get(level, {}).get("global", {}).get("set", [])) for r in rows]
        covered_all = [bool(observed_leaf[i] in sets[i]) for i in range(len(rows)) if valid[i]]
        covered_non_abstained = [bool(observed_leaf[i] in sets[i]) for i in range(len(rows)) if evaluated[i]]
        output.append({
            "endpoint": "any_readmission",
            "nominal_coverage": float(int(level) / 100),
            "n_evaluable": int(valid.sum()),
            "n_non_abstained": int(evaluated.sum()),
            "abstention_rate": float(abstain[valid].mean()),
            "observed_marginal_coverage": float(np.mean(covered_all)) if covered_all else None,
            "observed_non_abstained_coverage": float(np.mean(covered_non_abstained)) if covered_non_abstained else None,
            **m,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    rows = load_concatenated_json(args.predictions)
    overall = [bootstrap_metrics(rows, endpoint) for endpoint in ("any_readmission", "ap_specific_readmission")]
    dca = [item for endpoint in ("any_readmission", "ap_specific_readmission") for item in decision_curve(rows, endpoint)]
    groups = subgroup_rows(rows)
    selective = selective_rows(rows)
    summaries = {
        "status": "PASS_MIMIC_UPGRADE_AGGREGATE_ONLY",
        "source_predictions_sha256": sha256(args.predictions),
        "source_row_count": len(rows),
        "endpoint_contract": {"ap_specific_readmission": {"label_field": "AP_specific", "probability_field": "p_ap_specific_readmission_calibrated"}},
        "bootstrap": {"replicates": BOOTSTRAPS, "unit": "subject_id", "seed": RNG_SEED},
        "overall": overall,
        "baseline": {"type": "prevalence_only", "note": "No MIMIC fitting or recalibration was performed; model baselines require a separately locked NRD-trained artifact."},
        "privacy": {"row_level_outputs": False, "public_release": "aggregate_only"},
    }
    (args.output_dir / "upgrade_summary.json").write_text(stable_json(summaries), encoding="utf-8")
    (args.output_dir / "dca_curve.json").write_text(stable_json(dca), encoding="utf-8")
    (args.output_dir / "subgroup_metrics.json").write_text(stable_json(groups), encoding="utf-8")
    (args.output_dir / "selective_prediction.json").write_text(stable_json(selective), encoding="utf-8")
    (args.output_dir / "analysis_contract.md").write_text(
        "# Locked MIMIC upgrade analysis\n\n"
        "This aggregate-only analysis explicitly binds the AP-specific label to `AP_specific` and the frozen probability to `p_ap_specific_readmission_calibrated`. It reuses the existing frozen predictions without fitting, recalibrating, changing thresholds, growing the vocabulary, or releasing row-level data. Subject-cluster bootstrap uncertainty uses 1,000 replicates. The prevalence-only comparator is reported transparently; a NRD-trained model comparator requires a separately bound baseline artifact and is not inferred from MIMIC.\n",
        encoding="utf-8",
    )
    print(stable_json({"status": summaries["status"], "output": str(args.output_dir.resolve()), "rows": len(rows)}))


if __name__ == "__main__":
    main()
