#!/usr/bin/env python3
"""Extended restricted MIMIC analysis for IJMI revision.

Only aggregate outputs are written.  Row-level MIMIC records and predictions
remain in the restricted input directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


SEED = 20260921
OUTCOMES = {
    "any_readmission": ("any_readmission", "p_any_readmission_calibrated"),
    "ap_specific_readmission": ("AP_specific", "p_ap_specific_readmission_calibrated"),
}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_concatenated_json(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    result, pos = [], 0
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        value, pos = decoder.raw_decode(text, pos)
        if not isinstance(value, dict):
            raise RuntimeError("Expected object stream")
        result.append(value)
    return result


def cal_newton(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    x = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    b0, b1 = 0.0, 1.0
    for _ in range(30):
        mu = expit(b0 + b1 * x)
        v = np.clip(mu * (1 - mu), 1e-9, None)
        g0, g1 = np.sum(w * (y - mu)), np.sum(w * (y - mu) * x)
        h00, h01, h11 = np.sum(w * v), np.sum(w * v * x), np.sum(w * v * x * x)
        det = h00 * h11 - h01 * h01
        if det <= 1e-12:
            break
        d0, d1 = (h11 * g0 - h01 * g1) / det, (-h01 * g0 + h00 * g1) / det
        b0, b1 = b0 + d0, b1 + d1
        if max(abs(d0), abs(d1)) < 1e-9:
            break
    return float(b0), float(b1)


def metric(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    y, p = np.asarray(y, np.int8), np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    if w is None:
        w = np.ones(len(y), float)
    b0, b1 = cal_newton(y, p, w)
    obs, exp = np.average(y, weights=w), np.average(p, weights=w)
    return {
        "n": int(len(y)), "events": int(y.sum()), "prevalence": float(obs),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "auprc": float(average_precision_score(y, p, sample_weight=w)),
        "brier": float(np.average((y - p) ** 2, weights=w)),
        "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "calibration_intercept": b0, "calibration_slope": b1,
        "observed_expected_ratio": float(obs / exp),
    }


def subject_bootstrap_indices(subjects: np.ndarray, reps: int) -> tuple[list[np.ndarray], np.ndarray]:
    unique, inverse = np.unique(subjects, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(SEED)
    counts = rng.multinomial(len(unique), np.repeat(1 / len(unique), len(unique)), size=reps)
    return groups, counts


def bootstrap_metrics(y: np.ndarray, p: np.ndarray, groups: list[np.ndarray],
                      counts: np.ndarray) -> pd.DataFrame:
    rows = []
    for rep in range(len(counts)):
        w = np.zeros(len(y), float)
        for group_id, idx in enumerate(groups):
            if counts[rep, group_id]:
                w[idx] = counts[rep, group_id]
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        rows.append(metric(y, p, w))
    return pd.DataFrame(rows)


def ci(series: pd.Series | np.ndarray) -> tuple[float, float]:
    values = np.asarray(series, float)
    values = values[np.isfinite(values)]
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def operating(y: np.ndarray, p: np.ndarray, threshold: float, w: np.ndarray | None = None) -> dict[str, float]:
    if w is None:
        w = np.ones(len(y), float)
    flag = p >= threshold
    tp = np.sum(w * (flag & (y == 1))); fp = np.sum(w * (flag & (y == 0)))
    tn = np.sum(w * (~flag & (y == 0))); fn = np.sum(w * (~flag & (y == 1)))
    total = tp + fp + tn + fn
    return {
        "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp), "npv": tn / (tn + fn),
        "triggered_per_1000": 1000 * (tp + fp) / total,
        "net_benefit": tp / total - fp / total * threshold / (1 - threshold),
    }


def flatten(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        f, lab, pred = r["features"], r["label"], r["prediction"]
        rows.append({
            "subject_id": str(r["subject_id"]), "hadm_count": len(r["hadm_ids"]),
            "sex": str(r["sex"]), "age_group": str(r["age_group"]),
            "age": float(f["AGE"]), "los": float(f["LOS"]),
            "oov_total": int(f["oov_audit"]["total"]),
            "oov_diagnosis": int(f["oov_audit"]["diagnosis"]),
            "oov_procedure": int(f["oov_audit"]["procedure"]),
            "any_readmission": int(lab["any_readmission"]),
            "AP_specific": int(lab["AP_specific"]), "leaf": str(lab["leaf"]),
            "label_status": str(lab["label_status"]),
            "unknown_as_planned": int(lab["bounds"]["unknown_as_planned"]),
            "unknown_as_unplanned": int(lab["bounds"]["unknown_as_unplanned"]),
            "p_any_readmission_calibrated": float(pred["p_any_readmission_calibrated"]),
            "p_ap_specific_readmission_calibrated": float(pred["p_ap_specific_readmission_calibrated"]),
            "threshold_any": float(pred["thresholds"]["any_readmission"]),
            "threshold_ap": float(pred["thresholds"]["ap_specific_readmission"]),
            "model_id": str(pred["model_id"]),
            "conformal": pred["conformal"],
        })
    return pd.DataFrame(rows)


def subgroup_analysis(frame: pd.DataFrame, groups: list[np.ndarray], counts: np.ndarray) -> pd.DataFrame:
    rows = []
    for outcome, (ycol, pcol) in OUTCOMES.items():
        for variable in ("sex", "age_group", "oov_group"):
            for level in sorted(frame[variable].astype(str).unique()):
                m = frame[variable].astype(str).eq(level).to_numpy()
                if m.sum() < 40 or frame.loc[m, ycol].sum() < 10:
                    continue
                y, p = frame.loc[m, ycol].to_numpy(np.int8), frame.loc[m, pcol].to_numpy(float)
                point = metric(y, p)
                # Subject-cluster bootstrap within the subgroup.
                rng = np.random.default_rng(SEED + len(rows))
                subject_values = frame.loc[m, "subject_id"].to_numpy()
                unique_subjects = np.unique(subject_values)
                subject_rows = [np.flatnonzero(subject_values == s) for s in unique_subjects]
                vals = []
                for _ in range(1000):
                    sampled = rng.integers(0, len(unique_subjects), len(unique_subjects))
                    idx = np.concatenate([subject_rows[i] for i in sampled])
                    if len(np.unique(y[idx])) == 2:
                        vals.append(metric(y[idx], p[idx]))
                boot = pd.DataFrame(vals)
                row = {"outcome": outcome, "variable": variable, "level": level, **point}
                for name in ("auroc", "auprc", "brier", "log_loss", "calibration_intercept",
                             "calibration_slope", "observed_expected_ratio"):
                    row[f"{name}_ci_low"], row[f"{name}_ci_high"] = ci(boot[name])
                rows.append(row)
    return pd.DataFrame(rows)


def conformal_analysis(records: list[dict[str, Any]], frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, risk = [], []
    label_alias = {"AP_specific": "AP_specific", "ap_specific": "AP_specific"}
    for level in (80, 90, 95):
        for scope in ("global", "sex", "age_group"):
            sets, covered, abstain = [], [], []
            for r in records:
                item = r["prediction"]["conformal"][str(level)]
                branch = item if scope == "legacy" else item[scope]
                current = list(branch["set"])
                target = label_alias.get(str(r["label"]["leaf"]), str(r["label"]["leaf"]))
                sets.append(len(current)); covered.append(target in current)
                abstain.append(bool(branch["abstain"]))
            summary.append({
                "nominal": level / 100, "scope": scope, "n": len(records),
                "empirical_coverage": float(np.mean(covered)),
                "mean_set_size": float(np.mean(sets)),
                "abstention_rate": float(np.mean(abstain)),
                "task": "mutually exclusive five-class readmission leaf label",
                "guarantee": "empirical transfer coverage only; no exchangeability guarantee",
            })
        # Risk-coverage for each binary clinical endpoint using confidence =
        # singleton leaf-set status.  This does not redefine the conformal task.
        singleton = np.array([len(r["prediction"]["conformal"][str(level)]["global"]["set"]) == 1
                              for r in records])
        for outcome, (ycol, pcol) in OUTCOMES.items():
            y, p = frame[ycol].to_numpy(np.int8), frame[pcol].to_numpy(float)
            if singleton.sum() and len(np.unique(y[singleton])) == 2:
                risk.append({"nominal": level / 100, "outcome": outcome,
                             "coverage_fraction": float(singleton.mean()),
                             "selected_n": int(singleton.sum()),
                             "selected_error_rate": float(np.mean((p[singleton] >= 0.5) != y[singleton])),
                             "selected_auroc": float(roc_auc_score(y[singleton], p[singleton])),
                             "selected_auprc": float(average_precision_score(y[singleton], p[singleton]))})
    return pd.DataFrame(summary), pd.DataFrame(risk)


def calibration_curve(frame: pd.DataFrame, outcome: str, bins: int = 10) -> pd.DataFrame:
    ycol, pcol = OUTCOMES[outcome]
    f = frame[[ycol, pcol]].sort_values(pcol, kind="mergesort").reset_index(drop=True)
    f["bin"] = pd.qcut(np.arange(len(f)), bins, labels=False)
    rows = []
    rng = np.random.default_rng(SEED)
    for b, g in f.groupby("bin"):
        obs = float(g[ycol].mean()); pred = float(g[pcol].mean())
        boot = np.array([g.iloc[rng.integers(0, len(g), len(g))][ycol].mean() for _ in range(2000)])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append({"outcome": outcome, "bin": int(b + 1), "n": len(g),
                     "predicted": pred, "observed": obs,
                     "observed_ci_low": lo, "observed_ci_high": hi})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=5000)
    args = ap.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or list(output.parent.glob(output.name + ".partial-*")):
        raise SystemExit("Refusing to overwrite output/partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))

    records_all = read_concatenated_json(args.predictions)
    unknown_algorithm_n = sum(r["label"]["any_readmission"] is None for r in records_all)
    records = [r for r in records_all if r["label"]["any_readmission"] is not None]
    frame = flatten(records)
    frame["oov_group"] = np.where(frame.oov_total > 0, "OOV>0", "OOV=0")
    if not frame.model_id.eq("common_variable_only").all():
        raise RuntimeError("MIMIC model identity is not common_variable_only")
    bind_mismatch = int(((frame.leaf.eq("AP_specific")).astype(int) != frame.AP_specific).sum())
    if bind_mismatch:
        raise RuntimeError("AP-specific label/leaf binding mismatch")

    groups, counts = subject_bootstrap_indices(frame.subject_id.to_numpy(), args.reps)
    overall_rows, operation_rows = [], []
    for outcome, (ycol, pcol) in OUTCOMES.items():
        y, p = frame[ycol].to_numpy(np.int8), frame[pcol].to_numpy(float)
        point = metric(y, p)
        boot = bootstrap_metrics(y, p, groups, counts)
        row = {"outcome": outcome, "model": "common_variable_only Transformer", **point,
               "bootstrap_replicates": len(boot), "bootstrap_unit": "subject"}
        for name in ("auroc", "auprc", "brier", "log_loss", "calibration_intercept",
                     "calibration_slope", "observed_expected_ratio"):
            row[f"{name}_ci_low"], row[f"{name}_ci_high"] = ci(boot[name])
        overall_rows.append(row)
        threshold = float(frame["threshold_any" if outcome == "any_readmission" else "threshold_ap"].iloc[0])
        op = operating(y, p, threshold)
        op_boot = []
        for rep in range(min(2000, len(counts))):
            w = np.zeros(len(y), float)
            for group_id, idx in enumerate(groups):
                w[idx] = counts[rep, group_id]
            op_boot.append(operating(y, p, threshold, w))
        opdf = pd.DataFrame(op_boot)
        orow = {"outcome": outcome, "threshold": threshold, **op}
        for name in opdf:
            orow[f"{name}_ci_low"], orow[f"{name}_ci_high"] = ci(opdf[name])
        operation_rows.append(orow)
    pd.DataFrame(overall_rows).to_csv(temp / "mimic_overall_metrics_ci.csv", index=False)
    pd.DataFrame(operation_rows).to_csv(temp / "mimic_frozen_threshold_metrics_ci.csv", index=False)

    subgroup_analysis(frame, groups, counts).to_csv(temp / "mimic_sex_age_oov_subgroups_ci.csv", index=False)
    conformal, risk = conformal_analysis(records, frame)
    conformal.to_csv(temp / "mimic_conformal_leaf_coverage.csv", index=False)
    risk.to_csv(temp / "mimic_conformal_risk_coverage.csv", index=False)
    pd.concat([calibration_curve(frame, o) for o in OUTCOMES], ignore_index=True).to_csv(
        temp / "mimic_calibration_curves.csv", index=False)

    # Aggregate cohort/flow and mapping diagnostics.
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    flow = {
        "formal_endpoint_gated_episodes": int(len(frame)),
        "independent_subjects": int(frame.subject_id.nunique()),
        "subjects_with_multiple_collapsed_hadm_ids": int((frame.hadm_count > 1).sum()),
        "any_readmission_events": int(frame.any_readmission.sum()),
        "ap_specific_events": int(frame.AP_specific.sum()),
        "oov_any_episodes": int((frame.oov_total > 0).sum()),
        "oov_any_percent": float(100 * (frame.oov_total > 0).mean()),
        "label_probability_binding_mismatches": bind_mismatch,
        "pre_gate_algorithm_unknown_episodes": int(unknown_algorithm_n),
        "pre_gate_rows": int(len(records_all)),
        "model_id": "common_variable_only",
        "endpoint": "30-day same-system unplanned readmission",
        "calendar_year_interpretation": "MIMIC dates are shifted; displayed year is not used",
        "data_quality_audit": manifest.get("data_quality_audit", manifest.get("cohort", {})),
    }
    (temp / "mimic_cohort_flow_and_mapping.json").write_text(stable_json(flow), encoding="utf-8")
    pd.DataFrame([{
        "cohort": "MIMIC-IV restricted same-system transfer", "episodes": len(frame),
        "subjects": frame.subject_id.nunique(), "age_mean": frame.age.mean(),
        "female_percent": 100 * frame.sex.eq("1").mean(), "los_median": frame.los.median(),
        "any_readmission_percent": 100 * frame.any_readmission.mean(),
        "ap_readmission_percent": 100 * frame.AP_specific.mean(),
        "oov_any_percent": 100 * frame.oov_total.gt(0).mean(),
    }]).to_csv(temp / "mimic_cohort_characteristics.csv", index=False)

    release = {"status": "PASS_RESTRICTED_MIMIC_EXTENDED_ANALYSIS", "row_level_released": False,
               "post_unblinding": True, "model_id": "common_variable_only",
               "files": []}
    for path in sorted(temp.glob("*")):
        if path.is_file():
            release["files"].append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    (temp / "manifest.json").write_text(stable_json(release), encoding="utf-8")
    os.replace(temp, output)
    print(stable_json({"status": release["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
