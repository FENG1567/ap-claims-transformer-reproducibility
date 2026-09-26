#!/usr/bin/env python3
"""Additive IJMI reanalysis of the frozen/corrected NRD evaluation.

The script reads frozen predictions and source artifacts, never changes them,
and creates one new output directory.  It deliberately labels every result as
post-unblinding reanalysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


SEED = 20260921
OUTCOMES = {
    "any_readmission": "outcome_any_readmission",
    "ap_specific_readmission": "outcome_ap_specific_readmission",
}
PRIMARY_MODELS = ("joint_main", "lightgbm")
ABLATIONS = (
    "mlm_only", "scratch", "ap_only", "no_hierarchy_parameter_matched",
    "no_prday", "no_prior", "no_hospital_socioeconomic", "no_year",
    "common_variable_only",
)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                      allow_nan=False) + "\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(float)


def weighted_quantile(x: np.ndarray, q: Iterable[float], w: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs, ws = x[order], w[order]
    c = np.cumsum(ws) - 0.5 * ws
    c /= ws.sum()
    return np.interp(np.asarray(list(q), float), c, xs)


def cal_newton(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    x = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))
    b0, b1 = 0.0, 1.0
    for _ in range(30):
        mu = expit(b0 + b1 * x)
        v = np.clip(mu * (1 - mu), 1e-9, None)
        g0 = np.sum(w * (y - mu))
        g1 = np.sum(w * (y - mu) * x)
        h00 = np.sum(w * v)
        h01 = np.sum(w * v * x)
        h11 = np.sum(w * v * x * x)
        det = h00 * h11 - h01 * h01
        if det <= 1e-12:
            break
        d0 = (h11 * g0 - h01 * g1) / det
        d1 = (-h01 * g0 + h00 * g1) / det
        b0 += d0
        b1 += d1
        if max(abs(d0), abs(d1)) < 1e-9:
            break
    return float(b0), float(b1)


def metric_point(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, np.int8)
    w = np.asarray(w, float)
    b0, b1 = cal_newton(y, p, w)
    obs = float(np.sum(w * y) / np.sum(w))
    exp = float(np.sum(w * p) / np.sum(w))
    cuts = np.unique(weighted_quantile(p, np.linspace(0, 1, 11), w))
    if len(cuts) < 3:
        ici = float("nan")
    else:
        bins = np.clip(np.digitize(p, cuts[1:-1], right=True), 0, len(cuts) - 2)
        ici_num = 0.0
        for b in np.unique(bins):
            m = bins == b
            sw = w[m].sum()
            ici_num += sw * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
        ici = float(ici_num / w.sum())
    return {
        "n": int(len(y)),
        "events": int(y.sum()),
        "weighted_prevalence": obs,
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
        "auprc": float(average_precision_score(y, p, sample_weight=w)),
        "brier": float(np.sum(w * (y - p) ** 2) / np.sum(w)),
        "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "calibration_intercept": b0,
        "calibration_slope": b1,
        "observed_expected_ratio": float(obs / exp) if exp > 0 else float("nan"),
        "ici_decile": ici,
    }


def make_psu_bootstrap(frame: pd.DataFrame, reps: int, seed: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    psu = frame[["NRD_STRATUM", "hospital_hash"]].astype(str).drop_duplicates()
    psu = psu.sort_values(["NRD_STRATUM", "hospital_hash"], kind="mergesort").reset_index(drop=True)
    key_to_id = {(r.NRD_STRATUM, r.hospital_hash): i for i, r in psu.iterrows()}
    row_gid = np.fromiter((key_to_id[(str(s), str(h))] for s, h in
                           zip(frame["NRD_STRATUM"], frame["hospital_hash"])),
                          dtype=np.int32, count=len(frame))
    rng = np.random.default_rng(seed)
    mult = np.zeros((reps, len(psu)), dtype=np.uint16)
    for _, ids in psu.groupby("NRD_STRATUM", sort=False).groups.items():
        ids = np.asarray(list(ids), dtype=np.int32)
        m = len(ids)
        if m == 1:
            mult[:, ids[0]] = 1
        else:
            mult[:, ids] = rng.multinomial(m, np.repeat(1 / m, m), size=reps)
    return mult, row_gid, psu


def sorted_boot_metrics(y: np.ndarray, p: np.ndarray, base_w: np.ndarray,
                        mult: np.ndarray, gid: np.ndarray, batch: int = 24) -> dict[str, np.ndarray]:
    y = np.asarray(y, np.int8)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    base_w = np.asarray(base_w, float)
    asc = np.argsort(p, kind="mergesort")
    desc = asc[::-1]
    yy_a, yy_d = y[asc], y[desc]
    pp_a, pp_d = p[asc], p[desc]
    ww_a, ww_d = base_w[asc], base_w[desc]
    gg_a, gg_d = gid[asc], gid[desc]
    logloss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    result = {k: np.full(len(mult), np.nan) for k in ("auroc", "auprc", "brier", "log_loss", "prevalence", "eo")}
    for start in range(0, len(mult), batch):
        stop = min(start + batch, len(mult))
        b = stop - start
        wa = mult[start:stop, gg_a].T.astype(float) * ww_a[:, None]
        wd = mult[start:stop, gg_d].T.astype(float) * ww_d[:, None]
        sw = wa.sum(axis=0)
        pos_a = wa * yy_a[:, None]
        neg_a = wa * (1 - yy_a)[:, None]
        sp, sn = pos_a.sum(axis=0), neg_a.sum(axis=0)
        cneg = np.cumsum(neg_a, axis=0)
        result["auroc"][start:stop] = np.sum(pos_a * (cneg - 0.5 * neg_a), axis=0) / (sp * sn)
        pos_d = wd * yy_d[:, None]
        cum_pos = np.cumsum(pos_d, axis=0)
        cum_w = np.cumsum(wd, axis=0)
        precision = np.divide(cum_pos, cum_w, out=np.zeros_like(cum_pos), where=cum_w > 0)
        result["auprc"][start:stop] = np.sum(precision * pos_d, axis=0) / pos_d.sum(axis=0)
        row_w = mult[start:stop, gid].T.astype(float) * base_w[:, None]
        result["brier"][start:stop] = np.sum(row_w * (y - p)[:, None] ** 2, axis=0) / sw
        result["log_loss"][start:stop] = np.sum(row_w * logloss[:, None], axis=0) / sw
        obs = np.sum(row_w * y[:, None], axis=0) / sw
        exp = np.sum(row_w * p[:, None], axis=0) / sw
        result["prevalence"][start:stop] = obs
        result["eo"][start:stop] = obs / exp
    return result


def ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    return tuple(float(x) for x in np.quantile(vals, [alpha / 2, 1 - alpha / 2]))


def operation_point(y: np.ndarray, p: np.ndarray, w: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    tp = np.sum(w * (pred & (y == 1)))
    fp = np.sum(w * (pred & (y == 0)))
    tn = np.sum(w * (~pred & (y == 0)))
    fn = np.sum(w * (~pred & (y == 1)))
    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "ppv": float(tp / (tp + fp)),
        "npv": float(tn / (tn + fn)),
        "triggered_per_1000": float(1000 * (tp + fp) / (tp + fp + tn + fn)),
        "net_benefit": float(tp / w.sum() - fp / w.sum() * threshold / (1 - threshold)),
    }


def operation_boot(y: np.ndarray, p: np.ndarray, base_w: np.ndarray, threshold: float,
                   mult: np.ndarray, gid: np.ndarray) -> pd.DataFrame:
    pred = p >= threshold
    rows = []
    for start in range(0, len(mult), 32):
        stop = min(start + 32, len(mult))
        rw = mult[start:stop, gid].T.astype(float) * base_w[:, None]
        tp = np.sum(rw * (pred & (y == 1))[:, None], axis=0)
        fp = np.sum(rw * (pred & (y == 0))[:, None], axis=0)
        tn = np.sum(rw * (~pred & (y == 0))[:, None], axis=0)
        fn = np.sum(rw * (~pred & (y == 1))[:, None], axis=0)
        total = tp + fp + tn + fn
        for i in range(stop - start):
            rows.append({
                "sensitivity": tp[i] / (tp[i] + fn[i]),
                "specificity": tn[i] / (tn[i] + fp[i]),
                "ppv": tp[i] / (tp[i] + fp[i]),
                "npv": tn[i] / (tn[i] + fn[i]),
                "triggered_per_1000": 1000 * (tp[i] + fp[i]) / total[i],
                "net_benefit": tp[i] / total[i] - fp[i] / total[i] * threshold / (1 - threshold),
            })
    return pd.DataFrame(rows)


def calibration_curve(y: np.ndarray, p: np.ndarray, w: np.ndarray, mult: np.ndarray,
                      gid: np.ndarray, bins: int = 10) -> pd.DataFrame:
    cuts = np.unique(weighted_quantile(p, np.linspace(0, 1, bins + 1), w))
    bindex = np.clip(np.digitize(p, cuts[1:-1], right=True), 0, len(cuts) - 2)
    rows = []
    for b in sorted(np.unique(bindex)):
        m = bindex == b
        pred = np.average(p[m], weights=w[m])
        obs = np.average(y[m], weights=w[m])
        boot_obs, boot_pred = [], []
        for start in range(0, len(mult), 64):
            rw = mult[start:start + 64, gid[m]].T.astype(float) * w[m, None]
            sw = rw.sum(axis=0)
            boot_obs.extend(np.sum(rw * y[m, None], axis=0) / sw)
            boot_pred.extend(np.sum(rw * p[m, None], axis=0) / sw)
        lo, hi = ci(np.asarray(boot_obs))
        plo, phi = ci(np.asarray(boot_pred))
        rows.append({"bin": int(b + 1), "n": int(m.sum()), "predicted": pred,
                     "observed": obs, "observed_ci_low": lo, "observed_ci_high": hi,
                     "predicted_ci_low": plo, "predicted_ci_high": phi})
    return pd.DataFrame(rows)


def subgroup_metrics(frame: pd.DataFrame, outcome: str, model: str, w: np.ndarray,
                     mult: np.ndarray, gid: np.ndarray) -> pd.DataFrame:
    rows = []
    yall = frame[OUTCOMES[outcome]].to_numpy(np.int8)
    pall = frame[f"p_cal__{outcome}__{model}"].to_numpy(float)
    for variable in ("FEMALE", "age_group"):
        for level in sorted(frame[variable].astype(str).unique()):
            m = frame[variable].astype(str).eq(level).to_numpy()
            if m.sum() < 100 or yall[m].sum() < 20:
                continue
            point = metric_point(yall[m], pall[m], w[m])
            boot = sorted_boot_metrics(yall[m], pall[m], w[m], mult[:500], gid[m])
            row = {"outcome": outcome, "model": model, "variable": variable, "level": level, **point}
            for metric in ("auroc", "auprc", "brier", "log_loss", "eo"):
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = ci(boot[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def full_period_audit(root: Path) -> pd.DataFrame:
    """Independent, null-safe numeric reconstruction for 2018-2021."""
    rows = []
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    con.execute("SET memory_limit='24GB'")
    try:
        for year in (2018, 2019, 2020, 2021):
            adm = root / "data" / "nrd" / f"year={year}" / "admissions_model.parquet"
            eps = root / "data" / "nrd" / f"year={year}" / "ap_episodes.parquet"
            q = f"""
            WITH ordered AS (
              SELECT *, lead(NRD_DaysToEvent) OVER
                (PARTITION BY patient_hash ORDER BY NRD_DaysToEvent, encounter_hash)
                - NRD_DaysToEvent - greatest(LOS,0) AS immediate_gap
              FROM read_parquet('{adm}')
            ), base AS (
              SELECT * FROM ordered WHERE principal_ap AND AGE>=18 AND DIED=0
                AND DISPUNIFORM NOT IN (2,20) AND DMONTH BETWEEN 1 AND 11
                AND LOS>=0 AND (immediate_gap IS NULL OR immediate_gap>=1)
            ), cand AS (
              SELECT i.encounter_hash,
                min(r.NRD_DaysToEvent) FILTER (WHERE r.planned_status=-1) AS first_unknown,
                min(r.NRD_DaysToEvent) FILTER (WHERE r.planned_status=0) AS first_unplanned,
                arg_min(r.primary_cause_code, struct_pack(d:=r.NRD_DaysToEvent,e:=r.encounter_hash))
                  FILTER (WHERE r.planned_status=0) AS first_leaf
              FROM base i LEFT JOIN read_parquet('{adm}') r
                ON i.patient_hash=r.patient_hash
               AND r.NRD_DaysToEvent-i.NRD_DaysToEvent-greatest(i.LOS,0) BETWEEN 1 AND 30
              GROUP BY i.encounter_hash
            ), rebuilt AS (
              SELECT b.encounter_hash,
                (c.first_unplanned IS NOT NULL)::INTEGER AS y_any,
                (c.first_leaf=1 AND c.first_unplanned IS NOT NULL)::INTEGER AS y_ap
              FROM base b JOIN cand c USING(encounter_hash)
              WHERE c.first_unknown IS NULL OR
                    (c.first_unplanned IS NOT NULL AND c.first_unknown>c.first_unplanned)
            ), original AS (
              SELECT encounter_hash,
                any_unplanned_readmission_30d::INTEGER AS y_any,
                ap_specific_readmission_30d::INTEGER AS y_ap
              FROM read_parquet('{eps}') WHERE primary_analysis_eligible
            )
            SELECT {year} AS year,
              (SELECT count(*) FROM rebuilt) AS rebuilt_n,
              (SELECT count(*) FROM original) AS original_n,
              (SELECT sum(y_any) FROM rebuilt) AS rebuilt_any_events,
              (SELECT sum(y_any) FROM original) AS original_any_events,
              (SELECT sum(y_ap) FROM rebuilt) AS rebuilt_ap_events,
              (SELECT sum(y_ap) FROM original) AS original_ap_events,
              (SELECT count(*) FROM rebuilt r FULL JOIN original o USING(encounter_hash)
               WHERE r.encounter_hash IS NULL OR o.encounter_hash IS NULL) AS key_mismatch_n,
              (SELECT count(*) FROM rebuilt r JOIN original o USING(encounter_hash)
               WHERE r.y_any<>o.y_any OR r.y_ap<>o.y_ap) AS label_mismatch_n
            """
            rows.append(con.execute(q).fetchdf().iloc[0].to_dict())
    finally:
        con.close()
    return pd.DataFrame(rows)


def cohort_characteristics(root: Path, corrected_derivative: Path) -> pd.DataFrame:
    rows = []
    for year in (2018, 2019, 2020, 2021):
        eps = pq.read_table(root / "data" / "nrd" / f"year={year}" / "ap_episodes.parquet",
                            filters=[("primary_analysis_eligible", "=", True)]).to_pandas()
        if year <= 2020:
            groups = [("2018-2020 development", eps)]
        else:
            groups = [("2021A", eps[eps.analysis_partition.eq("2021A")]),
                      ("2021B", eps[eps.analysis_partition.eq("2021B")])]
        for name, f in groups:
            if f.empty:
                continue
            w = numeric(f.DISCWT)
            rows.append({
                "cohort": name, "year": year, "episodes": len(f),
                "patients": f.patient_hash.nunique(), "hospitals": f.hospital_hash.nunique(),
                "repeat_episode_percent": 100 * (1 - f.patient_hash.nunique() / len(f)),
                "weighted_population": w.sum(),
                "age_mean": np.average(numeric(f.AGE), weights=w),
                "female_percent": 100 * np.average(numeric(f.FEMALE), weights=w),
                "los_median": weighted_quantile(numeric(f.LOS), [0.5], w)[0],
                "any_readmission_percent": 100 * np.average(f.any_unplanned_readmission_30d.astype(int), weights=w),
                "ap_readmission_percent": 100 * np.average(f.ap_specific_readmission_30d.astype(int), weights=w),
                "severity_3_4_percent": 100 * np.average(numeric(f.APRDRG_Severity) >= 3, weights=w),
                "ed_percent": 100 * np.average(numeric(f.HCUP_ED) > 0, weights=w),
                "elective_percent": 100 * np.average(numeric(f.ELECTIVE) == 1, weights=w),
                "medicare_percent": 100 * np.average(numeric(f.PAY1) == 1, weights=w),
            })
    f = pq.read_table(corrected_derivative).to_pandas()
    w = numeric(f.DISCWT)
    rows.append({
        "cohort": "2022 temporal test", "year": 2022, "episodes": len(f),
        "patients": f.patient_hash.nunique(), "hospitals": f.hospital_hash.nunique(),
        "repeat_episode_percent": 100 * (1 - f.patient_hash.nunique() / len(f)),
        "weighted_population": w.sum(), "age_mean": np.average(numeric(f.AGE), weights=w),
        "female_percent": 100 * np.average(numeric(f.FEMALE), weights=w),
        "los_median": weighted_quantile(numeric(f.LOS), [0.5], w)[0],
        "any_readmission_percent": 100 * np.average(f.any_unplanned_readmission_30d.astype(int), weights=w),
        "ap_readmission_percent": 100 * np.average(f.ap_specific_readmission_30d.astype(int), weights=w),
        "severity_3_4_percent": 100 * np.average(numeric(f.APRDRG_Severity) >= 3, weights=w),
        "ed_percent": 100 * np.average(numeric(f.HCUP_ED) > 0, weights=w),
        "elective_percent": 100 * np.average(numeric(f.ELECTIVE) == 1, weights=w),
        "medicare_percent": 100 * np.average(numeric(f.PAY1) == 1, weights=w),
    })
    out = pd.DataFrame(rows)
    # Combine development-year rows by retaining each year and add a pooled row.
    dev = out[out.cohort.eq("2018-2020 development")]
    if len(dev) == 3:
        pooled = {"cohort": "2018-2020 development pooled", "year": "2018-2020"}
        for col in out.columns:
            if col in pooled or col == "cohort":
                continue
            if col in ("episodes", "patients", "hospitals", "weighted_population"):
                pooled[col] = dev[col].sum()
            else:
                pooled[col] = np.average(dev[col], weights=dev.episodes)
        out = pd.concat([out, pd.DataFrame([pooled])], ignore_index=True)
    return out


def first_index_sensitivity(frame: pd.DataFrame, derivative: pd.DataFrame) -> pd.DataFrame:
    # encounter_hash sorting is deterministic; the sensitivity estimand is one
    # eligible index episode per patient, independent of outcomes/predictions.
    order = derivative[["encounter_hash", "patient_hash"]].copy()
    order["encounter_hash"] = order.encounter_hash.astype(str)
    first_keys = (order.sort_values(["patient_hash", "encounter_hash"], kind="mergesort")
                       .drop_duplicates("patient_hash").encounter_hash)
    sub = frame[frame.encounter_hash.astype(str).isin(set(first_keys))].copy()
    w = numeric(sub.DISCWT)
    rows = []
    for outcome, ycol in OUTCOMES.items():
        y = sub[ycol].to_numpy(np.int8)
        for model in PRIMARY_MODELS:
            rows.append({"analysis": "first eligible index per patient",
                         "outcome": outcome, "model": model,
                         **metric_point(y, sub[f"p_cal__{outcome}__{model}"].to_numpy(float), w)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--derivative", type=Path, required=True)
    ap.add_argument("--evaluation-spec", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--primary-reps", type=int, default=5000)
    ap.add_argument("--secondary-reps", type=int, default=1000)
    args = ap.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    if output.exists() or list(output.parent.glob(output.name + ".partial-*")):
        raise SystemExit("Refusing to overwrite output/partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        frame = pq.read_table(args.predictions).to_pandas()
        derivative = pq.read_table(args.derivative).to_pandas()
        if len(frame) != 119114 or len(derivative) != len(frame):
            raise RuntimeError("Expected corrected 119,114-row cohort")
        if frame.encounter_hash.duplicated().any():
            raise RuntimeError("Duplicate prediction key")
        weights = numeric(frame.DISCWT)
        spec = json.loads(args.evaluation_spec.read_text())
        thresholds = {m: spec["models"][m]["thresholds"] for m in spec["models"]}

        mult, gid, psu = make_psu_bootstrap(frame, args.primary_reps, SEED)
        psu.to_csv(temp / "nrd_psu_inventory.csv", index=False)

        absolute_rows, primary_boot = [], {}
        for outcome, ycol in OUTCOMES.items():
            y = frame[ycol].to_numpy(np.int8)
            for model in PRIMARY_MODELS:
                p = frame[f"p_cal__{outcome}__{model}"].to_numpy(float)
                point = metric_point(y, p, weights)
                boot = sorted_boot_metrics(y, p, weights, mult, gid)
                primary_boot[(outcome, model)] = boot
                row = {"outcome": outcome, "model": model, **point,
                       "bootstrap_scheme": "hospital PSU within NRD_STRATUM",
                       "bootstrap_replicates": args.primary_reps}
                for metric in ("auroc", "auprc", "brier", "log_loss", "prevalence", "eo"):
                    row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = ci(boot[metric])
                absolute_rows.append(row)
        pd.DataFrame(absolute_rows).to_csv(temp / "primary_absolute_metrics_ci.csv", index=False)

        # Paired primary AUPRC differences and simultaneous max-T intervals.
        comparisons = []
        diffs, points, ses = {}, {}, {}
        for outcome in OUTCOMES:
            d = primary_boot[(outcome, "joint_main")]["auprc"] - primary_boot[(outcome, "lightgbm")]["auprc"]
            p_joint = frame[f"p_cal__{outcome}__joint_main"].to_numpy(float)
            p_lgb = frame[f"p_cal__{outcome}__lightgbm"].to_numpy(float)
            y = frame[OUTCOMES[outcome]].to_numpy(np.int8)
            point = (average_precision_score(y, p_joint, sample_weight=weights) -
                     average_precision_score(y, p_lgb, sample_weight=weights))
            diffs[outcome], points[outcome], ses[outcome] = d, point, float(np.std(d, ddof=1))
        centered_t = np.column_stack([(diffs[o] - points[o]) / ses[o] for o in OUTCOMES])
        qmax = float(np.quantile(np.max(np.abs(centered_t), axis=1), 0.95))
        for outcome in OUTCOMES:
            lo, hi = ci(diffs[outcome])
            max_abs = np.max(np.abs(centered_t), axis=1)
            obs_t = abs(points[outcome] / ses[outcome])
            comparisons.append({
                "outcome": outcome, "contrast": "joint_main - lightgbm",
                "estimate": points[outcome], "percentile_ci_low": lo,
                "percentile_ci_high": hi, "max_t_critical": qmax,
                "simultaneous_ci_low": points[outcome] - qmax * ses[outcome],
                "simultaneous_ci_high": points[outcome] + qmax * ses[outcome],
                "max_t_adjusted_p": (1 + np.sum(max_abs >= obs_t)) / (len(max_abs) + 1),
                "replicates": args.primary_reps,
            })
        pd.DataFrame(comparisons).to_csv(temp / "co_primary_simultaneous_inference.csv", index=False)

        # Exploratory paired ablations; Holm adjustment within each outcome.
        ablation_rows = []
        for outcome, ycol in OUTCOMES.items():
            y = frame[ycol].to_numpy(np.int8)
            joint_point = average_precision_score(y, frame[f"p_cal__{outcome}__joint_main"], sample_weight=weights)
            joint_boot = primary_boot[(outcome, "joint_main")]["auprc"][:args.secondary_reps]
            local = []
            for model in ABLATIONS:
                p = frame[f"p_cal__{outcome}__{model}"].to_numpy(float)
                point = average_precision_score(y, p, sample_weight=weights) - joint_point
                boot = sorted_boot_metrics(y, p, weights, mult[:args.secondary_reps], gid)["auprc"] - joint_boot
                lo, hi = ci(boot)
                pval = 2 * min((1 + np.sum(boot <= 0)) / (len(boot) + 1),
                               (1 + np.sum(boot >= 0)) / (len(boot) + 1))
                local.append({"outcome": outcome, "contrast": f"{model} - joint_main",
                              "estimate": point, "ci_low": lo, "ci_high": hi,
                              "unadjusted_p": min(1.0, pval), "replicates": args.secondary_reps,
                              "interpretation": "exploratory; no equivalence claim"})
            order = np.argsort([r["unadjusted_p"] for r in local])
            adjusted = np.empty(len(local))
            running = 0.0
            for rank, idx in enumerate(order):
                running = max(running, (len(local) - rank) * local[idx]["unadjusted_p"])
                adjusted[idx] = min(1.0, running)
            for r, adj in zip(local, adjusted):
                r["holm_adjusted_p"] = adj
            ablation_rows.extend(local)
        pd.DataFrame(ablation_rows).to_csv(temp / "transformer_ablation_paired_ci.csv", index=False)

        # Frozen operating points, clinical workload and DCA uncertainty.
        op_rows = []
        for outcome, ycol in OUTCOMES.items():
            y = frame[ycol].to_numpy(np.int8)
            for model in PRIMARY_MODELS:
                p = frame[f"p_cal__{outcome}__{model}"].to_numpy(float)
                threshold = float(thresholds[model][outcome])
                point = operation_point(y, p, weights, threshold)
                boot = operation_boot(y, p, weights, threshold, mult[:args.secondary_reps], gid)
                row = {"outcome": outcome, "model": model, **point,
                       "replicates": args.secondary_reps}
                for metric in boot.columns:
                    row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = ci(boot[metric].to_numpy())
                op_rows.append(row)
        pd.DataFrame(op_rows).to_csv(temp / "frozen_threshold_operating_metrics_ci.csv", index=False)

        # Curves and prespecified sex/age subgroups for the primary Transformer.
        subgroup_parts = []
        for outcome, ycol in OUTCOMES.items():
            y = frame[ycol].to_numpy(np.int8)
            for model in PRIMARY_MODELS:
                p = frame[f"p_cal__{outcome}__{model}"].to_numpy(float)
                curve = calibration_curve(y, p, weights, mult[:500], gid)
                curve.insert(0, "model", model); curve.insert(0, "outcome", outcome)
                curve.to_csv(temp / f"calibration_curve_{outcome}_{model}.csv", index=False)
                subgroup_parts.append(subgroup_metrics(frame, outcome, model, weights, mult, gid))
        pd.concat(subgroup_parts, ignore_index=True).to_csv(temp / "sex_age_subgroup_metrics_ci.csv", index=False)

        first_index_sensitivity(frame, derivative).to_csv(temp / "first_index_sensitivity.csv", index=False)
        audit = full_period_audit(root)
        audit.to_csv(temp / "full_period_episode_reconstruction_audit.csv", index=False)
        cohort_characteristics(root, args.derivative).to_csv(temp / "nrd_cohort_characteristics.csv", index=False)

        # Pre/post correction summary is documentary and never treated as a
        # second final test.  Values come from authenticated manifests.
        corrected_manifest = json.loads((args.derivative.parent.parent / "etl_technical_amendment_v8_corrected" / "manifest.json").read_text())
        qc = corrected_manifest["qc"]
        stable = {
            "status": "PASS_POST_UNBLINDING_IJMI_REANALYSIS",
            "post_unblinding": True,
            "frozen_models_changed": False,
            "cohort_correction": qc,
            "primary_bootstrap_replicates": args.primary_reps,
            "secondary_bootstrap_replicates": args.secondary_reps,
            "psu_count": int(len(psu)),
            "stratum_count": int(psu.NRD_STRATUM.nunique()),
            "patient_count": int(frame.patient_hash.nunique()),
            "episode_count": int(len(frame)),
            "hospital_count": int(frame.hospital_hash.nunique()),
            "limitations": [
                "The analysis is post-unblinding and cannot restore an untouched sealed test.",
                "Hospital-within-stratum bootstrap is primary; patient dependence is additionally described by first-index sensitivity.",
                "No 2022 result was used to change models, calibration, thresholds, ontology or endpoint mappings."
            ],
        }
        (temp / "reanalysis_summary.json").write_text(stable_json(stable), encoding="utf-8")
        manifest = {"status": stable["status"], "files": []}
        for path in sorted(temp.glob("*")):
            if path.is_file():
                manifest["files"].append({"name": path.name, "bytes": path.stat().st_size,
                                          "sha256": sha256(path)})
        (temp / "manifest.json").write_text(stable_json(manifest), encoding="utf-8")
        os.replace(temp, output)
        print(stable_json({"status": stable["status"], "output": str(output),
                           "files": len(manifest["files"]) - 1}))
    except Exception:
        raise


if __name__ == "__main__":
    main()
