#!/usr/bin/env python3
"""Build leakage-safe prior-encounter features for eligible AP index stays.

Only admissions beginning before the AP index admission are eligible history.
History is necessarily left-truncated at January because NRD linkage is
calendar-year specific; DMONTH-derived observability flags make this explicit.
No outcome or future-admission field is read by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


YEARS = (2018, 2019, 2020, 2021)
SPECIAL_TOKEN_MAX = 3
ADMISSION_COLUMNS = [
    "encounter_hash", "patient_hash", "NRD_DaysToEvent", "LOS", "DMONTH",
    "HCUP_ED", "ELECTIVE", "APRDRG_Severity", "APRDRG_Risk_Mortality",
    "principal_ap", "principal_biliary", "principal_sepsis_or_organ",
    "planned_status", "dx_tokens", "pr_tokens",
]
INDEX_COLUMNS = [
    "year", "encounter_hash", "patient_hash", "NRD_DaysToEvent", "DMONTH",
    "primary_analysis_eligible", "analysis_partition",
]
SUMMARY_COLUMNS = [
    "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d",
    "prior_ed_count_180d", "prior_nonelective_count_180d",
    "prior_ap_count_180d", "prior_biliary_count_180d",
    "prior_sepsis_or_organ_count_180d", "prior_los_sum_180d",
    "prior_max_severity_180d", "prior_max_mortality_risk_180d",
    "days_since_prior_discharge", "history_30d_fully_observable",
    "history_90d_fully_observable", "history_180d_fully_observable",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean_tokens(values: list[int] | None) -> set[int]:
    if values is None or len(values) == 0:
        return set()
    return {int(v) for v in values if v is not None and int(v) > SPECIAL_TOKEN_MAX}


def _safe_int(value, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return default
    return int(value)


def derive_patient_history(admissions: pd.DataFrame, indexes: pd.DataFrame) -> list[dict]:
    """Derive history for one patient; intended for deterministic unit testing."""
    admissions = admissions.sort_values(["NRD_DaysToEvent", "encounter_hash"], kind="mergesort")
    output: list[dict] = []
    for idx in indexes.sort_values(["NRD_DaysToEvent", "encounter_hash"], kind="mergesort").itertuples(index=False):
        index_day = int(idx.NRD_DaysToEvent)
        # Require a strictly earlier admission and a non-overlapping discharge.
        prior = admissions[
            (admissions["NRD_DaysToEvent"] < index_day)
            & (admissions["NRD_DaysToEvent"] + admissions["LOS"].fillna(0).clip(lower=0) <= index_day)
        ].copy()
        if len(prior):
            prior["gap"] = index_day - (
                prior["NRD_DaysToEvent"].astype(np.int64)
                + prior["LOS"].fillna(0).clip(lower=0).astype(np.int64)
            )
        else:
            prior["gap"] = pd.Series(index=prior.index, dtype=np.int64)
        rec: dict = {
            "year": int(idx.year),
            "encounter_hash": int(idx.encounter_hash),
            "patient_hash": int(idx.patient_hash),
            "analysis_partition": str(idx.analysis_partition),
            "prior_count_ytd": int(len(prior)),
            "days_since_prior_discharge": int(prior["gap"].min()) if len(prior) else -1,
            "history_30d_fully_observable": bool(_safe_int(idx.DMONTH) >= 2),
            "history_90d_fully_observable": bool(_safe_int(idx.DMONTH) >= 4),
            "history_180d_fully_observable": bool(_safe_int(idx.DMONTH) >= 7),
        }
        for window in (30, 90, 180):
            w = prior[prior["gap"].between(0, window, inclusive="both")]
            rec[f"prior_count_{window}d"] = int(len(w))
            if window == 180:
                rec["prior_ed_count_180d"] = int((w["HCUP_ED"].fillna(0) > 0).sum())
                rec["prior_nonelective_count_180d"] = int((w["ELECTIVE"].fillna(0) != 1).sum())
                rec["prior_ap_count_180d"] = int(w["principal_ap"].fillna(False).sum())
                rec["prior_biliary_count_180d"] = int(w["principal_biliary"].fillna(False).sum())
                rec["prior_sepsis_or_organ_count_180d"] = int(
                    w["principal_sepsis_or_organ"].fillna(False).sum()
                )
                rec["prior_los_sum_180d"] = int(w["LOS"].fillna(0).clip(lower=0).sum())
                rec["prior_max_severity_180d"] = int(
                    w["APRDRG_Severity"].dropna().max() if w["APRDRG_Severity"].notna().any() else 0
                )
                rec["prior_max_mortality_risk_180d"] = int(
                    w["APRDRG_Risk_Mortality"].dropna().max()
                    if w["APRDRG_Risk_Mortality"].notna().any() else 0
                )
                dx: set[int] = set()
                pr: set[int] = set()
                for vals in w["dx_tokens"]:
                    dx.update(_clean_tokens(vals))
                for vals in w["pr_tokens"]:
                    pr.update(_clean_tokens(vals))
                rec["prior_dx_tokens_180d"] = sorted(dx)
                rec["prior_pr_tokens_180d"] = sorted(pr)
        output.append(rec)
    return output


def read_relevant_admissions(path: Path, patient_ids: pa.Array) -> pa.Table:
    pf = pq.ParquetFile(path)
    chunks: list[pa.Table] = []
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=ADMISSION_COLUMNS)
        mask = pc.is_in(table["patient_hash"], value_set=patient_ids)
        filtered = table.filter(mask)
        if filtered.num_rows:
            chunks.append(filtered)
    if not chunks:
        raise RuntimeError(f"No linked admissions found in {path}")
    return pa.concat_tables(chunks, promote_options="default")


def build_year(root: Path, year: int) -> tuple[pa.Table, dict]:
    ap_path = root / "data" / "nrd" / f"year={year}" / "ap_episodes.parquet"
    adm_path = root / "data" / "nrd" / f"year={year}" / "admissions_model.parquet"
    idx_table = pq.read_table(ap_path, columns=INDEX_COLUMNS)
    idx_table = idx_table.filter(pc.equal(idx_table["primary_analysis_eligible"], True))
    idx = idx_table.drop(["primary_analysis_eligible"]).to_pandas()
    if idx["encounter_hash"].duplicated().any():
        raise RuntimeError(f"Duplicate eligible AP encounter hashes in {year}")
    patient_ids = pa.array(np.sort(idx["patient_hash"].unique()), type=pa.uint64())
    linked = read_relevant_admissions(adm_path, patient_ids).to_pandas()

    records: list[dict] = []
    index_groups = {int(k): v for k, v in idx.groupby("patient_hash", sort=False)}
    linked_groups = {int(k): v for k, v in linked.groupby("patient_hash", sort=False)}
    missing_patients = sorted(set(index_groups) - set(linked_groups))
    if missing_patients:
        raise RuntimeError(f"{year}: {len(missing_patients)} AP patients absent from admissions table")
    for patient, index_rows in index_groups.items():
        records.extend(derive_patient_history(linked_groups[patient], index_rows))

    result = pd.DataFrame.from_records(records)
    result = result.sort_values("encounter_hash", kind="mergesort").reset_index(drop=True)
    if len(result) != len(idx) or result["encounter_hash"].duplicated().any():
        raise RuntimeError(f"{year}: history row cardinality failure")
    expected = set(map(int, idx["encounter_hash"]))
    observed = set(map(int, result["encounter_hash"]))
    if expected != observed:
        raise RuntimeError(f"{year}: encounter hash set mismatch")

    qc = {
        "year": year,
        "eligible_index_rows": int(len(idx)),
        "eligible_index_patients": int(idx["patient_hash"].nunique()),
        "linked_admission_rows": int(len(linked)),
        "history_rows": int(len(result)),
        "rows_with_prior_ytd": int((result["prior_count_ytd"] > 0).sum()),
        "rows_with_prior_180d": int((result["prior_count_180d"] > 0).sum()),
        "prior_dx_token_cells": int(result["prior_dx_tokens_180d"].map(len).sum()),
        "prior_pr_token_cells": int(result["prior_pr_tokens_180d"].map(len).sum()),
        "partition_counts": {str(k): int(v) for k, v in result["analysis_partition"].value_counts().items()},
    }
    return pa.Table.from_pandas(result, preserve_index=False), qc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        raise SystemExit("--threads must be between 1 and 8")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    qcs = []
    for year in YEARS:
        table, qc = build_year(root, year)
        path = out / f"ap_history_{year}.parquet"
        pq.write_table(table, path, compression="zstd", compression_level=6, row_group_size=100_000)
        qc.update({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        qcs.append(qc)
        print(json.dumps(qc, ensure_ascii=False, sort_keys=True), flush=True)

    manifest = {
        "status": "PASS" if all(q["eligible_index_rows"] == q["history_rows"] for q in qcs) else "FAIL",
        "years": list(YEARS),
        "history_definition": "same-year nonoverlapping admissions discharged before index admission",
        "left_truncation": "NRD calendar-year linkage; DMONTH-derived observability flags retained",
        "outcome_fields_read": [],
        "sealed_year_2022_accessed": False,
        "special_token_ids_excluded_from_history_bags": [0, 1, 2, 3],
        "annual": qcs,
    }
    manifest_path = out / "history_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if manifest["status"] != "PASS":
        raise SystemExit("history manifest failed")


if __name__ == "__main__":
    main()
