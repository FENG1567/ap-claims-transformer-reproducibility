#!/usr/bin/env python3
"""Repair the post-unblinding 2022 episode-linkage implementation defects.

The frozen CSV ETL intentionally preserves projected HCUP fields as strings.
The historical episode routine (1) treated the no-candidate state as uncertain
because both missing first-event times were represented by infinity and
``inf <= inf`` is true, and (2) sorted ``NRD_DaysToEvent`` before converting it
to numeric, so e.g. ``"100"`` sorted before ``"90"``.  This additive repair
reuses the already published, de-identified admissions artifact and changes
only the temporal ordering representation.  Eligibility, endpoints, planned
readmission rules, models, thresholds, calibrators, ontology and mappings are
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STATUS = "PASS_2022_EPISODE_ORDER_TECHNICAL_AMENDMENT_V8"
LOCK_STATUS = "PASS_TECHNICAL_AMENDMENT_V8_NUMERIC_EPISODE_ORDER_LOCKED"
SOURCE_ETL_STATUS = "PASS_LOCKED_2022_LOCAL_ETL"
FORBIDDEN_IDENTIFIERS = {"KEY_NRD", "NRD_VISITLINK", "NRD_VisitLink", "HOSP_NRD"}
HISTORY_COLUMNS = (
    "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d",
    "prior_ed_count_180d", "prior_nonelective_count_180d", "prior_ap_count_180d",
    "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d",
    "prior_los_sum_180d", "prior_max_severity_180d", "prior_max_mortality_risk_180d",
    "days_since_prior_discharge", "history_30d_fully_observable",
    "history_90d_fully_observable", "history_180d_fully_observable",
    "prior_dx_tokens_180d", "prior_pr_tokens_180d",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing required file: {path}")
    return {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def assert_identity(path: Path, declared: dict[str, Any] | None, label: str) -> None:
    actual = identity(path)
    if (not isinstance(declared, dict) or declared.get("sha256") != actual["sha256"]
            or int(declared.get("bytes", -1)) != actual["bytes"]):
        raise RuntimeError(f"{label} bytes/SHA256 identity mismatch")


def assert_sidecar(path: Path, label: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != f"{sha256(path)}  {path.name}":
        raise RuntimeError(f"{label} SHA256 sidecar mismatch")


def reject_raw_identifiers(frame: pd.DataFrame, context: str) -> None:
    if survived := FORBIDDEN_IDENTIFIERS & set(frame.columns):
        raise RuntimeError(f"Raw NRD identifier survived in {context}: {sorted(survived)}")


def validate_preflight(unlock_lock: Path, etl_spec: Path, source_etl_dir: Path) -> dict[str, Any]:
    """Authenticate all inputs before opening the admissions parquet rows."""
    unlock_lock = unlock_lock.resolve()
    etl_spec = etl_spec.resolve()
    source_etl_dir = source_etl_dir.resolve()
    assert_sidecar(unlock_lock, "v8 unlock lock")
    lock = read_json(unlock_lock)
    amendment = lock.get("technical_amendment", {})
    if (lock.get("status") != "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
            or lock.get("sealed_test_year") != 2022
            or amendment.get("status") != LOCK_STATUS):
        raise RuntimeError("Supplied unlock lock is not the reviewed v8 repair lock")
    for path, label in ((Path(__file__).resolve(), "v8 repair runtime"), (etl_spec, "frozen ETL specification")):
        assert_identity(path, lock.get("frozen_code", {}).get(str(path)), label)
    source_manifest = source_etl_dir / "manifest.json"
    admissions = source_etl_dir / "admissions_model.parquet"
    broken_episodes = source_etl_dir / "ap_episodes.parquet"
    assert_sidecar(source_manifest, "source ETL manifest")
    assert_identity(source_manifest, amendment.get("source_etl_manifest"), "source ETL manifest")
    manifest = read_json(source_manifest)
    if (manifest.get("status") != SOURCE_ETL_STATUS or manifest.get("2022_accessed") is not True
            or manifest.get("data_dependent_adaptation") is not False):
        raise RuntimeError("Source ETL is not the immutable one-shot 2022 artifact")
    artifacts = manifest.get("artifacts", {})
    assert_identity(admissions, artifacts.get("admissions_model"), "source admissions_model")
    assert_identity(broken_episodes, artifacts.get("ap_episodes"), "source broken ap_episodes")
    return {"lock": lock, "spec": read_json(etl_spec), "manifest": manifest,
            "source_manifest": source_manifest, "admissions": admissions,
            "broken_episodes": broken_episodes}


def load_ap_related_admissions(admissions_path: Path, threads: int) -> pd.DataFrame:
    if not 1 <= threads <= 8:
        raise RuntimeError("Resource contract requires 1-8 threads")
    con = duckdb.connect()
    try:
        con.execute(f"PRAGMA threads={threads}")
        con.execute("SET memory_limit='16GB'")
        path = str(admissions_path.resolve()).replace("'", "''")
        query = (
            "SELECT a.* FROM read_parquet('" + path + "') a INNER JOIN "
            "(SELECT DISTINCT patient_hash FROM read_parquet('" + path + "') WHERE principal_ap) p "
            "USING (patient_hash)"
        )
        frame = con.execute(query).fetchdf()
    finally:
        con.close()
    if frame.empty:
        raise RuntimeError("AP-related admissions extraction is empty")
    reject_raw_identifiers(frame, "AP-related admissions")
    return frame


def _numeric(value: object) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def build_episodes_numeric(admissions: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the frozen episode rules after explicit numeric temporal ordering."""
    frame = admissions.copy()
    frame["__day_num"] = pd.to_numeric(frame["NRD_DaysToEvent"], errors="coerce")
    frame["__los_num"] = pd.to_numeric(frame["LOS"], errors="coerce")
    records: list[dict[str, Any]] = []
    qc = {"principal_ap_rows_seen": 0, "base_eligible_before_unknown_gate": 0,
          "excluded_unknown_priority": 0, "eligible_events": 0, "eligible_non_events": 0}
    ordered = frame.sort_values(["patient_hash", "__day_num", "encounter_hash"], kind="mergesort")
    for _, group in ordered.groupby("patient_hash", sort=False):
        group = group.reset_index(drop=True)
        for pos, index in group.iterrows():
            if not bool(index["principal_ap"]):
                continue
            qc["principal_ap_rows_seen"] += 1
            day, los = float(index["__day_num"]), float(index["__los_num"])
            if not np.isfinite(day) or not np.isfinite(los):
                continue
            later = group.iloc[pos + 1:].copy()
            gap = later["__day_num"] - day - max(0.0, los)
            immediate = gap.iloc[0] if len(gap) else np.nan
            base = (float(_numeric(index["AGE"])) >= 18
                    and int(_numeric(index["in_hospital_death_label"])) == 0
                    and int(_numeric(index["DISPUNIFORM"])) not in (2, 20)
                    and int(_numeric(index["DMONTH"])) <= 11
                    and (not np.isfinite(immediate) or immediate >= 1))
            if not base:
                continue
            qc["base_eligible_before_unknown_gate"] += 1
            candidates = later.loc[gap.between(1, 30, inclusive="both")]
            unknown = candidates.loc[candidates["planned_status"].eq(-1)]
            unplanned = candidates.loc[candidates["planned_status"].eq(0)]
            first_unknown = float(unknown["__day_num"].min()) if len(unknown) else math.inf
            first_unplanned = float(unplanned["__day_num"].min()) if len(unplanned) else math.inf
            # Absence of both candidate types is a valid no-readmission outcome,
            # not an algorithm-unknown outcome.  The historical ``inf <= inf``
            # comparison removed every such negative row.
            if len(unknown) and first_unknown <= first_unplanned:
                qc["excluded_unknown_priority"] += 1
                continue
            readmit = None if not len(unplanned) else unplanned.sort_values(
                ["__day_num", "encounter_hash"], kind="mergesort").iloc[0]
            leaf = 0 if readmit is None else int(readmit["primary_cause_code"])
            if readmit is not None and leaf not in (1, 2, 3, 4):
                raise RuntimeError("Frozen unplanned readmission has invalid primary cause leaf")
            row = index.drop(labels=["__day_num", "__los_num"]).to_dict()
            row.update({"analysis_partition": "test", "primary_analysis_eligible": True,
                        "any_unplanned_readmission_30d": int(readmit is not None), "readmission_leaf": leaf,
                        "ap_specific_readmission_30d": int(leaf == 1),
                        "biliary_readmission_30d": int(leaf == 2),
                        "sepsis_or_organ_readmission_30d": int(leaf == 3)})
            records.append(row)
            qc["eligible_events" if readmit is not None else "eligible_non_events"] += 1
    episodes = pd.DataFrame(records)
    if episodes.empty:
        raise RuntimeError("Numeric 2022 AP eligibility produced no episode rows")
    needed = set(spec["episode_output_columns"])
    if missing := needed - set(episodes.columns):
        raise RuntimeError(f"Episode output projection cannot be satisfied: {sorted(missing)}")
    episodes = episodes.loc[:, list(spec["episode_output_columns"])].copy()
    if episodes["encounter_hash"].duplicated().any() or not episodes["analysis_partition"].eq("test").all():
        raise RuntimeError("Corrected 2022 episode key/partition invariant failed")
    reject_raw_identifiers(episodes, "corrected ap_episodes")
    return episodes, qc


def build_history_numeric(admissions: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    frame = admissions.copy()
    frame["__day_num"] = pd.to_numeric(frame["NRD_DaysToEvent"], errors="coerce")
    frame["__los_num"] = pd.to_numeric(frame["LOS"], errors="coerce").fillna(0).clip(lower=0)
    by_patient = {patient: group.sort_values(["__day_num", "encounter_hash"], kind="mergesort")
                  for patient, group in frame.groupby("patient_hash", sort=False)}
    records: list[dict[str, Any]] = []
    for index in episodes.itertuples(index=False):
        group = by_patient[index.patient_hash]
        start = float(_numeric(index.NRD_DaysToEvent))
        prior = group[(group["__day_num"] < start) & (group["__day_num"] + group["__los_num"] <= start)].copy()
        prior["gap"] = start - (prior["__day_num"] + prior["__los_num"])
        record: dict[str, Any] = {
            "year": 2022, "encounter_hash": index.encounter_hash, "patient_hash": index.patient_hash,
            "analysis_partition": "test", "prior_count_ytd": int(len(prior)),
            "days_since_prior_discharge": int(prior["gap"].min()) if len(prior) else -1,
            "history_30d_fully_observable": int(_numeric(index.DMONTH)) >= 2,
            "history_90d_fully_observable": int(_numeric(index.DMONTH)) >= 4,
            "history_180d_fully_observable": int(_numeric(index.DMONTH)) >= 7,
        }
        for window in (30, 90, 180):
            current = prior[prior["gap"].between(0, window, inclusive="both")]
            record[f"prior_count_{window}d"] = int(len(current))
            if window == 180:
                record.update({
                    "prior_ed_count_180d": int((pd.to_numeric(current.get("HCUP_ED", 0), errors="coerce").fillna(0) > 0).sum()),
                    "prior_nonelective_count_180d": int((pd.to_numeric(current.get("ELECTIVE", 0), errors="coerce").fillna(0) != 1).sum()),
                    "prior_ap_count_180d": int(current["principal_ap"].sum()),
                    "prior_biliary_count_180d": int(current["principal_biliary"].sum()),
                    "prior_sepsis_or_organ_count_180d": int(current["principal_sepsis_or_organ"].sum()),
                    "prior_los_sum_180d": int(current["__los_num"].sum()),
                    "prior_max_severity_180d": int(pd.to_numeric(current["APRDRG_Severity"], errors="coerce").max()) if len(current) else 0,
                    "prior_max_mortality_risk_180d": int(pd.to_numeric(current["APRDRG_Risk_Mortality"], errors="coerce").max()) if len(current) else 0,
                    "prior_dx_tokens_180d": sorted({int(token) for values in current["dx_tokens"] for token in values if int(token) > 3}),
                    "prior_pr_tokens_180d": sorted({int(token) for values in current["pr_tokens"] for token in values if int(token) > 3}),
                })
        records.append(record)
    history = pd.DataFrame(records).loc[:, ["year", "encounter_hash", "patient_hash", "analysis_partition", *HISTORY_COLUMNS]]
    if len(history) != len(episodes) or history["encounter_hash"].duplicated().any():
        raise RuntimeError("Corrected 2022 history cardinality failure")
    reject_raw_identifiers(history, "corrected ap_history")
    return history


def write_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd",
                   compression_level=6, row_group_size=131072, use_dictionary=True, write_statistics=True)
    return identity(path)


def repair(unlock_lock: Path, etl_spec: Path, source_etl_dir: Path, output_dir: Path,
           *, threads: int = 8) -> dict[str, Any]:
    gate = validate_preflight(unlock_lock, etl_spec, source_etl_dir)
    output_dir = output_dir.resolve()
    partials = list(output_dir.parent.glob(output_dir.name + ".partial*")) if output_dir.parent.exists() else []
    if output_dir.exists() or partials:
        raise RuntimeError("Corrected v8 ETL output or partial already exists and is immutable")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent))
    try:
        admissions = load_ap_related_admissions(gate["admissions"], threads)
        episodes, qc = build_episodes_numeric(admissions, gate["spec"])
        if episodes["any_unplanned_readmission_30d"].nunique() != 2:
            raise RuntimeError("Corrected formal 2022 episode cohort must contain both endpoint classes")
        history = build_history_numeric(admissions, episodes)
        broken = pq.read_table(gate["broken_episodes"], columns=["encounter_hash", "any_unplanned_readmission_30d", "readmission_leaf"]).to_pandas()
        positives = episodes.loc[episodes["any_unplanned_readmission_30d"].eq(1), ["encounter_hash", "readmission_leaf"]]
        broken_ids = set(broken["encounter_hash"].astype(str))
        positive_ids = set(positives["encounter_hash"].astype(str))
        if not broken["any_unplanned_readmission_30d"].eq(1).all() or broken_ids != positive_ids:
            raise RuntimeError("Repair changed the historical positive-event set instead of only restoring negatives")
        old_leaf = broken.set_index("encounter_hash")["readmission_leaf"].astype(int).sort_index()
        new_leaf = positives.set_index("encounter_hash")["readmission_leaf"].astype(int).sort_index()
        if not old_leaf.equals(new_leaf):
            raise RuntimeError("Repair changed a historical positive-event leaf")
        episode_id = write_parquet(episodes, temporary / "ap_episodes.parquet")
        history_id = write_parquet(history, temporary / "ap_history_2022.parquet")
        manifest = {
            "status": STATUS, "schema_version": "stage7_2022_episode_order_repair_v8",
            "created_utc": datetime.now(timezone.utc).isoformat(), "sealed_test_year": 2022,
            "2022_accessed": True, "post_unblinding_technical_amendment": True,
            "data_dependent_adaptation": False, "scientific_contract_changed": False,
            "model_feature_threshold_calibration_ontology_mapping_changed": False,
            "repair": ("no-candidate episodes are retained as negatives; NRD_DaysToEvent and LOS are "
                       "converted to numeric before temporal ordering and linkage"),
            "source_etl_manifest": identity(gate["source_manifest"]),
            "source_admissions_model": identity(gate["admissions"]),
            "preserved_invalid_result": identity(gate["broken_episodes"]),
            "unlock_lock": identity(unlock_lock), "frozen_etl_spec": identity(etl_spec),
            "artifacts": {"ap_episodes": episode_id, "ap_history_2022": history_id},
            "qc": {**qc, "eligible_total": int(len(episodes)),
                   "historical_positive_set_preserved": True,
                   "restored_negative_rows": int((episodes["any_unplanned_readmission_30d"] == 0).sum())},
            "raw_identifiers_removed": True,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(stable_json(manifest), encoding="utf-8")
        digest = sha256(manifest_path)
        (temporary / "manifest.json.sha256").write_text(f"{digest}  manifest.json\n", encoding="ascii")
        temporary.replace(output_dir)
        return {**manifest, "manifest_sha256": digest, "output": str(output_dir)}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlock-lock", type=Path, required=True)
    parser.add_argument("--etl-spec", type=Path, required=True)
    parser.add_argument("--source-etl-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        gate = validate_preflight(args.unlock_lock, args.etl_spec, args.source_etl_dir)
        print(stable_json({"status": "PASS_V8_REPAIR_PREFLIGHT_WITHOUT_OPENING_ADMISSIONS_ROWS",
                           "sealed_test_year": gate["lock"]["sealed_test_year"]}), end="")
        return
    print(stable_json(repair(args.unlock_lock, args.etl_spec, args.source_etl_dir,
                             args.output_dir, threads=args.threads)), end="")


if __name__ == "__main__":
    main()
