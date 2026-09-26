from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(os.environ.get("AP_CLAIMS_WORKDIR", str(REPOSITORY_ROOT / "workdir"))) / "compact"
YEARS = [2018, 2019, 2020, 2021]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(partial, path)


def build(year: int, root: Path, threads: int) -> dict[str, object]:
    year_dir = root / f"year={year}"
    admissions = year_dir / "admissions_model.parquet"
    final = year_dir / "ap_episodes.parquet"
    partial = year_dir / "ap_episodes.partial.parquet"
    qc_path = year_dir / "ap_episodes_qc.json"
    if final.exists() or qc_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed {year} AP episodes")
    if partial.exists():
        partial.unlink()
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute("SET memory_limit='16GB'")
    con.execute("SET preserve_insertion_order=false")
    source = admissions.as_posix()
    started = time.perf_counter()
    con.execute(
        f"""
        CREATE TEMP TABLE sequence_flags AS
        SELECT encounter_hash,
               lead(NRD_DaysToEvent) OVER w AS next_admission_start,
               lead(NRD_DaysToEvent) OVER w - NRD_DaysToEvent - LOS AS immediate_gap_days
        FROM read_parquet('{source}')
        WINDOW w AS (
          PARTITION BY patient_hash
          ORDER BY NRD_DaysToEvent, encounter_hash
        )
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE indices AS
        SELECT a.*,
               s.immediate_gap_days,
               (a.DMONTH < 12) AS december_index_eligible,
               (
                 extract(doy from last_day(make_date(a.year, a.DMONTH, 1)))
                 + a.LOS + 30
                 <= extract(doy from make_date(a.year, 12, 31))
               ) AS guaranteed_30d_observable,
               (
                 a.principal_ap
                 AND a.AGE >= 18
                 AND a.DIED = 0
                 AND a.DISPUNIFORM <> 20
                 AND a.DISPUNIFORM <> 2
                 AND a.NRD_DaysToEvent IS NOT NULL
                 AND a.LOS >= 0
                 AND a.DMONTH BETWEEN 1 AND 11
                 AND (s.immediate_gap_days IS NULL OR s.immediate_gap_days >= 1)
               ) AS index_primary_base_eligible
        FROM read_parquet('{source}') a
        JOIN sequence_flags s USING(encounter_hash)
        WHERE a.any_ap AND a.AGE >= 18
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE candidate_summary AS
        SELECT i.encounter_hash AS index_encounter_hash,
               first(r.encounter_hash ORDER BY r.NRD_DaysToEvent, r.encounter_hash)
                 FILTER (WHERE r.planned_status=0) AS readmission_encounter_hash,
               min(r.NRD_DaysToEvent)
                 FILTER (WHERE r.planned_status=0) AS first_unplanned_start,
               min(r.NRD_DaysToEvent)
                 FILTER (WHERE r.planned_status=-1) AS first_unknown_start,
               count(*) FILTER (WHERE r.planned_status=1) AS planned_admissions_30d,
               count(*) FILTER (WHERE r.planned_status=-1) AS unknown_admissions_30d
        FROM indices i
        JOIN read_parquet('{source}') r
          ON r.patient_hash=i.patient_hash
        WHERE r.NRD_DaysToEvent-i.NRD_DaysToEvent-i.LOS BETWEEN 1 AND 30
        GROUP BY i.encounter_hash
        """
    )
    query = f"""
        SELECT i.*,
               CASE WHEN i.year=2021
                    THEN CASE WHEN i.patient_hash % 2=0 THEN '2021A' ELSE '2021B' END
                    ELSE 'development' END AS analysis_partition,
               c.readmission_encounter_hash,
               CASE WHEN c.readmission_encounter_hash IS NULL THEN NULL
                    ELSE r.NRD_DaysToEvent-i.NRD_DaysToEvent-i.LOS END AS readmission_gap_days,
               coalesce(c.planned_admissions_30d, 0) AS planned_admissions_30d,
               coalesce(c.unknown_admissions_30d, 0) AS unknown_admissions_30d,
               (
                 c.first_unknown_start IS NOT NULL
                 AND (
                   c.first_unplanned_start IS NULL
                   OR c.first_unknown_start <= c.first_unplanned_start
                 )
               ) AS outcome_algorithm_uncertain,
               (
                 i.index_primary_base_eligible
                 AND NOT (
                   c.first_unknown_start IS NOT NULL
                   AND (
                     c.first_unplanned_start IS NULL
                     OR c.first_unknown_start <= c.first_unplanned_start
                   )
                 )
               ) AS primary_analysis_eligible,
               (c.readmission_encounter_hash IS NOT NULL) AS any_unplanned_readmission_30d,
               CASE WHEN c.readmission_encounter_hash IS NULL THEN 0
                    ELSE r.primary_cause_code END AS readmission_leaf,
               (r.primary_cause_code=1 AND c.readmission_encounter_hash IS NOT NULL)
                 AS ap_specific_readmission_30d,
               (r.primary_cause_code=2 AND c.readmission_encounter_hash IS NOT NULL)
                 AS biliary_readmission_30d,
               (r.primary_cause_code=3 AND c.readmission_encounter_hash IS NOT NULL)
                 AS sepsis_or_organ_readmission_30d,
               (r.primary_cause_code=4 AND c.readmission_encounter_hash IS NOT NULL)
                 AS other_readmission_30d,
               r.any_ap AS readmission_any_ap,
               r.any_biliary AS readmission_any_biliary,
               r.any_sepsis_or_organ AS readmission_any_sepsis_or_organ,
               (i.LOS > 7) AS prolonged_los_fixed
        FROM indices i
        LEFT JOIN candidate_summary c ON i.encounter_hash=c.index_encounter_hash
        LEFT JOIN read_parquet('{source}') r
          ON c.readmission_encounter_hash=r.encounter_hash
    """
    con.execute(
        f"COPY ({query}) TO '{partial.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)"
    )
    metadata = pq.ParquetFile(partial).metadata
    qc = con.execute(
        f"""
        SELECT count(*) AS adult_any_ap,
               sum(principal_ap::int) AS principal_ap,
               sum(index_primary_base_eligible::int) AS base_eligible,
               sum(outcome_algorithm_uncertain::int) AS uncertain_outcome,
               sum(primary_analysis_eligible::int) AS primary_eligible,
               sum((primary_analysis_eligible AND any_unplanned_readmission_30d)::int)
                 AS primary_readmissions,
               sum((primary_analysis_eligible AND ap_specific_readmission_30d)::int)
                 AS primary_ap_readmissions,
               sum((primary_analysis_eligible AND biliary_readmission_30d)::int)
                 AS primary_biliary_readmissions,
               sum((primary_analysis_eligible AND sepsis_or_organ_readmission_30d)::int)
                 AS primary_sepsis_readmissions,
               sum((primary_analysis_eligible AND other_readmission_30d)::int)
                 AS primary_other_readmissions,
               sum((DISPUNIFORM=2)::int) AS transfer_disposition,
               sum((immediate_gap_days<=0)::int) AS overlap_or_contiguous,
               sum((NOT december_index_eligible)::int) AS december_indices,
               sum(guaranteed_30d_observable::int) AS guaranteed_window
        FROM read_parquet('{partial.as_posix()}')
        """
    ).fetchone()
    if metadata.num_rows != qc[0]:
        raise RuntimeError(f"{year} AP episode row mismatch")
    if qc[5] != sum(qc[6:10]):
        raise RuntimeError(f"{year} hierarchical leaf union mismatch: {qc}")
    if year == 2021:
        split = con.execute(
            f"""
            SELECT analysis_partition, count(*), count(distinct patient_hash)
            FROM read_parquet('{partial.as_posix()}')
            GROUP BY analysis_partition ORDER BY analysis_partition
            """
        ).fetchall()
        if {row[0] for row in split} != {"2021A", "2021B"}:
            raise RuntimeError(f"Invalid 2021 partitioning: {split}")
    else:
        split = [("development", qc[0], None)]
    os.replace(partial, final)
    keys = [
        "adult_any_ap",
        "principal_ap",
        "base_eligible",
        "uncertain_outcome",
        "primary_eligible",
        "primary_readmissions",
        "primary_ap_readmissions",
        "primary_biliary_readmissions",
        "primary_sepsis_readmissions",
        "primary_other_readmissions",
        "transfer_disposition",
        "overlap_or_contiguous",
        "december_indices",
        "guaranteed_window",
    ]
    counts = dict(zip(keys, map(int, qc)))
    counts["primary_readmission_prevalence"] = (
        counts["primary_readmissions"] / counts["primary_eligible"]
        if counts["primary_eligible"]
        else None
    )
    report = {
        "year": year,
        "status": "PASS",
        "rules": "outputs/stage3_build/episode_linkage_spec.md",
        "counts": counts,
        "partition_counts": [
            {"partition": row[0], "episodes": int(row[1]), "patients": row[2]}
            for row in split
        ],
        "threads": threads,
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": sha256(admissions),
        "output_bytes": final.stat().st_size,
        "output_sha256": sha256(final),
    }
    atomic_json(qc_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if any(year >= 2022 for year in args.years):
        raise SystemExit("2022 is sealed")
    if args.threads < 1 or args.threads > 8:
        raise SystemExit("Thread count must be between 1 and 8")
    for year in args.years:
        print(json.dumps(build(year, args.root, args.threads), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
