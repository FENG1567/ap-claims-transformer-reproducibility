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
CPI_U = {2018: 251.107, 2019: 255.657, 2020: 258.811, 2021: 270.970}


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


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build(year: int, root: Path, threads: int) -> dict[str, object]:
    year_dir = root / f"year={year}"
    paths = {
        name: (year_dir / name).as_posix()
        for name in [
            "core_compact.parquet",
            "severity_hashed.parquet",
            "hospital_compact.parquet",
            "ccr_compact.parquet",
            "exact_label_flags.parquet",
        ]
    }
    for name, path in paths.items():
        if not Path(path).is_file():
            raise RuntimeError(f"Missing {year} input {name}: {path}")
    final = year_dir / "admissions_model.parquet"
    partial = year_dir / "admissions_model.partial.parquet"
    qc_path = year_dir / "admissions_model_qc.json"
    if final.exists() or qc_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed {year} model admissions")
    if partial.exists():
        partial.unlink()
    core_columns = pq.ParquetFile(paths["core_compact.parquet"]).schema_arrow.names
    core_columns = [name for name in core_columns if name != "HOSP_NRD"]
    core_select = ",\n               ".join(f"c.{quote(name)}" for name in core_columns)
    factor = 270.970 / CPI_U[year]
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute("SET memory_limit='16GB'")
    con.execute("SET preserve_insertion_order=false")
    query = f"""
        SELECT {core_select},
               sha256(cast(c.year as varchar) || ':' || cast(c.HOSP_NRD as varchar)) AS hospital_hash,
               s.APRDRG,
               s.APRDRG_Risk_Mortality,
               s.APRDRG_Severity,
               h.HOSP_BEDSIZE,
               h.H_CONTRL,
               h.HOSP_URCAT4,
               h.HOSP_UR_TEACH,
               h.N_DISC_U,
               h.N_HOSP_U,
               h.S_DISC_U,
               h.S_HOSP_U,
               h.TOTAL_DISC,
               r.CCR_NRD,
               r.WAGEINDEX,
               CASE WHEN c.TOTCHG > 0 AND r.CCR_NRD > 0
                    THEN cast(c.TOTCHG * r.CCR_NRD AS DOUBLE)
                    ELSE NULL END AS nominal_cost,
               CASE WHEN c.TOTCHG > 0 AND r.CCR_NRD > 0
                    THEN cast(c.TOTCHG * r.CCR_NRD * {factor:.15f} AS DOUBLE)
                    ELSE NULL END AS cost_2021_usd,
               f.principal_ap,
               f.any_ap,
               f.principal_biliary,
               f.any_biliary,
               f.principal_sepsis_or_organ,
               f.any_sepsis_or_organ,
               f.primary_cause_code,
               f.planned_status,
               f.algorithm_unknown,
               f.unknown_principal_diagnosis,
               f.unknown_procedure_count,
               f.pra_pr1,
               f.pra_pr2,
               f.pra_pr3,
               f.pra_pr4
        FROM read_parquet('{paths['core_compact.parquet']}') c
        JOIN read_parquet('{paths['severity_hashed.parquet']}') s USING(encounter_hash)
        JOIN read_parquet('{paths['hospital_compact.parquet']}') h
          ON c.HOSP_NRD=h.HOSP_NRD
        JOIN read_parquet('{paths['ccr_compact.parquet']}') r
          ON c.HOSP_NRD=r.HOSP_NRD
        JOIN read_parquet('{paths['exact_label_flags.parquet']}') f USING(encounter_hash)
    """
    started = time.perf_counter()
    con.execute(
        f"COPY ({query}) TO '{partial.as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)"
    )
    metadata = pq.ParquetFile(partial).metadata
    expected = pq.ParquetFile(paths["core_compact.parquet"]).metadata.num_rows
    if metadata.num_rows != expected:
        raise RuntimeError(f"{year} sanitized admission row mismatch")
    schema_names = pq.ParquetFile(partial).schema_arrow.names
    forbidden = {"KEY_NRD", "NRD_VisitLink", "HOSP_NRD"} & set(schema_names)
    if forbidden:
        raise RuntimeError(f"{year} raw identifiers survived sanitization: {forbidden}")
    qc = con.execute(
        f"""SELECT
              count(*) AS rows,
              count(distinct encounter_hash) AS unique_encounters,
              sum(case when hospital_hash is null then 1 else 0 end) AS missing_hospital_hash,
              sum(case when nominal_cost is null then 1 else 0 end) AS missing_cost,
              sum(case when planned_status=-1 then 1 else 0 end) AS algorithm_unknown
            FROM read_parquet('{partial.as_posix()}')"""
    ).fetchone()
    if qc[0] != expected or qc[1] != expected or qc[2] != 0:
        raise RuntimeError(f"{year} sanitized admission QC failed: {qc}")
    os.replace(partial, final)
    report = {
        "year": year,
        "status": "PASS",
        "observed_rows": qc[0],
        "unique_encounters": qc[1],
        "missing_hospital_hash": qc[2],
        "missing_cost": qc[3],
        "algorithm_unknown": qc[4],
        "columns": len(schema_names),
        "forbidden_raw_identifier_columns": sorted(forbidden),
        "cpi_u": CPI_U[year],
        "cpi_to_2021_factor": factor,
        "threads": threads,
        "elapsed_seconds": time.perf_counter() - started,
        "input_sha256": {name: sha256(Path(path)) for name, path in paths.items()},
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
