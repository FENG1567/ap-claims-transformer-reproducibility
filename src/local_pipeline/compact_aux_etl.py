from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

try:
    import psutil
except ImportError:
    psutil = None


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SEVEN_ZIP = Path(os.environ.get("SEVEN_ZIP", "7z"))
NRD_ROOT = Path(os.environ.get("NRD_ROOT", str(REPOSITORY_ROOT / "data" / "nrd")))
SPEC_PATH = REPOSITORY_ROOT / "configs" / "reference" / "nrd_file_specs.json"
DEFAULT_OUT = Path(os.environ.get("AP_CLAIMS_WORKDIR", str(REPOSITORY_ROOT / "workdir"))) / "compact"
PROJECT_YEARS = [2018, 2019, 2020, 2021]
GROUPS = ["HOSPITAL", "SEVERITY", "CCR"]


GROUP_CONFIG: dict[str, dict[str, object]] = {
    "HOSPITAL": {
        "columns": [
            "HOSP_BEDSIZE",
            "H_CONTRL",
            "HOSP_NRD",
            "HOSP_URCAT4",
            "HOSP_UR_TEACH",
            "NRD_STRATUM",
            "N_DISC_U",
            "N_HOSP_U",
            "S_DISC_U",
            "S_HOSP_U",
            "TOTAL_DISC",
            "YEAR",
        ],
        "types": {
            "HOSP_BEDSIZE": pa.int8(),
            "H_CONTRL": pa.int8(),
            "HOSP_NRD": pa.int32(),
            "HOSP_URCAT4": pa.int8(),
            "HOSP_UR_TEACH": pa.int8(),
            "NRD_STRATUM": pa.int32(),
            "N_DISC_U": pa.float64(),
            "N_HOSP_U": pa.float64(),
            "S_DISC_U": pa.float64(),
            "S_HOSP_U": pa.float64(),
            "TOTAL_DISC": pa.int32(),
            "YEAR": pa.int16(),
        },
        "key": "HOSP_NRD",
        "encrypted": True,
    },
    "SEVERITY": {
        "columns": [
            "APRDRG",
            "APRDRG_Risk_Mortality",
            "APRDRG_Severity",
            "HOSP_NRD",
            "KEY_NRD",
        ],
        "types": {
            "APRDRG": pa.int16(),
            "APRDRG_Risk_Mortality": pa.int8(),
            "APRDRG_Severity": pa.int8(),
            "HOSP_NRD": pa.int32(),
            "KEY_NRD": pa.int64(),
        },
        "key": "KEY_NRD",
        "encrypted": True,
    },
    "CCR": {
        "columns": ["HOSP_NRD", "YEAR", "CCR_NRD", "WAGEINDEX"],
        "types": {
            "HOSP_NRD": pa.int32(),
            "YEAR": pa.int16(),
            "CCR_NRD": pa.float32(),
            "WAGEINDEX": pa.float32(),
        },
        "key": "HOSP_NRD",
        "encrypted": False,
    },
}


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


def load_specs() -> list[dict[str, object]]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def spec(year: int, group: str) -> dict[str, object]:
    return next(
        item for item in load_specs() if item["year"] == year and item["group"] == group
    )


def archive_path(year: int, group: str) -> Path:
    if group == "CCR":
        path = NRD_ROOT / f"cc{year}NRD.zip"
        if not path.is_file():
            raise RuntimeError(f"Missing CCR archive: {path}")
        return path
    year_dir = NRD_ROOT / f"NRD_{year}"
    candidates = [p for p in year_dir.glob("*.zip") if group.lower() in p.name.lower()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {group} archive for {year}, found {candidates}")
    return candidates[0]


def expected_rows(year: int, group: str) -> int | None:
    if group == "CCR":
        return None
    return int(spec(year, group)["observations"])


def validate_spec(year: int, group: str) -> list[str]:
    if group == "CCR":
        return list(GROUP_CONFIG[group]["columns"])
    available = [item["name"] for item in spec(year, group)["variables"]]
    required = list(GROUP_CONFIG[group]["columns"])
    if len(available) != len(required) or set(available) != set(required):
        raise RuntimeError(
            f"{year} {group} schema drift: observed={available}, required={required}"
        )
    return available


def process_table(
    year: int,
    group: str,
    out_root: Path,
    max_rows: int | None,
) -> dict[str, object]:
    source_columns = validate_spec(year, group)
    config = GROUP_CONFIG[group]
    columns = config["columns"]
    types = config["types"]
    key = str(config["key"])
    archive = archive_path(year, group)
    out_dir = out_root / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = group.lower()
    final_path = out_dir / f"{stem}_compact.parquet"
    partial_path = out_dir / f"{stem}_compact.partial.parquet"
    if final_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed output: {final_path}")
    if partial_path.exists():
        partial_path.unlink()

    command = [str(SEVEN_ZIP), "x", "-so", str(archive)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=None, stderr=None)
    assert process.stdout is not None
    if bool(config["encrypted"]):
        read_options = pacsv.ReadOptions(
            use_threads=True,
            block_size=16 * 1024 * 1024,
            column_names=source_columns,
        )
        parse_options = pacsv.ParseOptions(delimiter=",", quote_char='"')
    else:
        read_options = pacsv.ReadOptions(use_threads=True, block_size=4 * 1024 * 1024)
        parse_options = pacsv.ParseOptions(delimiter=",", quote_char="'")
    reader = pacsv.open_csv(
        process.stdout,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=pacsv.ConvertOptions(
            include_columns=columns,
            column_types=types,
            null_values=[
                "",
                ".",
                ".A",
                "#N/A",
                "#NA",
                "N/A",
                "NA",
                "NULL",
                "null",
                "NaN",
                "nan",
            ],
            strings_can_be_null=True,
            quoted_strings_can_be_null=True,
        ),
    )

    writer: pq.ParquetWriter | None = None
    started = time.perf_counter()
    rows = 0
    batches = 0
    peak_rss = 0
    null_keys = 0
    duplicate_keys = 0
    observed_years: set[int] = set()
    key_batches: list[np.ndarray] = []
    null_counts = {name: 0 for name in columns}
    try:
        for batch in reader:
            if max_rows is not None and rows >= max_rows:
                break
            if max_rows is not None and rows + batch.num_rows > max_rows:
                batch = batch.slice(0, max_rows - rows)
            table = pa.Table.from_batches([batch])
            for name in columns:
                null_counts[name] += int(table[name].null_count)
            key_array = table[key].combine_chunks()
            null_keys += int(key_array.null_count)
            if key_array.null_count == 0:
                key_batches.append(key_array.to_numpy(zero_copy_only=False))
            if "YEAR" in columns:
                observed_years.update(
                    int(value) for value in table["YEAR"].to_pylist() if value is not None
                )
            if writer is None:
                writer = pq.ParquetWriter(
                    partial_path,
                    table.schema,
                    compression="zstd",
                    compression_level=6,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(table, row_group_size=131072)
            rows += batch.num_rows
            batches += 1
            if psutil is not None:
                peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
    finally:
        if writer is not None:
            writer.close()
        process.stdout.close()
        if max_rows is not None and process.poll() is None:
            process.terminate()
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=30)

    if max_rows is None and return_code != 0:
        raise RuntimeError(f"7-Zip failed for {year} {group} with exit code {return_code}")
    required = expected_rows(year, group)
    if required is not None:
        required = min(required, max_rows) if max_rows is not None else required
        if rows != required:
            raise RuntimeError(f"{year} {group} row mismatch: {rows} != {required}")
    if rows == 0 or writer is None:
        raise RuntimeError(f"{year} {group} produced no rows")
    if not null_keys:
        all_keys = np.concatenate(key_batches)
        duplicate_keys = int(all_keys.size - np.unique(all_keys).size)
    if duplicate_keys:
        raise RuntimeError(f"{year} {group} has {duplicate_keys} duplicate {key} values")
    if null_keys:
        raise RuntimeError(f"{year} {group} has {null_keys} null {key} values")
    if observed_years and observed_years != {year}:
        raise RuntimeError(f"{year} {group} YEAR mismatch: {sorted(observed_years)}")
    metadata = pq.ParquetFile(partial_path).metadata
    if metadata.num_rows != rows:
        raise RuntimeError(f"{year} {group} Parquet row mismatch")
    os.replace(partial_path, final_path)
    elapsed = time.perf_counter() - started
    report = {
        "year": year,
        "group": group,
        "status": "PASS",
        "mode": "full" if max_rows is None else f"bounded_{max_rows}",
        "expected_rows": required,
        "observed_rows": rows,
        "batches": batches,
        "elapsed_seconds": elapsed,
        "rows_per_second": rows / elapsed,
        "peak_rss_bytes": peak_rss or None,
        "key": key,
        "null_keys": null_keys,
        "duplicate_keys": duplicate_keys,
        "observed_years": sorted(observed_years),
        "null_counts": null_counts,
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "output_bytes": final_path.stat().st_size,
        "output_sha256": sha256(final_path),
    }
    atomic_json(out_dir / f"{stem}_qc.json", report)
    return report


def validate_completed(out_root: Path, report: dict[str, object]) -> None:
    year = int(report["year"])
    group = str(report["group"])
    stem = group.lower()
    final_path = out_root / f"year={year}" / f"{stem}_compact.parquet"
    qc_path = out_root / f"year={year}" / f"{stem}_qc.json"
    if report.get("status") != "PASS" or not final_path.is_file() or not qc_path.is_file():
        raise RuntimeError(f"Incomplete prior report: {report}")
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if qc != report:
        raise RuntimeError(f"Progress/QC mismatch for {year} {group}")
    if pq.ParquetFile(final_path).metadata.num_rows != int(report["observed_rows"]):
        raise RuntimeError(f"Completed {year} {group} row count changed")
    if final_path.stat().st_size != int(report["output_bytes"]):
        raise RuntimeError(f"Completed {year} {group} byte count changed")
    if sha256(final_path) != report["output_sha256"]:
        raise RuntimeError(f"Completed {year} {group} SHA256 changed")


def load_progress(out_root: Path) -> list[dict[str, object]]:
    path = out_root / "aux_etl_progress.json"
    if not path.is_file():
        return []
    reports = json.loads(path.read_text(encoding="utf-8")).get("reports")
    if not isinstance(reports, list):
        raise RuntimeError("Malformed aux_etl_progress.json")
    for report in reports:
        validate_completed(out_root, report)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--years", nargs="+", type=int, default=PROJECT_YEARS)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if any(year >= 2022 for year in args.years):
        raise SystemExit("2022 is sealed; this Stage 3 program accepts only years through 2021")
    if args.years != sorted(set(args.years)):
        raise SystemExit("Years must be unique and chronological")
    if any(year not in PROJECT_YEARS for year in args.years):
        raise SystemExit(f"Years must be selected from {PROJECT_YEARS}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    reports = load_progress(args.out_root)
    completed = {(int(item["year"]), str(item["group"])) for item in reports}
    requested = [(year, group) for year in args.years for group in args.groups]
    for year, group in requested:
        if (year, group) in completed:
            print(f"RESUME_VERIFIED_SKIP year={year} group={group}", flush=True)
            continue
        if bool(GROUP_CONFIG[group]["encrypted"]):
            print(f"READY_FOR_PASSWORD year={year} group={group}", flush=True)
        report = process_table(year, group, args.out_root, args.max_rows)
        reports.append(report)
        completed.add((year, group))
        atomic_json(args.out_root / "aux_etl_progress.json", {"reports": reports})
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
