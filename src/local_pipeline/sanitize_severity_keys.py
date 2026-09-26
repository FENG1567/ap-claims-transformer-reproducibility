from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(os.environ.get("AP_CLAIMS_WORKDIR", str(REPOSITORY_ROOT / "workdir"))) / "compact"
PROJECT_YEARS = [2018, 2019, 2020, 2021]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_series(year: int, series: pd.Series) -> np.ndarray:
    material = str(year) + ":" + series.fillna("").astype(str)
    return pd.util.hash_pandas_object(material, index=False).to_numpy(dtype=np.uint64)


def atomic_json(path: Path, payload: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(partial, path)


def transform(year: int, root: Path) -> dict[str, object]:
    year_dir = root / f"year={year}"
    source = year_dir / "severity_compact.parquet"
    source_qc = year_dir / "severity_qc.json"
    final = year_dir / "severity_hashed.parquet"
    partial = year_dir / "severity_hashed.partial.parquet"
    qc_path = year_dir / "severity_hashed_qc.json"
    if final.exists() or qc_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed {year} hashed severity")
    if partial.exists():
        partial.unlink()
    qc = json.loads(source_qc.read_text(encoding="utf-8"))
    expected_rows = int(qc["observed_rows"])
    parquet = pq.ParquetFile(source)
    writer: pq.ParquetWriter | None = None
    rows = 0
    batches = 0
    started = time.perf_counter()
    for batch in parquet.iter_batches(batch_size=262144):
        table = pa.Table.from_batches([batch])
        keys = table["KEY_NRD"].to_pandas()
        if keys.isna().any():
            raise RuntimeError(f"{year} severity contains missing KEY_NRD")
        output = pa.table(
            {
                "year": pa.array(np.full(batch.num_rows, year, dtype=np.int16)),
                "encounter_hash": pa.array(hash_series(year, keys), type=pa.uint64()),
                "APRDRG": table["APRDRG"],
                "APRDRG_Risk_Mortality": table["APRDRG_Risk_Mortality"],
                "APRDRG_Severity": table["APRDRG_Severity"],
                "HOSP_NRD": table["HOSP_NRD"],
            }
        )
        if writer is None:
            writer = pq.ParquetWriter(
                partial,
                output.schema,
                compression="zstd",
                compression_level=6,
                use_dictionary=True,
                write_statistics=True,
            )
        writer.write_table(output, row_group_size=131072)
        rows += batch.num_rows
        batches += 1
    if writer is None:
        raise RuntimeError(f"{year} severity source is empty")
    writer.close()
    if rows != expected_rows or pq.ParquetFile(partial).metadata.num_rows != rows:
        raise RuntimeError(f"{year} hashed severity row mismatch")
    os.replace(partial, final)
    report = {
        "year": year,
        "status": "PASS",
        "observed_rows": rows,
        "batches": batches,
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": sha256(source),
        "output_bytes": final.stat().st_size,
        "output_sha256": sha256(final),
        "raw_KEY_NRD_removed": True,
    }
    atomic_json(qc_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--years", nargs="+", type=int, default=PROJECT_YEARS)
    args = parser.parse_args()
    if any(year >= 2022 for year in args.years):
        raise SystemExit("2022 is sealed")
    for year in args.years:
        final = args.root / f"year={year}" / "severity_hashed.parquet"
        qc_path = args.root / f"year={year}" / "severity_hashed_qc.json"
        if final.is_file() and qc_path.is_file():
            report = json.loads(qc_path.read_text(encoding="utf-8"))
            if (
                pq.ParquetFile(final).metadata.num_rows == int(report["observed_rows"])
                and sha256(final) == report["output_sha256"]
            ):
                print(f"RESUME_VERIFIED_SKIP year={year}", flush=True)
                continue
            raise RuntimeError(f"{year} hashed severity resume validation failed")
        print(json.dumps(transform(year, args.root), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
