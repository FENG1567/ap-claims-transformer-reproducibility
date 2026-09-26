from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
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
TRAIN_YEARS = {2018, 2019, 2020}
PROJECT_YEARS = [2018, 2019, 2020, 2021]
RESERVED = {"[PAD]": 0, "[MASK]": 1, "[OOV]": 2, "[MISSING]": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_code(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper().replace(".", "")


def hash_series(year: int, series: pd.Series) -> np.ndarray:
    material = str(year) + ":" + series.fillna("").astype(str)
    return pd.util.hash_pandas_object(material, index=False).to_numpy(dtype=np.uint64)


@dataclass
class Vocabulary:
    token_to_id: dict[str, int] = field(default_factory=lambda: dict(RESERVED))
    counts: collections.Counter[str] = field(default_factory=collections.Counter)
    oov_counts: collections.Counter[str] = field(default_factory=collections.Counter)

    def encode(self, frame: pd.DataFrame, update: bool) -> np.ndarray:
        flat = frame.fillna("").to_numpy(dtype=object).reshape(-1)
        local_codes, local_values = pd.factorize(flat, sort=False, use_na_sentinel=False)
        frequencies = np.bincount(local_codes, minlength=len(local_values))
        lookup = np.empty(len(local_values), dtype=np.int32)
        for i, raw in enumerate(local_values):
            code = normalize_code(raw)
            frequency = int(frequencies[i])
            if not code:
                token_id = RESERVED["[MISSING]"]
            elif code in self.token_to_id:
                token_id = self.token_to_id[code]
            elif update:
                token_id = len(self.token_to_id)
                self.token_to_id[code] = token_id
            else:
                token_id = RESERVED["[OOV]"]
                self.oov_counts[code] += frequency
            lookup[i] = token_id
            self.counts[code or "[MISSING]"] += frequency
        return lookup[local_codes].reshape(frame.shape)

    def save(self, path: Path, frozen_after_year: int, processed_years: list[int]) -> None:
        payload = {
            "reserved": RESERVED,
            "frozen_after_year": frozen_after_year,
            "processed_years": processed_years,
            "token_to_id": self.token_to_id,
            "counts": dict(self.counts),
            "oov_counts": dict(self.oov_counts),
        }
        atomic_json(path, payload)

    @classmethod
    def load(cls, path: Path, expected_processed_years: list[int]) -> "Vocabulary":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("reserved") != RESERVED:
            raise RuntimeError(f"Reserved-token mismatch in {path}")
        if payload.get("frozen_after_year") != 2020:
            raise RuntimeError(f"Unexpected vocabulary freeze year in {path}")
        if payload.get("processed_years") != expected_processed_years:
            raise RuntimeError(
                f"Vocabulary/progress year mismatch in {path}: "
                f"{payload.get('processed_years')} != {expected_processed_years}"
            )
        token_to_id = {str(k): int(v) for k, v in payload["token_to_id"].items()}
        if any(token_to_id.get(k) != v for k, v in RESERVED.items()):
            raise RuntimeError(f"Reserved-token IDs were modified in {path}")
        assigned = sorted(token_to_id.values())
        if assigned != list(range(len(assigned))):
            raise RuntimeError(f"Vocabulary IDs are not contiguous in {path}")
        return cls(
            token_to_id=token_to_id,
            counts=collections.Counter({str(k): int(v) for k, v in payload["counts"].items()}),
            oov_counts=collections.Counter(
                {str(k): int(v) for k, v in payload["oov_counts"].items()}
            ),
        )


def load_specs() -> list[dict[str, object]]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def spec(year: int, group: str) -> dict[str, object]:
    return next(item for item in load_specs() if item["year"] == year and item["group"] == group)


def archive_path(year: int, group: str) -> Path:
    year_dir = NRD_ROOT / f"NRD_{year}"
    candidates = [p for p in year_dir.glob("*.zip") if group.lower() in p.name.lower()]
    if group == "CORE":
        candidates = [p for p in candidates if "dx_pr" not in p.name.lower()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {group} archive for {year}, found {candidates}")
    return candidates[0]


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def validate_completed_report(out_root: Path, report: dict[str, object]) -> None:
    year = int(report["year"])
    if year not in PROJECT_YEARS or report.get("status") != "PASS":
        raise RuntimeError(f"Invalid completed-year report: {report}")
    final_path = out_root / f"year={year}" / "core_compact.parquet"
    qc_path = out_root / f"year={year}" / "core_qc.json"
    if not final_path.is_file() or not qc_path.is_file():
        raise RuntimeError(f"Completed {year} is missing Parquet or QC evidence")
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if qc != report:
        raise RuntimeError(f"Progress/QC mismatch for completed year {year}")
    metadata = pq.ParquetFile(final_path).metadata
    if metadata.num_rows != int(report["observed_rows"]):
        raise RuntimeError(f"Completed {year} Parquet row-count mismatch")
    if final_path.stat().st_size != int(report["output_bytes"]):
        raise RuntimeError(f"Completed {year} Parquet byte-count mismatch")
    if sha256(final_path) != report["output_sha256"]:
        raise RuntimeError(f"Completed {year} Parquet SHA256 mismatch")


def load_resume_state(
    out_root: Path,
) -> tuple[Vocabulary, Vocabulary, list[dict[str, object]], list[int]]:
    progress_path = out_root / "core_etl_progress.json"
    dx_path = out_root / "diagnosis_vocabulary_state.json"
    pr_path = out_root / "procedure_vocabulary_state.json"
    state_paths = [progress_path, dx_path, pr_path]
    present = [path.is_file() for path in state_paths]
    if any(present) and not all(present):
        raise RuntimeError(
            "Incomplete resume bundle: progress and both vocabulary states must all exist"
        )
    if not any(present):
        orphaned = sorted(out_root.glob("year=*/core_compact.parquet"))
        if orphaned:
            raise RuntimeError(f"Orphaned completed Parquet without resume state: {orphaned}")
        return Vocabulary(), Vocabulary(), [], []

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    reports = progress.get("reports")
    processed_years = progress.get("processed_years")
    if not isinstance(reports, list) or not isinstance(processed_years, list):
        raise RuntimeError("Malformed core_etl_progress.json")
    report_years = [int(report["year"]) for report in reports]
    processed_years = [int(year) for year in processed_years]
    if report_years != processed_years:
        raise RuntimeError("Progress reports and processed_years disagree")
    if processed_years != PROJECT_YEARS[: len(processed_years)]:
        raise RuntimeError(
            f"Completed years are not a chronological project prefix: {processed_years}"
        )
    for report in reports:
        validate_completed_report(out_root, report)
    return (
        Vocabulary.load(dx_path, processed_years),
        Vocabulary.load(pr_path, processed_years),
        reports,
        processed_years,
    )


def scalar_types() -> dict[str, pa.DataType]:
    return {
        "AGE": pa.int16(),
        "AWEEKEND": pa.int8(),
        "DIED": pa.int8(),
        "DISCWT": pa.float32(),
        "DISPUNIFORM": pa.int16(),
        "DMONTH": pa.int8(),
        "DRG": pa.int16(),
        "DRGVER": pa.int8(),
        "ELECTIVE": pa.int8(),
        "FEMALE": pa.int8(),
        "HCUP_ED": pa.int8(),
        "HOSP_NRD": pa.int32(),
        "KEY_NRD": pa.int64(),
        "LOS": pa.int32(),
        "MDC": pa.int8(),
        "I10_NDX": pa.int16(),
        "I10_NPR": pa.int16(),
        "NRD_DaysToEvent": pa.int32(),
        "NRD_STRATUM": pa.int32(),
        "PAY1": pa.int8(),
        "PL_NCHS": pa.int8(),
        "REHABTRANSFER": pa.int8(),
        "RESIDENT": pa.int8(),
        "SAMEDAYEVENT": pa.int8(),
        "TOTCHG": pa.int64(),
        "ZIPINC_QRTL": pa.int8(),
        "NRD_VisitLink": pa.string(),
    }


def process_year(
    year: int,
    out_root: Path,
    dx_vocab: Vocabulary,
    pr_vocab: Vocabulary,
    max_rows: int | None,
) -> dict[str, object]:
    year_spec = spec(year, "CORE")
    expected_rows = int(year_spec["observations"])
    all_columns = [v["name"] for v in year_spec["variables"]]
    dx_columns = [f"I10_DX{i}" for i in range(1, 41)]
    pr_columns = [f"I10_PR{i}" for i in range(1, 26)]
    prday_columns = [f"PRDAY{i}" for i in range(1, 26)]
    scalars = scalar_types()
    keep_columns = list(scalars) + dx_columns + pr_columns + prday_columns
    missing = sorted(set(keep_columns) - set(all_columns))
    if missing:
        raise RuntimeError(f"{year} missing required Core columns: {missing}")
    column_types = dict(scalars)
    column_types.update({c: pa.string() for c in dx_columns + pr_columns})
    column_types.update({c: pa.int16() for c in prday_columns})

    out_dir = out_root / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "core_compact.parquet"
    partial_path = out_dir / "core_compact.partial.parquet"
    if final_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed output: {final_path}")
    if partial_path.exists():
        partial_path.unlink()

    archive = archive_path(year, "CORE")
    process = subprocess.Popen(
        [str(SEVEN_ZIP), "x", "-so", str(archive)],
        stdout=subprocess.PIPE,
        stdin=None,
        stderr=None,
    )
    assert process.stdout is not None
    reader = pacsv.open_csv(
        process.stdout,
        read_options=pacsv.ReadOptions(
            use_threads=True,
            block_size=32 * 1024 * 1024,
            column_names=all_columns,
        ),
        parse_options=pacsv.ParseOptions(delimiter=",", quote_char='"'),
        convert_options=pacsv.ConvertOptions(
            include_columns=keep_columns,
            column_types=column_types,
            strings_can_be_null=True,
            quoted_strings_can_be_null=True,
        ),
    )
    writer: pq.ParquetWriter | None = None
    started = time.perf_counter()
    rows = 0
    batches = 0
    peak_rss = 0
    missing_key = 0
    missing_patient = 0
    invalid_prday = 0
    update_vocab = year in TRAIN_YEARS
    try:
        for batch in reader:
            if max_rows is not None and rows >= max_rows:
                break
            if max_rows is not None and rows + batch.num_rows > max_rows:
                batch = batch.slice(0, max_rows - rows)
            table = pa.Table.from_batches([batch])
            scalar_table = table.select(list(scalars))
            key_series = scalar_table["KEY_NRD"].to_pandas()
            patient_series = scalar_table["NRD_VisitLink"].to_pandas()
            missing_key += int(key_series.isna().sum())
            missing_patient += int(patient_series.isna().sum())
            encounter_hash = hash_series(year, key_series.fillna(-1).astype(str))
            patient_hash = hash_series(year, patient_series)
            dx = dx_vocab.encode(table.select(dx_columns).to_pandas(), update=update_vocab)
            pr = pr_vocab.encode(table.select(pr_columns).to_pandas(), update=update_vocab)
            prday = (
                table.select(prday_columns)
                .to_pandas()
                .fillna(-99)
                .to_numpy(dtype=np.int16, copy=False)
            )
            los = scalar_table["LOS"].to_pandas().fillna(-99).to_numpy(dtype=np.int32)
            valid_proc = pr != RESERVED["[MISSING]"]
            invalid_prday += int(((prday > los[:, None]) & valid_proc).sum())
            arrays: dict[str, pa.Array | pa.ChunkedArray] = {
                "year": pa.array(np.full(batch.num_rows, year, dtype=np.int16)),
                "encounter_hash": pa.array(encounter_hash, type=pa.uint64()),
                "patient_hash": pa.array(patient_hash, type=pa.uint64()),
            }
            for name in scalars:
                if name not in {"KEY_NRD", "NRD_VisitLink"}:
                    arrays[name] = scalar_table[name]
            arrays["dx_tokens"] = pa.FixedSizeListArray.from_arrays(
                pa.array(dx.reshape(-1), type=pa.int32()), len(dx_columns)
            )
            arrays["pr_tokens"] = pa.FixedSizeListArray.from_arrays(
                pa.array(pr.reshape(-1), type=pa.int32()), len(pr_columns)
            )
            arrays["prday"] = pa.FixedSizeListArray.from_arrays(
                pa.array(prday.reshape(-1), type=pa.int16()), len(prday_columns)
            )
            output = pa.table(arrays)
            if writer is None:
                writer = pq.ParquetWriter(
                    partial_path,
                    output.schema,
                    compression="zstd",
                    compression_level=6,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(output, row_group_size=65536)
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
        raise RuntimeError(f"7-Zip failed for {year} with exit code {return_code}")
    required_rows = min(expected_rows, max_rows) if max_rows is not None else expected_rows
    if rows != required_rows:
        raise RuntimeError(f"{year} row mismatch: {rows} != {required_rows}")
    metadata = pq.ParquetFile(partial_path).metadata
    if metadata.num_rows != rows:
        raise RuntimeError(f"{year} Parquet row mismatch: {metadata.num_rows} != {rows}")
    os.replace(partial_path, final_path)
    elapsed = time.perf_counter() - started
    report = {
        "year": year,
        "status": "PASS",
        "mode": "full" if max_rows is None else f"bounded_{max_rows}",
        "expected_rows": required_rows,
        "observed_rows": rows,
        "batches": batches,
        "elapsed_seconds": elapsed,
        "rows_per_second": rows / elapsed,
        "peak_rss_bytes": peak_rss or None,
        "missing_key": missing_key,
        "missing_visitlink": missing_patient,
        "prday_greater_than_los_cells": invalid_prday,
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "output_bytes": final_path.stat().st_size,
        "output_sha256": sha256(final_path),
        "dx_vocab_size": len(dx_vocab.token_to_id),
        "pr_vocab_size": len(pr_vocab.token_to_id),
        "vocabulary_update_allowed": update_vocab,
    }
    atomic_json(out_dir / "core_qc.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020, 2021])
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if any(year >= 2022 for year in args.years):
        raise SystemExit("2022 is sealed; this Stage 3 program accepts only years through 2021")
    if args.years != sorted(set(args.years)):
        raise SystemExit("Years must be unique and chronological")
    if any(year not in PROJECT_YEARS for year in args.years):
        raise SystemExit(f"Years must be selected from {PROJECT_YEARS}")
    args.out_root.mkdir(parents=True, exist_ok=True)
    dx_vocab, pr_vocab, reports, processed_years = load_resume_state(args.out_root)
    for year in args.years:
        if year in processed_years:
            print(f"RESUME_VERIFIED_SKIP year={year}", flush=True)
            continue
        expected_next = PROJECT_YEARS[len(processed_years)]
        if year != expected_next:
            raise RuntimeError(
                f"Unsafe year order: next required year is {expected_next}, requested {year}"
            )
        print(f"READY_FOR_PASSWORD year={year} group=CORE", flush=True)
        report = process_year(year, args.out_root, dx_vocab, pr_vocab, args.max_rows)
        reports.append(report)
        processed_years.append(year)
        dx_vocab.save(
            args.out_root / "diagnosis_vocabulary_state.json", 2020, processed_years
        )
        pr_vocab.save(
            args.out_root / "procedure_vocabulary_state.json", 2020, processed_years
        )
        atomic_json(
            args.out_root / "core_etl_progress.json",
            {"reports": reports, "processed_years": processed_years},
        )
        print(json.dumps(report, ensure_ascii=False), flush=True)

    vocabulary_lock = {
        "status": "FROZEN",
        "training_years": sorted(TRAIN_YEARS),
        "2021_behavior": "map codes unseen in 2018-2020 to OOV; do not extend active vocabulary",
        "2022_behavior": "sealed; map unseen codes to OOV only after formal unlock",
        "diagnosis_vocabulary_sha256": sha256(args.out_root / "diagnosis_vocabulary_state.json"),
        "procedure_vocabulary_sha256": sha256(args.out_root / "procedure_vocabulary_state.json"),
    }
    atomic_json(args.out_root / "vocabulary_lock.json", vocabulary_lock)
    print(json.dumps(vocabulary_lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
