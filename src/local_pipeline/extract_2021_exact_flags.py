from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SEVEN_ZIP = Path(os.environ.get("SEVEN_ZIP", "7z"))
NRD_ROOT = Path(os.environ.get("NRD_ROOT", str(REPOSITORY_ROOT / "data" / "nrd")))
SPEC_PATH = REPOSITORY_ROOT / "configs" / "reference" / "nrd_file_specs.json"
STAGE2 = REPOSITORY_ROOT / "configs" / "stage2_lock"
DEFAULT_OUT = Path(os.environ.get("AP_CLAIMS_WORKDIR", str(REPOSITORY_ROOT / "workdir"))) / "compact"
YEAR = 2021


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


def normalize_code(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper().replace(".", "")


def hash_series(year: int, series: pd.Series) -> np.ndarray:
    material = str(year) + ":" + series.fillna("").astype(str)
    return pd.util.hash_pandas_object(material, index=False).to_numpy(dtype=np.uint64)


def raw_archive() -> Path:
    candidates = [
        path
        for path in (NRD_ROOT / f"NRD_{YEAR}").glob("*.zip")
        if "core" in path.name.lower() and "dx_pr" not in path.name.lower()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one 2021 CORE archive, found {candidates}")
    return candidates[0]


def load_schema() -> tuple[list[str], int]:
    specs = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    item = next(x for x in specs if x["year"] == YEAR and x["group"] == "CORE")
    return [entry["name"] for entry in item["variables"]], int(item["observations"])


def load_sets() -> tuple[dict[str, set[str]], set[str], set[str]]:
    code_sets = pd.read_parquet(
        STAGE2 / "planned_readmission" / "annual_algorithm_code_sets.parquet"
    )
    mappings = pd.read_parquet(
        STAGE2 / "planned_readmission" / "annual_ccs_mapping.parquet"
    )
    code_sets = code_sets.loc[code_sets.calendar_year == YEAR]
    mappings = mappings.loc[mappings.calendar_year == YEAR]
    sets = {
        str(name): set(group.code.astype(str).map(normalize_code))
        for name, group in code_sets.groupby("table", sort=False)
    }
    label_table = pd.read_csv(STAGE2 / "ontology" / "icd_ccsr_labels.csv", dtype=str)
    for node, group in label_table.groupby("label_node", sort=False):
        sets[str(node)] = set(group.code.map(normalize_code))
    cm_mapped = set(
        mappings.loc[mappings.code_system == "CM", "code"].astype(str).map(normalize_code)
    )
    pcs_mapped = set(
        mappings.loc[mappings.code_system == "PCS", "code"].astype(str).map(normalize_code)
    )
    required = {
        "PR.1",
        "PR.2",
        "PR.3",
        "PR.4",
        "readmission.ap_specific",
        "readmission.biliary_event",
        "readmission.sepsis_or_acute_organ_dysfunction",
    }
    missing = required - sets.keys()
    if missing:
        raise RuntimeError(f"Missing locked code sets: {sorted(missing)}")
    return sets, cm_mapped, pcs_mapped


def normalize_factorized(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    flat = frame.fillna("").to_numpy(dtype=object).reshape(-1)
    codes, values = pd.factorize(flat, sort=False, use_na_sentinel=False)
    normalized = np.array([normalize_code(value) for value in values], dtype=object)
    return codes.reshape(frame.shape), normalized


def lookup_mask(
    factor_codes: np.ndarray, normalized_values: np.ndarray, target: set[str]
) -> np.ndarray:
    lookup = np.fromiter(
        (value in target for value in normalized_values),
        dtype=np.bool_,
        count=len(normalized_values),
    )
    return lookup[factor_codes]


def prefix_mask(
    factor_codes: np.ndarray, normalized_values: np.ndarray, prefix: str
) -> np.ndarray:
    lookup = np.fromiter(
        (value.startswith(prefix) for value in normalized_values),
        dtype=np.bool_,
        count=len(normalized_values),
    )
    return lookup[factor_codes]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    out_dir = args.out_root / f"year={YEAR}"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "exact_label_flags.parquet"
    partial_path = out_dir / "exact_label_flags.partial.parquet"
    qc_path = out_dir / "exact_label_flags_qc.json"
    if final_path.exists() or qc_path.exists():
        raise RuntimeError("Refusing to overwrite completed 2021 exact-label artifacts")
    if partial_path.exists():
        partial_path.unlink()

    all_columns, expected_rows = load_schema()
    dx_columns = [f"I10_DX{i}" for i in range(1, 41)]
    pr_columns = [f"I10_PR{i}" for i in range(1, 26)]
    keep_columns = ["KEY_NRD"] + dx_columns + pr_columns
    missing = set(keep_columns) - set(all_columns)
    if missing:
        raise RuntimeError(f"2021 CORE missing columns: {sorted(missing)}")
    sets, cm_mapped, pcs_mapped = load_sets()
    archive = raw_archive()
    print("READY_FOR_PASSWORD year=2021 group=CORE_EXACT_FLAGS", flush=True)
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
            column_types={
                "KEY_NRD": pa.int64(),
                **{column: pa.string() for column in dx_columns + pr_columns},
            },
            strings_can_be_null=True,
            quoted_strings_can_be_null=True,
        ),
    )
    writer: pq.ParquetWriter | None = None
    rows = 0
    batches = 0
    missing_key = 0
    unknown_rows = 0
    planned_rows = 0
    started = time.perf_counter()
    try:
        for batch in reader:
            if args.max_rows is not None and rows >= args.max_rows:
                break
            if args.max_rows is not None and rows + batch.num_rows > args.max_rows:
                batch = batch.slice(0, args.max_rows - rows)
            table = pa.Table.from_batches([batch])
            keys = table["KEY_NRD"].to_pandas()
            missing_key += int(keys.isna().sum())
            dx_codes, dx_values = normalize_factorized(table.select(dx_columns).to_pandas())
            pr_codes, pr_values = normalize_factorized(table.select(pr_columns).to_pandas())
            principal_lookup = dx_codes[:, 0]
            principal_values = dx_values[principal_lookup]

            principal_ap = np.fromiter(
                (value.startswith("K85") for value in principal_values),
                dtype=np.bool_,
                count=batch.num_rows,
            )
            any_ap = prefix_mask(dx_codes, dx_values, "K85").any(axis=1)
            principal_biliary = lookup_mask(
                principal_lookup,
                dx_values,
                sets["readmission.biliary_event"],
            )
            principal_sepsis = lookup_mask(
                principal_lookup,
                dx_values,
                sets["readmission.sepsis_or_acute_organ_dysfunction"],
            )
            any_biliary = lookup_mask(
                dx_codes,
                dx_values,
                sets["readmission.biliary_event"],
            ).any(axis=1)
            any_sepsis = lookup_mask(
                dx_codes,
                dx_values,
                sets["readmission.sepsis_or_acute_organ_dysfunction"],
            ).any(axis=1)

            principal_missing = principal_values == ""
            principal_unknown = np.fromiter(
                (bool(value) and value not in cm_mapped for value in principal_values),
                dtype=np.bool_,
                count=batch.num_rows,
            )
            pr_nonempty = pr_values[pr_codes] != ""
            pr_mapped = lookup_mask(pr_codes, pr_values, pcs_mapped)
            unknown_pr_count = (pr_nonempty & ~pr_mapped).sum(axis=1).astype(np.int16)
            pr1 = lookup_mask(pr_codes, pr_values, sets["PR.1"]).any(axis=1)
            pr3 = lookup_mask(pr_codes, pr_values, sets["PR.3"]).any(axis=1)
            pr2 = lookup_mask(principal_lookup, dx_values, sets["PR.2"])
            pr4 = lookup_mask(principal_lookup, dx_values, sets["PR.4"])
            algorithm_unknown = principal_missing | principal_unknown | (unknown_pr_count > 0)
            planned_bool = pr1 | pr2 | (pr3 & ~pr4)
            planned = np.where(algorithm_unknown, -1, planned_bool.astype(np.int8)).astype(
                np.int8
            )
            primary_cause = np.select(
                [principal_ap, principal_biliary, principal_sepsis],
                [1, 2, 3],
                default=4,
            ).astype(np.int8)

            output = pa.table(
                {
                    "year": pa.array(np.full(batch.num_rows, YEAR, dtype=np.int16)),
                    "encounter_hash": pa.array(hash_series(YEAR, keys), type=pa.uint64()),
                    "principal_ap": pa.array(principal_ap),
                    "any_ap": pa.array(any_ap),
                    "principal_biliary": pa.array(principal_biliary),
                    "any_biliary": pa.array(any_biliary),
                    "principal_sepsis_or_organ": pa.array(principal_sepsis),
                    "any_sepsis_or_organ": pa.array(any_sepsis),
                    "primary_cause_code": pa.array(primary_cause, type=pa.int8()),
                    "planned_status": pa.array(planned, type=pa.int8()),
                    "algorithm_unknown": pa.array(algorithm_unknown),
                    "unknown_principal_diagnosis": pa.array(
                        principal_missing | principal_unknown
                    ),
                    "unknown_procedure_count": pa.array(
                        unknown_pr_count, type=pa.int16()
                    ),
                    "pra_pr1": pa.array(pr1),
                    "pra_pr2": pa.array(pr2),
                    "pra_pr3": pa.array(pr3),
                    "pra_pr4": pa.array(pr4),
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    partial_path,
                    output.schema,
                    compression="zstd",
                    compression_level=6,
                    use_dictionary=True,
                    write_statistics=True,
                )
            writer.write_table(output, row_group_size=131072)
            rows += batch.num_rows
            batches += 1
            unknown_rows += int(algorithm_unknown.sum())
            planned_rows += int((planned == 1).sum())
    finally:
        if writer is not None:
            writer.close()
        process.stdout.close()
        if args.max_rows is not None and process.poll() is None:
            process.terminate()
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=30)
    if args.max_rows is None and return_code != 0:
        raise RuntimeError(f"7-Zip failed with exit code {return_code}")
    required = min(expected_rows, args.max_rows) if args.max_rows is not None else expected_rows
    if rows != required:
        raise RuntimeError(f"2021 exact-label row mismatch: {rows} != {required}")
    if missing_key:
        raise RuntimeError(f"2021 exact-label extraction has {missing_key} missing KEY_NRD")
    if writer is None or pq.ParquetFile(partial_path).metadata.num_rows != rows:
        raise RuntimeError("2021 exact-label Parquet validation failed")
    os.replace(partial_path, final_path)
    elapsed = time.perf_counter() - started
    report = {
        "year": YEAR,
        "status": "PASS",
        "purpose": "retain exact endpoint and planned-readmission classification for 2021 OOV codes",
        "mode": "full" if args.max_rows is None else f"bounded_{args.max_rows}",
        "expected_rows": required,
        "observed_rows": rows,
        "batches": batches,
        "missing_key": missing_key,
        "algorithm_unknown_rows": unknown_rows,
        "planned_rows": planned_rows,
        "elapsed_seconds": elapsed,
        "archive_name": archive.name,
        "archive_sha256": sha256(archive),
        "output_bytes": final_path.stat().st_size,
        "output_sha256": sha256(final_path),
    }
    atomic_json(qc_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
