from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = Path(os.environ.get("AP_CLAIMS_WORKDIR", str(REPOSITORY_ROOT / "workdir"))) / "compact"
TOKEN_SETS = REPOSITORY_ROOT / "configs" / "stage3_build" / "token_code_sets.json"
YEARS = [2018, 2019, 2020]
MISSING = 3
OOV = 2


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


def mask(values: np.ndarray, tokens: set[int]) -> np.ndarray:
    return np.isin(values, np.fromiter(tokens, dtype=np.int32))


def derive(year: int, root: Path, sets: dict[str, object]) -> dict[str, object]:
    year_dir = root / f"year={year}"
    source = year_dir / "core_compact.parquet"
    final = year_dir / "exact_label_flags.parquet"
    partial = year_dir / "exact_label_flags.partial.parquet"
    qc_path = year_dir / "exact_label_flags_qc.json"
    if final.exists() or qc_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed {year} exact-label flags")
    if partial.exists():
        partial.unlink()
    labels = sets["labels"]
    annual = sets["annual"][str(year)]
    ap_tokens = set(labels["principal_or_any_AP_prefix"]["token_ids"])
    biliary_tokens = set(labels["readmission.biliary_event"]["token_ids"])
    sepsis_tokens = set(
        labels["readmission.sepsis_or_acute_organ_dysfunction"]["token_ids"]
    )
    pr1_tokens = set(annual["PR.1"]["token_ids"])
    pr2_tokens = set(annual["PR.2"]["token_ids"])
    pr3_tokens = set(annual["PR.3"]["token_ids"])
    pr4_tokens = set(annual["PR.4"]["token_ids"])
    mapped_cm = set(annual["mapped_CM_token_ids"])
    mapped_pcs = set(annual["mapped_PCS_token_ids"])
    parquet = pq.ParquetFile(source)
    writer: pq.ParquetWriter | None = None
    rows = 0
    batches = 0
    unknown_rows = 0
    planned_rows = 0
    oov_dx_cells = 0
    oov_pr_cells = 0
    started = time.perf_counter()
    for batch in parquet.iter_batches(
        batch_size=262144,
        columns=["encounter_hash", "dx_tokens", "pr_tokens"],
    ):
        dx_array = batch.column(batch.schema.get_field_index("dx_tokens"))
        pr_array = batch.column(batch.schema.get_field_index("pr_tokens"))
        dx = dx_array.values.to_numpy(zero_copy_only=False).reshape(batch.num_rows, 40)
        pr = pr_array.values.to_numpy(zero_copy_only=False).reshape(batch.num_rows, 25)
        principal = dx[:, 0]
        oov_dx_cells += int((dx == OOV).sum())
        oov_pr_cells += int((pr == OOV).sum())
        principal_ap = mask(principal, ap_tokens)
        any_ap = mask(dx, ap_tokens).any(axis=1)
        principal_biliary = mask(principal, biliary_tokens)
        any_biliary = mask(dx, biliary_tokens).any(axis=1)
        principal_sepsis = mask(principal, sepsis_tokens)
        any_sepsis = mask(dx, sepsis_tokens).any(axis=1)
        pr1 = mask(pr, pr1_tokens).any(axis=1)
        pr2 = mask(principal, pr2_tokens)
        pr3 = mask(pr, pr3_tokens).any(axis=1)
        pr4 = mask(principal, pr4_tokens)
        unknown_principal = (principal == MISSING) | (principal == OOV) | ~mask(
            principal, mapped_cm
        )
        valid_pr = (pr != MISSING) & (pr != 0)
        unknown_pr_count = (valid_pr & ~mask(pr, mapped_pcs)).sum(axis=1).astype(np.int16)
        algorithm_unknown = unknown_principal | (unknown_pr_count > 0)
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
                "year": pa.array(np.full(batch.num_rows, year, dtype=np.int16)),
                "encounter_hash": batch.column(
                    batch.schema.get_field_index("encounter_hash")
                ),
                "principal_ap": pa.array(principal_ap),
                "any_ap": pa.array(any_ap),
                "principal_biliary": pa.array(principal_biliary),
                "any_biliary": pa.array(any_biliary),
                "principal_sepsis_or_organ": pa.array(principal_sepsis),
                "any_sepsis_or_organ": pa.array(any_sepsis),
                "primary_cause_code": pa.array(primary_cause, type=pa.int8()),
                "planned_status": pa.array(planned, type=pa.int8()),
                "algorithm_unknown": pa.array(algorithm_unknown),
                "unknown_principal_diagnosis": pa.array(unknown_principal),
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
        unknown_rows += int(algorithm_unknown.sum())
        planned_rows += int((planned == 1).sum())
    if writer is None:
        raise RuntimeError(f"{year} core is empty")
    writer.close()
    expected = parquet.metadata.num_rows
    if rows != expected or pq.ParquetFile(partial).metadata.num_rows != rows:
        raise RuntimeError(f"{year} exact-label flag row mismatch")
    if oov_dx_cells or oov_pr_cells:
        raise RuntimeError(
            f"Training-year OOV tokens found: dx={oov_dx_cells}, pr={oov_pr_cells}"
        )
    os.replace(partial, final)
    report = {
        "year": year,
        "status": "PASS",
        "purpose": "derive exact endpoint and planned-readmission flags from lossless training-year token IDs",
        "observed_rows": rows,
        "batches": batches,
        "algorithm_unknown_rows": unknown_rows,
        "planned_rows": planned_rows,
        "oov_dx_cells": oov_dx_cells,
        "oov_pr_cells": oov_pr_cells,
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": sha256(source),
        "token_code_sets_sha256": sha256(TOKEN_SETS),
        "output_bytes": final.stat().st_size,
        "output_sha256": sha256(final),
    }
    atomic_json(qc_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--years", nargs="+", type=int, default=YEARS)
    args = parser.parse_args()
    if any(year >= 2021 for year in args.years):
        raise SystemExit("This derivation is restricted to lossless-token years 2018-2020")
    sets = json.loads(TOKEN_SETS.read_text(encoding="utf-8"))
    for year in args.years:
        print(json.dumps(derive(year, args.root, sets), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
