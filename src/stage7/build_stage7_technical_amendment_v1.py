#!/usr/bin/env python3
"""Create the additive Stage 7 ETL vocabulary-compatibility amendment lock.

This builder preserves the original immutable unlock lock and records the
failed archive-opening attempt separately.  It never opens a 2022 archive or
patient-level source.  The replacement lock remains a one-evaluation lock and
adds identities for the amended ETL controller, its regression test, this
builder, and the zero-row failure record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "stage7_technical_amendment_v1"
EXPECTED_ORIGINAL_LOCK_SHA256 = "82f30ef13d7c0bd75a85edfcd673a44462f713993694eb22f262246c85dd0b65"
EXPECTED_ORIGINAL_ETL_SHA256 = "b219fd6317702e3637507c2b8fe15ddca6c7a2c478bf764881eca488eac6fcfb"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing amendment input: {path}")
    return {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def assert_identity(actual_path: Path, declared: dict[str, Any], label: str) -> None:
    actual = identity(actual_path)
    if (not isinstance(declared, dict) or declared.get("sha256") != actual["sha256"]
            or int(declared.get("bytes", -1)) != actual["bytes"]):
        raise RuntimeError(f"{label} path/bytes/SHA256 identity mismatch")


def write_json_with_sidecar(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.write_text(stable_json(value) + "\n", encoding="utf-8")
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return identity(path)


def build_amendment(original_lock: Path, original_controller: Path, amended_controller: Path,
                    amendment_test: Path, failed_output_dir: Path, output_dir: Path) -> dict[str, Any]:
    original_lock = original_lock.resolve()
    original_controller = original_controller.resolve()
    amended_controller = amended_controller.resolve()
    amendment_test = amendment_test.resolve()
    failed_output_dir = failed_output_dir.resolve()
    output_dir = output_dir.resolve()

    original_sidecar = original_lock.with_suffix(original_lock.suffix + ".sha256")
    if not original_lock.is_file() or not original_sidecar.is_file():
        raise RuntimeError("Original unlock lock or SHA256 sidecar is missing")
    original_digest = sha256(original_lock)
    if original_digest != EXPECTED_ORIGINAL_LOCK_SHA256:
        raise RuntimeError("Original unlock lock is not the reviewed immutable identity")
    if original_sidecar.read_text(encoding="ascii").strip() != f"{original_digest}  {original_lock.name}":
        raise RuntimeError("Original unlock lock SHA256 sidecar mismatch")
    original = read_json(original_lock)
    if (original.get("status") != "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
            or original.get("sealed_test_year") != 2022
            or original.get("2022_access_before_lock") is not False):
        raise RuntimeError("Original unlock lock does not preserve the sealed-test contract")
    if sha256(original_controller) != EXPECTED_ORIGINAL_ETL_SHA256:
        raise RuntimeError("Original ETL controller is not the reviewed immutable identity")
    declared_original = original.get("frozen_code", {}).get(str(original_controller))
    assert_identity(original_controller, declared_original, "Original ETL controller")
    assert_identity(original_controller, original.get("frozen_code_roles", {}).get("etl"), "Original ETL role")
    if original_controller == amended_controller:
        raise RuntimeError("Technical amendment must use a new controller path")
    amended_identity = identity(amended_controller)
    test_identity = identity(amendment_test)
    builder_identity = identity(Path(__file__).resolve())
    if not failed_output_dir.is_dir() or any(failed_output_dir.iterdir()):
        raise RuntimeError("Failed-attempt output directory must exist and remain empty")
    if output_dir.exists():
        raise RuntimeError("Technical-amendment output already exists and is immutable")

    output_dir.mkdir(parents=True)
    try:
        created_utc = datetime.now(timezone.utc).isoformat()
        failure_record_path = output_dir / "failed_pre_read_attempt_record.json"
        failure_record = {
            "status": "RECORDED_ZERO_PATIENT_ROW_PRE_READ_FAILURE",
            "schema_version": SCHEMA_VERSION,
            "created_utc": created_utc,
            "original_unlock_lock": identity(original_lock),
            "failure_stage": "frozen_vocabulary_metadata_validation_before_core_reader_construction",
            "archives_extracted_to_private_temporary_directory": True,
            "patient_rows_read": 0,
            "patient_rows_parsed": 0,
            "patient_rows_transformed": 0,
            "patient_values_inspected_or_summarized": False,
            "published_outputs": 0,
            "private_temporary_directory_removed_by_exception_cleanup": True,
            "failed_output_directory": str(failed_output_dir),
            "failed_output_directory_empty_at_recording": True,
            "failure_cause": [
                "The vocabulary remained frozen_after_year=2020 but processed_years also recorded no-growth use in 2021.",
                "Reserved [PAD]/[MASK]/[OOV]/[MISSING] entries used IDs 0-3 and were incorrectly checked as observed codes.",
            ],
            "repair_design_inputs": "Only frozen pre-2022 vocabulary metadata and controller control flow",
            "year_2022_values_used_to_design_or_test_repair": False,
        }
        failure_identity = write_json_with_sidecar(failure_record_path, failure_record)

        amended = copy.deepcopy(original)
        original_etl_role = copy.deepcopy(amended["frozen_code_roles"]["etl"])
        amended["created_utc"] = created_utc
        amended["frozen_code"][str(amended_controller)] = amended_identity
        amended["frozen_code"][str(amendment_test)] = test_identity
        amended["frozen_code"][str(Path(__file__).resolve())] = builder_identity
        amended["frozen_code"][str(failure_record_path.resolve())] = failure_identity
        amended["frozen_code_roles"]["etl_original_pre_amendment"] = original_etl_role
        amended["frozen_code_roles"]["etl"] = amended_identity
        amended["frozen_code_roles"]["technical_amendment_builder"] = builder_identity
        amended["frozen_code_roles"]["technical_amendment_test"] = test_identity
        amended["frozen_code_roles"]["failed_pre_read_attempt_record"] = failure_identity
        amended["technical_amendment"] = {
            "status": "PASS_TECHNICAL_AMENDMENT_V1_LOCKED_WITHOUT_2022_PATIENT_VALUES",
            "schema_version": SCHEMA_VERSION,
            "original_unlock_lock": identity(original_lock),
            "failed_pre_read_attempt_record": failure_identity,
            "original_etl_controller": identity(original_controller),
            "amended_etl_controller": amended_identity,
            "amendment_test": test_identity,
            "amendment_builder": builder_identity,
            "original_lock_preserved": True,
            "archive_access_before_amendment_recorded": True,
            "patient_row_access_before_amendment": False,
            "year_2022_values_used_to_design_or_test_repair": False,
            "replacement_attempt_authorized": True,
            "replacement_attempt_limit": 1,
            "allowed_semantic_changes": [
                "accept no-growth processed_years entries after frozen_after_year=2020",
                "validate and exclude exact reserved token names before observed-code ID checks",
                "identify amendment outputs with a versioned ETL schema string",
            ],
        }
        lock_path = output_dir / "pre2022_unlock_lock_technical_amendment_v1.json"
        lock_identity = write_json_with_sidecar(lock_path, amended)
        return {
            "status": amended["technical_amendment"]["status"],
            "lock": lock_identity,
            "failed_attempt_record": failure_identity,
            "original_lock_sha256": original_digest,
        }
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-lock", type=Path, required=True)
    parser.add_argument("--original-controller", type=Path, required=True)
    parser.add_argument("--amended-controller", type=Path, required=True)
    parser.add_argument("--amendment-test", type=Path, required=True)
    parser.add_argument("--failed-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(stable_json(build_amendment(args.original_lock, args.original_controller,
                                      args.amended_controller, args.amendment_test,
                                      args.failed_output_dir, args.output_dir)))


if __name__ == "__main__":
    main()
