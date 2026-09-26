#!/usr/bin/env python3
"""Create the additive Stage 7 exact-membership performance amendment lock."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "stage7_technical_amendment_v2"
EXPECTED_PREVIOUS_LOCK_SHA256 = "54711c340114a97ea78c0efaf00ad4ef7abe7ff47e1553377529ed6c35d72c43"
EXPECTED_PREVIOUS_ETL_SHA256 = "c279a1af05bc8540a0ca7a014cc67821e1c27475ce443cf47373a3969977b987"


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


def assert_identity(path: Path, declared: dict[str, Any], label: str) -> None:
    actual = identity(path)
    if (not isinstance(declared, dict) or declared.get("sha256") != actual["sha256"]
            or int(declared.get("bytes", -1)) != actual["bytes"]):
        raise RuntimeError(f"{label} path/bytes/SHA256 identity mismatch")


def write_json_with_sidecar(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.write_text(stable_json(value) + "\n", encoding="utf-8")
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return identity(path)


def build_amendment(previous_lock: Path, previous_controller: Path, amended_controller: Path,
                    amendment_test: Path, failed_output_dir: Path, output_dir: Path) -> dict[str, Any]:
    previous_lock = previous_lock.resolve()
    previous_controller = previous_controller.resolve()
    amended_controller = amended_controller.resolve()
    amendment_test = amendment_test.resolve()
    failed_output_dir = failed_output_dir.resolve()
    output_dir = output_dir.resolve()

    sidecar = previous_lock.with_suffix(previous_lock.suffix + ".sha256")
    if not previous_lock.is_file() or not sidecar.is_file():
        raise RuntimeError("Previous amendment lock or SHA256 sidecar is missing")
    previous_digest = sha256(previous_lock)
    if previous_digest != EXPECTED_PREVIOUS_LOCK_SHA256:
        raise RuntimeError("Previous amendment lock is not the reviewed immutable identity")
    if sidecar.read_text(encoding="ascii").strip() != f"{previous_digest}  {previous_lock.name}":
        raise RuntimeError("Previous amendment lock SHA256 sidecar mismatch")
    previous = read_json(previous_lock)
    previous_amendment = previous.get("technical_amendment")
    if (previous.get("status") != "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
            or previous.get("sealed_test_year") != 2022
            or previous.get("2022_access_before_lock") is not False
            or not isinstance(previous_amendment, dict)
            or previous_amendment.get("status") != "PASS_TECHNICAL_AMENDMENT_V1_LOCKED_WITHOUT_2022_PATIENT_VALUES"):
        raise RuntimeError("Previous amendment does not preserve the sealed-test contract")
    if sha256(previous_controller) != EXPECTED_PREVIOUS_ETL_SHA256:
        raise RuntimeError("Previous ETL controller is not the reviewed v1 identity")
    assert_identity(previous_controller, previous.get("frozen_code", {}).get(str(previous_controller)),
                    "Previous ETL controller")
    assert_identity(previous_controller, previous.get("frozen_code_roles", {}).get("etl"),
                    "Previous ETL role")
    if previous_controller == amended_controller:
        raise RuntimeError("Performance amendment must use a new controller path")
    amended_identity = identity(amended_controller)
    test_identity = identity(amendment_test)
    builder_identity = identity(Path(__file__).resolve())
    if not failed_output_dir.is_dir() or any(failed_output_dir.iterdir()):
        raise RuntimeError("Interrupted-attempt output directory must exist and remain empty after cleanup")
    if output_dir.exists():
        raise RuntimeError("Technical-amendment v2 output already exists and is immutable")

    output_dir.mkdir(parents=True)
    try:
        created_utc = datetime.now(timezone.utc).isoformat()
        failure_record_path = output_dir / "interrupted_performance_attempt_record.json"
        failure_record = {
            "status": "RECORDED_NO_PUBLISHED_OUTPUT_PERFORMANCE_INTERRUPTION",
            "schema_version": SCHEMA_VERSION,
            "created_utc": created_utc,
            "previous_amendment_lock": identity(previous_lock),
            "failure_stage": "first_bounded_core_chunk_frozen_code_set_membership",
            "controller_traceback_location": "_flags_from_raw_codes -> np.isin -> numpy._in1d while evaluating frozen PR.3",
            "elapsed_minutes_before_controlled_interrupt": 55,
            "core_reader_byte_offset_at_diagnosis": 47972352,
            "core_file_bytes": 6079981917,
            "scientific_values_viewed_or_summarized": False,
            "patient_level_output_published": False,
            "manifest_published": False,
            "interrupted_process_terminated": True,
            "private_temporary_directory_removed_after_exact_path_validation": True,
            "failed_output_directory": str(failed_output_dir),
            "failed_output_directory_empty_at_recording": True,
            "root_cause": "Object-array np.isin compares each element against every value in large frozen ICD/PRA catalogs.",
            "repair_basis": "Static complexity analysis, frozen catalog identities, process CPU/RSS, and byte offset only",
            "year_2022_patient_values_used_to_design_or_test_repair": False,
        }
        failure_identity = write_json_with_sidecar(failure_record_path, failure_record)

        amended = copy.deepcopy(previous)
        previous_etl_role = copy.deepcopy(amended["frozen_code_roles"]["etl"])
        amended["created_utc"] = created_utc
        amended["frozen_code"][str(amended_controller)] = amended_identity
        amended["frozen_code"][str(amendment_test)] = test_identity
        amended["frozen_code"][str(Path(__file__).resolve())] = builder_identity
        amended["frozen_code"][str(failure_record_path.resolve())] = failure_identity
        amended["frozen_code_roles"]["etl_previous_amendment_v1"] = previous_etl_role
        amended["frozen_code_roles"]["etl"] = amended_identity
        amended["frozen_code_roles"]["technical_amendment_v2_builder"] = builder_identity
        amended["frozen_code_roles"]["technical_amendment_v2_test"] = test_identity
        amended["frozen_code_roles"]["interrupted_performance_attempt_record"] = failure_identity
        history = amended.get("technical_amendment_history", [])
        if not isinstance(history, list):
            raise RuntimeError("Technical amendment history is malformed")
        history.append(copy.deepcopy(previous_amendment))
        amended["technical_amendment_history"] = history
        amended["technical_amendment"] = {
            "status": "PASS_TECHNICAL_AMENDMENT_V2_LOCKED_WITHOUT_2022_PATIENT_VALUE_ADAPTATION",
            "schema_version": SCHEMA_VERSION,
            "previous_amendment_lock": identity(previous_lock),
            "original_unlock_lock": copy.deepcopy(previous_amendment["original_unlock_lock"]),
            "interrupted_performance_attempt_record": failure_identity,
            "previous_etl_controller": identity(previous_controller),
            "amended_etl_controller": amended_identity,
            "amendment_test": test_identity,
            "amendment_builder": builder_identity,
            "previous_locks_preserved": True,
            "patient_level_output_before_amendment": False,
            "year_2022_patient_values_used_to_design_or_test_repair": False,
            "replacement_attempt_authorized": True,
            "replacement_attempt_limit": 1,
            "allowed_semantic_changes": [
                "replace object-array np.isin against large frozen catalogs with exact membership in the same frozen sets",
                "identify amendment outputs with a versioned ETL schema string",
            ],
            "equivalence_requirement": "All emitted flags must match the v1 implementation exactly on synthetic predeclared-code tests.",
        }
        lock_path = output_dir / "pre2022_unlock_lock_technical_amendment_v2.json"
        lock_identity = write_json_with_sidecar(lock_path, amended)
        return {
            "status": amended["technical_amendment"]["status"],
            "lock": lock_identity,
            "interrupted_attempt_record": failure_identity,
            "previous_lock_sha256": previous_digest,
        }
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-lock", type=Path, required=True)
    parser.add_argument("--previous-controller", type=Path, required=True)
    parser.add_argument("--amended-controller", type=Path, required=True)
    parser.add_argument("--amendment-test", type=Path, required=True)
    parser.add_argument("--failed-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(stable_json(build_amendment(args.previous_lock, args.previous_controller,
                                      args.amended_controller, args.amendment_test,
                                      args.failed_output_dir, args.output_dir)))


if __name__ == "__main__":
    main()
