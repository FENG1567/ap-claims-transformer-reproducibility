#!/usr/bin/env python3
"""Create the additive Stage 7 fine-tune-manifest compatibility amendment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "stage7_technical_amendment_v4"
EXPECTED_PREVIOUS_LOCK_SHA256 = "cf215c7304fe8a67c68e242c91d159df1b12b3e7b13082c769e66936ca015fd0"
EXPECTED_PREVIOUS_PREDICTOR_SHA256 = "711ce7684bd3bbea84be27757d30f2a6331e264caec8c0257de545ce6600ba32"


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


def build_amendment(previous_lock: Path, previous_predictor: Path, amended_predictor: Path,
                    amendment_test: Path, predictions_output_dir: Path, output_dir: Path) -> dict[str, Any]:
    previous_lock = previous_lock.resolve()
    previous_predictor = previous_predictor.resolve()
    amended_predictor = amended_predictor.resolve()
    amendment_test = amendment_test.resolve()
    predictions_output_dir = predictions_output_dir.resolve()
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
            or previous.get("sealed_test_year") != 2022 or previous.get("2022_access_before_lock") is not False
            or not isinstance(previous_amendment, dict)
            or previous_amendment.get("status") != "PASS_TECHNICAL_AMENDMENT_V3_LOCKED_WITHOUT_2022_PATIENT_VALUE_ADAPTATION"):
        raise RuntimeError("Previous amendment does not preserve the sealed-test contract")
    if sha256(previous_predictor) != EXPECTED_PREVIOUS_PREDICTOR_SHA256:
        raise RuntimeError("Previous prediction producer is not the reviewed v3 identity")
    assert_identity(previous_predictor, previous.get("frozen_code", {}).get(str(previous_predictor)),
                    "Previous prediction producer")
    assert_identity(previous_predictor, previous.get("frozen_code_roles", {}).get("prediction"),
                    "Previous prediction role")
    if previous_predictor == amended_predictor:
        raise RuntimeError("Manifest compatibility amendment must use a new producer path")
    amended_identity = identity(amended_predictor)
    test_identity = identity(amendment_test)
    builder_identity = identity(Path(__file__).resolve())
    if predictions_output_dir.exists():
        raise RuntimeError("Failed prediction attempt must not have published an output directory")
    if output_dir.exists():
        raise RuntimeError("Technical-amendment v4 output already exists and is immutable")

    output_dir.mkdir(parents=True)
    try:
        created_utc = datetime.now(timezone.utc).isoformat()
        failure_path = output_dir / "pre_prediction_finetune_manifest_failure_record.json"
        failure = {
            "status": "RECORDED_PRE_PREDICTION_FINETUNE_MANIFEST_COMPATIBILITY_FAILURE",
            "schema_version": SCHEMA_VERSION,
            "created_utc": created_utc,
            "previous_amendment_lock": identity(previous_lock),
            "failure_stage": "frozen_finetune_manifest_validation_before_derivative_parquet_open_or_model_inference",
            "failure_message": "Fine-tune manifest/checkpoint mismatch: joint_main",
            "root_cause": "Frozen fine-tune manifests hash-bind checkpoint files but omit a redundant byte-count field required by the producer.",
            "derivative_patient_values_read_by_prediction_attempt": False,
            "models_executed": 0,
            "predictions_published": False,
            "repair_basis": "Frozen 2021A fine-tune manifest and registry identities only",
            "year_2022_patient_values_used_to_design_or_test_repair": False,
        }
        failure_identity = write_json_with_sidecar(failure_path, failure)
        amended = copy.deepcopy(previous)
        previous_role = copy.deepcopy(amended["frozen_code_roles"]["prediction"])
        amended["created_utc"] = created_utc
        amended["frozen_code"][str(amended_predictor)] = amended_identity
        amended["frozen_code"][str(amendment_test)] = test_identity
        amended["frozen_code"][str(Path(__file__).resolve())] = builder_identity
        amended["frozen_code"][str(failure_path.resolve())] = failure_identity
        amended["frozen_code_roles"]["prediction_previous_amendment_v3"] = previous_role
        amended["frozen_code_roles"]["prediction"] = amended_identity
        amended["frozen_code_roles"]["technical_amendment_v4_builder"] = builder_identity
        amended["frozen_code_roles"]["technical_amendment_v4_test"] = test_identity
        amended["frozen_code_roles"]["pre_prediction_finetune_manifest_failure_record"] = failure_identity
        history = amended.get("technical_amendment_history", [])
        if not isinstance(history, list):
            raise RuntimeError("Technical amendment history is malformed")
        history.append(copy.deepcopy(previous_amendment))
        amended["technical_amendment_history"] = history
        amended["technical_amendment"] = {
            "status": "PASS_TECHNICAL_AMENDMENT_V4_LOCKED_WITHOUT_2022_PATIENT_VALUE_ADAPTATION",
            "schema_version": SCHEMA_VERSION,
            "previous_amendment_lock": identity(previous_lock),
            "original_unlock_lock": copy.deepcopy(previous_amendment["original_unlock_lock"]),
            "pre_prediction_failure_record": failure_identity,
            "previous_prediction_producer": identity(previous_predictor),
            "amended_prediction_producer": amended_identity,
            "amendment_test": test_identity,
            "amendment_builder": builder_identity,
            "previous_locks_preserved": True,
            "year_2022_patient_values_used_to_design_or_test_repair": False,
            "replacement_prediction_attempt_authorized": True,
            "replacement_prediction_attempt_limit": 1,
            "allowed_semantic_changes": [
                "require the frozen manifest checkpoint SHA256 and declared file path",
                "validate checkpoint bytes when the optional redundant bytes field is present",
                "accept reviewed frozen manifests that omit only that redundant bytes field",
            ],
        }
        lock_path = output_dir / "pre2022_unlock_lock_technical_amendment_v4.json"
        lock_identity = write_json_with_sidecar(lock_path, amended)
        return {"status": amended["technical_amendment"]["status"], "lock": lock_identity,
                "failure_record": failure_identity, "previous_lock_sha256": previous_digest}
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-lock", type=Path, required=True)
    parser.add_argument("--previous-predictor", type=Path, required=True)
    parser.add_argument("--amended-predictor", type=Path, required=True)
    parser.add_argument("--amendment-test", type=Path, required=True)
    parser.add_argument("--predictions-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(stable_json(build_amendment(args.previous_lock, args.previous_predictor,
                                      args.amended_predictor, args.amendment_test,
                                      args.predictions_output_dir, args.output_dir)))


if __name__ == "__main__":
    main()
