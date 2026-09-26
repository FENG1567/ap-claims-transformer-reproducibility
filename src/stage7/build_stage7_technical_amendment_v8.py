#!/usr/bin/env python3
"""Freeze the additive post-unblinding 2022 episode-linkage repair."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PREVIOUS_LOCK_SHA256 = "de29c0ce7cab5c1e4b5248b9672f4a7efec1a8abf3698260c4fd8bfc7643413b"
PREVIOUS_STATUS = "PASS_TECHNICAL_AMENDMENT_V5_ALL_MODEL_INPUT_CONTRACT_LOCKED"
STATUS = "PASS_TECHNICAL_AMENDMENT_V8_NUMERIC_EPISODE_ORDER_LOCKED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing amendment input: {path}")
    return {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def assert_sidecar(path: Path, label: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").strip() != f"{sha256(path)}  {path.name}":
        raise RuntimeError(f"{label} SHA256 sidecar mismatch")


def write_json_with_sidecar(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.write_text(stable_json(value), encoding="utf-8")
    digest = sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return identity(path)


def build(previous_lock: Path, repair_runtime: Path, repair_test: Path,
          source_etl_manifest: Path, output_dir: Path) -> dict[str, Any]:
    previous_lock = previous_lock.resolve()
    repair_runtime = repair_runtime.resolve()
    repair_test = repair_test.resolve()
    source_etl_manifest = source_etl_manifest.resolve()
    output_dir = output_dir.resolve()
    assert_sidecar(previous_lock, "previous v5 lock")
    if sha256(previous_lock) != EXPECTED_PREVIOUS_LOCK_SHA256:
        raise RuntimeError("Previous v5 lock is not the reviewed immutable identity")
    previous = read_json(previous_lock)
    prior_amendment = previous.get("technical_amendment", {})
    if (previous.get("status") != "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
            or previous.get("sealed_test_year") != 2022
            or previous.get("2022_access_before_lock") is not False
            or prior_amendment.get("status") != PREVIOUS_STATUS):
        raise RuntimeError("Previous v5 lock does not preserve the sealed-test contract")
    assert_sidecar(source_etl_manifest, "source 2022 ETL manifest")
    source = read_json(source_etl_manifest)
    if (source.get("status") != "PASS_LOCKED_2022_LOCAL_ETL" or source.get("2022_accessed") is not True):
        raise RuntimeError("Source ETL manifest is not the preserved one-shot 2022 run")
    if output_dir.exists():
        raise RuntimeError("Technical-amendment v8 output already exists and is immutable")
    output_dir.mkdir(parents=True)
    try:
        created = datetime.now(timezone.utc).isoformat()
        defect = {
            "status": "CONFIRMED_2022_EPISODE_LINKAGE_IMPLEMENTATION_DEFECTS",
            "schema_version": "stage7_technical_amendment_v8", "created_utc": created,
            "failure": [
                "no unknown and no unplanned candidate produced inf <= inf and was wrongly excluded",
                "NRD_DaysToEvent was sorted as VARCHAR before numeric gap calculations",
            ],
            "deterministic_example": {"lexical_order": ["100", "90"], "numeric_order": [90, 100]},
            "affected_stage": "episode eligibility and same-year linkage only",
            "scientific_contract_changed": False,
            "model_feature_threshold_calibration_ontology_mapping_changed": False,
            "prior_outputs_preserved_as_invalid_audit_evidence": True,
        }
        defect_path = output_dir / "episode_order_defect_record.json"
        defect_id = write_json_with_sidecar(defect_path, defect)
        builder_id, runtime_id, test_id = identity(Path(__file__).resolve()), identity(repair_runtime), identity(repair_test)
        source_id = identity(source_etl_manifest)
        amended = copy.deepcopy(previous)
        amended["created_utc"] = created
        for item in (builder_id, runtime_id, test_id, defect_id, source_id):
            amended["frozen_code"][item["file"]] = item
        amended.setdefault("frozen_code_roles", {}).update({
            "technical_amendment_v8_builder": builder_id,
            "technical_amendment_v8_repair_runtime": runtime_id,
            "technical_amendment_v8_test": test_id,
            "technical_amendment_v8_defect_record": defect_id,
        })
        amended.setdefault("frozen_artifacts", {})["invalid_2022_etl_v5_manifest"] = source_id
        history = amended.get("technical_amendment_history", [])
        if not isinstance(history, list):
            raise RuntimeError("Technical amendment history is malformed")
        history.append(copy.deepcopy(prior_amendment))
        amended["technical_amendment_history"] = history
        amended["technical_amendment"] = {
            "status": STATUS, "schema_version": "stage7_technical_amendment_v8",
            "previous_amendment_lock": identity(previous_lock),
            "original_unlock_lock": copy.deepcopy(prior_amendment["original_unlock_lock"]),
            "source_etl_manifest": source_id, "defect_record": defect_id,
            "repair_runtime": runtime_id, "repair_test": test_id, "amendment_builder": builder_id,
            "post_unblinding_technical_amendment": True,
            "repair_basis": "data-type/ordering contract and deterministic synthetic counterexample",
            "allowed_semantic_changes": [
                "treat absence of both unknown and unplanned candidates as an eligible negative outcome",
                "convert NRD_DaysToEvent and LOS to numeric before temporal ordering and linkage",
                "restore eligible no-readmission index admissions excluded by lexical ordering",
                "retain the original eligibility, endpoint, algorithm_unknown priority, model, feature, threshold, calibration, ontology and mapping contracts",
            ],
            "year_2022_results_used_to_select_or_change_models_thresholds_or_features": False,
            "previous_locks_specs_and_invalid_outputs_preserved": True,
            "one_corrected_2022_evaluation_authorized": True,
        }
        lock_path = output_dir / "pre2022_unlock_lock_technical_amendment_v8.json"
        lock_id = write_json_with_sidecar(lock_path, amended)
        return {"status": STATUS, "lock": lock_id, "defect_record": defect_id,
                "previous_lock_sha256": EXPECTED_PREVIOUS_LOCK_SHA256}
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-lock", type=Path, required=True)
    parser.add_argument("--repair-runtime", type=Path, required=True)
    parser.add_argument("--repair-test", type=Path, required=True)
    parser.add_argument("--source-etl-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(stable_json(build(args.previous_lock, args.repair_runtime, args.repair_test,
                            args.source_etl_manifest, args.output_dir)), end="")


if __name__ == "__main__":
    main()
