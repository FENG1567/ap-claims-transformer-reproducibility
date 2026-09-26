#!/usr/bin/env python3
"""Create the additive Stage 7 all-model derivative-contract amendment."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "stage7_technical_amendment_v5"
EXPECTED_PREVIOUS_LOCK_SHA256 = "801dd1f3e226958023e61d38644554cc73835ac892498ff233f48a4466c93fc3"
EXPECTED_ETL_SPEC_SHA256 = "2f17173e8698024aed29131a7ee77edd21a8c442fe26f0ad818f6dbed3153ec8"
EXPECTED_DERIVATIVE_SPEC_SHA256 = "2f8018cc657886c74cb33b4d9898cc197f63770b832f538a4e8310ae220e2b05"
BASELINE_MISSING_FIELDS = (
    "APRDRG", "DISPUNIFORM", "DRG", "DRGVER", "MDC", "N_DISC_U", "N_HOSP_U",
    "S_DISC_U", "S_HOSP_U", "TOTAL_DISC", "TOTCHG", "cost_2021_usd",
    "primary_analysis_eligible",
)
CORE_SOURCE_ADDITIONS = ("DRG", "DRGVER", "MDC")


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


def load_baseline_required_fields(baseline_support: Path) -> set[str]:
    tree = ast.parse(baseline_support.resolve().read_text(encoding="utf-8"))
    values: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {"BASE_COLUMNS", "HISTORY_COLUMNS"}:
                value = ast.literal_eval(value_node)
                if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
                    raise RuntimeError(f"Frozen baseline {target.id} is not a literal string list")
                values[target.id] = list(value)
    if set(values) != {"BASE_COLUMNS", "HISTORY_COLUMNS"}:
        raise RuntimeError("Frozen baseline support lacks literal feature contracts")
    return set(values["BASE_COLUMNS"]) | set(values["HISTORY_COLUMNS"])


def amend_specs(etl_spec: dict[str, Any], derivative_spec: dict[str, Any],
                baseline_required: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return deterministic spec copies with only the frozen baseline inputs added."""
    etl = copy.deepcopy(etl_spec)
    derivative = copy.deepcopy(derivative_spec)
    projected = (
        set(derivative["episode_columns"])
        | set(derivative["history_columns"])
        | {"analysis_year"}
    ) - {derivative["eligibility_column"]}
    missing = baseline_required - projected
    if missing != set(BASELINE_MISSING_FIELDS):
        raise RuntimeError(f"Unexpected frozen baseline/derivative mismatch: {sorted(missing)}")
    all_core = list(etl["source_all_columns"]["core"])
    if not set(CORE_SOURCE_ADDITIONS).issubset(all_core):
        raise RuntimeError("Frozen Core layout lacks required baseline source fields")
    core_selected = set(etl["source_columns"]["core"]) | set(CORE_SOURCE_ADDITIONS)
    etl["source_columns"]["core"] = [column for column in all_core if column in core_selected]
    for column in CORE_SOURCE_ADDITIONS:
        if column not in etl["core_passthrough_columns"]:
            etl["core_passthrough_columns"].append(column)
    for column in BASELINE_MISSING_FIELDS:
        if column not in etl["episode_output_columns"]:
            etl["episode_output_columns"].append(column)
        if column not in derivative["episode_columns"]:
            derivative["episode_columns"].append(column)
        if column not in derivative["static_columns"]:
            derivative["static_columns"].append(column)
    amendment = {
        "schema_version": SCHEMA_VERSION,
        "reason": "Complete the predeclared 13-model derivative input contract",
        "added_baseline_fields": list(BASELINE_MISSING_FIELDS),
        "added_core_source_fields": list(CORE_SOURCE_ADDITIONS),
        "data_dependent_adaptation": False,
        "year_2022_patient_values_used": False,
    }
    etl["technical_amendment"] = amendment
    derivative["technical_amendment"] = copy.deepcopy(amendment)
    return etl, derivative


def build_amendment(previous_lock: Path, original_etl_spec: Path, original_derivative_spec: Path,
                    baseline_lock: Path, baseline_support: Path, amendment_test: Path, predictions_output_dir: Path,
                    output_dir: Path) -> dict[str, Any]:
    previous_lock = previous_lock.resolve()
    original_etl_spec = original_etl_spec.resolve()
    original_derivative_spec = original_derivative_spec.resolve()
    baseline_lock = baseline_lock.resolve()
    baseline_support = baseline_support.resolve()
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
            or previous_amendment.get("status") != "PASS_TECHNICAL_AMENDMENT_V4_LOCKED_WITHOUT_2022_PATIENT_VALUE_ADAPTATION"):
        raise RuntimeError("Previous amendment does not preserve the sealed-test contract")
    if sha256(original_etl_spec) != EXPECTED_ETL_SPEC_SHA256 or sha256(original_derivative_spec) != EXPECTED_DERIVATIVE_SPEC_SHA256:
        raise RuntimeError("Original data specifications are not the reviewed immutable identities")
    assert_identity(original_etl_spec, previous["frozen_artifacts"]["etl_spec"], "Original ETL specification")
    assert_identity(original_derivative_spec, previous["frozen_artifacts"]["derivative_spec"],
                    "Original derivative specification")
    assert_identity(baseline_lock, previous["upstream_gates"].get("baseline_lock"), "Frozen baseline lock")
    baseline_gate = read_json(baseline_lock)
    if (baseline_gate.get("status") != "PASS_BASELINES_FROZEN_PRE_2021B_PRE_2022"
            or baseline_gate.get("script_sha256") != sha256(baseline_support)):
        raise RuntimeError("Frozen baseline support does not match the baseline lock")
    test_identity = identity(amendment_test)
    builder_identity = identity(Path(__file__).resolve())
    if predictions_output_dir.exists():
        raise RuntimeError("Failed prediction attempt must not have published an output directory")
    if output_dir.exists():
        raise RuntimeError("Technical-amendment v5 output already exists and is immutable")
    baseline_required = load_baseline_required_fields(baseline_support)
    etl, derivative = amend_specs(read_json(original_etl_spec), read_json(original_derivative_spec), baseline_required)

    output_dir.mkdir(parents=True)
    try:
        created_utc = datetime.now(timezone.utc).isoformat()
        etl_path = output_dir / "frozen_2022_etl_spec_technical_amendment_v5.json"
        derivative_path = output_dir / "frozen_2022_derivative_spec_technical_amendment_v5.json"
        etl_identity = write_json_with_sidecar(etl_path, etl)
        derivative_identity = write_json_with_sidecar(derivative_path, derivative)
        failure_path = output_dir / "pre_baseline_derivative_contract_failure_record.json"
        failure = {
            "status": "RECORDED_UNPUBLISHED_ALL_MODEL_DERIVATIVE_CONTRACT_FAILURE",
            "schema_version": SCHEMA_VERSION,
            "created_utc": created_utc,
            "previous_amendment_lock": identity(previous_lock),
            "failure_stage": "after_ten_serial_transformers_before_first_baseline_prediction",
            "transformer_models_completed_in_private_temporary_output": 10,
            "baseline_models_executed": 0,
            "predictions_published": False,
            "private_temporary_predictions_removed_by_exception_cleanup": True,
            "missing_fields": list(BASELINE_MISSING_FIELDS),
            "repair_basis": "Frozen Stage 4 baseline feature contract and frozen HCUP source layout only",
            "year_2022_patient_values_used_to_select_or_change_fields": False,
        }
        failure_identity = write_json_with_sidecar(failure_path, failure)
        amended = copy.deepcopy(previous)
        amended["created_utc"] = created_utc
        baseline_support_identity = identity(baseline_support)
        for item in (etl_identity, derivative_identity, baseline_support_identity,
                     test_identity, builder_identity, failure_identity):
            amended["frozen_code"][item["file"]] = item
        amended["frozen_artifacts"]["etl_spec_original_pre_amendment"] = copy.deepcopy(
            amended["frozen_artifacts"]["etl_spec"])
        amended["frozen_artifacts"]["derivative_spec_original_pre_amendment"] = copy.deepcopy(
            amended["frozen_artifacts"]["derivative_spec"])
        amended["frozen_artifacts"]["etl_spec"] = etl_identity
        amended["frozen_artifacts"]["derivative_spec"] = derivative_identity
        amended["frozen_code_roles"]["technical_amendment_v5_builder"] = builder_identity
        amended["frozen_code_roles"]["technical_amendment_v5_test"] = test_identity
        amended["frozen_code_roles"]["pre_baseline_derivative_contract_failure_record"] = failure_identity
        history = amended.get("technical_amendment_history", [])
        if not isinstance(history, list):
            raise RuntimeError("Technical amendment history is malformed")
        history.append(copy.deepcopy(previous_amendment))
        amended["technical_amendment_history"] = history
        amended["technical_amendment"] = {
            "status": "PASS_TECHNICAL_AMENDMENT_V5_ALL_MODEL_INPUT_CONTRACT_LOCKED",
            "schema_version": SCHEMA_VERSION,
            "previous_amendment_lock": identity(previous_lock),
            "original_unlock_lock": copy.deepcopy(previous_amendment["original_unlock_lock"]),
            "failure_record": failure_identity,
            "amended_etl_spec": etl_identity,
            "amended_derivative_spec": derivative_identity,
            "amendment_test": test_identity,
            "amendment_builder": builder_identity,
            "previous_locks_and_specs_preserved": True,
            "year_2022_patient_values_used_to_select_or_change_fields": False,
            "replacement_etl_and_prediction_attempt_authorized": True,
            "allowed_semantic_changes": [
                "project the frozen Stage 4 baseline input fields alongside the existing Transformer inputs",
                "add only DRG, DRGVER, and MDC to the Core CSV source projection",
                "retain all original eligibility, outcome, ontology, vocabulary, threshold, and model contracts",
            ],
        }
        lock_path = output_dir / "pre2022_unlock_lock_technical_amendment_v5.json"
        lock_identity = write_json_with_sidecar(lock_path, amended)
        return {"status": amended["technical_amendment"]["status"], "lock": lock_identity,
                "etl_spec": etl_identity, "derivative_spec": derivative_identity,
                "failure_record": failure_identity, "previous_lock_sha256": previous_digest}
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-lock", type=Path, required=True)
    parser.add_argument("--original-etl-spec", type=Path, required=True)
    parser.add_argument("--original-derivative-spec", type=Path, required=True)
    parser.add_argument("--baseline-lock", type=Path, required=True)
    parser.add_argument("--baseline-support", type=Path, required=True)
    parser.add_argument("--amendment-test", type=Path, required=True)
    parser.add_argument("--predictions-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(stable_json(build_amendment(args.previous_lock, args.original_etl_spec,
                                      args.original_derivative_spec, args.baseline_lock, args.baseline_support,
                                      args.amendment_test, args.predictions_output_dir, args.output_dir)))


if __name__ == "__main__":
    main()
