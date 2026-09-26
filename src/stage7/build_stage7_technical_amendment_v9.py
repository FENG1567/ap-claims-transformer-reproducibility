#!/usr/bin/env python3
"""Freeze the report-only v9 coverage-stratum amendment for corrected 2022 drift.

This builder is intentionally additive.  It copies the reviewed v8 unlock lock,
records the narrowly scoped coverage-reporting gap, and hash-binds the amended
drift analyzer before that analyzer is allowed to open the corrected 2022 table.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_PREVIOUS_LOCK_SHA256 = "ea907782875b345c35f3de658709b7845e033ed1a3ef92db297f7d0f277f06c0"
PREVIOUS_STATUS = "PASS_TECHNICAL_AMENDMENT_V8_NUMERIC_EPISODE_ORDER_LOCKED"
STATUS = "PASS_TECHNICAL_AMENDMENT_V9_COVERAGE_PREDECLARED_NEW_STRATA_REPORT_ONLY_LOCKED"


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


def validate_predeclared_coverage_gap(evaluation_spec: dict[str, Any]) -> dict[str, Any]:
    """Prove that ``-8`` is frozen as a category but lacks a coverage row.

    The distinction from ``-9`` is retained exactly.  A pre-existing ``-8``
    coverage row, an absent declaration, or a malformed reference indicates
    this technical amendment is not the reviewed one and therefore fails closed.
    """
    if (evaluation_spec.get("status") != "FROZEN_2022_EVALUATION_SPEC"
            or evaluation_spec.get("sealed_test_year") != 2022):
        raise RuntimeError("Evaluation specification is not the frozen 2022 contract")
    reference = evaluation_spec.get("drift_reference")
    if not isinstance(reference, dict) or reference.get("status") != "FROZEN_2022_DRIFT_REFERENCE":
        raise RuntimeError("Frozen evaluation specification lacks its drift reference")
    covariates = reference.get("covariates")
    coverage = reference.get("coverage")
    if not isinstance(covariates, dict) or not isinstance(coverage, list):
        raise RuntimeError("Frozen drift reference is malformed")
    matching = [item for item in covariates.values()
                if isinstance(item, dict) and item.get("column") == "ZIPINC_QRTL"
                and item.get("kind") == "categorical"]
    if len(matching) != 1 or not isinstance(matching[0].get("categories"), list):
        raise RuntimeError("Frozen drift reference lacks exactly one categorical ZIPINC_QRTL contract")
    declared = {str(value) for value in matching[0]["categories"]}
    if "-8" not in declared:
        raise RuntimeError("ZIPINC_QRTL=-8 is not predeclared in the frozen category contract")
    if "-9" not in declared:
        raise RuntimeError("ZIPINC_QRTL=-9 is not predeclared in the frozen category contract")
    coverage_levels: set[str] = set()
    for item in coverage:
        if not isinstance(item, dict):
            raise RuntimeError("Frozen coverage reference contains a malformed row")
        if str(item.get("subgroup_dimension")) == "ZIPINC_QRTL":
            coverage_levels.add(str(item.get("subgroup_value")))
    if "-8" in coverage_levels:
        raise RuntimeError("Frozen coverage reference already contains ZIPINC_QRTL=-8; v9 amendment is inapplicable")
    if "-9" not in coverage_levels:
        raise RuntimeError("Frozen coverage reference lacks expected independent ZIPINC_QRTL=-9 row")
    return {
        "status": "CONFIRMED_FROZEN_COVERAGE_PREDECLARED_NEW_STRATUM_GAP",
        "schema_version": "stage7_technical_amendment_v9",
        "subgroup_dimension": "ZIPINC_QRTL",
        "predeclared_new_category": "-8",
        "separate_existing_category": "-9",
        "declared_categories": sorted(declared),
        "coverage_reference_levels": sorted(coverage_levels),
        "gap": "ZIPINC_QRTL=-8 is predeclared by the frozen categorical contract but has no pre-2022 coverage reference row",
        "allowed_reporting_change": "emit independent report-only rows with null reference and change fields",
        "forbidden_reporting_change": "do not merge -8 with -9; unpredeclared levels remain fail closed",
        "model_probability_outcome_threshold_calibration_conformal_ontology_mapping_changed": False,
    }


def build(previous_lock: Path, drift_analyzer: Path, drift_test: Path,
          evaluation_spec: Path, output_dir: Path) -> dict[str, Any]:
    previous_lock, drift_analyzer, drift_test, evaluation_spec, output_dir = (
        previous_lock.resolve(), drift_analyzer.resolve(), drift_test.resolve(),
        evaluation_spec.resolve(), output_dir.resolve(),
    )
    assert_sidecar(previous_lock, "previous v8 lock")
    if sha256(previous_lock) != EXPECTED_PREVIOUS_LOCK_SHA256:
        raise RuntimeError("Previous v8 lock is not the reviewed immutable identity")
    previous = read_json(previous_lock)
    prior_amendment = previous.get("technical_amendment", {})
    if (previous.get("status") != "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION"
            or previous.get("sealed_test_year") != 2022
            or previous.get("2022_access_before_lock") is not False
            or prior_amendment.get("status") != PREVIOUS_STATUS):
        raise RuntimeError("Previous v8 lock does not preserve the sealed-test contract")
    assert_sidecar(evaluation_spec, "frozen evaluation specification")
    expected_spec = previous.get("frozen_code", {}).get(str(evaluation_spec))
    actual_spec = identity(evaluation_spec)
    if not isinstance(expected_spec, dict) or any(expected_spec.get(key) != actual_spec[key] for key in ("bytes", "sha256")):
        raise RuntimeError("Frozen evaluation specification is not exactly bound by the v8 lock")
    if output_dir.exists() or list(output_dir.parent.glob(output_dir.name + ".partial*")):
        raise RuntimeError("Technical-amendment v9 output already exists or has a stale partial and is immutable")
    gap = validate_predeclared_coverage_gap(read_json(evaluation_spec))
    output_dir.mkdir(parents=True)
    try:
        created = datetime.now(timezone.utc).isoformat()
        gap["created_utc"] = created
        gap_path = output_dir / "coverage_predeclared_new_stratum_gap_record.json"
        gap_id = write_json_with_sidecar(gap_path, gap)
        builder_id = identity(Path(__file__).resolve())
        analyzer_id, test_id, spec_id = identity(drift_analyzer), identity(drift_test), identity(evaluation_spec)
        amended = copy.deepcopy(previous)
        amended["created_utc"] = created
        for item in (builder_id, analyzer_id, test_id, gap_id, spec_id):
            amended["frozen_code"][item["file"]] = item
        amended.setdefault("frozen_code_roles", {}).update({
            "technical_amendment_v9_builder": builder_id,
            "technical_amendment_v9_drift_analyzer": analyzer_id,
            "technical_amendment_v9_drift_test": test_id,
            "technical_amendment_v9_coverage_gap_record": gap_id,
            "technical_amendment_v9_frozen_evaluation_spec_revalidated": spec_id,
        })
        history = amended.get("technical_amendment_history", [])
        if not isinstance(history, list):
            raise RuntimeError("Technical amendment history is malformed")
        history.append(copy.deepcopy(prior_amendment))
        amended["technical_amendment_history"] = history
        amended["technical_amendment"] = {
            "status": STATUS,
            "schema_version": "stage7_technical_amendment_v9",
            "previous_amendment_lock": identity(previous_lock),
            "original_unlock_lock": copy.deepcopy(prior_amendment["original_unlock_lock"]),
            "evaluation_spec": spec_id,
            "coverage_gap_record": gap_id,
            "drift_analyzer": analyzer_id,
            "drift_test": test_id,
            "amendment_builder": builder_id,
            "post_unblinding_technical_amendment": True,
            "repair_basis": "frozen categorical contract predeclared ZIPINC_QRTL=-8 but frozen coverage reference has no -8 row",
            "allowed_semantic_changes": [
                "for a frozen categorical level observed in 2022 without a frozen coverage reference row, emit an independent report-only coverage row",
                "set reference and change fields null only for that independent report-only row",
                "retain ZIPINC_QRTL=-8 and ZIPINC_QRTL=-9 as separate values",
            ],
            "forbidden_semantic_changes": [
                "no category coalescing or coverage-reference estimation from 2022",
                "unpredeclared coverage levels fail closed",
                "no model, probability, outcome, threshold, calibration, conformal, ontology or mapping change",
            ],
            "year_2022_results_used_to_select_or_change_models_thresholds_or_features": False,
            "corrected_2022_primary_evaluation_reauthorized": False,
            "one_corrected_2022_drift_rerun_authorized": True,
            "previous_locks_specs_and_outputs_preserved": True,
        }
        lock_path = output_dir / "pre2022_unlock_lock_technical_amendment_v9.json"
        lock_id = write_json_with_sidecar(lock_path, amended)
        return {"status": STATUS, "lock": lock_id, "coverage_gap_record": gap_id,
                "previous_lock_sha256": EXPECTED_PREVIOUS_LOCK_SHA256}
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-lock", type=Path, required=True)
    parser.add_argument("--drift-analyzer", type=Path, required=True)
    parser.add_argument("--drift-test", type=Path, required=True)
    parser.add_argument("--evaluation-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(stable_json(build(args.previous_lock, args.drift_analyzer, args.drift_test,
                            args.evaluation_spec, args.output_dir)), end="")


if __name__ == "__main__":
    main()
