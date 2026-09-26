#!/usr/bin/env python3
"""Create the once-only, fully specified pre-2022 unlock lock.

This is deliberately role based.  An arbitrary list of code files cannot show
that every production pathway is fixed before the sealed temporal test opens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRANSFORMER_ROSTER = ("joint_main", "mlm_only", "scratch", "ap_only", "no_hierarchy_parameter_matched", "no_prday", "no_prior", "no_hospital_socioeconomic", "no_year", "common_variable_only")
GATE_ROLES = ("analysis_lock", "baseline_lock", "model_selection_lock", "ablation_registry", "operating_point_manifest", "stage6_registry")
ARTIFACT_ROLES = ("all_transformer_registry", "drift_reference", "evaluation_spec", "etl_spec", "derivative_spec", "pra_lock")
CODE_ROLES = ("etl", "derivative", "prediction", "evaluation", "drift_reference_builder", "drift_analyzer", "data_spec_builder", "evaluation_spec_builder")
ETL_DEPENDENCY_ROLES = ("stage2_ontology", "pra_code_sets", "diagnosis_vocabulary", "procedure_vocabulary", "auxiliary_thresholds", "cpi_constants")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable JSON lock input: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing lock input: {path}")
    return {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _assert_identity(path: Path, declared: dict[str, Any], label: str) -> None:
    actual = identity(path)
    if not isinstance(declared, dict) or declared.get("sha256") != actual["sha256"] or int(declared.get("bytes", -1)) != actual["bytes"]:
        raise RuntimeError(f"{label} path/bytes/SHA256 identity mismatch")


def _validate_sidecar(path: Path, label: str) -> Path:
    path = path.resolve(); sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"Missing {label} or SHA256 sidecar")
    if sidecar.read_text(encoding="ascii").strip() != f"{sha256(path)}  {path.name}":
        raise RuntimeError(f"{label} SHA256 sidecar mismatch")
    return sidecar


def _require_exact_roles(value: dict[str, Path], roles: tuple[str, ...], label: str) -> None:
    if set(value) != set(roles):
        missing, extra = sorted(set(roles) - set(value)), sorted(set(value) - set(roles))
        raise RuntimeError(f"{label} roles are incomplete or mixed; missing={missing}, extra={extra}")


def _no_duplicate_paths(value: dict[str, Path], label: str) -> None:
    paths = [path.resolve() for path in value.values()]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"{label} roles must name unique paths")


def validate_gate(name: str, value: dict[str, Any]) -> None:
    validators = {
        "analysis_lock": value.get("status") == "LOCKED_PRE_2022" and value.get("sealed_test_year") == 2022 and value.get("conformal_only_partition") == "2021B",
        "baseline_lock": value.get("status") == "PASS_BASELINES_FROZEN_PRE_2021B_PRE_2022" and value.get("2021B_outcomes_accessed") is False and value.get("year_2022_accessed") is False,
        "model_selection_lock": value.get("status") == "LOCKED_ON_2021A_PRE_2021B_PRE_2022" and value.get("selection_partition") == "2021A" and value.get("2021B_accessed") is False and value.get("year_2022_accessed") is False,
        "ablation_registry": value.get("status") == "PASS_PRE_2021B_PRE_2022" and value.get("2021B_accessed") is False and value.get("year_2022_accessed") is False and isinstance(value.get("operating_point"), dict),
        "operating_point_manifest": value.get("status") == "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A" and value.get("selection_partition") == "2021A" and value.get("2021B_accessed") is False and value.get("year_2022_accessed") is False,
        "stage6_registry": value.get("status") == "PASS_2021B_CONFORMAL_LOCKED_PRE_2022" and value.get("partition") == "2021B only" and value.get("model_or_threshold_selection_on_2021B") is False and value.get("year_2022_accessed") is False,
    }
    if name not in validators or not validators[name]:
        raise RuntimeError(f"Pre-2022 gate failed: {name}")


def _validate_all_transformer_registry(path: Path) -> list[Path]:
    registry = read_json(path)
    if (registry.get("status") != "PASS_ALL_TRANSFORMER_CALIBRATIONS_AND_THRESHOLDS_LOCKED_ON_2021A" or registry.get("selection_partition") != "2021A" or registry.get("2021B_accessed") is not False or registry.get("year_2022_accessed") is not False or registry.get("exact_roster") != list(TRANSFORMER_ROSTER)):
        raise RuntimeError("All-transformer operating-point registry is not sealed 2021A-only")
    runs = registry.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(TRANSFORMER_ROSTER):
        raise RuntimeError("All-transformer operating-point registry roster is incomplete")
    directories: set[Path] = set(); manifests: list[Path] = []
    for run in TRANSFORMER_ROSTER:
        entry = runs[run]
        if not isinstance(entry, dict) or not isinstance(entry.get("operating_point_directory"), str):
            raise RuntimeError(f"Missing operating-point directory: {run}")
        directory = Path(entry["operating_point_directory"]).resolve(); manifest = directory / "transformer_operating_point_manifest.json"
        if not directory.is_dir() or directory in directories or not manifest.is_file():
            raise RuntimeError(f"Unsafe, duplicate, or missing operating point: {run}")
        directories.add(directory); _assert_identity(manifest, entry.get("operating_point_manifest"), f"{run} operating-point manifest")
        validate_gate("operating_point_manifest", read_json(manifest)); manifests.append(manifest)
    return manifests


def _validate_drift_reference(path: Path) -> Path:
    sidecar = _validate_sidecar(path, "Frozen drift reference"); value = read_json(path)
    source = value.get("source_partitions", {})
    if (value.get("status") != "FROZEN_2022_DRIFT_REFERENCE" or value.get("year_2022_accessed") is not False or value.get("model_or_threshold_selection_on_2021B") is not False or source.get("selection_partition") != "2021A" or source.get("conformal_partition") != "2021B only"):
        raise RuntimeError("Frozen drift reference seal/access contract failed")
    return sidecar


def _validate_evaluation_spec(path: Path) -> Path:
    sidecar = _validate_sidecar(path, "Frozen evaluation specification"); value = read_json(path)
    if (value.get("status") != "FROZEN_2022_EVALUATION_SPEC" or value.get("sealed_test_year") != 2022 or value.get("2021B_model_or_threshold_selection") is not False or value.get("year_2022_accessed_before_unlock") is not False or value.get("transformer_roster") != list(TRANSFORMER_ROSTER)):
        raise RuntimeError("Frozen evaluation specification seal/access/roster failed")
    return sidecar


def _validate_data_spec(path: Path, status: str, label: str) -> Path:
    sidecar = _validate_sidecar(path, label); value = read_json(path)
    if (value.get("status") != status or value.get("sealed_test_year") != 2022 or value.get("test_partition") != "test" or value.get("data_dependent_adaptation") is not False):
        raise RuntimeError(f"{label} seal/adaptation contract failed")
    return sidecar


def _validate_pra_lock(path: Path) -> tuple[Path, list[Path]]:
    sidecar = _validate_sidecar(path, "2022 PRA lock"); value = read_json(path)
    if value.get("status") != "PASS_2022_PRA_LOCKED_PRE_TEST_ACCESS" or value.get("calendar_year") != 2022 or value.get("nrd_2022_accessed") is not False:
        raise RuntimeError("2022 PRA lock seal/access gate failed")
    sources, artifacts = value.get("sources"), value.get("artifacts")
    if not isinstance(sources, dict) or "yale_modified_ccs_2024_mapping" not in sources:
        raise RuntimeError("2022 PRA lock lacks declared mapping dependency")
    if not isinstance(artifacts, dict) or not {"ccs_mapping", "algorithm_code_sets"}.issubset(artifacts):
        raise RuntimeError("2022 PRA lock lacks declared mapping/code-set artifacts")
    bound: list[Path] = []
    for name, declared in sources.items():
        if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
            raise RuntimeError(f"Invalid 2022 PRA source identity: {name}")
        target = Path(declared["path"]).resolve(); _assert_identity(target, declared, f"2022 PRA source {name}"); bound.append(target)
    for name, declared in artifacts.items():
        if not isinstance(declared, dict) or not isinstance(declared.get("file"), str):
            raise RuntimeError(f"Invalid 2022 PRA artifact identity: {name}")
        target = path.resolve().parent / declared["file"]
        if target.resolve().parent != path.resolve().parent:
            raise RuntimeError(f"Unsafe 2022 PRA artifact path: {name}")
        _assert_identity(target, declared, f"2022 PRA artifact {name}"); bound.append(target)
    return sidecar, bound


def _etl_dependencies(path: Path) -> list[Path]:
    dependencies = read_json(path).get("frozen_dependencies")
    if not isinstance(dependencies, dict) or set(dependencies) != set(ETL_DEPENDENCY_ROLES):
        raise RuntimeError("Frozen ETL specification dependency roster is incomplete")
    targets = [Path(raw).resolve() for raw in dependencies.values() if isinstance(raw, str) and raw]
    if len(targets) != len(ETL_DEPENDENCY_ROLES):
        raise RuntimeError("Frozen ETL specification has an invalid dependency path")
    for target in targets: identity(target)
    return targets


def _atomic_publish_lock(output: Path, payload: dict[str, Any]) -> str:
    """Publish lock and sidecar as one recoverable, no-overwrite transaction."""
    output = output.resolve(); sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists():
        raise RuntimeError("Pre-2022 lock already exists and is immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_fd, lock_tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    sidecar_fd, sidecar_tmp_name = tempfile.mkstemp(prefix=f".{sidecar.name}.", suffix=".tmp", dir=output.parent)
    lock_tmp, sidecar_tmp = Path(lock_tmp_name), Path(sidecar_tmp_name)
    try:
        with os.fdopen(lock_fd, "w", encoding="utf-8") as handle:
            handle.write(stable_json(payload)); handle.flush(); os.fsync(handle.fileno())
        digest = sha256(lock_tmp)
        with os.fdopen(sidecar_fd, "w", encoding="ascii") as handle:
            handle.write(f"{digest}  {output.name}\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(lock_tmp, output)
        try:
            os.link(sidecar_tmp, sidecar)
        except Exception:
            # Remove only the first hard link made by this call.  If another
            # writer changed the target, leave it untouched for diagnosis.
            if output.exists() and os.path.samefile(lock_tmp, output):
                output.unlink()
            raise
        return digest
    finally:
        if lock_tmp.exists(): lock_tmp.unlink()
        if sidecar_tmp.exists(): sidecar_tmp.unlink()


def create_lock(inputs: dict[str, Path], artifacts: dict[str, Path], code_roles: dict[str, Path], output: Path) -> dict[str, Any]:
    """Validate every fixed role and exclusively publish the immutable lock."""
    _require_exact_roles(inputs, GATE_ROLES, "Upstream gate"); _require_exact_roles(artifacts, ARTIFACT_ROLES, "Frozen artifact"); _require_exact_roles(code_roles, CODE_ROLES, "Final code")
    _no_duplicate_paths(artifacts, "Frozen artifact"); _no_duplicate_paths(code_roles, "Final code")
    output = output.resolve(); sidecar = output.with_suffix(output.suffix + ".sha256")
    if output.exists() or sidecar.exists(): raise RuntimeError("Pre-2022 lock already exists and is immutable")
    gates: dict[str, dict[str, Any]] = {}
    for role in GATE_ROLES:
        path = inputs[role].resolve(); value = read_json(path); validate_gate(role, value); gates[role] = {**identity(path), "status": value["status"]}
    artifact_paths = {role: artifacts[role].resolve() for role in ARTIFACT_ROLES}
    transformer_manifests = _validate_all_transformer_registry(artifact_paths["all_transformer_registry"])
    drift_sidecar = _validate_drift_reference(artifact_paths["drift_reference"]); evaluation_sidecar = _validate_evaluation_spec(artifact_paths["evaluation_spec"])
    etl_sidecar = _validate_data_spec(artifact_paths["etl_spec"], "FROZEN_2022_LOCAL_ETL_SPEC", "Frozen ETL specification")
    derivative_sidecar = _validate_data_spec(artifact_paths["derivative_spec"], "FROZEN_2022_DERIVATIVE_SPEC", "Frozen derivative specification")
    pra_sidecar, pra_dependencies = _validate_pra_lock(artifact_paths["pra_lock"]); etl_dependencies = _etl_dependencies(artifact_paths["etl_spec"])
    frozen_code: dict[str, dict[str, Any]] = {}
    for path in (*inputs.values(), *artifact_paths.values(), drift_sidecar, evaluation_sidecar, etl_sidecar, derivative_sidecar, pra_sidecar, *transformer_manifests, *pra_dependencies, *etl_dependencies, *code_roles.values()):
        resolved = path.resolve(); frozen_code[str(resolved)] = identity(resolved)
    lock = {
        "status": "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION", "created_utc": datetime.now(timezone.utc).isoformat(), "sealed_test_year": 2022, "2022_access_before_lock": False,
        "authorized_uses": ["one immutable primary prediction pass for all frozen models", "locked primary metrics, paired patient bootstrap, calibration, decision curves and conformal coverage", "prespecified covariate/label/calibration/coverage drift summaries"],
        "prohibited_after_unlock": ["model, hyperparameter, feature, ontology, calibration-method or threshold selection", "overwriting the primary 2022 prediction or report", "presenting later recalibration or code changes as prespecified primary analysis"],
        "upstream_gates": gates, "frozen_artifacts": {role: identity(path) for role, path in artifact_paths.items()}, "frozen_code_roles": {role: identity(path) for role, path in code_roles.items()}, "frozen_code": frozen_code,
    }
    digest = _atomic_publish_lock(output, lock)
    return {**lock, "lock_sha256": digest, "sidecar": str(sidecar)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in GATE_ROLES: parser.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
    for role in ARTIFACT_ROLES: parser.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
    for role in CODE_ROLES: parser.add_argument("--" + role.replace("_", "-") + "-code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    print(stable_json(create_lock({r: getattr(args, r) for r in GATE_ROLES}, {r: getattr(args, r) for r in ARTIFACT_ROLES}, {r: getattr(args, r + "_code") for r in CODE_ROLES}, args.output)))


if __name__ == "__main__": main()
