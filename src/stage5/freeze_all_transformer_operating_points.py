#!/usr/bin/env python3
"""Freeze every pre-specified Transformer 2021A operating point before 2021B.

This controller is deliberately a narrow, fail-closed bridge between the
completed fine-tuning ablation registry and downstream conformal calibration.
It contains no cohort reader and never accepts a future-cohort input.  Each
non-main run is delegated to ``freeze_transformer_calibration.py`` in an
exclusive directory; the main-model artifact produced by the original grid
controller is revalidated and reused rather than regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROSTER = (
    "joint_main", "mlm_only", "scratch", "ap_only",
    "no_hierarchy_parameter_matched", "no_prday", "no_prior",
    "no_hospital_socioeconomic", "no_year", "common_variable_only",
)
EXPECTED_ABLATIONS = {
    "joint_main": frozenset(),
    "mlm_only": frozenset(),
    "scratch": frozenset(),
    "ap_only": frozenset(),
    "no_hierarchy_parameter_matched": frozenset({"no_hierarchy"}),
    "no_prday": frozenset({"no_prday"}),
    "no_prior": frozenset({"no_prior"}),
    "no_hospital_socioeconomic": frozenset({"no_hospital_socioeconomic"}),
    "no_year": frozenset({"no_year"}),
    "common_variable_only": frozenset({"common_variable_only"}),
}


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
        raise RuntimeError(f"Unreadable JSON seal: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def artifact_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing required artifact: {path}")
    return {"file": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def _regular_child(directory: Path, name: object, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise RuntimeError(f"Invalid {label} filename")
    candidate = directory / name
    if not candidate.is_file() or candidate.resolve().parent != directory.resolve():
        raise RuntimeError(f"Missing or unsafe {label}: {candidate}")
    return candidate


def _assert_identity(path: Path, declared: object, label: str, require_bytes: bool = True) -> None:
    if not isinstance(declared, dict):
        raise RuntimeError(f"Missing {label} identity")
    if declared.get("sha256") != sha256(path):
        raise RuntimeError(f"{label} SHA256 mismatch: {path}")
    if require_bytes and int(declared.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"{label} size mismatch: {path}")


def _assert_finetune_gate(manifest: dict[str, Any], run_name: str) -> None:
    if (manifest.get("status") != "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022"
            or manifest.get("prediction_partition") != "2021A"
            or manifest.get("2021B_accessed") is not False
            or manifest.get("year_2022_accessed") is not False):
        raise RuntimeError(f"Fine-tuning partition/access gate failed: {run_name}")
    ablations = manifest.get("ablations")
    if not isinstance(ablations, dict):
        raise RuntimeError(f"Missing ablation declaration: {run_name}")
    active = {name for name, enabled in ablations.items() if enabled is True}
    if active != EXPECTED_ABLATIONS[run_name]:
        raise RuntimeError(f"Unexpected ablation declaration: {run_name}")


def validate_ablation_registry(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate exact roster and bind every registry entry to its on-disk run."""
    registry = read_json(path)
    if (registry.get("status") != "PASS_PRE_2021B_PRE_2022"
            or registry.get("2021B_accessed") is not False
            or registry.get("year_2022_accessed") is not False):
        raise RuntimeError("Ablation registry is not a final pre-future sealed registry")
    runs = registry.get("runs")
    if not isinstance(runs, dict) or tuple(sorted(runs)) != tuple(sorted(ROSTER)):
        missing = sorted(set(ROSTER) - set(runs or {}))
        extra = sorted(set(runs or {}) - set(ROSTER))
        raise RuntimeError(f"Incomplete or mixed ablation roster; missing={missing}, extra={extra}")
    validated: dict[str, dict[str, Any]] = {}
    seen_outputs: set[Path] = set()
    for run_name in ROSTER:
        entry = runs[run_name]
        if not isinstance(entry, dict) or not isinstance(entry.get("output"), str) or not isinstance(entry.get("manifest"), dict):
            raise RuntimeError(f"Invalid registry run entry: {run_name}")
        output = Path(entry["output"]).expanduser().resolve()
        if not output.is_dir():
            raise RuntimeError(f"Missing run output directory: {run_name}")
        if output in seen_outputs:
            raise RuntimeError(f"Two prespecified runs share one output directory: {run_name}")
        seen_outputs.add(output)
        manifest_path = output / "finetune_manifest.json"
        disk_manifest = read_json(manifest_path)
        if disk_manifest != entry["manifest"]:
            raise RuntimeError(f"Embedded and on-disk fine-tune manifest differ: {run_name}")
        _assert_finetune_gate(disk_manifest, run_name)
        prediction_declared = disk_manifest.get("prediction")
        if not isinstance(prediction_declared, dict):
            raise RuntimeError(f"Missing prediction identity: {run_name}")
        prediction = _regular_child(output, prediction_declared.get("file"), "prediction")
        _assert_identity(prediction, prediction_declared, f"prediction for {run_name}")
        checkpoint_declared = disk_manifest.get("checkpoint")
        if not isinstance(checkpoint_declared, dict):
            raise RuntimeError(f"Missing checkpoint identity: {run_name}")
        checkpoint = _regular_child(output, checkpoint_declared.get("file"), "checkpoint")
        _assert_identity(checkpoint, checkpoint_declared, f"checkpoint for {run_name}", require_bytes=False)
        validated[run_name] = {
            "output": output, "manifest_path": manifest_path, "manifest": disk_manifest,
            "prediction": prediction, "checkpoint": checkpoint,
        }
    return registry, validated


def validate_operating_point(directory: Path, source: dict[str, Any], *,
                             allow_grid_controller_log: bool = False) -> dict[str, Any]:
    """Validate a complete freezer output and bind it to one 2021A run."""
    directory = directory.resolve()
    manifest_path = directory / "transformer_operating_point_manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A"
            or manifest.get("selection_partition") != "2021A"
            or manifest.get("2021B_accessed") is not False
            or manifest.get("year_2022_accessed") is not False):
        raise RuntimeError(f"Operating-point partition/access gate failed: {directory}")
    manifest_source = manifest.get("source")
    if not isinstance(manifest_source, dict):
        raise RuntimeError("Operating-point source identity missing")
    if (manifest_source.get("file") != str(source["prediction"])
            or manifest_source.get("manifest") != str(source["manifest_path"])
            or manifest_source.get("sha256") != sha256(source["prediction"])
            or int(manifest_source.get("bytes", -1)) != source["prediction"].stat().st_size
            or manifest_source.get("manifest_sha256") != sha256(source["manifest_path"])):
        raise RuntimeError(f"Operating-point source does not bind to run: {directory}")
    artifacts = manifest.get("artifacts")
    expected = {
        "predictions_2021A_calibrated.parquet",
        "transformer_binary_calibrators.joblib",
        "transformer_operating_thresholds_2021A.json",
        "transformer_calibration_selection_2021A.json",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise RuntimeError(f"Operating-point artifact roster is incomplete or mixed: {directory}")
    allowed = set(expected) | {manifest_path.name}
    # The pre-existing main artifact is produced by run_finetuning_grid.py,
    # which writes this command transcript before invoking the same freezer.
    # It is not a calibrated artifact and is never allowed for new outputs.
    actual = {item.name for item in directory.iterdir()}
    if allow_grid_controller_log and "controller.log" in actual:
        allowed.add("controller.log")
    if actual != allowed:
        raise RuntimeError(f"Operating-point directory is partial or mixed: {directory}")
    for name in sorted(expected):
        artifact = _regular_child(directory, name, "operating-point artifact")
        _assert_identity(artifact, artifacts[name], f"operating-point artifact {name}")
    return manifest


def validate_reused_main(registry: dict[str, Any], source: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    entry = registry.get("operating_point")
    if not isinstance(entry, dict) or not isinstance(entry.get("output"), str) or not isinstance(entry.get("manifest"), dict):
        raise RuntimeError("Main-model operating point is absent from final ablation registry")
    directory = Path(entry["output"]).expanduser().resolve()
    manifest = validate_operating_point(directory, source, allow_grid_controller_log=True)
    if manifest != entry["manifest"]:
        raise RuntimeError("Embedded and on-disk main operating-point manifests differ")
    return directory, manifest


def _exclusive_run_directory(output_root: Path, run_name: str) -> Path:
    root = output_root.resolve()
    target = (root / run_name).resolve()
    if target.parent != root:
        raise RuntimeError("Unsafe per-run output location")
    if target.exists():
        if not target.is_dir():
            raise RuntimeError(f"Per-run output is not a directory: {target}")
        return target
    root.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Concurrent or ambiguous per-run output creation: {target}") from exc
    return target


def freeze_one(freezer_script: Path, output_root: Path, run_name: str,
               source: dict[str, Any]) -> tuple[Path, dict[str, Any], bool]:
    """Return a sealed run, or fail rather than repairing any partial output."""
    target = _exclusive_run_directory(output_root, run_name)
    if any(target.iterdir()):
        if not (target / "transformer_operating_point_manifest.json").is_file():
            raise RuntimeError(f"Operating-point directory is partial or mixed: {target}")
        return target, validate_operating_point(target, source), True
    command = [sys.executable, str(freezer_script.resolve()),
               "--predictions-2021a", str(source["prediction"]),
               "--finetune-manifest", str(source["manifest_path"]),
               "--output-dir", str(target)]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, check=False)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Operating-point freezer failed for {run_name} (exit {completed.returncode}): {detail}")
    return target, validate_operating_point(target, source), False


def source_identity(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_output": str(source["output"]),
        "finetune_manifest": artifact_identity(source["manifest_path"]),
        "prediction": artifact_identity(source["prediction"]),
        "checkpoint": artifact_identity(source["checkpoint"]),
        "active_ablations": sorted(name for name, enabled in source["manifest"]["ablations"].items()
                                     if enabled is True),
    }


def write_final_registry(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish once: no replacement and no ambiguous resume state."""
    path = path.resolve()
    if path.exists():
        raise RuntimeError("Final all-operating-points registry already exists and is immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(stable_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError("Final all-operating-points registry appeared concurrently") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def run_controller(ablation_registry: Path, output_root: Path, final_registry: Path,
                   freezer_script: Path) -> dict[str, Any]:
    if not freezer_script.is_file():
        raise RuntimeError(f"Missing single-run calibration freezer: {freezer_script}")
    registry_path = ablation_registry.resolve()
    registry, sources = validate_ablation_registry(registry_path)
    main_directory, main_manifest = validate_reused_main(registry, sources["joint_main"])
    results: dict[str, dict[str, Any]] = {
        "joint_main": {
            "source": source_identity(sources["joint_main"]),
            "operating_point_directory": str(main_directory),
            "operating_point_manifest": artifact_identity(main_directory / "transformer_operating_point_manifest.json"),
            "reused_existing_main_artifact": True,
        }
    }
    # Ordered iteration is an intentional one-process/one-writer execution contract.
    for run_name in ROSTER[1:]:
        target, manifest, resumed = freeze_one(freezer_script, output_root, run_name, sources[run_name])
        results[run_name] = {
            "source": source_identity(sources[run_name]),
            "operating_point_directory": str(target),
            "operating_point_manifest": artifact_identity(target / "transformer_operating_point_manifest.json"),
            "reused_existing_main_artifact": False,
            "resumed_verified_seal": resumed,
        }
        # The manifest is parsed only for validation; never trust a subprocess exit alone.
        if manifest.get("source", {}).get("sha256") != results[run_name]["source"]["prediction"]["sha256"]:
            raise RuntimeError(f"Post-freeze source seal mismatch: {run_name}")
    final = {
        "status": "PASS_ALL_TRANSFORMER_CALIBRATIONS_AND_THRESHOLDS_LOCKED_ON_2021A",
        "selection_partition": "2021A",
        "exact_roster": list(ROSTER),
        "ablation_registry": artifact_identity(registry_path),
        "runs": results,
        "2021B_accessed": False,
        "year_2022_accessed": False,
    }
    # Check the reused manifest remains source-bound immediately before publication.
    if main_manifest.get("source", {}).get("sha256") != results["joint_main"]["source"]["prediction"]["sha256"]:
        raise RuntimeError("Main operating-point source changed during controller execution")
    write_final_registry(final_registry, final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="exclusive per-ablation operating-point directories")
    parser.add_argument("--final-registry", type=Path, required=True)
    parser.add_argument("--freezer-script", type=Path,
                        default=Path(__file__).with_name("freeze_transformer_calibration.py"))
    args = parser.parse_args()
    result = run_controller(args.ablation_registry, args.output_root, args.final_registry,
                            args.freezer_script)
    print(stable_json(result))


if __name__ == "__main__":
    main()
