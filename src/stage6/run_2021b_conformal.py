#!/usr/bin/env python3
"""Run the serial 2021B prediction and hierarchical conformal calibration stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


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
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(stable_json(value), encoding="utf-8")
    temporary.replace(path)


def validate_prediction(output: Path, checkpoint: Path,
                        operating_manifest: Path) -> dict[str, Any] | None:
    manifest_path = output / "prediction_manifest.json"
    prediction_path = output / "predictions_2021B.parquet"
    if not manifest_path.exists() and not prediction_path.exists():
        return None
    if not manifest_path.is_file() or not prediction_path.is_file():
        raise RuntimeError("Partial 2021B prediction output exists")
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS_PREDICTIONS_2021B_ONLY"
            or manifest.get("partition") != "2021B"
            or manifest.get("model_or_threshold_selection_on_2021B") is not False
            or manifest.get("year_2022_accessed") is not False
            or manifest.get("model", {}).get("sha256") != sha256(checkpoint)
            or manifest.get("operating_point", {}).get("manifest_sha256") != sha256(operating_manifest)
            or manifest.get("artifact", {}).get("sha256") != sha256(prediction_path)):
        raise RuntimeError("2021B prediction output fails seal or identity validation")
    return manifest


def validate_conformal(output: Path, prediction: Path,
                       operating_manifest: Path) -> dict[str, Any] | None:
    calibrator_path = output / "conformal_calibrator.json"
    set_path = output / "conformal_sets_2021B.parquet"
    if not calibrator_path.exists() and not set_path.exists():
        return None
    if not calibrator_path.is_file() or not set_path.is_file():
        raise RuntimeError("Partial conformal output exists")
    manifest = read_json(calibrator_path)
    operating = manifest.get("operating_point_source", {})
    if (manifest.get("status") != "PASS_LOCKED_PRE_2022"
            or manifest.get("calibration_partition") != "2021B only"
            or manifest.get("model_or_threshold_selection_on_2021B") is not False
            or manifest.get("year_2022_accessed") is not False
            or manifest.get("source", {}).get("sha256") != sha256(prediction)
            or manifest.get("artifact", {}).get("sha256") != sha256(set_path)
            or operating.get("manifest_sha256") != sha256(operating_manifest)):
        raise RuntimeError("Conformal output fails seal or identity validation")
    return manifest


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": command}, ensure_ascii=False) + "\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {command[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-dir", type=Path, required=True)
    parser.add_argument("--model-selection-lock", type=Path, required=True)
    parser.add_argument("--selected-model-dir", type=Path, required=True)
    parser.add_argument("--operating-point-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--min-mondrian-rows", type=int, default=200)
    parser.add_argument("--predict-script", type=Path,
                        default=Path(__file__).with_name("predict_locked_2021b.py"))
    parser.add_argument("--conformal-script", type=Path,
                        default=Path(__file__).with_name("fit_hierarchical_conformal.py"))
    args = parser.parse_args()
    if not 1 <= args.threads <= 8 or not 0 <= args.workers <= 8:
        raise SystemExit("Invalid resource arguments")

    root = args.root.resolve()
    history = args.history_dir.resolve()
    hierarchy = args.hierarchy_dir.resolve()
    model_lock = args.model_selection_lock.resolve()
    selected_model = args.selected_model_dir.resolve()
    operating = args.operating_point_dir.resolve()
    output_root = args.output_root.resolve()
    prediction_output = output_root / "predictions_2021B"
    conformal_output = output_root / "conformal_2021B"
    checkpoint = selected_model / "best_checkpoint.pt"
    operating_manifest = operating / "transformer_operating_point_manifest.json"
    prediction_path = prediction_output / "predictions_2021B.parquet"

    prediction_manifest = validate_prediction(prediction_output, checkpoint, operating_manifest)
    if prediction_manifest is None:
        command = [sys.executable, str(args.predict_script.resolve()),
                   "--root", str(root), "--history-dir", str(history),
                   "--hierarchy-dir", str(hierarchy), "--model-selection-lock", str(model_lock),
                   "--selected-model-dir", str(selected_model), "--operating-point-dir", str(operating),
                   "--output-dir", str(prediction_output), "--batch-size", str(args.batch_size),
                   "--workers", str(args.workers), "--threads", str(args.threads)]
        run_logged(command, prediction_output / "controller.log")
        prediction_manifest = validate_prediction(prediction_output, checkpoint, operating_manifest)
        if prediction_manifest is None:
            raise RuntimeError("2021B prediction exited without a complete manifest")

    conformal_manifest = validate_conformal(conformal_output, prediction_path, operating_manifest)
    if conformal_manifest is None:
        command = [sys.executable, str(args.conformal_script.resolve()),
                   "--predictions-2021b", str(prediction_path),
                   "--output-dir", str(conformal_output),
                   "--transformer-operating-point-dir", str(operating),
                   "--min-mondrian-rows", str(args.min_mondrian_rows)]
        run_logged(command, conformal_output / "controller.log")
        conformal_manifest = validate_conformal(conformal_output, prediction_path, operating_manifest)
        if conformal_manifest is None:
            raise RuntimeError("Conformal calibration exited without a complete manifest")

    registry = {
        "status": "PASS_2021B_CONFORMAL_LOCKED_PRE_2022",
        "partition": "2021B only",
        "model_or_threshold_selection_on_2021B": False,
        "year_2022_accessed": False,
        "prediction": {"manifest": str(prediction_output / "prediction_manifest.json"),
                       "manifest_sha256": sha256(prediction_output / "prediction_manifest.json"),
                       "artifact": str(prediction_path), "artifact_sha256": sha256(prediction_path)},
        "conformal": {"manifest": str(conformal_output / "conformal_calibrator.json"),
                      "manifest_sha256": sha256(conformal_output / "conformal_calibrator.json"),
                      "artifact": str(conformal_output / "conformal_sets_2021B.parquet"),
                      "artifact_sha256": sha256(conformal_output / "conformal_sets_2021B.parquet")},
        "nominal_coverages": conformal_manifest["nominal_coverages"],
    }
    atomic_json(output_root / "stage6_registry.json", registry)
    print(stable_json(registry))


if __name__ == "__main__":
    main()
