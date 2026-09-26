"""Run the locked 2021A-only Transformer selection and ablation program.

The controller never reads 2021B or 2022.  It first selects one bounded
fine-tuning configuration for the joint-pretrained encoder on 2021A, then
applies that exact configuration to all representation and component
comparators.  Jobs are serial and resumable to preserve the one-GPU/one-writer
contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SEED = 20260912
GRID = (
    {"id": "lr1e4_d010", "learning_rate": 1e-4, "dropout": 0.10},
    {"id": "lr2e4_d010", "learning_rate": 2e-4, "dropout": 0.10},
    {"id": "lr1e4_d020", "learning_rate": 1e-4, "dropout": 0.20},
    {"id": "lr2e4_d020", "learning_rate": 2e-4, "dropout": 0.20},
)
BASE = {
    "epochs": 8, "batch_size": 64, "grad_accum": 1, "weight_decay": 0.01,
    "max_tokens": 128, "d_model": 256, "nhead": 8, "num_layers": 4,
    "dim_feedforward": 1024, "threads": 8, "workers": 4, "seed": SEED,
    "checkpoint_every": 250, "patience": 2, "selection_min_delta": 1e-4,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def validate_pretraining_checkpoint(path: Path, expected_kind: str) -> dict[str, Any]:
    path = path.resolve()
    manifest_path = path.parent / "pretraining_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Missing checkpoint or pretraining manifest: {path}")
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS" or manifest.get("year_2021_accessed") is not False
            or manifest.get("year_2022_accessed") is not False):
        raise RuntimeError(f"Unsealed or incomplete pretraining source: {manifest_path}")
    final = manifest.get("final_checkpoint", {})
    if final.get("sha256") != sha256(path) or int(final.get("bytes", -1)) != path.stat().st_size:
        raise RuntimeError(f"Pretraining checkpoint hash/size mismatch: {path}")
    objectives = set(manifest.get("objectives", []))
    if expected_kind == "joint" and not {"high_cost", "prolonged_los", "death"}.issubset(objectives):
        raise RuntimeError("Joint checkpoint lacks the locked auxiliary objectives")
    if expected_kind in {"mlm_only", "ap_only"} and not objectives.issubset(
            {"masked_code", "masked_hierarchy_category"}):
        raise RuntimeError(f"{expected_kind} checkpoint contains unmatched auxiliary objectives")
    if expected_kind == "ap_only" and manifest.get("pretraining_population") != "AP_development_2018_2020":
        raise RuntimeError("AP-only checkpoint population is not locked")
    return {"path": str(path), "sha256": final["sha256"], "bytes": path.stat().st_size,
            "manifest": str(manifest_path), "kind": expected_kind}


def option(name: str) -> str:
    return "--" + name.replace("_", "-")


def finetune_command(script: Path, root: Path, history: Path, hierarchy: Path,
                     output: Path, config: dict[str, Any], pretrained: Path | None,
                     flags: tuple[str, ...], resume: Path | None = None) -> list[str]:
    command = [sys.executable, str(script), "--root", str(root), "--history-dir", str(history),
               "--hierarchy-dir", str(hierarchy), "--output-dir", str(output)]
    for name, value in config.items():
        command.extend((option(name), str(value)))
    if pretrained is not None:
        command.extend(("--pretrained-checkpoint", str(pretrained)))
    if resume is not None:
        command.extend(("--resume", str(resume)))
    command.extend(option(flag) for flag in flags)
    return command


def validate_completed_run(output: Path, expected_pretrained: Path | None,
                           expected_flags: tuple[str, ...]) -> dict[str, Any] | None:
    manifest_path = output / "finetune_manifest.json"
    prediction_path = output / "predictions_2021A.parquet"
    if not manifest_path.exists() or not prediction_path.exists():
        return None
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022"
            or manifest.get("prediction_partition") != "2021A"
            or manifest.get("2021B_accessed") is not False
            or manifest.get("year_2022_accessed") is not False):
        raise RuntimeError(f"Invalid fine-tuning partition gate: {manifest_path}")
    if manifest.get("prediction", {}).get("sha256") != sha256(prediction_path):
        raise RuntimeError(f"Prediction hash mismatch: {prediction_path}")
    source = manifest.get("pretrained_checkpoint")
    if expected_pretrained is None:
        if source is not None:
            raise RuntimeError("Scratch model unexpectedly used a pretrained checkpoint")
    elif source is None or source.get("sha256") != sha256(expected_pretrained):
        raise RuntimeError("Fine-tuning source checkpoint mismatch")
    active = {name for name, enabled in manifest.get("ablations", {}).items() if enabled}
    if active != set(expected_flags):
        raise RuntimeError(f"Ablation flags mismatch: {active} != {set(expected_flags)}")
    return manifest


def validate_operating_point(output: Path, selected_prediction: Path) -> dict[str, Any] | None:
    manifest_path = output / "transformer_operating_point_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    if (manifest.get("status") != "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A"
            or manifest.get("selection_partition") != "2021A"
            or manifest.get("2021B_accessed") is not False
            or manifest.get("year_2022_accessed") is not False
            or manifest.get("source", {}).get("sha256") != sha256(selected_prediction)):
        raise RuntimeError("Invalid or unsealed Transformer operating-point manifest")
    for name, identity in manifest.get("artifacts", {}).items():
        artifact = output / name
        if (not artifact.is_file() or identity.get("sha256") != sha256(artifact)
                or int(identity.get("bytes", -1)) != artifact.stat().st_size):
            raise RuntimeError(f"Transformer operating-point artifact mismatch: {artifact}")
    return manifest


def freeze_operating_point(script: Path, selected_output: Path, output: Path) -> dict[str, Any]:
    prediction = selected_output / "predictions_2021A.parquet"
    completed = validate_operating_point(output, prediction)
    if completed is not None:
        return completed
    command = [sys.executable, str(script), "--predictions-2021a", str(prediction),
               "--finetune-manifest", str(selected_output / "finetune_manifest.json"),
               "--output-dir", str(output)]
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / f"{output.name}.controller.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": command}, ensure_ascii=False) + "\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Transformer operating-point freeze failed with exit {result.returncode}")
    completed = validate_operating_point(output, prediction)
    if completed is None:
        raise RuntimeError("Transformer operating-point freeze exited without a complete manifest")
    return completed


def run_one(script: Path, root: Path, history: Path, hierarchy: Path, output: Path,
            config: dict[str, Any], pretrained: Path | None, flags: tuple[str, ...]) -> dict[str, Any]:
    completed = validate_completed_run(output, pretrained, flags)
    if completed is not None:
        return completed
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "last_checkpoint.pt"
    command = finetune_command(script, root, history, hierarchy, output, config, pretrained, flags,
                               checkpoint if checkpoint.exists() else None)
    with (output / "controller.log").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": command}, ensure_ascii=False) + "\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Fine-tuning failed with exit {result.returncode}: {output}")
    completed = validate_completed_run(output, pretrained, flags)
    if completed is None:
        raise RuntimeError(f"Fine-tuning exited without a complete manifest: {output}")
    return completed


def selection_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, int, str]:
    candidate_id, manifest = item
    selection = manifest["model_selection"]
    return (-float(selection["best_score"]), float(selection["best_mean_brier"]),
            int(selection["best_epoch"]), candidate_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--hierarchy-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--joint-checkpoint", type=Path, required=True)
    parser.add_argument("--mlm-checkpoint", type=Path, required=True)
    parser.add_argument("--ap-only-checkpoint", type=Path, required=True)
    parser.add_argument("--finetune-script", type=Path,
                        default=Path(__file__).with_name("finetune_ap_transformer_v2.py"))
    parser.add_argument("--operating-point-script", type=Path,
                        default=Path(__file__).with_name("freeze_transformer_calibration.py"))
    args = parser.parse_args()
    root, history, hierarchy, output_root = (args.root.resolve(), args.history_dir.resolve(),
                                              args.hierarchy_dir.resolve(), args.output_root.resolve())
    script = args.finetune_script.resolve()
    operating_point_script = args.operating_point_script.resolve()
    sources = {
        "joint": validate_pretraining_checkpoint(args.joint_checkpoint, "joint"),
        "mlm_only": validate_pretraining_checkpoint(args.mlm_checkpoint, "mlm_only"),
        "ap_only": validate_pretraining_checkpoint(args.ap_only_checkpoint, "ap_only"),
    }
    checkpoints = {name: Path(source["path"]) for name, source in sources.items()}

    candidates: dict[str, dict[str, Any]] = {}
    for grid in GRID:
        config = {**BASE, "learning_rate": grid["learning_rate"], "dropout": grid["dropout"]}
        candidates[grid["id"]] = run_one(script, root, history, hierarchy,
                                              output_root / "selection_grid" / grid["id"], config,
                                              checkpoints["joint"], ())
    selected_id, selected_manifest = sorted(candidates.items(), key=selection_key)[0]
    selected_grid = next(item for item in GRID if item["id"] == selected_id)
    selected_config = {**BASE, "learning_rate": selected_grid["learning_rate"],
                       "dropout": selected_grid["dropout"]}
    lock = {"status": "LOCKED_ON_2021A_PRE_2021B_PRE_2022", "selection_partition": "2021A",
            "criterion": "mean_unweighted_co_primary_auprc",
            "tie_breaker": "lower_mean_co_primary_brier_then_earlier_epoch_then_candidate_id",
            "grid": list(GRID), "selected_id": selected_id, "selected_config": selected_config,
            "selected_metrics": selected_manifest["model_selection"], "sources": sources,
            "2021B_accessed": False, "year_2022_accessed": False}
    atomic_json(output_root / "model_selection_lock.json", lock)

    variants: dict[str, tuple[Path | None, tuple[str, ...]]] = {
        "mlm_only": (checkpoints["mlm_only"], ()),
        "scratch": (None, ()),
        "ap_only": (checkpoints["ap_only"], ()),
        "no_hierarchy_parameter_matched": (checkpoints["joint"], ("no_hierarchy",)),
        "no_prday": (checkpoints["joint"], ("no_prday",)),
        "no_prior": (checkpoints["joint"], ("no_prior",)),
        "no_hospital_socioeconomic": (checkpoints["joint"], ("no_hospital_socioeconomic",)),
        "no_year": (checkpoints["joint"], ("no_year",)),
        "common_variable_only": (checkpoints["joint"], ("common_variable_only",)),
    }
    runs: dict[str, dict[str, Any]] = {
        "joint_main": {"output": str(output_root / "selection_grid" / selected_id),
                       "manifest": selected_manifest}
    }
    main_parameter_count = int(selected_manifest["parameter_count"])
    for name, (checkpoint, flags) in variants.items():
        output = output_root / "ablations" / name
        manifest = run_one(script, root, history, hierarchy, output, selected_config, checkpoint, flags)
        if name in {"mlm_only", "scratch", "ap_only", "no_hierarchy_parameter_matched", "no_prday", "no_prior", "no_year"}:
            if int(manifest["parameter_count"]) != main_parameter_count:
                raise RuntimeError(f"Parameter-matched comparator changed parameter count: {name}")
        runs[name] = {"output": str(output), "manifest": manifest}
        atomic_json(output_root / "ablation_registry.json",
                    {"status": "IN_PROGRESS_PRE_2021B_PRE_2022", "model_selection_lock": str(output_root / "model_selection_lock.json"),
                     "runs": runs, "2021B_accessed": False, "year_2022_accessed": False})
    operating_point_output = output_root / "operating_point_2021A"
    operating_point = freeze_operating_point(
        operating_point_script, output_root / "selection_grid" / selected_id, operating_point_output
    )
    atomic_json(output_root / "ablation_registry.json",
                {"status": "PASS_PRE_2021B_PRE_2022", "model_selection_lock": str(output_root / "model_selection_lock.json"),
                 "runs": runs, "operating_point": {"output": str(operating_point_output),
                                                     "manifest": operating_point},
                 "2021B_accessed": False, "year_2022_accessed": False,
                 })


if __name__ == "__main__":
    main()
