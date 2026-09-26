import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from freeze_all_transformer_operating_points import ROSTER, run_controller, validate_ablation_registry


def test_rejects_incomplete_roster_before_any_freezer_execution(tmp_path: Path):
    registry = {
        "status": "PASS_PRE_2021B_PRE_2022",
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "runs": {name: {} for name in ROSTER[:-1]},
    }
    path = tmp_path / "ablation_registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Incomplete or mixed ablation roster"):
        validate_ablation_registry(path)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _operating_manifest(directory: Path, prediction: Path, finetune_manifest: Path) -> dict:
    artifact_names = (
        "predictions_2021A_calibrated.parquet",
        "transformer_binary_calibrators.joblib",
        "transformer_operating_thresholds_2021A.json",
        "transformer_calibration_selection_2021A.json",
    )
    for name in artifact_names:
        (directory / name).write_bytes(name.encode("ascii"))
    return {
        "status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A",
        "selection_partition": "2021A",
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "source": {
            "file": str(prediction.resolve()), "manifest": str(finetune_manifest.resolve()),
            "sha256": _hash(prediction), "bytes": prediction.stat().st_size,
            "manifest_sha256": _hash(finetune_manifest),
        },
        "artifacts": {name: {"sha256": _hash(directory / name),
                              "bytes": (directory / name).stat().st_size}
                      for name in artifact_names},
    }


def _source_run(root: Path, name: str) -> tuple[dict, Path, Path]:
    output = root / name
    output.mkdir()
    prediction = output / "predictions_2021A.parquet"
    checkpoint = output / "best_checkpoint.pt"
    prediction.write_bytes(f"prediction-{name}".encode("ascii"))
    checkpoint.write_bytes(f"checkpoint-{name}".encode("ascii"))
    active = {
        "no_hierarchy_parameter_matched": "no_hierarchy",
        "no_prday": "no_prday",
        "no_prior": "no_prior",
        "no_hospital_socioeconomic": "no_hospital_socioeconomic",
        "no_year": "no_year",
        "common_variable_only": "common_variable_only",
    }.get(name)
    manifest = {
        "status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022",
        "prediction_partition": "2021A",
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "prediction": {"file": prediction.name, "sha256": _hash(prediction),
                       "bytes": prediction.stat().st_size},
        "checkpoint": {"file": checkpoint.name, "sha256": _hash(checkpoint)},
        "ablations": ({active: True} if active else {}),
    }
    manifest_path = output / "finetune_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return {"output": str(output), "manifest": manifest}, prediction, manifest_path


def _complete_registry(tmp_path: Path) -> tuple[Path, dict[str, tuple[Path, Path]]]:
    sources = {}
    runs = {}
    for name in ROSTER:
        entry, prediction, manifest = _source_run(tmp_path, name)
        runs[name] = entry
        sources[name] = (prediction, manifest)
    main_dir = tmp_path / "main_operating"
    main_dir.mkdir()
    main = _operating_manifest(main_dir, *sources["joint_main"])
    (main_dir / "transformer_operating_point_manifest.json").write_text(
        json.dumps(main, sort_keys=True), encoding="utf-8")
    registry = {
        "status": "PASS_PRE_2021B_PRE_2022", "2021B_accessed": False,
        "year_2022_accessed": False, "runs": runs,
        "operating_point": {"output": str(main_dir), "manifest": main},
    }
    path = tmp_path / "ablation_registry.json"
    path.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")
    return path, sources


@pytest.mark.parametrize("legacy_controller_log", [False, True])
def test_serial_controller_seals_every_non_main_run_and_writes_immutable_registry(
        tmp_path: Path, monkeypatch, legacy_controller_log: bool):
    registry, _ = _complete_registry(tmp_path)
    # Compatibility with the existing grid controller's command transcript is
    # explicit and limited to reuse of the already sealed main artifact.
    if legacy_controller_log:
        (tmp_path / "main_operating" / "controller.log").write_text(
            "existing command", encoding="utf-8")
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        prediction = Path(command[command.index("--predictions-2021a") + 1])
        finetune = Path(command[command.index("--finetune-manifest") + 1])
        target = Path(command[command.index("--output-dir") + 1])
        manifest = _operating_manifest(target, prediction, finetune)
        (target / "transformer_operating_point_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("freeze_all_transformer_operating_points.subprocess.run", fake_run)
    freezer = tmp_path / "freezer.py"
    freezer.write_text("# synthetic", encoding="utf-8")
    final = tmp_path / "final.json"
    result = run_controller(registry, tmp_path / "sealed", final, freezer)
    assert result["status"] == "PASS_ALL_TRANSFORMER_CALIBRATIONS_AND_THRESHOLDS_LOCKED_ON_2021A"
    assert result["exact_roster"] == list(ROSTER)
    assert len(calls) == len(ROSTER) - 1
    saved = json.loads(final.read_text(encoding="utf-8"))
    assert list(saved["runs"]) == sorted(ROSTER)
    with pytest.raises(RuntimeError, match="already exists"):
        run_controller(registry, tmp_path / "sealed", final, freezer)


def test_existing_partial_per_run_output_is_rejected_not_repaired(tmp_path: Path):
    registry, _ = _complete_registry(tmp_path)
    output = tmp_path / "sealed" / "mlm_only"
    output.mkdir(parents=True)
    (output / "stray.txt").write_text("partial", encoding="utf-8")
    freezer = tmp_path / "freezer.py"
    freezer.write_text("# never invoked", encoding="utf-8")
    with pytest.raises(RuntimeError, match="partial or mixed"):
        run_controller(registry, tmp_path / "sealed", tmp_path / "final.json", freezer)


def test_tampered_registered_prediction_is_rejected_before_any_output_creation(tmp_path: Path):
    registry, sources = _complete_registry(tmp_path)
    sources["joint_main"][0].write_bytes(b"tampered-after-registry")
    freezer = tmp_path / "freezer.py"
    freezer.write_text("# never invoked", encoding="utf-8")
    with pytest.raises(RuntimeError, match="prediction for joint_main SHA256 mismatch"):
        run_controller(registry, tmp_path / "sealed", tmp_path / "final.json", freezer)
    assert not (tmp_path / "sealed").exists()
