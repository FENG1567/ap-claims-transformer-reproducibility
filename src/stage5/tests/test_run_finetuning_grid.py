import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_finetuning_grid import (
    BASE, finetune_command, freeze_operating_point, selection_key, validate_completed_run,
    validate_operating_point, validate_pretraining_checkpoint,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_command_is_argument_list_and_contains_only_requested_ablation(tmp_path):
    command = finetune_command(tmp_path/"fine.py", tmp_path/"root", tmp_path/"history",
                               tmp_path/"hierarchy", tmp_path/"out", BASE,
                               tmp_path/"pre.pt", ("no_prday",), tmp_path/"last.pt")
    assert command[0] == sys.executable
    assert "--no-prday" in command and "--resume" in command and "--pretrained-checkpoint" in command
    assert "--no-year" not in command and "2021B" not in " ".join(command) and "2022" not in " ".join(command)


def test_selection_key_prefers_score_then_brier_then_earlier_epoch_and_id():
    def manifest(score, brier, epoch):
        return {"model_selection": {"best_score": score, "best_mean_brier": brier, "best_epoch": epoch}}
    items = [("b", manifest(.5, .2, 2)), ("a", manifest(.5, .2, 2)),
             ("c", manifest(.5, .1, 3)), ("d", manifest(.6, .4, 8))]
    assert sorted(items, key=selection_key)[0][0] == "d"
    assert sorted(items[:-1], key=selection_key)[0][0] == "c"
    assert sorted(items[:2], key=selection_key)[0][0] == "a"


@pytest.mark.parametrize("kind,objectives,population", [
    ("joint", ["masked_code", "masked_hierarchy_category", "high_cost", "prolonged_los", "death"], None),
    ("mlm_only", ["masked_code", "masked_hierarchy_category"], None),
    ("ap_only", ["masked_code", "masked_hierarchy_category"], "AP_development_2018_2020"),
])
def test_pretraining_checkpoint_requires_pass_hash_seal_and_expected_objectives(tmp_path, kind, objectives, population):
    checkpoint = tmp_path/"pretrained_encoder_final.pt"; checkpoint.write_bytes(b"weights")
    manifest = {"status": "PASS", "year_2021_accessed": False, "year_2022_accessed": False,
                "objectives": objectives, "final_checkpoint": {"sha256": _hash(checkpoint), "bytes": checkpoint.stat().st_size}}
    if population: manifest["pretraining_population"] = population
    (tmp_path/"pretraining_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_pretraining_checkpoint(checkpoint, kind)["sha256"] == _hash(checkpoint)
    manifest["year_2022_accessed"] = True
    (tmp_path/"pretraining_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unsealed"):
        validate_pretraining_checkpoint(checkpoint, kind)


def test_completed_run_checks_prediction_hash_source_and_exact_flags(tmp_path):
    output = tmp_path/"run"; output.mkdir()
    prediction = output/"predictions_2021A.parquet"; prediction.write_bytes(b"prediction")
    source = tmp_path/"source.pt"; source.write_bytes(b"source")
    manifest = {"status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022",
                "prediction_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False,
                "prediction": {"sha256": _hash(prediction)},
                "pretrained_checkpoint": {"sha256": _hash(source)},
                "ablations": {"no_prday": True, "no_year": False}}
    (output/"finetune_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_completed_run(output, source, ("no_prday",)) == manifest
    with pytest.raises(RuntimeError, match="Ablation flags mismatch"):
        validate_completed_run(output, source, ("no_year",))


def test_operating_point_manifest_is_partition_and_artifact_hash_bound(tmp_path):
    selected = tmp_path / "predictions_2021A.parquet"
    selected.write_bytes(b"selected")
    output = tmp_path / "operating"
    output.mkdir()
    artifact = output / "transformer_operating_thresholds_2021A.json"
    artifact.write_text("{}", encoding="utf-8")
    manifest = {
        "status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A",
        "selection_partition": "2021A",
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "source": {"sha256": _hash(selected)},
        "artifacts": {artifact.name: {"bytes": artifact.stat().st_size, "sha256": _hash(artifact)}},
    }
    (output / "transformer_operating_point_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_operating_point(output, selected) == manifest
    manifest["year_2022_accessed"] = True
    (output / "transformer_operating_point_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid or unsealed"):
        validate_operating_point(output, selected)


def test_freeze_operating_point_keeps_controller_log_outside_frozen_directory(tmp_path, monkeypatch):
    selected_output = tmp_path / "selected"
    selected_output.mkdir()
    prediction = selected_output / "predictions_2021A.parquet"
    prediction.write_bytes(b"selected")
    (selected_output / "finetune_manifest.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "operating_point_2021A"

    def fake_run(command, stdout, stderr, check):
        assert not output.exists()
        assert stdout.name == str(tmp_path / "operating_point_2021A.controller.log")
        output.mkdir()
        artifact = output / "transformer_operating_thresholds_2021A.json"
        artifact.write_text("{}", encoding="utf-8")
        manifest = {
            "status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A",
            "selection_partition": "2021A",
            "2021B_accessed": False,
            "year_2022_accessed": False,
            "source": {"sha256": _hash(prediction)},
            "artifacts": {
                artifact.name: {"bytes": artifact.stat().st_size, "sha256": _hash(artifact)}
            },
        }
        (output / "transformer_operating_point_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("run_finetuning_grid.subprocess.run", fake_run)
    manifest = freeze_operating_point(tmp_path / "freeze.py", selected_output, output)
    assert manifest["status"] == "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A"
    assert not (output / "controller.log").exists()
