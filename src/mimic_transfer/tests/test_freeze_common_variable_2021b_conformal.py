import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import freeze_common_variable_2021b_conformal as controller  # noqa: E402
from freeze_common_variable_2021b_conformal import (  # noqa: E402
    FINAL_STATUS,
    TRANSFER_BINDING_STATUS,
    preflight,
    run_controller,
    validate_transfer_binding,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _hash(path)}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _make_all_operating_registry(tmp_path: Path) -> tuple[Path, dict]:
    model_dir = tmp_path / "models" / "common_variable_only"
    model_dir.mkdir(parents=True)
    checkpoint = model_dir / "best_checkpoint.pt"
    prediction = model_dir / "predictions_2021A.parquet"
    checkpoint.write_bytes(b"frozen common checkpoint")
    prediction.write_bytes(b"frozen 2021A predictions")
    finetune_path = model_dir / "finetune_manifest.json"
    finetune = {
        "status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022",
        "prediction_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False,
        "ablations": {"common_variable_only": True},
        "checkpoint": {"sha256": _hash(checkpoint)}, "prediction": {"sha256": _hash(prediction)},
    }
    _write_json(finetune_path, finetune)
    operating_dir = tmp_path / "operating" / "common_variable_only"
    operating_dir.mkdir(parents=True)
    calibrator = operating_dir / "transformer_binary_calibrators.joblib"
    threshold = operating_dir / "transformer_operating_thresholds_2021A.json"
    calibrator.write_bytes(b"calibrators")
    threshold.write_bytes(b'{"any_readmission":{"threshold":0.2}}')
    operating_path = operating_dir / "transformer_operating_point_manifest.json"
    operating = {
        "status": "PASS_TRANSFORMER_CALIBRATION_AND_THRESHOLDS_LOCKED_ON_2021A",
        "selection_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False,
        "source": {"sha256": _hash(prediction), "manifest_sha256": _hash(finetune_path)},
        "artifacts": {
            calibrator.name: _identity(calibrator), threshold.name: _identity(threshold),
        },
    }
    _write_json(operating_path, operating)
    registry = {
        "status": "PASS_ALL_TRANSFORMER_CALIBRATIONS_AND_THRESHOLDS_LOCKED_ON_2021A",
        "selection_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False,
        "exact_roster": ["common_variable_only"],
        "runs": {"common_variable_only": {
            "source": {"active_ablations": ["common_variable_only"], "checkpoint": _identity(checkpoint),
                       "finetune_manifest": _identity(finetune_path), "prediction": _identity(prediction)},
            "operating_point_directory": str(operating_dir.resolve()),
            "operating_point_manifest": _identity(operating_path),
        }},
    }
    registry_path = tmp_path / "all_operating_points.json"
    _write_json(registry_path, registry)
    return registry_path, {"checkpoint": checkpoint, "operating": operating_path}


def _fake_stage6(command, **_kwargs):
    def value(flag: str) -> Path:
        return Path(command[command.index(flag) + 1])

    assert int(command[command.index("--threads") + 1]) <= 8
    assert int(command[command.index("--workers") + 1]) <= 8
    lock = json.loads(value("--model-selection-lock").read_text(encoding="utf-8"))
    assert lock["selected_id"] == "common_variable_only"
    output = value("--output-root")
    prediction_dir, conformal_dir = output / "predictions_2021B", output / "conformal_2021B"
    prediction_dir.mkdir(parents=True); conformal_dir.mkdir(parents=True)
    prediction = prediction_dir / "predictions_2021B.parquet"
    prediction.write_bytes(b"2021B predictions")
    operating = value("--operating-point-dir") / "transformer_operating_point_manifest.json"
    prediction_manifest = {
        "status": "PASS_PREDICTIONS_2021B_ONLY", "partition": "2021B",
        "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False,
        "model": {"sha256": lock["selected_model"]["checkpoint"]["sha256"]},
        "operating_point": {"manifest_sha256": _hash(operating)},
        "artifact": {"sha256": _hash(prediction)},
    }
    _write_json(prediction_dir / "prediction_manifest.json", prediction_manifest)
    sets = conformal_dir / "conformal_sets_2021B.parquet"
    sets.write_bytes(b"2021B conformal sets")
    global_outputs = {coverage: {"leaf_set_mask": f"leaf_set_mask_{coverage}"}
                      for coverage in ("80", "90", "95")}
    all_dimensions = {name: {"levels": {coverage: {"leaf_set_mask": f"{name}_{coverage}"}
                                              for coverage in ("80", "90", "95")}}
                      for name in ("sex", "age_group", "payer", "zip_income_quartile")}
    calibration_global = {
        "80": {"alpha": 0.20, "n": 25, "q": 0.31},
        "90": {"alpha": 0.10, "n": 25, "q": 0.43},
        "95": {"alpha": 0.05, "n": 25, "q": 0.57},
    }

    def groups(name: str, base: float):
        return {
            "80": {"0": {"n": 12, "q": base, "status": "CALIBRATED", "events_any_readmission": 3},
                   "1": {"n": 13, "q": base + 0.01, "status": "GLOBAL_FALLBACK_SMALL_CALIBRATION_GROUP", "events_any_readmission": 4}},
            "90": {"0": {"n": 12, "q": base + 0.10, "status": "CALIBRATED", "events_any_readmission": 3},
                   "1": {"n": 13, "q": base + 0.11, "status": "GLOBAL_FALLBACK_SMALL_CALIBRATION_GROUP", "events_any_readmission": 4}},
            "95": {"0": {"n": 12, "q": base + 0.20, "status": "CALIBRATED", "events_any_readmission": 3},
                   "1": {"n": 13, "q": base + 0.21, "status": "GLOBAL_FALLBACK_SMALL_CALIBRATION_GROUP", "events_any_readmission": 4}},
        }

    calibration_mondrian = {"sex": groups("sex", 0.20), "age_group": groups("age_group", 0.25),
                            "payer": groups("payer", 0.30), "zip_income_quartile": groups("zip", 0.35)}
    conformal_manifest = {
        "status": "PASS_LOCKED_PRE_2022", "calibration_partition": "2021B only",
        "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False,
        "source": {"sha256": _hash(prediction)}, "artifact": {"sha256": _hash(sets)},
        "operating_point_source": {"manifest_sha256": _hash(operating)},
        "nominal_coverages": [0.90, 0.80, 0.95],
        "calibration": {"global": calibration_global, "mondrian": calibration_mondrian},
        "risk_set_outputs": {"global": global_outputs, "mondrian": all_dimensions},
    }
    _write_json(conformal_dir / "conformal_calibrator.json", conformal_manifest)
    registry = {
        "status": "PASS_2021B_CONFORMAL_LOCKED_PRE_2022", "partition": "2021B only",
        "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False,
        "prediction": {"manifest_sha256": _hash(prediction_dir / "prediction_manifest.json"), "artifact_sha256": _hash(prediction)},
        "conformal": {"manifest_sha256": _hash(conformal_dir / "conformal_calibrator.json"), "artifact_sha256": _hash(sets)},
    }
    _write_json(output / "stage6_registry.json", registry)
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_preflight_is_no_data_and_binds_the_prespecified_common_model(tmp_path: Path, monkeypatch):
    registry, artifacts = _make_all_operating_registry(tmp_path)
    opened = []
    original_sha256 = controller.sha256

    def spy_sha256(path: Path) -> str:
        opened.append(Path(path).resolve())
        return original_sha256(path)

    monkeypatch.setattr(controller, "sha256", spy_sha256)
    result = preflight(registry)
    assert result["status"] == "PASS_PREFLIGHT_PRE_2021B_PRE_2022_NO_DATA_ACCESS"
    assert result["mimic_archive_opened"] is False
    assert result["nrd_patient_rows_opened"] is False
    assert result["model_lock"]["selected_id"] == "common_variable_only"
    assert result["model_lock"]["2021B_accessed"] is False
    assert result["model_lock"]["year_2022_accessed"] is False
    assert artifacts["checkpoint"].resolve() in opened
    assert not any(path.name == "predictions_2021A.parquet" for path in opened)


def test_controller_freezes_lock_validates_stage6_and_exports_only_allowed_dimensions(tmp_path: Path, monkeypatch):
    registry, _ = _make_all_operating_registry(tmp_path)
    monkeypatch.setattr("freeze_common_variable_2021b_conformal.subprocess.run", _fake_stage6)
    stage6 = tmp_path / "existing_stage6.py"; stage6.write_text("# synthetic", encoding="utf-8")
    root = tmp_path / "nrd_root"; history = tmp_path / "history"; hierarchy = tmp_path / "hierarchy"
    root.mkdir(); history.mkdir(); hierarchy.mkdir()
    output = tmp_path / "frozen"
    result = run_controller(registry, root, history, hierarchy, output, threads=8, workers=4, stage6_controller=stage6)
    assert result["status"] == FINAL_STATUS
    assert result["2021B_accessed_before_run"] is False
    lock = json.loads((output / "common_variable_model_lock_2021A.json").read_text(encoding="utf-8"))
    assert lock["selected_id"] == "common_variable_only"
    transfer_path = output / "mimic_conformal_binding.json"
    transfer = validate_transfer_binding(transfer_path)
    assert transfer["status"] == TRANSFER_BINDING_STATUS
    assert set(transfer["mondrian"]) == {"sex", "age_group"}
    assert transfer["nominal_coverages"] == [0.8, 0.9, 0.95]
    assert transfer["global"] == {
        "80": {"alpha": 0.2, "n": 25, "q": 0.31},
        "90": {"alpha": 0.1, "n": 25, "q": 0.43},
        "95": {"alpha": 0.05, "n": 25, "q": 0.57},
    }
    assert transfer["mondrian"]["sex"]["90"]["1"] == {
        "n": 13, "q": 0.31, "status": "GLOBAL_FALLBACK_SMALL_CALIBRATION_GROUP", "events_any_readmission": 4,
    }
    assert transfer["mondrian"]["age_group"]["95"]["0"]["q"] == 0.45
    assert "payer" not in transfer_path.read_text(encoding="utf-8").lower()
    assert "zip_income" not in transfer_path.read_text(encoding="utf-8").lower()


def test_controller_fails_closed_before_stage6_for_tampered_common_configuration(tmp_path: Path, monkeypatch):
    registry, _ = _make_all_operating_registry(tmp_path)
    raw = json.loads(registry.read_text(encoding="utf-8"))
    raw["runs"]["common_variable_only"]["source"]["active_ablations"] = ["no_year"]
    _write_json(registry, raw)
    called = False

    def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("freeze_common_variable_2021b_conformal.subprocess.run", should_not_run)
    stage6 = tmp_path / "existing_stage6.py"; stage6.write_text("# synthetic", encoding="utf-8")
    with pytest.raises(RuntimeError, match="configuration"):
        run_controller(registry, tmp_path, tmp_path, tmp_path, tmp_path / "out", stage6_controller=stage6)
    assert called is False
    assert not (tmp_path / "out").exists()


def test_existing_output_and_prohibited_transfer_dimension_are_rejected(tmp_path: Path):
    registry, _ = _make_all_operating_registry(tmp_path)
    output = tmp_path / "out"; output.mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        run_controller(registry, tmp_path, tmp_path, tmp_path, output)
    prohibited = tmp_path / "bad_binding.json"
    _write_json(prohibited, {
            "status": TRANSFER_BINDING_STATUS, "calibration_partition": "2021B only",
            "model_or_threshold_selection_on_2021B": False, "year_2022_accessed": False,
            "nominal_coverages": [0.8, 0.9, 0.95],
            "allowed_mondrian_dimensions": ["sex", "age_group"], "mondrian": {"sex": {}, "age_group": {}},
        "global": {}, "payer": "forbidden",
    })
    with pytest.raises(RuntimeError, match="prohibited"):
        validate_transfer_binding(prohibited)


def test_transfer_binding_rejects_tampered_q_and_incomplete_coverage_roster(tmp_path: Path, monkeypatch):
    registry, _ = _make_all_operating_registry(tmp_path)
    monkeypatch.setattr("freeze_common_variable_2021b_conformal.subprocess.run", _fake_stage6)
    stage6 = tmp_path / "existing_stage6.py"; stage6.write_text("# synthetic", encoding="utf-8")
    result_root = tmp_path / "frozen"
    run_controller(registry, tmp_path, tmp_path, tmp_path, result_root, stage6_controller=stage6)
    binding_path = result_root / "mimic_conformal_binding.json"
    tampered = json.loads(binding_path.read_text(encoding="utf-8"))
    tampered["global"]["80"]["q"] = 1.01
    _write_json(binding_path, tampered)
    with pytest.raises(RuntimeError, match="invalid global q"):
        validate_transfer_binding(binding_path)
    tampered["global"]["80"]["q"] = 0.31
    tampered["mondrian"]["sex"].pop("95")
    _write_json(binding_path, tampered)
    with pytest.raises(RuntimeError, match="q roster is incomplete"):
        validate_transfer_binding(binding_path)
