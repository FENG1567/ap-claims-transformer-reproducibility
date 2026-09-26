import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_2021b_conformal import validate_conformal, validate_prediction


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prediction_validator_rejects_2021b_selection_or_hash_drift(tmp_path):
    output = tmp_path / "prediction"
    output.mkdir()
    checkpoint = tmp_path / "best.pt"
    operating = tmp_path / "operating.json"
    prediction = output / "predictions_2021B.parquet"
    checkpoint.write_bytes(b"model")
    operating.write_bytes(b"operating")
    prediction.write_bytes(b"prediction")
    manifest = {
        "status": "PASS_PREDICTIONS_2021B_ONLY",
        "partition": "2021B",
        "model_or_threshold_selection_on_2021B": False,
        "year_2022_accessed": False,
        "model": {"sha256": _hash(checkpoint)},
        "operating_point": {"manifest_sha256": _hash(operating)},
        "artifact": {"sha256": _hash(prediction)},
    }
    (output / "prediction_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_prediction(output, checkpoint, operating) == manifest
    manifest["model_or_threshold_selection_on_2021B"] = True
    (output / "prediction_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fails seal"):
        validate_prediction(output, checkpoint, operating)


def test_conformal_validator_binds_prediction_set_and_operating_point(tmp_path):
    output = tmp_path / "conformal"
    output.mkdir()
    prediction = tmp_path / "prediction.parquet"
    operating = tmp_path / "operating.json"
    sets = output / "conformal_sets_2021B.parquet"
    prediction.write_bytes(b"prediction")
    operating.write_bytes(b"operating")
    sets.write_bytes(b"sets")
    manifest = {
        "status": "PASS_LOCKED_PRE_2022",
        "calibration_partition": "2021B only",
        "model_or_threshold_selection_on_2021B": False,
        "year_2022_accessed": False,
        "source": {"sha256": _hash(prediction)},
        "artifact": {"sha256": _hash(sets)},
        "operating_point_source": {"manifest_sha256": _hash(operating)},
    }
    (output / "conformal_calibrator.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_conformal(output, prediction, operating) == manifest
    prediction.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="fails seal"):
        validate_conformal(output, prediction, operating)
