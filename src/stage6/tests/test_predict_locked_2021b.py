import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_locked_2021b import completed_output, validate_model_lock


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_lock_must_match_selected_directory_and_remain_sealed(tmp_path):
    selected = tmp_path / "lr1e4_d010"
    selected.mkdir()
    lock_path = tmp_path / "model_selection_lock.json"
    lock = {
        "status": "LOCKED_ON_2021A_PRE_2021B_PRE_2022",
        "selection_partition": "2021A",
        "selected_id": selected.name,
        "2021B_accessed": False,
        "year_2022_accessed": False,
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    assert validate_model_lock(lock_path, selected)["selected_id"] == selected.name
    lock["year_2022_accessed"] = True
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not sealed"):
        validate_model_lock(lock_path, selected)


def test_existing_2021b_output_is_idempotent_only_when_hash_bound(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    prediction = output / "predictions_2021B.parquet"
    prediction.write_bytes(b"prediction")
    manifest = {
        "status": "PASS_PREDICTIONS_2021B_ONLY",
        "partition": "2021B",
        "year_2022_accessed": False,
        "model": {"sha256": "model"},
        "operating_point": {"manifest_sha256": "operating"},
        "artifact": {"sha256": _hash(prediction)},
    }
    (output / "prediction_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert completed_output(output, "model", "operating") == manifest
    manifest["partition"] = "2021A"
    (output / "prediction_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fails identity"):
        completed_output(output, "model", "operating")
