import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from freeze_transformer_calibration import (
    apply_calibrator,
    capacity_constrained_flags,
    choose_calibrator,
    endpoint_outcome,
    patient_calibration_split,
    sha256,
    validate_selected_model,
)


def test_patient_split_is_deterministic_and_non_degenerate():
    values = np.arange(32, dtype=np.uint64)
    first = patient_calibration_split(values)
    second = patient_calibration_split(values.copy())
    assert np.array_equal(first, second)
    assert first.any() and (~first).any()


def test_capacity_is_exact_and_ties_use_hash():
    probability = np.array([0.8, 0.8, 0.8, 0.1, 0.1])
    hashes = np.array([30, 10, 20, 40, 50], dtype=np.uint64)
    flags, report = capacity_constrained_flags(probability, hashes, 0.40)
    assert flags.tolist() == [False, True, True, False, False]
    assert report["capacity_rows_requested"] == 2
    assert report["boundary_tie_rows"] == 3
    assert report["boundary_tie_rows_selected"] == 2


def test_calibrator_selection_and_application_are_finite():
    rng = np.random.default_rng(7)
    probability = np.linspace(0.01, 0.99, 200)
    outcome = rng.binomial(1, probability).astype(np.int8)
    fit = np.arange(200) % 2 == 0
    method, model, report = choose_calibrator(probability, outcome, fit)
    result = apply_calibrator(method, model, probability)
    assert method in {"none", "platt", "isotonic"}
    assert report["fit_rows"] == 100 and report["selection_rows"] == 100
    assert np.isfinite(result).all() and ((0 <= result) & (result <= 1)).all()


def test_ap_endpoint_uses_leaf_one_only():
    frame = pd.DataFrame({"readmission_leaf": [0, 1, 2, 3, 4]})
    assert endpoint_outcome(frame, "ap_specific_readmission", "readmission_leaf").tolist() == [0, 1, 0, 0, 0]


def test_selected_model_manifest_is_hash_bound(tmp_path: Path):
    prediction = tmp_path / "predictions_2021A.parquet"
    pq.write_table(pa.table({"x": [1]}), prediction)
    manifest = tmp_path / "finetune_manifest.json"
    value = {
        "status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022",
        "prediction_partition": "2021A",
        "2021B_accessed": False,
        "year_2022_accessed": False,
        "prediction": {"bytes": prediction.stat().st_size, "sha256": sha256(prediction)},
    }
    manifest.write_text(json.dumps(value), encoding="utf-8")
    assert validate_selected_model(manifest, prediction)["prediction_partition"] == "2021A"
    value["2021B_accessed"] = True
    manifest.write_text(json.dumps(value), encoding="utf-8")
    try:
        validate_selected_model(manifest, prediction)
    except RuntimeError as exc:
        assert "not frozen" in str(exc)
    else:
        raise AssertionError("Unsealed manifest was accepted")
