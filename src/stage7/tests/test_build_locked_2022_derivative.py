"""Synthetic regression tests for the lock-first 2022 minimum derivative."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "build_locked_2022_derivative.py"
SPEC = importlib.util.spec_from_file_location("build_locked_2022_derivative", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def frozen_spec(path: Path) -> dict:
    outcomes = {
        "any_readmission": "any_unplanned_readmission_30d",
        "ap_specific_readmission": "ap_specific_readmission_30d",
        "biliary_event": "biliary_readmission_30d",
        "sepsis_or_organ_complication": "sepsis_or_organ_readmission_30d",
        "high_cost": "high_cost_label",
        "prolonged_los": "prolonged_los_label",
        "in_hospital_death": "in_hospital_death_label",
    }
    token = ["dx_tokens", "pr_tokens", "prday"]
    # Mirror the complete frozen Stage5 input interface. Prior-state fields
    # come only from history; all other non-identity fields come from episodes.
    history_static = [column for column in builder.STAGE5_STATIC_INPUTS if column.startswith("prior_") or column.startswith("history_") or column == "days_since_prior_discharge"]
    static = [column for column in builder.STAGE5_STATIC_INPUTS if column not in history_static and column not in builder.IDENTITY_COLUMNS]
    episode = list(dict.fromkeys((
        *builder.IDENTITY_COLUMNS, "year", "analysis_partition", "primary_analysis_eligible", "readmission_leaf",
        *token, *static, *outcomes.values(),
    )))
    history = ["encounter_hash", "patient_hash", "analysis_partition", *history_static]
    value = {
        "status": "FROZEN_2022_DERIVATIVE_SPEC", "sealed_test_year": 2022,
        "test_partition": "test", "data_dependent_adaptation": False,
        "year_column": "year", "partition_column": "analysis_partition",
        "eligibility_column": "primary_analysis_eligible", "eligibility_value": True,
        "leaf_column": "readmission_leaf", "episode_columns": episode,
        "history_columns": history, "token_sequence_columns": token,
        "static_columns": static, "outcomes": outcomes,
    }
    write_json(path, value)
    return value


def unlock_with_exact_identities(tmp_path: Path, spec_path: Path) -> Path:
    lock_path = tmp_path / "unlock.json"
    lock = {
        "status": "PASS_UNLOCK_AUTHORIZED_FOR_ONE_2022_PRIMARY_EVALUATION",
        "sealed_test_year": 2022, "2022_access_before_lock": False,
        "frozen_code": {str(MODULE_PATH.resolve()): identity(MODULE_PATH), str(spec_path.resolve()): identity(spec_path)},
    }
    write_json(lock_path, lock)
    lock_path.with_suffix(".json.sha256").write_text(f"{digest(lock_path)}  {lock_path.name}\n", encoding="ascii")
    return lock_path


def synthetic_inputs(episode_path: Path, history_path: Path, spec: dict) -> None:
    rows = []
    leaves = [0, 1, 2, 3, 4]
    for index, leaf in enumerate(leaves):
        row = {column: 1 for column in spec["episode_columns"]}
        row.update({
            "encounter_hash": f"e{index}", "patient_hash": f"p{index}", "hospital_hash": f"h{index % 2}",
            "NRD_STRATUM": f"s{index % 2}", "DISCWT": 1.5, "AGE": 50 + index, "FEMALE": index % 2,
            "PAY1": 1, "ZIPINC_QRTL": 2, "year": 2022, "analysis_partition": "test",
            "primary_analysis_eligible": True, "readmission_leaf": leaf,
            "dx_tokens": [4, 5], "pr_tokens": [8], "prday": [0], "LOS": 3, "CCR_NRD": 0.5,
            "any_unplanned_readmission_30d": int(leaf > 0),
            "ap_specific_readmission_30d": int(leaf == 1),
            "biliary_readmission_30d": int(leaf == 2),
            "sepsis_or_organ_readmission_30d": int(leaf == 3),
            "high_cost_label": index % 2, "prolonged_los_label": 0, "in_hospital_death_label": 0,
        })
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), episode_path)
    history = [
        {"encounter_hash": f"e{index}", "patient_hash": f"p{index}", "analysis_partition": "test",
         **{column: index for column in spec["history_columns"][3:]}}
        for index in range(len(rows))
    ]
    pq.write_table(pa.Table.from_pylist(history), history_path)


def test_validate_only_checks_lock_and_self_hash_without_opening_any_data(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"; frozen_spec(spec_path)
    lock_path = unlock_with_exact_identities(tmp_path, spec_path)
    lock = builder.validate_preflight(lock_path, spec_path)
    assert lock["sealed_test_year"] == 2022
    bad = json.loads(lock_path.read_text(encoding="utf-8"))
    bad["frozen_code"][str(MODULE_PATH.resolve())]["sha256"] = "0" * 64
    write_json(lock_path, bad)
    lock_path.with_suffix(".json.sha256").write_text(f"{digest(lock_path)}  {lock_path.name}\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="Derivative script path/bytes/SHA256"):
        builder.validate_preflight(lock_path, spec_path)


def test_preflight_rejects_spec_tampering_before_data_open(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"; frozen_spec(spec_path)
    lock_path = unlock_with_exact_identities(tmp_path, spec_path)
    spec_path.write_text(spec_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="specification is not hash-locked"):
        builder.validate_preflight(lock_path, spec_path)


def test_one_shot_atomic_minimum_derivative_and_manifest_no_small_cells(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"; spec = frozen_spec(spec_path)
    lock_path = unlock_with_exact_identities(tmp_path, spec_path)
    episode_path, history_path = tmp_path / "episodes.parquet", tmp_path / "history.parquet"
    synthetic_inputs(episode_path, history_path, spec)
    output = tmp_path / "locked_output"
    result = builder.build_locked_2022_derivative(episode_path, history_path, lock_path, spec_path, output, threads=1)
    assert result["status"] == "PASS_LOCKED_2022_MINIMUM_DERIVATIVE"
    assert result["2022_accessed"] is True and result["row_count"] == 5
    assert (output / "locked_2022_derivative.parquet").is_file()
    assert (output / "manifest.json.sha256").is_file()
    frame = pd.read_parquet(output / "locked_2022_derivative.parquet")
    assert frame["hospital_hash"].tolist() == ["h0", "h1", "h0", "h1", "h0"]
    assert set(builder.IDENTITY_COLUMNS).issubset(frame.columns)
    assert set(builder.STAGE5_STATIC_INPUTS).issubset(frame.columns)
    assert not (set(builder.STAGE5_STATIC_INPUTS) & set(spec["outcomes"].values()))
    assert set(spec["outcomes"].values()).issubset(frame.columns)
    assert "primary_analysis_eligible" not in frame.columns
    assert "high_cost_label" not in json.dumps({"counts": "no outcome counts"})
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_outcome_counts_or_small_cells_logged"] is True
    assert "outcome_counts" not in manifest
    with pytest.raises(RuntimeError, match="immutable"):
        builder.build_locked_2022_derivative(episode_path, history_path, lock_path, spec_path, output, threads=1)


def test_join_year_partition_and_patient_mismatch_fail_closed(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"; spec = frozen_spec(spec_path)
    lock_path = unlock_with_exact_identities(tmp_path, spec_path)
    episode_path, history_path = tmp_path / "episodes.parquet", tmp_path / "history.parquet"
    synthetic_inputs(episode_path, history_path, spec)
    episodes = pq.read_table(episode_path).to_pandas()
    episodes.loc[0, "year"] = 2021
    pq.write_table(pa.Table.from_pandas(episodes, preserve_index=False), episode_path)
    with pytest.raises(RuntimeError, match="non-2022"):
        builder.build_locked_2022_derivative(episode_path, history_path, lock_path, spec_path, tmp_path / "bad_year", threads=1)
    synthetic_inputs(episode_path, history_path, spec)
    history = pq.read_table(history_path).to_pandas(); history.loc[0, "patient_hash"] = "wrong"
    pq.write_table(pa.Table.from_pandas(history, preserve_index=False), history_path)
    with pytest.raises(RuntimeError, match="patient_hash mismatch"):
        builder.build_locked_2022_derivative(episode_path, history_path, lock_path, spec_path, tmp_path / "bad_patient", threads=1)


def test_duplicate_and_partial_outputs_fail_closed(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"; spec = frozen_spec(spec_path)
    lock_path = unlock_with_exact_identities(tmp_path, spec_path)
    episode_path, history_path = tmp_path / "episodes.parquet", tmp_path / "history.parquet"
    synthetic_inputs(episode_path, history_path, spec)
    history = pq.read_table(history_path).to_pandas()
    history = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    pq.write_table(pa.Table.from_pandas(history, preserve_index=False), history_path)
    with pytest.raises(RuntimeError, match="duplicate encounter_hash"):
        builder.build_locked_2022_derivative(episode_path, history_path, lock_path, spec_path, tmp_path / "duplicate", threads=1)
    synthetic_inputs(episode_path, history_path, spec)
    partial = tmp_path / "partial_output.partial-stale"; partial.mkdir()
    with pytest.raises(RuntimeError, match="partial output"):
        builder.build_locked_2022_derivative(episode_path, history_path, lock_path, spec_path, tmp_path / "partial_output", threads=1)


def test_only_high_cost_outcome_may_be_missing_and_mask_is_preserved(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.json"; spec = frozen_spec(spec_path); lock_path = unlock_with_exact_identities(tmp_path, spec_path)
    episode_path, history_path = tmp_path / "episodes.parquet", tmp_path / "history.parquet"; synthetic_inputs(episode_path, history_path, spec)
    frame = pq.read_table(episode_path).to_pandas(); frame.loc[0, "high_cost_label"] = pd.NA; pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), episode_path)
    result = builder.build_locked_2022_derivative(episode_path, history_path, lock_path, spec_path, tmp_path / "masked", threads=1)
    derived = pd.read_parquet(Path(result["output"]) / "locked_2022_derivative.parquet")
    assert pd.isna(derived.loc[0, "high_cost_label"]) and result["high_cost_missingness_preserved_as_mask"] is True
