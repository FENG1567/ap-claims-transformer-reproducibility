import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_ap_only_pretraining import (
    EXPECTED_AP_ROWS, PRETRAIN_COLUMNS, exposure_plan, finalize_manifest,
    prepare_ap_root,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exposure_plan_matches_joint_budget_within_one_percent():
    plan = exposure_plan(sum(EXPECTED_AP_ROWS.values()), 52_512_061)
    assert plan["target_joint_updates"] == 51_282
    assert plan["ap_updates_per_epoch"] == 396
    assert plan["ap_epochs"] == 129
    assert plan["ap_planned_updates"] == 51_084
    assert plan["relative_update_difference"] < 0.01


def test_prepare_ap_root_filters_partition_and_eligibility(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow"); pq = pytest.importorskip("pyarrow.parquet")
    source, derived = tmp_path/"source", tmp_path/"derived"
    nrd = source/"data"/"nrd"; nrd.mkdir(parents=True)
    for name in ("diagnosis_vocabulary_state.json", "procedure_vocabulary_state.json"):
        (nrd/name).write_text('{"token_to_id": {}}', encoding="utf-8")
    monkeypatch.setattr("run_ap_only_pretraining.EXPECTED_AP_ROWS", {2018: 1, 2019: 1, 2020: 1})
    for year in (2018, 2019, 2020):
        folder=nrd/f"year={year}"; folder.mkdir()
        rows=[]
        for partition, eligible in (("development", True), ("development", False), ("2021A", True)):
            row={name: 0 for name in PRETRAIN_COLUMNS}; row.update({"dx_tokens": [4], "pr_tokens": [5],
                "prday": [0], "analysis_partition": partition, "primary_analysis_eligible": eligible})
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), folder/"ap_episodes.parquet")
    manifest=prepare_ap_root(source, derived)
    assert manifest["total_rows"] == 3 and manifest["year_2021_accessed"] is False
    for year in (2018, 2019, 2020):
        table=pq.read_table(derived/"data"/"nrd"/f"year={year}"/"admissions_model.parquet")
        assert table.num_rows == 1 and "analysis_partition" not in table.column_names


def test_finalize_manifest_adds_population_only_after_full_contract_pass(tmp_path):
    output=tmp_path/"out"; output.mkdir(); checkpoint=output/"pretrained_encoder_final.pt"; checkpoint.write_bytes(b"x")
    plan={"ap_planned_updates": 10}; data={"total_rows": 3, "derived_root": str(tmp_path/"root")}
    manifest={"status":"PASS","year_2021_accessed":False,"year_2022_accessed":False,
        "objectives":["masked_code","masked_hierarchy_category"],"optimizer_updates_completed":10,
        "all_admission_rows_available":3,"final_checkpoint":{"sha256":_sha(checkpoint),"bytes":1}}
    (output/"pretraining_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    final=finalize_manifest(output,data,plan)
    assert final["pretraining_population"] == "AP_development_2018_2020"
    manifest["optimizer_updates_completed"] = 9
    (output/"pretraining_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked exposure"):
        finalize_manifest(output,data,plan)
