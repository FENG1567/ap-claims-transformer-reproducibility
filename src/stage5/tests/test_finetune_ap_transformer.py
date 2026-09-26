import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claims_transformer import ClaimsTransformer, ClaimsTransformerConfig
from finetune_ap_transformer import (
    DEV_YEARS, ExampleDataset, StaticPreprocessor, collate_examples, make_example,
    make_model, predict, read_partition, train_epoch,
)
from torch.utils.data import DataLoader


def base_row(**changes):
    row = {
        "year": 2018, "encounter_hash": 11, "patient_hash": 22, "analysis_partition": "development",
        "any_unplanned_readmission_30d": True, "readmission_leaf": 1, "LOS": 3,
        "dx_tokens": [4], "pr_tokens": [5, 6], "prday": [None],
        "prior_dx_tokens_180d": [7], "prior_pr_tokens_180d": [8], "static_raw": np.zeros(2, dtype=np.float32),
    }
    row.update(changes)
    return row


def small_model(static_dim=2):
    cfg = ClaimsTransformerConfig(
        unified_vocab_size=30, category_vocab_size=8, domain_vocab_size=6, static_dim=static_dim,
        admission_time_static_dim=0, d_model=16, nhead=4, num_layers=1, dim_feedforward=32,
        dropout=0.0, max_encounters=2,
    )
    return ClaimsTransformer(cfg, torch.arange(30) % 8, torch.arange(30) % 6)


def test_token_assembly_preserves_missing_prday_offset_and_prior_bag():
    example = make_example(base_row(), dx_size=10, max_tokens=8)
    assert example["token_ids"] == [4, 15, 16, 7, 18]
    assert example["token_type"] == [0, 1, 1, 0, 1]
    assert example["timing_bucket"] == [0, 0, 0, 0, 0]
    assert example["encounter_index"] == [0, 0, 0, 1, 1]


def test_static_preprocessor_is_fit_only_on_development_values():
    development = pd.DataFrame({"AGE": [10.0, np.nan], "LOS": [2.0, 2.0]})
    validation = pd.DataFrame({"AGE": [1000.0], "LOS": [2.0]})
    prep = StaticPreprocessor.fit(development, columns=("AGE", "LOS"))
    transformed = prep.transform(validation)
    assert prep.median[0] == 10.0
    assert transformed.shape == (1, 4)
    assert transformed[0, 0] > 100


def test_ablation_switches_are_transferred_to_transformer_configuration():
    args = Namespace(d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.0,
                     no_hierarchy=True, no_prday=True, no_prior=True, no_static=True, no_year=True)
    bundle = {"unified_vocab_size": 30, "category_vocab_size": 8, "domain_vocab_size": 6,
              "token_to_category": torch.arange(30) % 8, "token_to_domain": torch.arange(30) % 6}
    config = make_model(bundle, static_dim=8, args=args).config
    assert config.static_dim == 0
    assert not config.use_hierarchy and not config.use_prday and not config.use_prior_encounters
    assert not config.use_static_context and not config.use_year_version
    model = make_model(bundle, static_dim=8, args=args).eval()
    inputs = dict(
        token_ids=torch.tensor([[4, 5]]), token_type=torch.tensor([[0, 1]]),
        timing_bucket=torch.tensor([[0, 0]]), encounter_index=torch.tensor([[0, 1]]),
        year_index=torch.tensor([[0, 0]]), token_mask=torch.tensor([[True, True]]),
        static_features=torch.zeros((1, 8)), admission_time_static=torch.zeros((1, 1)),
    )
    with torch.no_grad():
        output = model(**inputs, return_mlm=False)
    assert output["leaf_logits"].shape == (1, 5)


def test_train_final_gradient_accumulation_and_hierarchical_prediction_consistency():
    examples = []
    for index in range(5):
        row = make_example(base_row(encounter_hash=index, patient_hash=index, readmission_leaf=index % 2,
                                    any_unplanned_readmission_30d=bool(index % 2)), 10, 8)
        row["static"] = row.pop("static_raw")
        examples.append(row)
    loader = DataLoader(ExampleDataset(examples), batch_size=1, shuffle=False, collate_fn=collate_examples)
    model = small_model(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    updates, cursor = train_epoch(model, loader, optimizer, torch.device("cpu"), grad_accum=4, amp=False)
    assert (updates, cursor) == (2, 5)
    frame = predict(model, loader, torch.device("cpu"), amp=False)
    leaves = ["p_leaf_none", "p_leaf_ap", "p_leaf_biliary", "p_leaf_sepsis_or_organ", "p_leaf_other"]
    assert np.allclose(frame[leaves].sum(axis=1), 1.0)
    assert np.allclose(frame["p_any_readmission"], frame[leaves[1:]].sum(axis=1))


def _write_partition(root: Path, history: Path, year: int, partition: str, include_2021b=False):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    episode_dir = root / "data" / "nrd" / f"year={year}"; episode_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset, part in enumerate([partition] + (["2021B"] if include_2021b else [])):
        rows.append({"year": year, "encounter_hash": year * 10 + offset, "patient_hash": year * 100 + offset,
                     "analysis_partition": part, "primary_analysis_eligible": True,
                     "any_unplanned_readmission_30d": False, "readmission_leaf": 0, "dx_tokens": [4],
                     "pr_tokens": [5], "prday": [None], "LOS": 2, "AGE": 50})
    pq.write_table(pa.Table.from_pylist(rows), episode_dir / "ap_episodes.parquet")
    history_rows = []
    for row in rows:
        hist = {"encounter_hash": row["encounter_hash"], "patient_hash": row["patient_hash"],
                "analysis_partition": row["analysis_partition"]}
        for name in (
            "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d", "prior_ed_count_180d",
            "prior_nonelective_count_180d", "prior_ap_count_180d", "prior_biliary_count_180d",
            "prior_sepsis_or_organ_count_180d", "prior_los_sum_180d", "prior_max_severity_180d",
            "prior_max_mortality_risk_180d", "days_since_prior_discharge",
        ):
            hist[name] = 0
        hist.update({"history_30d_fully_observable": True, "history_90d_fully_observable": True,
                     "history_180d_fully_observable": True, "prior_dx_tokens_180d": [], "prior_pr_tokens_180d": []})
        history_rows.append(hist)
    pq.write_table(pa.Table.from_pylist(history_rows), history / f"ap_history_{year}.parquet")


def test_partition_reader_is_strictly_gated_to_development_or_2021a(tmp_path):
    pytest.importorskip("pyarrow")
    root, history = tmp_path / "root", tmp_path / "history"; history.mkdir()
    for year in DEV_YEARS:
        _write_partition(root, history, year, "development")
    _write_partition(root, history, 2021, "2021A", include_2021b=True)
    val = read_partition(root, history, 2021, "2021A")
    assert val["analysis_partition"].tolist() == ["2021A"]
    with pytest.raises(RuntimeError, match="Only 2021A"):
        read_partition(root, history, 2021, "2021B")
    with pytest.raises(RuntimeError, match="sealed"):
        read_partition(root, history, 2022, "development")
