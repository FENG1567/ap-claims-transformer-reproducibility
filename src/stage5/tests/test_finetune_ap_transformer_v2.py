import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claims_transformer import ClaimsTransformer, ClaimsTransformerConfig
from finetune_ap_transformer_v2 import (
    COMMON_VARIABLE_CATEGORICAL_COLUMNS, COMMON_VARIABLE_NUMERIC_COLUMNS, DEV_YEARS,
    HOSPITAL_SOCIOECONOMIC_CATEGORICAL_COLUMNS, HOSPITAL_SOCIOECONOMIC_NUMERIC_COLUMNS,
    ExampleDataset, StaticPreprocessor, collate_examples, make_example, load_pretrained,
    average_precision_binary, finetuning_contract, make_model, predict, read_partition,
    make_examples, selection_metrics, static_feature_columns, train_epoch,
    unique_preserving_order,
)
from torch.utils.data import DataLoader


def base_row(**changes):
    row = {
        "year": 2018, "encounter_hash": 11, "patient_hash": 22, "analysis_partition": "development",
        "any_unplanned_readmission_30d": True, "readmission_leaf": 1, "LOS": 3,
        "dx_tokens": [4], "pr_tokens": [5, 6], "prday": [None],
        "prior_dx_tokens_180d": [7], "prior_pr_tokens_180d": [8],
        "static_raw": np.zeros(2, dtype=np.float32),
    }
    row.update(changes)
    return row


def small_model(static_dim=2):
    cfg = ClaimsTransformerConfig(unified_vocab_size=30, category_vocab_size=8, domain_vocab_size=6,
        static_dim=static_dim, admission_time_static_dim=0, d_model=16, nhead=4, num_layers=1,
        dim_feedforward=32, dropout=0.0, max_encounters=2)
    return ClaimsTransformer(cfg, torch.arange(30) % 8, torch.arange(30) % 6)


def test_unique_projection_and_frozen_static_levels_with_oov_and_missing():
    assert unique_preserving_order(["LOS", "AGE", "LOS", "PAY1", "AGE"]) == ["LOS", "AGE", "PAY1"]
    development = pd.DataFrame({"AGE": [10.0, np.nan, 20.0], "PAY1": [1, 2, None], "FEMALE": [0, 1, 0]})
    validation = pd.DataFrame({"AGE": [1000.0, np.nan], "PAY1": [99, None], "FEMALE": [1, 9]})
    prep = StaticPreprocessor.fit(development, numeric_columns=("AGE",), categorical_columns=("PAY1", "FEMALE"))
    assert prep.levels == {"PAY1": ["1", "2"], "FEMALE": ["0", "1"]}
    transformed = prep.transform(validation)
    assert transformed.shape == (2, prep.dimension)
    # AGE was fitted on 10/20 only.  PAY1=99 and FEMALE=9 are frozen-state OOV;
    # validation can never add a level.
    assert transformed[0, 0] > 100
    pay1_start = 2
    assert transformed[0, pay1_start + 3] == 1  # PAY1 OOV after levels + missing
    assert transformed[1, pay1_start + 2] == 1  # PAY1 explicit missing
    assert StaticPreprocessor.from_dict(prep.to_dict()) == prep


def test_token_assembly_preserves_missing_prday_offset_and_prior_bag():
    example = make_example(base_row(), dx_size=10, max_tokens=8)
    assert example["token_ids"] == [4, 15, 16, 7, 18]
    assert example["token_type"] == [0, 1, 1, 0, 1]
    assert example["timing_bucket"] == [0, 0, 0, 0, 0]
    assert example["encounter_index"] == [0, 0, 0, 1, 1]


def test_future_year_oov_inference_is_explicit_fail_closed_and_known_years_are_unchanged():
    with pytest.raises(RuntimeError, match="Unauthorized year 2022"):
        make_example(base_row(year=2022), dx_size=10, max_tokens=8)
    with pytest.raises(RuntimeError, match="Missing or invalid year"):
        make_example(base_row(year=None), dx_size=10, max_tokens=8)
    known = make_example(base_row(year=2021), dx_size=10, max_tokens=8)
    assert known["year_index"] == [3] * len(known["token_ids"])
    future = make_example(base_row(year=2022), dx_size=10, max_tokens=8, allow_future_oov_year=True)
    assert future["year_index"] == [4] * len(future["token_ids"])
    # The batch helper must forward the opt-in rather than silently relaxing its
    # default training path.
    frame = pd.DataFrame([base_row(year=2022)])
    with pytest.raises(RuntimeError, match="Unauthorized year 2022"):
        make_examples(frame, dx_size=10, max_tokens=8, static_values=np.zeros((1, 2), dtype=np.float32))
    examples = make_examples(frame, dx_size=10, max_tokens=8, static_values=np.zeros((1, 2), dtype=np.float32),
                             allow_future_oov_year=True)
    assert examples[0]["year_index"] == [4] * len(examples[0]["token_ids"])


def test_no_static_forward_and_ablation_flags_are_structural():
    args = Namespace(d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.0,
                     no_hierarchy=True, no_prday=True, no_prior=True, no_static=True, no_year=True)
    bundle = {"unified_vocab_size": 30, "category_vocab_size": 8, "domain_vocab_size": 6,
              "token_to_category": torch.arange(30) % 8, "token_to_domain": torch.arange(30) % 6}
    model = make_model(bundle, static_dim=12, args=args).eval()
    assert model.config.static_dim == 0 and not model.config.use_static_context
    inputs = {"token_ids": torch.tensor([[4, 5]]), "token_type": torch.tensor([[0, 1]]),
        "timing_bucket": torch.tensor([[0, 0]]), "encounter_index": torch.tensor([[0, 1]]),
        "year_index": torch.tensor([[0, 0]]), "token_mask": torch.tensor([[True, True]]),
        "static_features": torch.randn((1, 12)), "admission_time_static": torch.zeros((1, 1))}
    with torch.no_grad(): first = model(**inputs, return_mlm=False)["leaf_logits"]
    inputs["static_features"] = torch.randn((1, 12)) * 1000
    with torch.no_grad(): second = model(**inputs, return_mlm=False)["leaf_logits"]
    assert torch.equal(first, second)


def test_hospital_socioeconomic_and_common_variable_projections_are_distinct_and_frozen():
    base = Namespace(no_hospital_socioeconomic=False, common_variable_only=False)
    all_numeric, all_categorical = static_feature_columns(base)
    no_context_numeric, no_context_categorical = static_feature_columns(
        Namespace(no_hospital_socioeconomic=True, common_variable_only=False))
    common_numeric, common_categorical = static_feature_columns(
        Namespace(no_hospital_socioeconomic=False, common_variable_only=True))
    assert set(HOSPITAL_SOCIOECONOMIC_NUMERIC_COLUMNS).isdisjoint(no_context_numeric)
    assert set(HOSPITAL_SOCIOECONOMIC_CATEGORICAL_COLUMNS).isdisjoint(no_context_categorical)
    assert set(common_numeric) == set(COMMON_VARIABLE_NUMERIC_COLUMNS)
    assert set(common_categorical) == set(COMMON_VARIABLE_CATEGORICAL_COLUMNS)
    assert set(common_numeric).issubset(all_numeric)
    assert set(common_categorical).issubset(all_categorical)
    assert (no_context_numeric, no_context_categorical) != (common_numeric, common_categorical)
    with pytest.raises(ValueError, match="already excludes"):
        static_feature_columns(Namespace(no_hospital_socioeconomic=True, common_variable_only=True))


def test_common_variable_model_disables_nrd_year_token_but_keeps_common_static_context():
    args = Namespace(d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.0,
                     no_hierarchy=False, no_prday=False, no_prior=False, no_static=False,
                     no_year=False, common_variable_only=True)
    bundle = {"unified_vocab_size": 30, "category_vocab_size": 8, "domain_vocab_size": 6,
              "token_to_category": torch.arange(30) % 8, "token_to_domain": torch.arange(30) % 6}
    model = make_model(bundle, static_dim=7, args=args)
    assert model.config.static_dim == 7 and model.config.use_static_context
    assert not model.config.use_year_version
    assert model.config.year_vocab_size == 5


def test_training_remainder_prediction_hierarchy_and_checkpoint_resume_state(tmp_path):
    examples = []
    for index in range(5):
        row = make_example(base_row(encounter_hash=index, patient_hash=index, readmission_leaf=index % 2,
            any_unplanned_readmission_30d=bool(index % 2)), 10, 8)
        row["static"] = row.pop("static_raw"); examples.append(row)
    loader = DataLoader(ExampleDataset(examples), batch_size=1, shuffle=False, collate_fn=collate_examples)
    model = small_model(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    updates, cursor = train_epoch(model, loader, optimizer, torch.device("cpu"), grad_accum=4, amp=False)
    assert (updates, cursor) == (2, 5)
    frame = predict(model, loader, torch.device("cpu"), amp=False)
    leaves = ["p_leaf_none", "p_leaf_ap", "p_leaf_biliary", "p_leaf_sepsis_or_organ", "p_leaf_other"]
    assert np.allclose(frame[leaves].sum(axis=1), 1.0)
    assert np.allclose(frame["p_any_readmission"], frame[leaves[1:]].sum(axis=1))
    prep = StaticPreprocessor.fit(pd.DataFrame({"AGE": [1.0, 2.0]}), numeric_columns=("AGE",), categorical_columns=())
    # Serialized checkpoint state used by --resume contains the frozen preprocessor.
    from finetune_ap_transformer_v2 import checkpoint_payload
    payload = checkpoint_payload(model, optimizer, 1, 3, 7, prep, Namespace(seed=1))
    checkpoint = tmp_path / "checkpoint.pt"; torch.save(payload, checkpoint)
    restored = torch.load(checkpoint, weights_only=False)
    assert restored["epoch"] == 1 and restored["batch_cursor"] == 3 and restored["global_step"] == 7
    assert StaticPreprocessor.from_dict(restored["static_preprocessor"]) == prep
    assert restored["training_contract"]["seed"] == 1


def test_average_precision_and_2021a_selection_metric_are_deterministic_and_partition_locked():
    assert average_precision_binary(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.7])) == pytest.approx((1.0+2/3)/2)
    frame = pd.DataFrame({"analysis_partition": ["2021A"]*4, "year": [2021]*4,
        "any_unplanned_readmission_30d": [1, 0, 1, 0], "readmission_leaf": [1, 0, 2, 0],
        "p_any_readmission": [0.9, 0.2, 0.7, 0.1], "p_leaf_ap": [0.8, 0.1, 0.2, 0.05]})
    metrics = selection_metrics(frame)
    assert metrics["auprc_any"] == 1.0 and metrics["auprc_ap"] == 1.0
    assert metrics["selection_score"] == 1.0 and metrics["mean_brier"] >= 0
    bad = frame.copy(); bad["analysis_partition"] = "2021B"
    with pytest.raises(RuntimeError, match="2021A only"): selection_metrics(bad)


def test_finetuning_contract_captures_resume_sensitive_settings():
    first = Namespace(seed=1, learning_rate=1e-4, no_year=False, pretrained_checkpoint=None)
    second = Namespace(seed=1, learning_rate=2e-4, no_year=False, pretrained_checkpoint=None)
    assert finetuning_contract(first)["learning_rate"] == 1e-4
    assert finetuning_contract(first) != finetuning_contract(second)


def _pretrain_and_finetune_models():
    common = dict(unified_vocab_size=30, category_vocab_size=8, domain_vocab_size=6,
        d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.0, max_encounters=2)
    source = ClaimsTransformer(ClaimsTransformerConfig(static_dim=0, admission_time_static_dim=0, **common),
                               torch.arange(30) % 8, torch.arange(30) % 6)
    # Fine-tuning owns different static and admission-time preprocessing.  The
    # shared token/hierarchy/encoder geometry deliberately remains identical.
    target = ClaimsTransformer(ClaimsTransformerConfig(static_dim=4, admission_time_static_dim=3, **common),
                               torch.arange(30) % 8, torch.arange(30) % 6)
    return source, target


@pytest.mark.parametrize("state_key", ["model_state", "model_state_dict", "model"])
def test_pretraining_checkpoint_keys_load_shared_representation_and_skip_only_nonshared_mismatch(tmp_path, state_key):
    source, target = _pretrain_and_finetune_models()
    state = dict(source.state_dict())
    # This malformed auxiliary task head may be discarded, just like a shape
    # change induced by a task-specific checkpoint revision.
    state["aux_high_cost_head.weight"] = torch.zeros((7, 7))
    checkpoint = tmp_path / f"{state_key}.pt"; torch.save({state_key: state}, checkpoint)
    expected_embedding = source.code_embedding.weight.detach().clone()
    load_pretrained(target, checkpoint)
    assert torch.equal(target.code_embedding.weight, expected_embedding)
    assert torch.equal(target.encoder.layers[0].linear1.weight, source.encoder.layers[0].linear1.weight)


def test_pretraining_loader_rejects_nonshared_representation_shape_mismatch(tmp_path):
    source, target = _pretrain_and_finetune_models()
    state = dict(source.state_dict())
    state["code_embedding.weight"] = torch.zeros((3, 3))
    checkpoint = tmp_path / "bad_shared.pt"; torch.save({"model_state": state}, checkpoint)
    with pytest.raises(RuntimeError, match="Incompatible pretrained tensor code_embedding.weight"):
        load_pretrained(target, checkpoint)


def test_production_make_model_preserves_pretrained_encounter_geometry_for_two_bag_examples(tmp_path):
    args = Namespace(d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.0,
                     no_hierarchy=False, no_prday=False, no_prior=False, no_static=False,
                     no_year=False, common_variable_only=False)
    bundle = {"unified_vocab_size": 30, "category_vocab_size": 8, "domain_vocab_size": 6,
              "token_to_category": torch.arange(30) % 8, "token_to_domain": torch.arange(30) % 6}
    source = ClaimsTransformer(ClaimsTransformerConfig(
        static_dim=0, admission_time_static_dim=0, year_vocab_size=4,
        unified_vocab_size=30, category_vocab_size=8, domain_vocab_size=6,
        d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.0, max_encounters=9),
        bundle["token_to_category"], bundle["token_to_domain"])
    checkpoint = tmp_path / "pretrained_max_encounters_9.pt"
    torch.save({"model_state": source.state_dict()}, checkpoint)

    target = make_model(bundle, static_dim=2, args=args).eval()
    assert target.config.max_encounters == 9
    load_pretrained(target, checkpoint)
    assert torch.equal(target.encounter_embedding.weight, source.encounter_embedding.weight)

    example = make_example(base_row(), dx_size=10, max_tokens=8)
    example["static"] = example.pop("static_raw")
    batch = collate_examples([example])
    assert set(example["encounter_index"]) == {0, 1}
    assert int(batch["model"]["encounter_index"].max()) == 1
    with torch.no_grad():
        assert target(**batch["model"], return_mlm=False)["leaf_logits"].shape == (1, 5)


def test_pretraining_loader_expands_future_year_slots_with_development_mean(tmp_path):
    common = dict(unified_vocab_size=30, category_vocab_size=8, domain_vocab_size=6,
                  static_dim=0, admission_time_static_dim=0, d_model=16, nhead=4,
                  num_layers=1, dim_feedforward=32, dropout=0.0, max_encounters=2)
    source = ClaimsTransformer(ClaimsTransformerConfig(year_vocab_size=4, **common),
                               torch.arange(30) % 8, torch.arange(30) % 6)
    target = ClaimsTransformer(ClaimsTransformerConfig(year_vocab_size=5, **common),
                               torch.arange(30) % 8, torch.arange(30) % 6)
    checkpoint = tmp_path / "pretrained.pt"
    torch.save({"model_state": source.state_dict()}, checkpoint)
    expected = source.year_embedding.weight.detach()[:3].mean(dim=0)
    load_pretrained(target, checkpoint)
    assert torch.equal(target.year_embedding.weight[:3], source.year_embedding.weight[:3])
    assert torch.allclose(target.year_embedding.weight[3], expected)
    assert torch.allclose(target.year_embedding.weight[4], expected)


def _write_partition(root: Path, history: Path, year: int, partition: str, include_2021b=False):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    episode_dir = root / "data" / "nrd" / f"year={year}"; episode_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset, part in enumerate([partition] + (["2021B"] if include_2021b else [])):
        rows.append({"year": year, "encounter_hash": year * 10 + offset, "patient_hash": year * 100 + offset,
            "analysis_partition": part, "primary_analysis_eligible": True, "any_unplanned_readmission_30d": False,
            "readmission_leaf": 0, "dx_tokens": [4], "pr_tokens": [5], "prday": [None], "LOS": 2,
            "AGE": 50, "FEMALE": 1, "PAY1": 1, "ZIPINC_QRTL": 2})
    pq.write_table(pa.Table.from_pylist(rows), episode_dir / "ap_episodes.parquet")
    history_rows = []
    for row in rows:
        hist = {"encounter_hash": row["encounter_hash"], "patient_hash": row["patient_hash"], "analysis_partition": row["analysis_partition"]}
        for name in ("prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d", "prior_ed_count_180d",
                     "prior_nonelective_count_180d", "prior_ap_count_180d", "prior_biliary_count_180d",
                     "prior_sepsis_or_organ_count_180d", "prior_los_sum_180d", "prior_max_severity_180d",
                     "prior_max_mortality_risk_180d", "days_since_prior_discharge"):
            hist[name] = 0
        hist.update({"history_30d_fully_observable": True, "history_90d_fully_observable": True,
            "history_180d_fully_observable": True, "prior_dx_tokens_180d": [], "prior_pr_tokens_180d": []})
        history_rows.append(hist)
    pq.write_table(pa.Table.from_pylist(history_rows), history / f"ap_history_{year}.parquet")


def test_partition_reader_de_duplicates_real_schema_request_and_blocks_sealed_data(tmp_path):
    pytest.importorskip("pyarrow")
    root, history = tmp_path / "root", tmp_path / "history"; history.mkdir()
    for year in DEV_YEARS: _write_partition(root, history, year, "development")
    _write_partition(root, history, 2021, "2021A", include_2021b=True)
    validation = read_partition(root, history, 2021, "2021A")
    assert validation.columns.is_unique
    assert validation["analysis_partition"].tolist() == ["2021A"]
    with pytest.raises(RuntimeError, match="Only 2021A"): read_partition(root, history, 2021, "2021B")
    with pytest.raises(RuntimeError, match="sealed"): read_partition(root, history, 2022, "development")
