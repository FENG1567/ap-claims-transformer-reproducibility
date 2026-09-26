import sys
import random
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pretrain_claims_transformer import (
    AdmissionStream,
    MLMCollator,
    admission_static_dimension,
    flush_partial_gradient,
    is_early_procedure,
    planned_optimizer_updates,
    restore_rng_state,
    training_contract,
    timing_bucket,
)
from prepare_pretraining_auxiliary_spec import bounded_pos_weight, weighted_quantile


def auxiliary_spec():
    return {
        "age_scaling": {"mean": 50.0, "scale": 20.0},
        "high_cost": {"threshold_2021_usd": 1000.0},
    }


def collator_row(token_ids, token_type, timing, year_index, auxiliary_mask):
    return {
        "token_ids": token_ids,
        "token_type": token_type,
        "timing_bucket": timing,
        "year_index": year_index,
        "auxiliary_token_mask": auxiliary_mask,
        "admission_time_static": [0.0] * admission_static_dimension(),
        "auxiliary_labels": {"high_cost": 0, "prolonged_los": 0, "death": 0},
        "auxiliary_masks": {"high_cost": True, "prolonged_los": True, "death": True},
    }


def test_timing_bucket_marks_after_discharge_invalid():
    assert timing_bucket(None, 3) == 0
    assert timing_bucket(-99, 3) == 0
    assert timing_bucket(-2, 3) == 1
    assert timing_bucket(0, 3) == 2
    assert timing_bucket(2, 3) == 3
    assert timing_bucket(3, 3) == 4
    assert timing_bucket(4, 3) == 6


def test_collator_masks_only_real_tokens_and_returns_flat_labels():
    mapping = torch.arange(30) % 7
    collate = MLMCollator(30, mapping, seed=1)
    rows = [
        collator_row([4, 5, 3], [0, 1, 0], [0, 2, 0], 0, [False, True, False]),
        collator_row([6], [0], [0], 1, [False]),
    ]
    batch = collate(rows)
    assert batch["code_labels"].ndim == 1
    assert batch["category_labels"].shape == batch["code_labels"].shape
    assert not batch["model"]["mlm_mask"][0, 2]
    assert batch["model"]["admission_time_static"].shape == (2, admission_static_dimension())
    assert batch["model"]["auxiliary_token_mask"].sum() == 1


def test_early_procedure_gate_excludes_missing_and_sentinel_days():
    assert is_early_procedure(0)
    assert is_early_procedure(-2)
    assert not is_early_procedure(1)
    assert not is_early_procedure(None)
    assert not is_early_procedure(-99)
    assert not is_early_procedure(-66)


def test_admission_stream_keeps_procedures_when_prday_is_missing(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "year=2018" / "admissions_model.parquet"
    path.parent.mkdir()
    table = pa.table({
        "dx_tokens": pa.array([[4], [4]], type=pa.list_(pa.int32())),
        "pr_tokens": pa.array([[5, 6], [7]], type=pa.list_(pa.int32())),
        "prday": pa.array([[None, 1], None], type=pa.list_(pa.int16())),
        "LOS": pa.array([3, 3], type=pa.int16()),
        "AGE": [50, 70], "AWEEKEND": [0, 1], "ELECTIVE": [0, 0],
        "FEMALE": [1, 0], "HCUP_ED": [1, 1], "RESIDENT": [1, 1],
        "PAY1": [1, 2], "ZIPINC_QRTL": [2, 3], "PL_NCHS": [1, 2],
        "HOSP_BEDSIZE": [2, 3], "H_CONTRL": [1, 2], "HOSP_URCAT4": [1, 2],
        "HOSP_UR_TEACH": [1, 2], "DMONTH": [1, 2],
        "cost_2021_usd": [500.0, 2000.0], "DIED": [0, 1],
    })
    pq.write_table(table, path)

    rows = list(AdmissionStream([path], dx_size=10, max_tokens=8, epoch=0,
                                auxiliary_spec=auxiliary_spec(), seed=1))
    by_tokens = {tuple(row["token_ids"]): row for row in rows}
    assert by_tokens[(4, 15, 16)]["token_type"] == [0, 1, 1]
    assert by_tokens[(4, 15, 16)]["timing_bucket"] == [0, 0, 3]
    assert by_tokens[(4, 17)]["token_type"] == [0, 1]
    assert by_tokens[(4, 17)]["timing_bucket"] == [0, 0]
    assert by_tokens[(4, 15, 16)]["auxiliary_token_mask"] == [False, False, False]
    assert by_tokens[(4, 15, 16)]["auxiliary_masks"] == {
        "high_cost": True, "prolonged_los": True, "death": True,
    }


def test_planned_updates_include_final_partial_gradient_accumulation():
    assert planned_optimizer_updates(total_rows=5, batch_size=1, grad_accum=4, epochs=1) == 2
    assert planned_optimizer_updates(total_rows=5, batch_size=1, grad_accum=4, epochs=3) == 6


def test_auxiliary_spec_statistics_are_deterministic():
    np = pytest.importorskip("numpy")
    values = np.asarray([1.0, 2.0, 10.0])
    weights = np.asarray([1.0, 1.0, 8.0])
    assert weighted_quantile(values, weights, 0.90) == 10.0
    assert bounded_pos_weight(1, 100) == 20.0


def test_final_partial_gradient_flushes_once():
    updates = []
    flush_partial_gradient(5, 4, lambda: updates.append("step"))
    assert updates == ["step"]
    flush_partial_gradient(4, 4, lambda: updates.append("unexpected"))
    assert updates == ["step"]


def test_resume_with_no_new_batches_cannot_flush_an_empty_update():
    updates = []
    flush_partial_gradient(0, 4, lambda: updates.append("unexpected"))
    assert updates == []


def test_training_contract_captures_all_resume_sensitive_arguments():
    args = Namespace(
        epochs=1, batch_size=128, grad_accum=2, max_tokens=96, workers=8,
        threads=8, learning_rate=2e-4, auxiliary_weight=0.25,
        max_steps=0, checkpoint_every=1000,
    )
    assert training_contract(args) == {
        "epochs": 1, "batch_size": 128, "grad_accum": 2, "max_tokens": 96,
        "workers": 8, "threads": 8, "learning_rate": 2e-4,
        "auxiliary_weight": 0.25, "max_steps": 0, "checkpoint_every": 1000,
    }


def _valid_rng_state(device: str = "cpu"):
    state = torch.get_rng_state().clone().to(device)
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": state, "torch_cuda": [state.clone()],
    }


def test_restore_rng_state_normalizes_device_mapped_byte_tensors_to_cpu():
    # CUDA checkpoints loaded with map_location=cuda carry RNG byte tensors on
    # CUDA.  Exercise that exact form when a CUDA runtime is present; CPU still
    # verifies the same API contract in local test environments.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = _valid_rng_state(device)
    with patch("pretrain_claims_transformer.torch.set_rng_state") as set_cpu, \
         patch("pretrain_claims_transformer.torch.cuda.set_rng_state_all") as set_cuda:
        restore_rng_state(state)
    cpu_argument = set_cpu.call_args.args[0]
    cuda_arguments = set_cuda.call_args.args[0]
    assert cpu_argument.device.type == "cpu" and cpu_argument.dtype == torch.uint8
    assert all(value.device.type == "cpu" and value.dtype == torch.uint8 for value in cuda_arguments)
    assert torch.equal(cpu_argument, torch.get_rng_state())


def test_restore_rng_state_rejects_invalid_or_incomplete_device_state_before_restoration():
    bad_cpu = _valid_rng_state()
    bad_cpu["torch_cpu"] = torch.ones(4, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="CPU RNG state is not a 1D ByteTensor"):
        restore_rng_state(bad_cpu)
    bad_cuda = _valid_rng_state()
    bad_cuda["torch_cuda"] = torch.ones(4, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="CUDA RNG state is not a list"):
        restore_rng_state(bad_cuda)
    with pytest.raises(RuntimeError, match="incomplete"):
        restore_rng_state({})
