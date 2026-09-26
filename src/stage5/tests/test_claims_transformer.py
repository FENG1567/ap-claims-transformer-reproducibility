import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from claims_transformer import (
    ClaimsTransformer, ClaimsTransformerConfig, multitask_pretraining_loss,
)


def make_model(use_prior=True):
    cfg = ClaimsTransformerConfig(
        unified_vocab_size=20, category_vocab_size=8, domain_vocab_size=5,
        static_dim=4, admission_time_static_dim=3, d_model=16, nhead=4,
        num_layers=1, dim_feedforward=32, dropout=0.0, max_encounters=3,
        use_prior_encounters=use_prior,
    )
    return ClaimsTransformer(cfg, torch.arange(20) % 8, torch.arange(20) % 5).eval()


def inputs():
    return dict(
        token_ids=torch.tensor([[4, 5, 6, 0], [7, 8, 9, 10]]),
        token_type=torch.tensor([[0, 0, 1, 0], [0, 1, 1, 0]]),
        timing_bucket=torch.tensor([[0, 0, 2, 0], [0, 3, 4, 0]]),
        encounter_index=torch.tensor([[0, 1, 1, 0], [0, 0, 2, 2]]),
        year_index=torch.tensor([[0, 0, 0, 0], [2, 2, 2, 2]]),
        token_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool),
        static_features=torch.ones(2, 4),
        admission_time_static=torch.ones(2, 3),
    )


def test_forward_shapes():
    out = make_model()(**inputs())
    assert out["readmission_logit"].shape == (2,)
    assert out["leaf_logits"].shape == (2, 5)
    assert out["mlm_code_logits"].shape == (2, 4, 20)
    leaves = torch.softmax(out["leaf_logits"], dim=1)
    assert torch.allclose(torch.sigmoid(out["readmission_logit"]), leaves[:, 1:].sum(dim=1), atol=1e-6)


def test_auxiliary_heads_are_claim_token_invariant():
    model = make_model()
    a = inputs()
    b = {k: v.clone() for k, v in a.items()}
    b["token_ids"] = torch.tensor([[19, 18, 17, 0], [16, 15, 14, 13]])
    with torch.no_grad():
        out_a, out_b = model(**a), model(**b)
    for name in ("high_cost_logit", "prolonged_los_logit", "death_logit"):
        assert torch.equal(out_a[name], out_b[name])


def test_no_prior_ablation_masks_prior_tokens():
    model = make_model(use_prior=False)
    a = inputs()
    b = {k: v.clone() for k, v in a.items()}
    b["token_ids"][a["encounter_index"] > 0] = 19
    with torch.no_grad():
        out_a, out_b = model(**a), model(**b)
    assert torch.allclose(out_a["readmission_logit"], out_b["readmission_logit"], atol=1e-6)


def test_masked_only_mlm_avoids_full_sequence_logits():
    model = make_model()
    x = inputs()
    mask = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 0]], dtype=torch.bool)
    out = model(**x, mlm_mask=mask)
    assert out["mlm_code_logits"].shape == (3, 20)
    assert out["mlm_category_logits"].shape == (3, 8)


def test_masked_joint_auxiliary_ignores_forbidden_tokens_and_uses_shared_encoder():
    cfg = ClaimsTransformerConfig(
        unified_vocab_size=20, category_vocab_size=8, domain_vocab_size=5,
        static_dim=3, admission_time_static_dim=3, d_model=16, nhead=4,
        num_layers=1, dim_feedforward=32, dropout=0.0, max_encounters=3,
        auxiliary_uses_masked_joint=True,
    )
    model = ClaimsTransformer(cfg, torch.arange(20) % 8, torch.arange(20) % 5).eval()
    x = inputs()
    x["static_features"] = torch.ones(2, 3)
    auxiliary_mask = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)
    altered_forbidden = {key: value.clone() for key, value in x.items()}
    altered_forbidden["token_ids"][~auxiliary_mask & x["token_mask"]] = 19
    with torch.no_grad():
        original = model(**x, auxiliary_token_mask=auxiliary_mask)
        changed = model(**altered_forbidden, auxiliary_token_mask=auxiliary_mask)
    for name in ("high_cost_logit", "prolonged_los_logit", "death_logit"):
        assert torch.allclose(original[name], changed[name], atol=1e-6)
    model.zero_grad(set_to_none=True)
    model(**x, auxiliary_token_mask=auxiliary_mask)["high_cost_logit"].sum().backward()
    assert model.encoder.layers[0].linear1.weight.grad.norm() > 0


def test_multitask_pretraining_loss_masks_missing_outcomes():
    model = make_model()
    x = inputs()
    mlm_mask = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 0]], dtype=torch.bool)
    outputs = model(**x, mlm_mask=mlm_mask)
    losses = multitask_pretraining_loss(
        outputs,
        code_labels=x["token_ids"][mlm_mask],
        category_labels=model.token_to_category[x["token_ids"][mlm_mask]],
        auxiliary_labels={
            "high_cost": torch.tensor([0.0, 1.0]),
            "prolonged_los": torch.tensor([1.0, 0.0]),
            "death": torch.tensor([1.0, 1.0]),
        },
        auxiliary_masks={
            "high_cost": torch.tensor([1, 1], dtype=torch.bool),
            "prolonged_los": torch.tensor([1, 0], dtype=torch.bool),
            "death": torch.tensor([0, 0], dtype=torch.bool),
        },
    )
    assert torch.isfinite(losses["loss"])
    assert "high_cost_loss" in losses and "prolonged_los_loss" in losses
    assert "death_loss" not in losses
