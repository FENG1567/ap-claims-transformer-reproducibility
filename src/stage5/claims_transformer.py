#!/usr/bin/env python3
"""Knowledge-augmented claims Transformer architecture.

The auxiliary admission-time heads are structurally isolated from current-stay
diagnosis/procedure tokens, preventing the absence of code timestamps from
creating outcome leakage for cost, prolonged LOS, and in-hospital mortality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ClaimsTransformerConfig:
    unified_vocab_size: int
    category_vocab_size: int
    domain_vocab_size: int
    static_dim: int
    admission_time_static_dim: int
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.10
    max_encounters: int = 9
    year_vocab_size: int = 4
    timing_vocab_size: int = 7
    token_type_vocab_size: int = 3
    leaf_classes: int = 5
    pad_token_id: int = 0
    use_hierarchy: bool = True
    use_prday: bool = True
    use_prior_encounters: bool = True
    use_static_context: bool = True
    use_year_version: bool = True
    # During large-scale pretraining, auxiliary outcome reconstruction can use
    # the same encoder under a stricter, admission-time token mask.  Downstream
    # models keep the legacy static-only auxiliary branch unless enabled.
    auxiliary_uses_masked_joint: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClaimsTransformer(nn.Module):
    def __init__(
        self,
        config: ClaimsTransformerConfig,
        token_to_category: torch.Tensor,
        token_to_domain: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        if token_to_category.shape != (config.unified_vocab_size,):
            raise ValueError("token_to_category shape mismatch")
        if token_to_domain.shape != (config.unified_vocab_size,):
            raise ValueError("token_to_domain shape mismatch")
        self.register_buffer("token_to_category", token_to_category.long(), persistent=True)
        self.register_buffer("token_to_domain", token_to_domain.long(), persistent=True)

        d = config.d_model
        self.code_embedding = nn.Embedding(config.unified_vocab_size, d, padding_idx=config.pad_token_id)
        self.category_embedding = nn.Embedding(config.category_vocab_size, d, padding_idx=0)
        self.domain_embedding = nn.Embedding(config.domain_vocab_size, d, padding_idx=0)
        self.type_embedding = nn.Embedding(config.token_type_vocab_size, d)
        self.timing_embedding = nn.Embedding(config.timing_vocab_size, d, padding_idx=0)
        self.encounter_embedding = nn.Embedding(config.max_encounters, d)
        self.year_embedding = nn.Embedding(config.year_vocab_size, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.input_norm = nn.LayerNorm(d)
        self.input_dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.nhead, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers, norm=nn.LayerNorm(d))
        static_hidden = max(32, d // 2)
        self.static_encoder = nn.Sequential(
            nn.Linear(max(1, config.static_dim), static_hidden), nn.GELU(), nn.Dropout(config.dropout),
            nn.LayerNorm(static_hidden),
        )
        joint_dim = d + static_hidden
        self.leaf_head = nn.Linear(joint_dim, config.leaf_classes)
        self.mlm_code_head = nn.Linear(d, config.unified_vocab_size)
        self.mlm_category_head = nn.Linear(d, config.category_vocab_size)

        # This branch never consumes claim-token representations.
        aux_hidden = max(32, d // 2)
        self.admission_time_encoder = nn.Sequential(
            nn.Linear(max(1, config.admission_time_static_dim), aux_hidden),
            nn.GELU(), nn.Dropout(config.dropout), nn.LayerNorm(aux_hidden),
        )
        if config.auxiliary_uses_masked_joint and config.static_dim != config.admission_time_static_dim:
            raise ValueError("masked-joint auxiliary mode requires matching static dimensions")
        aux_input_dim = joint_dim if config.auxiliary_uses_masked_joint else aux_hidden
        self.aux_high_cost_head = nn.Linear(aux_input_dim, 1)
        self.aux_prolonged_los_head = nn.Linear(aux_input_dim, 1)
        self.aux_death_head = nn.Linear(aux_input_dim, 1)
        nn.init.normal_(self.cls_token, std=0.02)

    def _effective_mask(self, token_mask: torch.Tensor, encounter_index: torch.Tensor) -> torch.Tensor:
        mask = token_mask.bool()
        if not self.config.use_prior_encounters:
            mask = mask & encounter_index.eq(0)
        return mask

    def encode(
        self,
        token_ids: torch.Tensor,
        token_type: torch.Tensor,
        timing_bucket: torch.Tensor,
        encounter_index: torch.Tensor,
        year_index: torch.Tensor,
        token_mask: torch.Tensor,
        static_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        if encounter_index.max().item() >= cfg.max_encounters:
            raise ValueError("encounter_index exceeds max_encounters")
        effective_mask = self._effective_mask(token_mask, encounter_index)
        x = self.code_embedding(token_ids) + self.type_embedding(token_type)
        if cfg.use_hierarchy:
            x = x + self.category_embedding(self.token_to_category[token_ids])
            x = x + self.domain_embedding(self.token_to_domain[token_ids])
        if cfg.use_prday:
            x = x + self.timing_embedding(timing_bucket)
        x = x + self.encounter_embedding(encounter_index)
        if cfg.use_year_version:
            x = x + self.year_embedding(year_index)
        x = self.input_dropout(self.input_norm(x))

        batch = token_ids.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(batch, 1, dtype=torch.bool, device=token_ids.device)
        full_mask = torch.cat([cls_mask, effective_mask], dim=1)
        encoded = self.encoder(x, src_key_padding_mask=~full_mask)
        cls_encoded = encoded[:, 0]
        token_encoded = encoded[:, 1:]
        if cfg.static_dim == 0 or not cfg.use_static_context:
            static_input = torch.zeros(batch, 1, dtype=cls_encoded.dtype, device=cls_encoded.device)
        else:
            static_input = static_features
        static_encoded = self.static_encoder(static_input)
        return cls_encoded, token_encoded, torch.cat([cls_encoded, static_encoded], dim=-1)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_type: torch.Tensor,
        timing_bucket: torch.Tensor,
        encounter_index: torch.Tensor,
        year_index: torch.Tensor,
        token_mask: torch.Tensor,
        static_features: torch.Tensor,
        admission_time_static: torch.Tensor,
        mlm_mask: torch.Tensor | None = None,
        auxiliary_token_mask: torch.Tensor | None = None,
        return_mlm: bool = True,
    ) -> dict[str, torch.Tensor]:
        _, token_encoded, joint = self.encode(
            token_ids, token_type, timing_bucket, encounter_index, year_index,
            token_mask, static_features,
        )
        if self.config.auxiliary_uses_masked_joint:
            if auxiliary_token_mask is None:
                raise ValueError("auxiliary_token_mask is required in masked-joint auxiliary mode")
            _, _, aux = self.encode(
                token_ids, token_type, timing_bucket, encounter_index, year_index,
                token_mask.bool() & auxiliary_token_mask.bool(), admission_time_static,
            )
        else:
            if self.config.admission_time_static_dim == 0:
                aux_input = torch.zeros(
                    token_ids.shape[0], 1, dtype=joint.dtype, device=joint.device
                )
            else:
                aux_input = admission_time_static
            aux = self.admission_time_encoder(aux_input)
        leaf_logits = self.leaf_head(joint)
        # The root event is exactly the union of all non-zero cause leaves.
        # This log-odds identity makes sigmoid(root) equal to the summed leaf
        # softmax probability and prevents contradictory hierarchy outputs.
        readmission_logit = torch.logsumexp(leaf_logits[:, 1:], dim=1) - leaf_logits[:, 0]
        result = {
            "readmission_logit": readmission_logit,
            "leaf_logits": leaf_logits,
            "high_cost_logit": self.aux_high_cost_head(aux).squeeze(-1),
            "prolonged_los_logit": self.aux_prolonged_los_head(aux).squeeze(-1),
            "death_logit": self.aux_death_head(aux).squeeze(-1),
        }
        if return_mlm:
            mlm_features = token_encoded[mlm_mask.bool()] if mlm_mask is not None else token_encoded
            result["mlm_code_logits"] = self.mlm_code_head(mlm_features)
            result["mlm_category_logits"] = self.mlm_category_head(mlm_features)
        return result


def masked_pretraining_loss(
    outputs: dict[str, torch.Tensor], code_labels: torch.Tensor,
    category_labels: torch.Tensor, category_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    code = F.cross_entropy(outputs["mlm_code_logits"].reshape(-1, outputs["mlm_code_logits"].shape[-1]),
                           code_labels.reshape(-1), ignore_index=-100)
    category = F.cross_entropy(outputs["mlm_category_logits"].reshape(-1, outputs["mlm_category_logits"].shape[-1]),
                               category_labels.reshape(-1), ignore_index=-100)
    return {"loss": code + category_weight * category, "code_loss": code, "category_loss": category}


def multitask_pretraining_loss(
    outputs: dict[str, torch.Tensor],
    code_labels: torch.Tensor,
    category_labels: torch.Tensor,
    auxiliary_labels: dict[str, torch.Tensor],
    auxiliary_masks: dict[str, torch.Tensor],
    positive_weights: dict[str, float] | None = None,
    category_weight: float = 0.25,
    auxiliary_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Joint MLM and masked auxiliary loss for large-scale pretraining.

    Missing outcomes are excluded rather than coerced to the negative class.
    Auxiliary heads use the model's stricter admission-time representation; the
    caller is responsible for supplying its token mask.
    """
    losses = masked_pretraining_loss(outputs, code_labels, category_labels, category_weight)
    positive_weights = positive_weights or {}
    task_map = {
        "high_cost": "high_cost_logit",
        "prolonged_los": "prolonged_los_logit",
        "death": "death_logit",
    }
    auxiliary: list[torch.Tensor] = []
    for task, output_name in task_map.items():
        if task not in auxiliary_labels or task not in auxiliary_masks:
            continue
        valid = auxiliary_masks[task].bool()
        if not valid.any():
            continue
        target = auxiliary_labels[task][valid].float()
        logit = outputs[output_name][valid]
        weight = logit.new_tensor(float(positive_weights.get(task, 1.0)))
        task_loss = F.binary_cross_entropy_with_logits(logit, target, pos_weight=weight)
        losses[f"{task}_loss"] = task_loss
        auxiliary.append(task_loss)
    auxiliary_loss = torch.stack(auxiliary).mean() if auxiliary else losses["loss"].new_zeros(())
    losses["auxiliary_loss"] = auxiliary_loss
    losses["loss"] = losses["loss"] + auxiliary_weight * auxiliary_loss
    return losses


def supervised_loss(
    outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor],
    auxiliary_weight: float = 0.20,
) -> dict[str, torch.Tensor]:
    any_loss = F.binary_cross_entropy_with_logits(outputs["readmission_logit"], labels["any_readmission"].float())
    leaf_loss = F.cross_entropy(outputs["leaf_logits"], labels["leaf"].long())
    aux_losses = []
    for output_name, label_name in [
        ("high_cost_logit", "high_cost"),
        ("prolonged_los_logit", "prolonged_los"),
        ("death_logit", "death"),
    ]:
        if label_name in labels:
            aux_losses.append(F.binary_cross_entropy_with_logits(outputs[output_name], labels[label_name].float()))
    aux = torch.stack(aux_losses).mean() if aux_losses else any_loss.new_zeros(())
    total = any_loss + leaf_loss + auxiliary_weight * aux
    return {"loss": total, "any_loss": any_loss, "leaf_loss": leaf_loss, "auxiliary_loss": aux}
