#!/usr/bin/env python3
"""Leakage-guarded AP claims Transformer fine-tuning (v2).

Only 2018--2020 eligible AP episodes fit representation preprocessing and model
parameters.  This program may materialize 2021A only; it explicitly refuses
2021B and 2022.  v2 replaces the superseded v1 reader and static preprocessing:
column requests are de-duplicated, numeric state is development-frozen median/
scale, and administrative categories use development-frozen one-hot levels with
explicit missing and OOV columns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from claims_transformer import ClaimsTransformer, ClaimsTransformerConfig, supervised_loss

SEED = 20260912
DEV_YEARS = (2018, 2019, 2020)
VALIDATION_YEAR = 2021
SPECIAL_TOKEN_MAX = 3
LEAVES = ("none", "ap", "biliary", "sepsis_or_organ", "other")
YEAR_TO_INDEX = {2018: 0, 2019: 1, 2020: 2, 2021: 3}
FUTURE_OOV_YEAR_INDEX = 4

# Numeric is reserved for quantities with defensible intervals.  Ordered
# clinical/admin codes remain categorical: a neural model must not assume that
# adjacent code values have adjacent clinical meaning.
NUMERIC_STATIC_COLUMNS = (
    "AGE", "LOS", "I10_NDX", "I10_NPR", "CCR_NRD", "WAGEINDEX",
    "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d",
    "prior_ed_count_180d", "prior_nonelective_count_180d", "prior_ap_count_180d",
    "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d", "prior_los_sum_180d",
    "prior_max_severity_180d", "prior_max_mortality_risk_180d", "days_since_prior_discharge",
)
CATEGORICAL_STATIC_COLUMNS = (
    "APRDRG_Severity", "APRDRG_Risk_Mortality", "AWEEKEND", "DMONTH", "ELECTIVE",
    "FEMALE", "HCUP_ED", "PAY1", "PL_NCHS", "RESIDENT", "ZIPINC_QRTL", "HOSP_BEDSIZE",
    "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH", "history_30d_fully_observable",
    "history_90d_fully_observable", "history_180d_fully_observable",
)
STATIC_COLUMNS = NUMERIC_STATIC_COLUMNS + CATEGORICAL_STATIC_COLUMNS
HOSPITAL_SOCIOECONOMIC_NUMERIC_COLUMNS = ("CCR_NRD", "WAGEINDEX")
HOSPITAL_SOCIOECONOMIC_CATEGORICAL_COLUMNS = (
    "PAY1", "PL_NCHS", "RESIDENT", "ZIPINC_QRTL", "HOSP_BEDSIZE",
    "H_CONTRL", "HOSP_URCAT4", "HOSP_UR_TEACH",
)
# The transport model is deliberately narrower than "all variables that could
# be approximated".  These fields have the same patient/episode meaning in NRD
# and MIMIC-IV and do not require an NRD-specific payer, ZIP, hospital, APR-DRG,
# or calendar-year mapping.  Diagnosis/procedure tokens, PRDAY and encounter
# order remain model inputs outside this static projection.
COMMON_VARIABLE_NUMERIC_COLUMNS = (
    "AGE", "LOS", "I10_NDX", "I10_NPR",
    "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d",
    "prior_ed_count_180d", "prior_nonelective_count_180d", "prior_ap_count_180d",
    "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d", "prior_los_sum_180d",
    "days_since_prior_discharge",
)
COMMON_VARIABLE_CATEGORICAL_COLUMNS = (
    "FEMALE", "history_30d_fully_observable", "history_90d_fully_observable",
    "history_180d_fully_observable",
)
HISTORY_COLUMNS = (
    "prior_count_ytd", "prior_count_30d", "prior_count_90d", "prior_count_180d",
    "prior_ed_count_180d", "prior_nonelective_count_180d", "prior_ap_count_180d",
    "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d", "prior_los_sum_180d",
    "prior_max_severity_180d", "prior_max_mortality_risk_180d", "days_since_prior_discharge",
    "history_30d_fully_observable", "history_90d_fully_observable", "history_180d_fully_observable",
    "prior_dx_tokens_180d", "prior_pr_tokens_180d",
)
REQUIRED_EPISODE_COLUMNS = (
    "year", "encounter_hash", "patient_hash", "analysis_partition", "primary_analysis_eligible",
    "any_unplanned_readmission_30d", "readmission_leaf", "dx_tokens", "pr_tokens", "prday", "LOS",
)
OUTPUT_CONTEXT_COLUMNS = ("AGE", "FEMALE", "PAY1", "ZIPINC_QRTL", "DISCWT")


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    """Return a deterministic unique projection suitable for PyArrow columns."""
    return list(dict.fromkeys(values))


def static_feature_columns(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the prespecified static projection for one structural ablation."""
    common = bool(getattr(args, "common_variable_only", False))
    no_hospital_socioeconomic = bool(getattr(args, "no_hospital_socioeconomic", False))
    if common and no_hospital_socioeconomic:
        raise ValueError("--common-variable-only already excludes hospital/socioeconomic variables")
    if common:
        return COMMON_VARIABLE_NUMERIC_COLUMNS, COMMON_VARIABLE_CATEGORICAL_COLUMNS
    if no_hospital_socioeconomic:
        numeric = tuple(c for c in NUMERIC_STATIC_COLUMNS if c not in HOSPITAL_SOCIOECONOMIC_NUMERIC_COLUMNS)
        categorical = tuple(c for c in CATEGORICAL_STATIC_COLUMNS
                            if c not in HOSPITAL_SOCIOECONOMIC_CATEGORICAL_COLUMNS)
        return numeric, categorical
    return NUMERIC_STATIC_COLUMNS, CATEGORICAL_STATIC_COLUMNS


def require_unique_columns(frame: pd.DataFrame, context: str) -> None:
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise RuntimeError(f"{context} has duplicate columns: {duplicates}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def average_precision_binary(y_true: np.ndarray, probability: np.ndarray) -> float:
    """Dependency-free average precision with deterministic stable tie order."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Invalid binary labels or probabilities")
    positives = int(y.sum())
    if positives == 0:
        raise ValueError("Average precision requires at least one positive")
    order = np.argsort(-p, kind="stable")
    ordered = y[order]
    true_positive = np.cumsum(ordered)
    precision = true_positive / np.arange(1, len(y) + 1)
    return float(precision[ordered == 1].sum() / positives)


def selection_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Prespecified unweighted 2021A early-stopping metrics for both co-primary outcomes."""
    if set(predictions["analysis_partition"]) != {"2021A"} or set(predictions["year"].astype(int)) != {2021}:
        raise RuntimeError("Selection metrics may use 2021A only")
    y_any = predictions["any_unplanned_readmission_30d"].astype(int).to_numpy()
    y_ap = (predictions["readmission_leaf"].astype(int).to_numpy() == LEAVES.index("ap")).astype(int)
    p_any = predictions["p_any_readmission"].to_numpy(dtype=np.float64)
    p_ap = predictions["p_leaf_ap"].to_numpy(dtype=np.float64)
    ap_any = average_precision_binary(y_any, p_any)
    ap_ap = average_precision_binary(y_ap, p_ap)
    brier_any = float(np.mean((p_any-y_any) ** 2))
    brier_ap = float(np.mean((p_ap-y_ap) ** 2))
    return {"auprc_any": ap_any, "auprc_ap": ap_ap,
            "selection_score": (ap_any+ap_ap)/2.0,
            "brier_any": brier_any, "brier_ap": brier_ap,
            "mean_brier": (brier_any+brier_ap)/2.0}


def set_seed(seed: int = SEED, threads: int = 8) -> None:
    if not 1 <= threads <= 8:
        raise ValueError("--threads must be between 1 and 8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)


def timing_bucket(day: Any, los: Any) -> int:
    """Preserve the procedure while representing absent/invalid PRDAY as 0."""
    if day is None or pd.isna(day): return 0
    day_i = int(day); los_i = max(0, int(los)) if los is not None and not pd.isna(los) else 0
    if day_i <= -90 or day_i == -66: return 0
    if day_i < 0: return 1
    if day_i > los_i: return 6
    if day_i == 0: return 2
    if day_i <= 2: return 3
    if day_i <= 7: return 4
    return 5


def _token_list(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)): return []
    return [int(x) for x in value if x is not None and not pd.isna(x)]


def _leaf_label(row: dict[str, Any]) -> int:
    leaf = int(row["readmission_leaf"]); parent = int(bool(row["any_unplanned_readmission_30d"]))
    if leaf not in range(len(LEAVES)) or parent != int(leaf > 0):
        raise RuntimeError("Parent/leaf label consistency failure")
    return leaf


def _year_index_for_example(value: Any, *, allow_future_oov_year: bool = False) -> tuple[int, int]:
    """Resolve a token year index without silently widening the training contract.

    The future/OOD bucket is an inference-only compatibility path for a model
    whose embedding table was trained with a reserved fifth index.  Callers
    must opt in explicitly; it never changes what ``read_partition`` may read.
    """
    if value is None or isinstance(value, bool):
        raise RuntimeError("Missing or invalid year")
    try:
        if pd.isna(value):
            raise RuntimeError("Missing or invalid year")
        year = int(value)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError("Missing or invalid year") from None
    if isinstance(value, float) and not value.is_integer():
        raise RuntimeError("Missing or invalid year")
    if year in YEAR_TO_INDEX:
        return year, YEAR_TO_INDEX[year]
    if allow_future_oov_year:
        return year, FUTURE_OOV_YEAR_INDEX
    raise RuntimeError(f"Unauthorized year {year}")


def make_example(row: dict[str, Any], dx_size: int, max_tokens: int, *,
                 allow_future_oov_year: bool = False) -> dict[str, Any]:
    """Current-stay bag is encounter 0; 180-day prior bag is encounter 1.

    ``allow_future_oov_year`` is deliberately keyword-only and defaults to
    false.  It is for a post-lock inference caller only, never development or
    2021A training/selection.
    """
    if max_tokens < 2: raise ValueError("max_tokens must reserve current and prior bags")
    current_dx = [x for x in _token_list(row.get("dx_tokens")) if x > SPECIAL_TOKEN_MAX]
    current_pr = [x for x in _token_list(row.get("pr_tokens")) if x > SPECIAL_TOKEN_MAX]
    prday = _token_list(row.get("prday"))
    current = [(x, 0, 0, 0) for x in current_dx]
    current.extend((x + dx_size, 1, timing_bucket(prday[i] if i < len(prday) else None, row.get("LOS")), 0)
                   for i, x in enumerate(current_pr))
    prior_dx = [x for x in _token_list(row.get("prior_dx_tokens_180d")) if x > SPECIAL_TOKEN_MAX]
    prior_pr = [x for x in _token_list(row.get("prior_pr_tokens_180d")) if x > SPECIAL_TOKEN_MAX]
    prior = [(x, 0, 0, 1) for x in prior_dx]
    prior.extend((x + dx_size, 1, 0, 1) for x in prior_pr)
    current_cap = min(len(current), max_tokens - (1 if prior else 0))
    chosen = current[:current_cap] + prior[:max_tokens-current_cap]
    if not chosen: chosen = [(3, 0, 0, 0)]
    year, year_index = _year_index_for_example(row.get("year"), allow_future_oov_year=allow_future_oov_year)
    leaf = _leaf_label(row)
    return {
        "token_ids": [x[0] for x in chosen], "token_type": [x[1] for x in chosen],
        "timing_bucket": [x[2] for x in chosen], "encounter_index": [x[3] for x in chosen],
        "year_index": [year_index] * len(chosen), "static_raw": row["static_raw"],
        "leaf": leaf, "any_readmission": int(leaf > 0),
        "metadata": {key: row[key] for key in ("year", "encounter_hash", "patient_hash", "analysis_partition",
                      "readmission_leaf", "any_unplanned_readmission_30d", *OUTPUT_CONTEXT_COLUMNS) if key in row},
    }


def _category_value(value: Any) -> str:
    if value is None or pd.isna(value): return "__MISSING__"
    if isinstance(value, float) and value.is_integer(): return str(int(value))
    return str(value)


@dataclass(frozen=True)
class StaticPreprocessor:
    """Development-frozen, serializable static feature state.

    Numeric columns yield standardized value plus explicit missingness.  Each
    categorical column yields frozen observed levels plus distinct MISSING/OOV
    buckets.  Level ordering is lexicographic, therefore checkpoint identity is
    independent of input row order.
    """
    numeric_columns: list[str]
    categorical_columns: list[str]
    median: dict[str, float]
    scale: dict[str, float]
    levels: dict[str, list[str]]
    version: int = 2

    @classmethod
    def fit(cls, frame: pd.DataFrame, numeric_columns: Iterable[str] = NUMERIC_STATIC_COLUMNS,
            categorical_columns: Iterable[str] = CATEGORICAL_STATIC_COLUMNS) -> "StaticPreprocessor":
        require_unique_columns(frame, "StaticPreprocessor.fit input")
        numeric = [name for name in numeric_columns if name in frame.columns]
        categorical = [name for name in categorical_columns if name in frame.columns]
        median: dict[str, float] = {}; scale: dict[str, float] = {}; levels: dict[str, list[str]] = {}
        for name in numeric:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
            med = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
            filled = np.where(np.isfinite(values), values, med)
            sd = float(filled.std())
            median[name] = med; scale[name] = sd if sd > 1e-8 else 1.0
        for name in categorical:
            observed = {_category_value(v) for v in frame[name].tolist()} - {"__MISSING__"}
            # Reserve both states even when absent in development; this preserves
            # semantics rather than allowing a future validation value to extend state.
            levels[name] = sorted(observed)
        return cls(numeric, categorical, median, scale, levels)

    @property
    def dimension(self) -> int:
        return 2 * len(self.numeric_columns) + sum(len(self.levels[c]) + 2 for c in self.categorical_columns)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        require_unique_columns(frame, "StaticPreprocessor.transform input")
        columns: list[np.ndarray] = []
        n = len(frame)
        for name in self.numeric_columns:
            raw = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64) if name in frame else np.full(n, np.nan)
            missing = ~np.isfinite(raw); filled = np.where(missing, self.median[name], raw)
            columns.extend(((filled-self.median[name])/self.scale[name], missing.astype(np.float64)))
        for name in self.categorical_columns:
            values = [_category_value(v) for v in frame[name].tolist()] if name in frame else ["__MISSING__"] * n
            allowed = self.levels[name]
            emitted = allowed + ["__MISSING__", "__OOV__"]
            normalized = [v if v in allowed or v == "__MISSING__" else "__OOV__" for v in values]
            columns.extend(np.asarray([v == level for v in normalized], dtype=np.float64) for level in emitted)
        return np.column_stack(columns).astype(np.float32) if columns else np.zeros((n, 0), dtype=np.float32)

    def to_dict(self) -> dict[str, Any]: return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StaticPreprocessor": return cls(**value)


class ExampleDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]) -> None: self.examples = examples
    def __len__(self) -> int: return len(self.examples)
    def __getitem__(self, index: int) -> dict[str, Any]: return self.examples[index]


def collate_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    size, length = len(rows), max(len(row["token_ids"]) for row in rows)
    ids = torch.zeros((size, length), dtype=torch.long); types = torch.zeros_like(ids)
    timing = torch.zeros_like(ids); encounter = torch.zeros_like(ids); years = torch.zeros_like(ids)
    mask = torch.zeros((size, length), dtype=torch.bool)
    for i, row in enumerate(rows):
        count = len(row["token_ids"])
        for destination, source in ((ids, "token_ids"), (types, "token_type"), (timing, "timing_bucket"),
                                    (encounter, "encounter_index"), (years, "year_index")):
            destination[i, :count] = torch.tensor(row[source], dtype=torch.long)
        mask[i, :count] = True
    static = torch.tensor(np.stack([row["static"] for row in rows]), dtype=torch.float32)
    return {"model": {"token_ids": ids, "token_type": types, "timing_bucket": timing, "encounter_index": encounter,
             "year_index": years, "token_mask": mask, "static_features": static,
             "admission_time_static": torch.zeros((size, 1), dtype=torch.float32)},
            "labels": {"any_readmission": torch.tensor([r["any_readmission"] for r in rows]),
                       "leaf": torch.tensor([r["leaf"] for r in rows])}, "metadata": [r["metadata"] for r in rows]}


def _available_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq
    return set(pq.ParquetFile(path).schema_arrow.names)


def read_partition(root: Path, history_dir: Path, year: int, partition: str,
                   static_columns: Iterable[str] = STATIC_COLUMNS) -> pd.DataFrame:
    """Read exactly one permitted partition, with a unique physical column list."""
    import pyarrow.parquet as pq
    if year in DEV_YEARS:
        if partition != "development": raise RuntimeError("Development years must use development partition")
    elif year == VALIDATION_YEAR:
        if partition != "2021A": raise RuntimeError("Only 2021A may be materialized by finetuning")
    else: raise RuntimeError("2021B and 2022 are sealed from this script")
    episode_path = root / "data" / "nrd" / f"year={year}" / "ap_episodes.parquet"
    history_path = history_dir / f"ap_history_{year}.parquet"
    available = _available_columns(episode_path)
    needed = unique_preserving_order((*REQUIRED_EPISODE_COLUMNS,
                                     *(c for c in (*static_columns, *OUTPUT_CONTEXT_COLUMNS) if c in available)))
    table = pq.read_table(episode_path, columns=needed, filters=[("primary_analysis_eligible", "=", True),
                                                                   ("analysis_partition", "=", partition)])
    frame = table.to_pandas(); require_unique_columns(frame, f"{year} episode partition")
    if frame.empty or set(frame["year"].astype(int)) != {year} or set(frame["analysis_partition"]) != {partition}:
        raise RuntimeError("Partition gate failed")
    history_columns = unique_preserving_order(("encounter_hash", "patient_hash", "analysis_partition", *HISTORY_COLUMNS))
    history = pq.read_table(history_path, columns=history_columns, filters=[("analysis_partition", "=", partition)]).to_pandas()
    require_unique_columns(history, f"{year} history partition")
    h = history.drop(columns=["patient_hash", "analysis_partition"])
    if h["encounter_hash"].duplicated().any(): raise RuntimeError("Duplicate history keys")
    merged = frame.merge(h, on="encounter_hash", how="left", validate="one_to_one")
    require_unique_columns(merged, f"{year} merged partition")
    if merged[list(HISTORY_COLUMNS)].isna().all(axis=1).any(): raise RuntimeError("Missing history join")
    return merged


def load_unified_hierarchy(root: Path, hierarchy_dir: Path) -> dict[str, Any]:
    arrays = np.load(hierarchy_dir / "hierarchy_arrays.npz")
    vocab = json.loads((hierarchy_dir / "hierarchy_vocabularies.json").read_text(encoding="utf-8"))
    dx_size = len(json.loads((root / "data/nrd/diagnosis_vocabulary_state.json").read_text(encoding="utf-8"))["token_to_id"])
    pr_size = len(json.loads((root / "data/nrd/procedure_vocabulary_state.json").read_text(encoding="utf-8"))["token_to_id"])
    if len(arrays["dx_category"]) != dx_size or len(arrays["pr_category"]) != pr_size:
        raise RuntimeError("Vocabulary/hierarchy cardinality mismatch")
    dx_cat_n, dx_dom_n = len(vocab["diagnosis_category"]), len(vocab["diagnosis_domain"])
    pr_cat = np.where(arrays["pr_category"] > 0, arrays["pr_category"] + dx_cat_n - 1, 0)
    pr_dom = np.where(arrays["pr_domain"] > 0, arrays["pr_domain"] + dx_dom_n - 1, 0)
    return {"dx_size": dx_size, "unified_vocab_size": dx_size + pr_size,
            "category_vocab_size": dx_cat_n + len(vocab["procedure_category"]) - 1,
            "domain_vocab_size": dx_dom_n + len(vocab["procedure_domain"]) - 1,
            "token_to_category": torch.from_numpy(np.concatenate([arrays["dx_category"], pr_cat]).astype(np.int64)),
            "token_to_domain": torch.from_numpy(np.concatenate([arrays["dx_domain"], pr_dom]).astype(np.int64))}


def make_model(bundle: dict[str, Any], static_dim: int, args: argparse.Namespace) -> ClaimsTransformer:
    common_variable_only = bool(getattr(args, "common_variable_only", False))
    config = ClaimsTransformerConfig(unified_vocab_size=int(bundle["unified_vocab_size"]),
        category_vocab_size=int(bundle["category_vocab_size"]), domain_vocab_size=int(bundle["domain_vocab_size"]),
        static_dim=0 if args.no_static else static_dim, admission_time_static_dim=0, d_model=args.d_model,
        nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        # Fine-tuning examples currently use only current/prior bags (indices
        # 0/1), but encounter embeddings are part of the shared pretrained
        # representation and must retain the pretraining geometry.
        max_encounters=9, year_vocab_size=5, use_hierarchy=not args.no_hierarchy, use_prday=not args.no_prday,
        use_prior_encounters=not args.no_prior, use_static_context=not args.no_static,
        use_year_version=not args.no_year and not common_variable_only)
    return ClaimsTransformer(config, bundle["token_to_category"], bundle["token_to_domain"])


def move_to_device(value: Any, device: torch.device) -> Any:
    return {key: item.to(device) if isinstance(item, torch.Tensor) else item for key, item in value.items()}


def finetuning_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Immutable model/fine-tuning settings required for fail-closed resume."""
    names = ("epochs", "batch_size", "grad_accum", "learning_rate", "weight_decay", "max_tokens",
             "d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "threads", "workers",
             "seed", "checkpoint_every", "max_updates", "patience", "selection_min_delta",
             "no_hierarchy", "no_prday", "no_prior", "no_static", "no_year",
             "no_hospital_socioeconomic", "common_variable_only")
    contract = {name: getattr(args, name, None) for name in names}
    pretrained = getattr(args, "pretrained_checkpoint", None)
    contract["pretrained_checkpoint"] = str(Path(pretrained).resolve()) if pretrained else None
    return contract


def checkpoint_payload(model: ClaimsTransformer, optimizer: torch.optim.Optimizer, epoch: int, batch_cursor: int,
                       global_step: int, static: StaticPreprocessor, args: argparse.Namespace,
                       selection_state: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch,
            "batch_cursor": batch_cursor, "global_step": global_step, "static_preprocessor": static.to_dict(),
            "model_config": model.config.to_dict(), "training_contract": finetuning_contract(args),
            "selection_state": selection_state or {}, "args": vars(args), "seed": args.seed}


def save_checkpoint(path: Path, model: ClaimsTransformer, optimizer: torch.optim.Optimizer, epoch: int,
                    batch_cursor: int, global_step: int, static: StaticPreprocessor, args: argparse.Namespace,
                    selection_state: dict[str, Any] | None = None) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(model, optimizer, epoch, batch_cursor, global_step, static, args,
                                  selection_state), temporary)
    os.replace(temporary, path)


def load_pretrained(model: ClaimsTransformer, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # Current Stage 5 pretraining writes ``model_state``.  Earlier validated
    # runs used either ``model_state_dict`` or ``model``; a raw state_dict is
    # also accepted for an explicitly supplied legacy artifact.  The order is
    # intentional so a checkpoint carrying more than one representation has a
    # single canonical source.
    state: dict[str, torch.Tensor] | None = None
    if isinstance(payload, dict):
        for key in ("model_state", "model_state_dict", "model"):
            if key in payload:
                if not isinstance(payload[key], dict):
                    raise RuntimeError(f"Pretrained checkpoint {key} is not a state dictionary")
                state = dict(payload[key]); break
    if state is None:
        if not isinstance(payload, dict):
            raise RuntimeError("Pretrained checkpoint is not a state dictionary")
        state = dict(payload)
    nonshared_prefixes = (
        "static_encoder.", "admission_time_encoder.", "aux_high_cost_head.",
        "aux_prolonged_los_head.", "aux_death_head.",
    )
    for name, tensor in list(state.items()):
        target = model.state_dict().get(name)
        if (name == "year_embedding.weight" and target is not None
                and tensor.ndim == 2 and target.ndim == 2
                and tensor.shape[1] == target.shape[1]
                and tensor.shape[0] >= len(DEV_YEARS)
                and target.shape[0] > len(DEV_YEARS)):
            # The pretraining contract only exposes indices 0--2 even when an
            # older checkpoint allocated an unused fourth row.  Never inherit
            # that random, untrained row as the 2021 representation.
            expanded = target.detach().clone()
            expanded[:len(DEV_YEARS)] = tensor[:len(DEV_YEARS)]
            expanded[len(DEV_YEARS):] = tensor[:len(DEV_YEARS)].mean(dim=0, keepdim=True)
            state[name] = expanded
            tensor = expanded
        if target is not None and target.shape != tensor.shape:
            # These layers depend on static/admission-time preprocessing or
            # auxiliary tasks.  All code, hierarchy and Transformer tensors
            # are shared representation parameters and must match exactly.
            if name.startswith(nonshared_prefixes): del state[name]
            else: raise RuntimeError(f"Incompatible pretrained tensor {name}: {tensor.shape} != {target.shape}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {"readmission_head.weight", "readmission_head.bias", "leaf_head.weight", "leaf_head.bias",
               "aux_high_cost_head.weight", "aux_high_cost_head.bias", "aux_prolonged_los_head.weight",
               "aux_prolonged_los_head.bias", "aux_death_head.weight", "aux_death_head.bias"}
    allowed |= {name for name in model.state_dict() if name.startswith(("static_encoder.", "admission_time_encoder."))}
    if set(missing) - allowed or unexpected: raise RuntimeError(f"Incompatible pretrained checkpoint: {missing=}, {unexpected=}")


def train_epoch(model: ClaimsTransformer, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
                grad_accum: int, amp: bool, start_batch: int = 0, max_updates: int = 0, on_update=None) -> tuple[int, int]:
    """Step final partial gradient accumulation rather than silently dropping it."""
    if grad_accum < 1: raise ValueError("grad_accum must be positive")
    model.train(); optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    updates = accumulated = 0; cursor = start_batch; total = len(loader)
    for batch_index, batch in enumerate(loader):
        if batch_index < start_batch: continue
        cursor = batch_index + 1
        inputs, labels = move_to_device(batch["model"], device), move_to_device(batch["labels"], device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            loss = supervised_loss(model(**inputs, return_mlm=False), labels)["loss"] / grad_accum
        scaler.scale(loss).backward(); accumulated += 1
        if accumulated == grad_accum or cursor == total:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); updates += 1; accumulated = 0
            if on_update is not None: on_update(cursor)
            if max_updates and updates >= max_updates: return updates, cursor
    return updates, cursor


@torch.no_grad()
def predict(model: ClaimsTransformer, loader: DataLoader, device: torch.device, amp: bool) -> pd.DataFrame:
    model.eval(); records: list[dict[str, Any]] = []
    for batch in loader:
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            output = model(**move_to_device(batch["model"], device), return_mlm=False)
        leaves = torch.softmax(output["leaf_logits"], dim=1).cpu().numpy()
        for metadata, probability in zip(batch["metadata"], leaves):
            record = dict(metadata); record["p_any_readmission"] = float(probability[1:].sum())
            record.update({f"p_leaf_{name}": float(value) for name, value in zip(LEAVES, probability)})
            records.append(record)
    return pd.DataFrame.from_records(records)


def make_examples(frame: pd.DataFrame, dx_size: int, max_tokens: int, static_values: np.ndarray, *,
                  allow_future_oov_year: bool = False) -> list[dict[str, Any]]:
    require_unique_columns(frame, "example source")
    if len(frame) != len(static_values): raise RuntimeError("Static feature row count mismatch")
    needed = unique_preserving_order((*REQUIRED_EPISODE_COLUMNS, *HISTORY_COLUMNS, *OUTPUT_CONTEXT_COLUMNS))
    present = [name for name in needed if name in frame.columns]
    rows: list[dict[str, Any]] = []
    # Projection follows the duplicate guard; do not turn an unvalidated whole frame into dict records.
    for raw, static_raw in zip(frame.loc[:, present].to_dict("records"), static_values):
        raw["static_raw"] = static_raw
        item = make_example(raw, dx_size, max_tokens, allow_future_oov_year=allow_future_oov_year)
        item["static"] = item.pop("static_raw"); rows.append(item)
    return rows


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--history-dir", type=Path, required=True)
    p.add_argument("--hierarchy-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pretrained-checkpoint", type=Path); p.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, default=4); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4); p.add_argument("--weight-decay", type=float, default=0.01); p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256); p.add_argument("--nhead", type=int, default=8); p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=1024); p.add_argument("--dropout", type=float, default=0.10); p.add_argument("--threads", type=int, default=8)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--seed", type=int, default=SEED); p.add_argument("--checkpoint-every", type=int, default=250); p.add_argument("--max-updates", type=int, default=0)
    p.add_argument("--patience", type=int, default=2); p.add_argument("--selection-min-delta", type=float, default=1e-4)
    p.add_argument("--no-hierarchy", action="store_true"); p.add_argument("--no-prday", action="store_true"); p.add_argument("--no-prior", action="store_true")
    p.add_argument("--no-static", action="store_true"); p.add_argument("--no-year", action="store_true")
    p.add_argument("--no-hospital-socioeconomic", action="store_true")
    p.add_argument("--common-variable-only", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    if not (0 <= args.workers <= 8 and args.batch_size > 0 and args.epochs > 0 and args.max_tokens >= 2
            and args.patience >= 1 and args.selection_min_delta >= 0): raise SystemExit("Invalid loader/training arguments")
    try: numeric_static, categorical_static = static_feature_columns(args)
    except ValueError as exc: raise SystemExit(str(exc)) from exc
    set_seed(args.seed, args.threads)
    root, history_dir, hierarchy_dir, out = args.root.resolve(), args.history_dir.resolve(), args.hierarchy_dir.resolve(), args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    selected_static = (*numeric_static, *categorical_static)
    development = pd.concat([read_partition(root, history_dir, year, "development", selected_static) for year in DEV_YEARS], ignore_index=True)
    validation = read_partition(root, history_dir, VALIDATION_YEAR, "2021A", selected_static)
    require_unique_columns(development, "development concatenation"); require_unique_columns(validation, "2021A validation")
    static = StaticPreprocessor.fit(development, numeric_static, categorical_static)
    development_static, validation_static = static.transform(development), static.transform(validation)
    bundle = load_unified_hierarchy(root, hierarchy_dir)
    train_rows = make_examples(development, bundle["dx_size"], args.max_tokens, development_static)
    val_rows = make_examples(validation, bundle["dx_size"], args.max_tokens, validation_static)
    val_loader = DataLoader(ExampleDataset(val_rows), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_examples)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = make_model(bundle, static.dimension, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    start_epoch = start_cursor = global_step = 0
    selection_state: dict[str, Any] = {"history": [], "best_score": -1.0, "best_mean_brier": float("inf"),
                                       "best_epoch": None, "stale_epochs": 0}
    if args.pretrained_checkpoint: load_pretrained(model, args.pretrained_checkpoint.resolve())
    if args.resume:
        resume = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        if (resume["model_config"] != model.config.to_dict()
                or StaticPreprocessor.from_dict(resume["static_preprocessor"]) != static
                or resume.get("training_contract") != finetuning_contract(args)):
            raise RuntimeError("Resume checkpoint configuration, training contract or frozen static state does not match")
        model.load_state_dict(resume["model_state"]); optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch, start_cursor, global_step = int(resume["epoch"]), int(resume["batch_cursor"]), int(resume["global_step"])
        selection_state = dict(resume.get("selection_state", selection_state))
    checkpoint = out / "last_checkpoint.pt"
    best_checkpoint = out / "best_checkpoint.pt"
    for epoch in range(start_epoch, args.epochs):
        # Epoch-indexed seed gives a resumed run exactly the same batch order.
        train_loader = DataLoader(ExampleDataset(train_rows), batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + epoch), num_workers=args.workers, collate_fn=collate_examples)
        cursor = start_cursor if epoch == start_epoch else 0
        def on_update(next_cursor: int, epoch: int = epoch) -> None:
            nonlocal global_step
            global_step += 1
            if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                save_checkpoint(checkpoint, model, optimizer, epoch, next_cursor, global_step, static, args,
                                selection_state)
        remaining = max(0, args.max_updates-global_step) if args.max_updates else 0
        _, cursor = train_epoch(model, train_loader, optimizer, device, args.grad_accum, True, cursor, remaining, on_update)
        if args.max_updates and global_step >= args.max_updates:
            save_checkpoint(checkpoint, model, optimizer, epoch, cursor, global_step, static, args,
                            selection_state); return
        start_cursor = 0
        epoch_predictions = predict(model, val_loader, device, True)
        metrics = selection_metrics(epoch_predictions)
        record = {"epoch": epoch+1, **metrics}
        selection_state["history"] = [*selection_state["history"], record]
        improved = (metrics["selection_score"] > float(selection_state["best_score"])+args.selection_min_delta
                    or (abs(metrics["selection_score"]-float(selection_state["best_score"])) <= args.selection_min_delta
                        and metrics["mean_brier"] < float(selection_state["best_mean_brier"])-1e-12))
        if improved:
            selection_state.update({"best_score": metrics["selection_score"],
                                    "best_mean_brier": metrics["mean_brier"],
                                    "best_epoch": epoch+1, "stale_epochs": 0})
            save_checkpoint(best_checkpoint, model, optimizer, epoch+1, 0, global_step, static, args,
                            selection_state)
        else:
            selection_state["stale_epochs"] = int(selection_state["stale_epochs"])+1
        save_checkpoint(checkpoint, model, optimizer, epoch+1, 0, global_step, static, args,
                        selection_state)
        print(stable_json(record), flush=True)
        if int(selection_state["stale_epochs"]) >= args.patience:
            break
    if not best_checkpoint.exists(): raise RuntimeError("No finite 2021A early-stopping candidate was produced")
    selected = torch.load(best_checkpoint, map_location=device, weights_only=False)
    if selected.get("training_contract") != finetuning_contract(args):
        raise RuntimeError("Best checkpoint training contract mismatch")
    model.load_state_dict(selected["model_state"])
    predictions = predict(model, val_loader, device, True)
    if set(predictions["analysis_partition"]) != {"2021A"} or set(predictions["year"].astype(int)) != {2021}: raise RuntimeError("Prediction partition gate failed")
    import pyarrow as pa
    import pyarrow.parquet as pq
    prediction_path = out / "predictions_2021A.parquet"; pq.write_table(pa.Table.from_pandas(predictions, preserve_index=False), prediction_path, compression="zstd", compression_level=6)
    manifest = {"status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022", "version": 2, "seed": args.seed,
        "development_years": list(DEV_YEARS), "prediction_partition": "2021A", "2021B_accessed": False, "year_2022_accessed": False,
        "prevalence_resampling": "none", "discwt_predictor": False, "static_preprocessor_fit_years": list(DEV_YEARS),
        "static_preprocessor": static.to_dict(), "history": {"encounter_index": 1, "left_truncation_indicators": [x for x in CATEGORICAL_STATIC_COLUMNS if x.startswith("history_")]},
        "model_config": model.config.to_dict(), "training_contract": finetuning_contract(args),
        "year_version_contract": {"mapping": {str(k): v for k, v in YEAR_TO_INDEX.items()},
                                  "future_or_oov_index": FUTURE_OOV_YEAR_INDEX,
                                  "future_initialization": "mean of available development-year pretrained embeddings",
                                  "test_year_not_used_for_initialization": True},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "ablations": {"no_hierarchy": args.no_hierarchy, "no_prday": args.no_prday, "no_prior": args.no_prior,
                      "no_static": args.no_static, "no_year": args.no_year,
                      "no_hospital_socioeconomic": args.no_hospital_socioeconomic,
                      "common_variable_only": args.common_variable_only},
        "static_feature_set": {"numeric": list(numeric_static), "categorical": list(categorical_static),
                               "dimension": static.dimension},
        "model_selection": {"partition": "2021A", "criterion": "mean_unweighted_co_primary_auprc",
                            "tie_breaker": "lower_mean_co_primary_brier_then_earlier_epoch",
                            "patience": args.patience, "minimum_delta": args.selection_min_delta,
                            **selection_state},
        "amp": True, "threads": args.threads, "optimizer_updates": global_step,
        "pretrained_checkpoint": ({"path": str(args.pretrained_checkpoint.resolve()),
                                   "sha256": sha256(args.pretrained_checkpoint.resolve())}
                                  if args.pretrained_checkpoint else None),
        "prediction": {"file": prediction_path.name, "bytes": prediction_path.stat().st_size, "sha256": sha256(prediction_path)},
        "checkpoint": {"file": best_checkpoint.name, "sha256": sha256(best_checkpoint)}}
    (out / "finetune_manifest.json").write_text(stable_json(manifest), encoding="utf-8"); print(stable_json(manifest))


if __name__ == "__main__": main()
