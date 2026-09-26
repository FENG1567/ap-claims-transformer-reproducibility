#!/usr/bin/env python3
"""Fit one frozen AP claims-Transformer configuration and emit raw 2021A risk.

The script deliberately does *not* tune a grid.  Invoke it once per candidate
configuration; an external controller may compare the resulting 2021A files.
Only 2018--2020 eligible episodes are development data.  The only 2021 data
that this module materializes are rows explicitly filtered to ``2021A``.
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
STATIC_COLUMNS = (
    "AGE", "LOS", "I10_NDX", "I10_NPR", "APRDRG_Severity", "APRDRG_Risk_Mortality",
    "AWEEKEND", "DMONTH", "ELECTIVE", "FEMALE", "HCUP_ED", "PAY1", "PL_NCHS",
    "RESIDENT", "ZIPINC_QRTL", "HOSP_BEDSIZE", "H_CONTRL", "HOSP_URCAT4",
    "HOSP_UR_TEACH", "CCR_NRD", "WAGEINDEX", "prior_count_ytd", "prior_count_30d",
    "prior_count_90d", "prior_count_180d", "prior_ed_count_180d", "prior_nonelective_count_180d",
    "prior_ap_count_180d", "prior_biliary_count_180d", "prior_sepsis_or_organ_count_180d",
    "prior_los_sum_180d", "prior_max_severity_180d", "prior_max_mortality_risk_180d",
    "days_since_prior_discharge", "history_30d_fully_observable",
    "history_90d_fully_observable", "history_180d_fully_observable",
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
    "any_unplanned_readmission_30d", "readmission_leaf", "dx_tokens", "pr_tokens", "prday",
    "LOS",
)
OUTPUT_CONTEXT_COLUMNS = ("AGE", "FEMALE", "PAY1", "ZIPINC_QRTL", "DISCWT")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def set_seed(seed: int = SEED, threads: int = 8) -> None:
    if not 1 <= threads <= 8:
        raise ValueError("--threads must be between 1 and 8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)


def timing_bucket(day: Any, los: Any) -> int:
    """Map missing procedure time to 0 without dropping the procedure token."""
    if day is None or pd.isna(day):
        return 0
    day_i, los_i = int(day), max(0, int(los)) if los is not None and not pd.isna(los) else 0
    if day_i <= -90 or day_i == -66:
        return 0
    if day_i < 0:
        return 1
    if day_i > los_i:
        return 6
    if day_i == 0:
        return 2
    if day_i <= 2:
        return 3
    if day_i <= 7:
        return 4
    return 5


def _token_list(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [int(x) for x in value if x is not None and not pd.isna(x)]


def _leaf_label(row: dict[str, Any]) -> int:
    leaf = int(row["readmission_leaf"])
    any_readmission = int(bool(row["any_unplanned_readmission_30d"]))
    if leaf not in range(len(LEAVES)) or any_readmission != int(leaf > 0):
        raise RuntimeError("Parent/leaf label consistency failure")
    return leaf


def make_example(row: dict[str, Any], dx_size: int, max_tokens: int) -> dict[str, Any]:
    """Create discharge-time current and prior bags with a correct PR offset."""
    if max_tokens < 2:
        raise ValueError("max_tokens must reserve room for both current and prior bags")
    current_dx = [x for x in _token_list(row.get("dx_tokens")) if x > SPECIAL_TOKEN_MAX]
    current_pr = [x for x in _token_list(row.get("pr_tokens")) if x > SPECIAL_TOKEN_MAX]
    prday = _token_list(row.get("prday"))
    current: list[tuple[int, int, int, int]] = [(x, 0, 0, 0) for x in current_dx]
    # The position pairing is intentional: an absent/short PRDAY list remains missing-time (0).
    current.extend((x + dx_size, 1, timing_bucket(prday[i] if i < len(prday) else None, row.get("LOS")), 0)
                   for i, x in enumerate(current_pr))
    prior_dx = [x for x in _token_list(row.get("prior_dx_tokens_180d")) if x > SPECIAL_TOKEN_MAX]
    prior_pr = [x for x in _token_list(row.get("prior_pr_tokens_180d")) if x > SPECIAL_TOKEN_MAX]
    prior = [(x, 0, 0, 1) for x in prior_dx]
    prior.extend((x + dx_size, 1, 0, 1) for x in prior_pr)
    # Preserve current admission first; retain at least one historical token when available.
    current_cap = min(len(current), max_tokens - (1 if prior else 0))
    chosen = current[:current_cap] + prior[:max_tokens - current_cap]
    if not chosen:
        chosen = [(3, 0, 0, 0)]  # non-padding placeholder for code-less eligible stays
    year = int(row["year"])
    if year not in (*DEV_YEARS, VALIDATION_YEAR):
        raise RuntimeError(f"Unauthorized year {year}")
    return {
        "token_ids": [v[0] for v in chosen], "token_type": [v[1] for v in chosen],
        "timing_bucket": [v[2] for v in chosen], "encounter_index": [v[3] for v in chosen],
        "year_index": [year - 2018] * len(chosen), "static_raw": row["static_raw"],
        "leaf": _leaf_label(row), "any_readmission": int(_leaf_label(row) > 0),
        "metadata": {k: row[k] for k in (
            "year", "encounter_hash", "patient_hash", "analysis_partition", "readmission_leaf",
            "any_unplanned_readmission_30d", *OUTPUT_CONTEXT_COLUMNS,
        ) if k in row},
    }


@dataclass
class StaticPreprocessor:
    columns: list[str]
    median: list[float]
    scale: list[float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Iterable[str] = STATIC_COLUMNS) -> "StaticPreprocessor":
        cols = [name for name in columns if name in frame.columns]
        if not cols:
            return cls([], [], [])
        matrix = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        med = np.nanmedian(matrix, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        filled = np.where(np.isfinite(matrix), matrix, med)
        scale = filled.std(axis=0); scale = np.where(scale > 1e-8, scale, 1.0)
        return cls(cols, med.astype(float).tolist(), scale.astype(float).tolist())

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.columns:
            return np.zeros((len(frame), 0), dtype=np.float32)
        matrix = frame.reindex(columns=self.columns).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        median, scale = np.asarray(self.median), np.asarray(self.scale)
        missing = ~np.isfinite(matrix)
        filled = np.where(missing, median, matrix)
        # Concatenated missingness indicators keep administrative missingness explicit.
        return np.concatenate([(filled - median) / scale, missing.astype(np.float64)], axis=1).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExampleDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    size, length = len(rows), max(len(row["token_ids"]) for row in rows)
    ids = torch.zeros((size, length), dtype=torch.long); types = torch.zeros_like(ids)
    timing = torch.zeros_like(ids); encounter = torch.zeros_like(ids); years = torch.zeros_like(ids)
    mask = torch.zeros((size, length), dtype=torch.bool)
    for i, row in enumerate(rows):
        n = len(row["token_ids"])
        for destination, source in ((ids, "token_ids"), (types, "token_type"), (timing, "timing_bucket"),
                                    (encounter, "encounter_index"), (years, "year_index")):
            destination[i, :n] = torch.tensor(row[source], dtype=torch.long)
        mask[i, :n] = True
    static = torch.tensor(np.stack([row["static"] for row in rows]), dtype=torch.float32)
    return {"model": {"token_ids": ids, "token_type": types, "timing_bucket": timing,
                       "encounter_index": encounter, "year_index": years, "token_mask": mask,
                       "static_features": static, "admission_time_static": torch.zeros((size, 1), dtype=torch.float32)},
            "labels": {"any_readmission": torch.tensor([r["any_readmission"] for r in rows]),
                       "leaf": torch.tensor([r["leaf"] for r in rows])},
            "metadata": [row["metadata"] for row in rows]}


def _available_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq
    return set(pq.ParquetFile(path).schema_arrow.names)


def read_partition(root: Path, history_dir: Path, year: int, partition: str) -> pd.DataFrame:
    """Read precisely one authorized partition; never accept 2021B/2022."""
    import pyarrow.parquet as pq
    if year in DEV_YEARS:
        if partition != "development":
            raise RuntimeError("Development years must use the development partition")
    elif year == VALIDATION_YEAR:
        if partition != "2021A":
            raise RuntimeError("Only 2021A may be materialized by finetuning")
    else:
        raise RuntimeError("2021B and 2022 are sealed from this script")
    episode_path = root / "data" / "nrd" / f"year={year}" / "ap_episodes.parquet"
    history_path = history_dir / f"ap_history_{year}.parquet"
    available = _available_columns(episode_path)
    needed = list(REQUIRED_EPISODE_COLUMNS) + [c for c in (*STATIC_COLUMNS, *OUTPUT_CONTEXT_COLUMNS) if c in available]
    table = pq.read_table(episode_path, columns=needed,
                          filters=[("primary_analysis_eligible", "=", True), ("analysis_partition", "=", partition)])
    frame = table.to_pandas()
    if frame.empty or set(frame["year"].astype(int)) != {year} or set(frame["analysis_partition"]) != {partition}:
        raise RuntimeError("Partition gate failed")
    history_columns = ["encounter_hash", "patient_hash", "analysis_partition", *HISTORY_COLUMNS]
    history = pq.read_table(history_path, columns=history_columns, filters=[("analysis_partition", "=", partition)])
    h = history.drop(["patient_hash", "analysis_partition"]).to_pandas()
    if h["encounter_hash"].duplicated().any():
        raise RuntimeError("Duplicate history keys")
    merged = frame.merge(h, on="encounter_hash", how="left", validate="one_to_one")
    if merged[list(HISTORY_COLUMNS)].isna().all(axis=1).any():
        raise RuntimeError("Missing history join")
    # DISCWT can travel in the restricted 2021A prediction table for later
    # authorized weighted evaluation, but is intentionally absent from STATIC_COLUMNS.
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
    effective_static_dim = 0 if args.no_static else static_dim
    config = ClaimsTransformerConfig(
        unified_vocab_size=int(bundle["unified_vocab_size"]), category_vocab_size=int(bundle["category_vocab_size"]),
        domain_vocab_size=int(bundle["domain_vocab_size"]), static_dim=effective_static_dim, admission_time_static_dim=0,
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout, max_encounters=2, year_vocab_size=4, use_hierarchy=not args.no_hierarchy,
        use_prday=not args.no_prday, use_prior_encounters=not args.no_prior,
        use_static_context=not args.no_static, use_year_version=not args.no_year,
    )
    return ClaimsTransformer(config, bundle["token_to_category"], bundle["token_to_domain"])


def move_to_device(value: Any, device: torch.device) -> Any:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in value.items()}


def checkpoint_payload(model: ClaimsTransformer, optimizer: torch.optim.Optimizer, epoch: int, batch_cursor: int,
                       global_step: int, static: StaticPreprocessor, args: argparse.Namespace) -> dict[str, Any]:
    return {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch,
            "batch_cursor": batch_cursor, "global_step": global_step, "static_preprocessor": static.to_dict(),
            "model_config": model.config.to_dict(), "args": vars(args), "seed": args.seed}


def save_checkpoint(path: Path, model: ClaimsTransformer, optimizer: torch.optim.Optimizer, epoch: int,
                    batch_cursor: int, global_step: int, static: StaticPreprocessor, args: argparse.Namespace) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint_payload(model, optimizer, epoch, batch_cursor, global_step, static, args), temporary)
    os.replace(temporary, path)


def load_pretrained(model: ClaimsTransformer, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = dict(payload.get("model_state", payload.get("model_state_dict", payload)))
    # Pretraining has static_dim=0, so only its first static projection differs.
    for name, tensor in list(state.items()):
        target = model.state_dict().get(name)
        if target is not None and target.shape != tensor.shape:
            if name.startswith("static_encoder.0."):
                del state[name]
            else:
                raise RuntimeError(f"Incompatible pretrained tensor {name}: {tensor.shape} != {target.shape}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"readmission_head.weight", "readmission_head.bias", "leaf_head.weight", "leaf_head.bias",
                       "aux_high_cost_head.weight", "aux_high_cost_head.bias", "aux_prolonged_los_head.weight",
                       "aux_prolonged_los_head.bias", "aux_death_head.weight", "aux_death_head.bias"}
    allowed_missing |= {"static_encoder.0.weight", "static_encoder.0.bias"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(f"Incompatible pretrained checkpoint: missing={missing}, unexpected={unexpected}")


def train_epoch(model: ClaimsTransformer, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device,
                grad_accum: int, amp: bool, start_batch: int = 0, max_updates: int = 0,
                on_update=None) -> tuple[int, int]:
    """Return (optimizer updates, next batch cursor), including final accumulation remainder."""
    if grad_accum < 1:
        raise ValueError("grad_accum must be positive")
    model.train(); optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    updates, accumulated, cursor = 0, 0, start_batch
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader):
        if batch_index < start_batch:
            continue
        cursor = batch_index + 1
        model_inputs, labels = move_to_device(batch["model"], device), move_to_device(batch["labels"], device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            loss = supervised_loss(model(**model_inputs, return_mlm=False), labels)["loss"] / grad_accum
        scaler.scale(loss).backward(); accumulated += 1
        final_remainder = cursor == total_batches
        if accumulated == grad_accum or final_remainder:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            updates += 1; accumulated = 0
            if on_update is not None:
                on_update(cursor)
            if max_updates and updates >= max_updates:
                return updates, cursor
    return updates, cursor


@torch.no_grad()
def predict(model: ClaimsTransformer, loader: DataLoader, device: torch.device, amp: bool) -> pd.DataFrame:
    model.eval(); records: list[dict[str, Any]] = []
    for batch in loader:
        inputs = move_to_device(batch["model"], device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(**inputs, return_mlm=False)
        leaf = torch.softmax(outputs["leaf_logits"], dim=1).cpu().numpy()
        # Parent is the exact union of exclusive leaf probabilities, not a competing binary score.
        for meta, probabilities in zip(batch["metadata"], leaf):
            record = dict(meta)
            record["p_any_readmission"] = float(probabilities[1:].sum())
            for name, probability in zip(LEAVES, probabilities):
                record[f"p_leaf_{name}"] = float(probability)
            records.append(record)
    return pd.DataFrame.from_records(records)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--history-dir", type=Path, required=True)
    p.add_argument("--hierarchy-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--pretrained-checkpoint", type=Path); p.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, default=4); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1); p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01); p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256); p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4); p.add_argument("--dim-feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10); p.add_argument("--threads", type=int, default=8)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--checkpoint-every", type=int, default=250); p.add_argument("--max-updates", type=int, default=0)
    p.add_argument("--no-hierarchy", action="store_true"); p.add_argument("--no-prday", action="store_true")
    p.add_argument("--no-prior", action="store_true"); p.add_argument("--no-static", action="store_true")
    p.add_argument("--no-year", action="store_true"); return p


def main() -> None:
    args = parser().parse_args()
    if not (0 <= args.workers <= 8 and args.batch_size > 0 and args.epochs > 0 and args.max_tokens >= 2):
        raise SystemExit("Invalid loader/training arguments")
    set_seed(args.seed, args.threads)
    root, history_dir, hierarchy_dir, out = (args.root.resolve(), args.history_dir.resolve(),
                                               args.hierarchy_dir.resolve(), args.output_dir.resolve())
    out.mkdir(parents=True, exist_ok=True)
    development = pd.concat([read_partition(root, history_dir, year, "development") for year in DEV_YEARS], ignore_index=True)
    validation = read_partition(root, history_dir, VALIDATION_YEAR, "2021A")
    static = StaticPreprocessor.fit(development)  # fitting is intentionally development-only
    development["static_raw"] = list(static.transform(development)); validation["static_raw"] = list(static.transform(validation))
    bundle = load_unified_hierarchy(root, hierarchy_dir)
    train_rows = [make_example(r, bundle["dx_size"], args.max_tokens) for r in development.to_dict("records")]
    val_rows = [make_example(r, bundle["dx_size"], args.max_tokens) for r in validation.to_dict("records")]
    for row in train_rows + val_rows:
        row["static"] = row.pop("static_raw")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(ExampleDataset(train_rows), batch_size=args.batch_size, shuffle=True, generator=generator,
                              num_workers=args.workers, collate_fn=collate_examples)
    val_loader = DataLoader(ExampleDataset(val_rows), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_examples)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = make_model(bundle, static_dim=0 if args.no_static else len(static.columns) * 2, args=args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    start_epoch = start_cursor = global_step = 0
    if args.pretrained_checkpoint:
        load_pretrained(model, args.pretrained_checkpoint.resolve())
    if args.resume:
        resume = torch.load(args.resume.resolve(), map_location=device, weights_only=False)
        if resume["model_config"] != model.config.to_dict():
            raise RuntimeError("Resume checkpoint configuration does not match locked run")
        model.load_state_dict(resume["model_state"]); optimizer.load_state_dict(resume["optimizer_state"])
        start_epoch, start_cursor, global_step = int(resume["epoch"]), int(resume["batch_cursor"]), int(resume["global_step"])
    checkpoint = out / "last_checkpoint.pt"
    for epoch in range(start_epoch, args.epochs):
        cursor = start_cursor if epoch == start_epoch else 0
        def on_update(next_cursor: int, epoch=epoch) -> None:
            nonlocal global_step
            global_step += 1
            if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                save_checkpoint(checkpoint, model, optimizer, epoch, next_cursor, global_step, static, args)
        remaining = max(0, args.max_updates - global_step) if args.max_updates else 0
        updates, cursor = train_epoch(model, train_loader, optimizer, device, args.grad_accum, True, cursor,
                                      max_updates=remaining, on_update=on_update)
        if args.max_updates and global_step >= args.max_updates:
            save_checkpoint(checkpoint, model, optimizer, epoch, cursor, global_step, static, args)
            return
        start_cursor = 0
    save_checkpoint(checkpoint, model, optimizer, args.epochs, 0, global_step, static, args)
    predictions = predict(model, val_loader, device, True)
    if set(predictions["analysis_partition"]) != {"2021A"} or set(predictions["year"].astype(int)) != {2021}:
        raise RuntimeError("Prediction partition gate failed")
    prediction_path = out / "predictions_2021A.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pandas(predictions, preserve_index=False), prediction_path, compression="zstd", compression_level=6)
    manifest = {"status": "PASS_SINGLE_CONFIGURATION_PRE_2021B_PRE_2022", "seed": args.seed,
                "development_years": list(DEV_YEARS), "prediction_partition": "2021A",
                "2021B_accessed": False, "year_2022_accessed": False, "prevalence_resampling": "none",
                "discwt_predictor": False, "static_preprocessor_fit_years": list(DEV_YEARS),
                "history": {"encounter_index": 1, "left_truncation_indicators": [x for x in STATIC_COLUMNS if x.startswith("history_")]},
                "model_config": model.config.to_dict(), "parameter_count": int(sum(p.numel() for p in model.parameters())),
                "ablations": {"no_hierarchy": args.no_hierarchy, "no_prday": args.no_prday, "no_prior": args.no_prior,
                              "no_static": args.no_static, "no_year": args.no_year}, "amp": True, "threads": args.threads,
                "optimizer_updates": global_step, "prediction": {"file": prediction_path.name, "bytes": prediction_path.stat().st_size,
                "sha256": sha256(prediction_path)}, "checkpoint": {"file": checkpoint.name, "sha256": sha256(checkpoint)}}
    (out / "finetune_manifest.json").write_text(stable_json(manifest), encoding="utf-8")
    print(stable_json(manifest))


if __name__ == "__main__":
    main()
