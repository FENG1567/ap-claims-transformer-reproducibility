#!/usr/bin/env python3
"""Stream all 2018-2020 NRD admissions for masked hierarchical pretraining."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from claims_transformer import ClaimsTransformer, ClaimsTransformerConfig, multitask_pretraining_loss


SEED = 20260912
YEARS = (2018, 2019, 2020)
SPECIAL_MAX = 3

# Fixed semantic level sets avoid fitting administrative-code geometry on an
# evaluation year.  Each field also receives explicit MISSING and OOV columns.
ADMISSION_CATEGORY_LEVELS = {
    "AWEEKEND": (0, 1),
    "ELECTIVE": (0, 1),
    "FEMALE": (0, 1),
    "HCUP_ED": (0, 1),
    "RESIDENT": (0, 1),
    "PAY1": (1, 2, 3, 4, 5, 6),
    "ZIPINC_QRTL": (1, 2, 3, 4),
    "PL_NCHS": (1, 2, 3, 4, 5, 6),
    "HOSP_BEDSIZE": (1, 2, 3),
    "H_CONTRL": (1, 2, 3),
    "HOSP_URCAT4": (1, 2, 3, 4),
    "HOSP_UR_TEACH": (1, 2, 3),
    "DMONTH": tuple(range(1, 13)),
}
ADMISSION_COLUMNS = ("AGE", *ADMISSION_CATEGORY_LEVELS, "cost_2021_usd", "DIED")


def admission_static_dimension() -> int:
    return 2 + sum(len(levels) + 2 for levels in ADMISSION_CATEGORY_LEVELS.values())


def _valid_number(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def encode_admission_static(values: dict[str, object], spec: dict) -> list[float]:
    age = values.get("AGE")
    age_valid = _valid_number(age) and 0 <= float(age) <= 120
    age_state = spec["age_scaling"]
    vector = [
        (float(age) - float(age_state["mean"])) / float(age_state["scale"]) if age_valid else 0.0,
        float(not age_valid),
    ]
    for name, levels in ADMISSION_CATEGORY_LEVELS.items():
        raw = values.get(name)
        valid = _valid_number(raw)
        normalized = int(float(raw)) if valid and float(raw).is_integer() else None
        vector.extend(float(normalized == level) for level in levels)
        vector.append(float(not valid))
        vector.append(float(valid and normalized not in levels))
    if len(vector) != admission_static_dimension():
        raise RuntimeError("admission-time static feature dimension mismatch")
    return vector


def is_early_procedure(day) -> bool:
    if not _valid_number(day):
        return False
    value = int(float(day))
    return value <= 0 and value > -90 and value != -66


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def timing_bucket(day: int | None, los: int) -> int:
    if day is None:
        return 0
    if day <= -90 or day == -66:
        return 0
    if day < 0:
        return 1
    if day > max(0, los):
        return 6
    if day == 0:
        return 2
    if day <= 2:
        return 3
    if day <= 7:
        return 4
    return 5


def planned_optimizer_updates(total_rows: int, batch_size: int, grad_accum: int, epochs: int) -> int:
    batches_per_epoch = math.ceil(total_rows / batch_size)
    return math.ceil(batches_per_epoch / grad_accum) * epochs


def flush_partial_gradient(batch_count: int, grad_accum: int, step_fn: Callable[[], None]) -> None:
    if batch_count and batch_count % grad_accum:
        step_fn()


def training_contract(args: argparse.Namespace) -> dict[str, int | float]:
    """Immutable settings required for an exact, fail-closed resume."""
    return {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "max_tokens": int(args.max_tokens),
        "workers": int(args.workers),
        "threads": int(args.threads),
        "learning_rate": float(args.learning_rate),
        "auxiliary_weight": float(args.auxiliary_weight),
        "max_steps": int(args.max_steps),
        "checkpoint_every": int(args.checkpoint_every),
    }


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _cpu_byte_rng_state(value: object, label: str) -> torch.Tensor:
    """Validate and device-normalize a serialized PyTorch RNG byte state.

    ``torch.load(..., map_location=cuda)`` moves every tensor in a checkpoint,
    including CPU/CUDA RNG byte tensors.  PyTorch's RNG restoration APIs require
    CPU ByteTensors regardless of the model device, so copy bytes back to CPU
    without casting or otherwise transforming their contents.
    """
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"resume checkpoint {label} RNG state is not a tensor")
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0:
        raise RuntimeError(f"resume checkpoint {label} RNG state is not a 1D ByteTensor")
    return value.detach().to(device="cpu").contiguous()


def restore_rng_state(state: dict) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise RuntimeError("resume checkpoint RNG state is incomplete")
    cpu_state = _cpu_byte_rng_state(state["torch_cpu"], "CPU")
    cuda_raw = state["torch_cuda"]
    if not isinstance(cuda_raw, (list, tuple)):
        raise RuntimeError("resume checkpoint CUDA RNG state is not a list")
    cuda_states = [_cpu_byte_rng_state(value, f"CUDA[{index}]") for index, value in enumerate(cuda_raw)]
    # Validate all device-sensitive state before changing any RNG generator.
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_states)


def load_unified_hierarchy(root: Path, hierarchy_dir: Path):
    arrays = np.load(hierarchy_dir / "hierarchy_arrays.npz")
    vocab = json.loads((hierarchy_dir / "hierarchy_vocabularies.json").read_text(encoding="utf-8"))
    dx_size = len(json.loads((root / "data/nrd/diagnosis_vocabulary_state.json").read_text())["token_to_id"])
    pr_size = len(json.loads((root / "data/nrd/procedure_vocabulary_state.json").read_text())["token_to_id"])
    dx_cat_n = len(vocab["diagnosis_category"])
    dx_dom_n = len(vocab["diagnosis_domain"])
    pr_cat = arrays["pr_category"].astype(np.int64)
    pr_dom = arrays["pr_domain"].astype(np.int64)
    pr_cat = np.where(pr_cat > 0, pr_cat + dx_cat_n - 1, 0)
    pr_dom = np.where(pr_dom > 0, pr_dom + dx_dom_n - 1, 0)
    token_to_category = np.concatenate([arrays["dx_category"].astype(np.int64), pr_cat])
    token_to_domain = np.concatenate([arrays["dx_domain"].astype(np.int64), pr_dom])
    return {
        "dx_size": dx_size, "pr_size": pr_size,
        "unified_vocab_size": dx_size + pr_size,
        "category_vocab_size": dx_cat_n + len(vocab["procedure_category"]) - 1,
        "domain_vocab_size": dx_dom_n + len(vocab["procedure_domain"]) - 1,
        "token_to_category": torch.from_numpy(token_to_category),
        "token_to_domain": torch.from_numpy(token_to_domain),
    }


class AdmissionStream(IterableDataset):
    def __init__(self, paths: list[Path], dx_size: int, max_tokens: int, epoch: int,
                 auxiliary_spec: dict, seed: int = SEED):
        self.paths = paths
        self.dx_size = dx_size
        self.max_tokens = max_tokens
        self.epoch = epoch
        self.auxiliary_spec = auxiliary_spec
        self.seed = seed

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        units = []
        for path in self.paths:
            pf = pq.ParquetFile(path)
            year = int(path.parent.name.split("=")[1])
            units.extend((path, rg, year) for rg in range(pf.num_row_groups))
        rng = random.Random(self.seed + self.epoch * 1009)
        rng.shuffle(units)
        units = units[worker_id::worker_count]
        for path, rg, year in units:
            pf = pq.ParquetFile(path)
            table = pf.read_row_group(
                rg, columns=["dx_tokens", "pr_tokens", "prday", "LOS", *ADMISSION_COLUMNS]
            )
            order = list(range(table.num_rows))
            random.Random(self.seed + self.epoch * 1009 + rg * 9173 + year).shuffle(order)
            dx_col = table["dx_tokens"].to_pylist()
            pr_col = table["pr_tokens"].to_pylist()
            day_col = table["prday"].to_pylist()
            los_col = table["LOS"].to_pylist()
            static_columns = {name: table[name].to_pylist() for name in ADMISSION_COLUMNS}
            for i in order:
                dx = [int(v) for v in (dx_col[i] or []) if v is not None and int(v) > SPECIAL_MAX]
                pr_values = pr_col[i] or []
                day_values = day_col[i] or []
                pr_pairs = []
                for pos, tok in enumerate(pr_values):
                    if tok is None or int(tok) <= SPECIAL_MAX:
                        continue
                    day = day_values[pos] if pos < len(day_values) else None
                    pr_pairs.append(
                        (int(tok) + self.dx_size, timing_bucket(day, int(los_col[i] or 0)), is_early_procedure(day))
                    )
                # Preserve principal/position order and reserve at least one slot for procedures.
                dx_cap = min(len(dx), min(64, self.max_tokens))
                pr_cap = min(len(pr_pairs), max(0, self.max_tokens - dx_cap))
                tokens = dx[:dx_cap] + [p[0] for p in pr_pairs[:pr_cap]]
                types = [0] * dx_cap + [1] * pr_cap
                times = [0] * dx_cap + [p[1] for p in pr_pairs[:pr_cap]]
                aux_mask = [False] * dx_cap + [p[2] for p in pr_pairs[:pr_cap]]
                if not tokens:
                    tokens, types, times, aux_mask = [3], [0], [0], [False]
                values = {name: column[i] for name, column in static_columns.items()}
                cost = values["cost_2021_usd"]
                los = los_col[i]
                died = values["DIED"]
                cost_valid = _valid_number(cost) and float(cost) > 0
                los_valid = _valid_number(los) and float(los) >= 0
                death_valid = _valid_number(died) and float(died) in (0.0, 1.0)
                yield {
                    "token_ids": tokens, "token_type": types, "timing_bucket": times,
                    "year_index": year - 2018,
                    "auxiliary_token_mask": aux_mask,
                    "admission_time_static": encode_admission_static(values, self.auxiliary_spec),
                    "auxiliary_labels": {
                        "high_cost": int(cost_valid and float(cost) > float(self.auxiliary_spec["high_cost"]["threshold_2021_usd"])),
                        "prolonged_los": int(los_valid and float(los) > 7),
                        "death": int(death_valid and int(float(died)) == 1),
                    },
                    "auxiliary_masks": {
                        "high_cost": cost_valid, "prolonged_los": los_valid, "death": death_valid,
                    },
                }


class MLMCollator:
    def __init__(self, unified_vocab_size: int, token_to_category: torch.Tensor, seed: int = SEED):
        self.vocab_size = unified_vocab_size
        self.token_to_category = token_to_category
        self.seed = seed
        self._rng = None

    def __call__(self, rows: list[dict]):
        if self._rng is None:
            worker = get_worker_info()
            self._rng = np.random.default_rng(self.seed + (worker.id if worker else 0))
        batch, length = len(rows), max(len(r["token_ids"]) for r in rows)
        ids = torch.zeros((batch, length), dtype=torch.long)
        types = torch.zeros_like(ids)
        times = torch.zeros_like(ids)
        years = torch.zeros_like(ids)
        mask = torch.zeros((batch, length), dtype=torch.bool)
        auxiliary_token_mask = torch.zeros((batch, length), dtype=torch.bool)
        for i, row in enumerate(rows):
            n = len(row["token_ids"])
            ids[i, :n] = torch.tensor(row["token_ids"])
            types[i, :n] = torch.tensor(row["token_type"])
            times[i, :n] = torch.tensor(row["timing_bucket"])
            years[i, :n] = int(row["year_index"])
            mask[i, :n] = True
            auxiliary_token_mask[i, :n] = torch.tensor(row["auxiliary_token_mask"], dtype=torch.bool)
        selectable = mask & ids.gt(SPECIAL_MAX)
        selected = torch.from_numpy(self._rng.random(ids.shape) < 0.15) & selectable
        for i in range(batch):
            if selectable[i].any() and not selected[i].any():
                choices = torch.nonzero(selectable[i], as_tuple=False).flatten().numpy()
                selected[i, int(self._rng.choice(choices))] = True
        original = ids.clone()
        draw = torch.from_numpy(self._rng.random(ids.shape))
        ids[selected & (draw < 0.80)] = 1
        random_mask = selected & (draw >= 0.80) & (draw < 0.90)
        if random_mask.any():
            ids[random_mask] = torch.from_numpy(
                self._rng.integers(SPECIAL_MAX + 1, self.vocab_size, size=int(random_mask.sum()), dtype=np.int64)
            )
        code_labels = original[selected]
        category_labels = self.token_to_category[original[selected]]
        admission_static = torch.tensor(
            [row["admission_time_static"] for row in rows], dtype=torch.float32
        )
        return {
            "model": {
                "token_ids": ids, "token_type": types, "timing_bucket": times,
                "encounter_index": torch.zeros_like(ids), "year_index": years,
                "token_mask": mask, "static_features": admission_static,
                "admission_time_static": admission_static, "mlm_mask": selected,
                "auxiliary_token_mask": auxiliary_token_mask,
            },
            "code_labels": code_labels, "category_labels": category_labels,
            "auxiliary_labels": {
                name: torch.tensor([row["auxiliary_labels"][name] for row in rows], dtype=torch.float32)
                for name in ("high_cost", "prolonged_los", "death")
            },
            "auxiliary_masks": {
                name: torch.tensor([row["auxiliary_masks"][name] for row in rows], dtype=torch.bool)
                for name in ("high_cost", "prolonged_los", "death")
            },
            "tokens": int(mask.sum()), "masked_tokens": int(selected.sum()),
            "auxiliary_early_tokens": int(auxiliary_token_mask.sum()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hierarchy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--auxiliary-spec", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full planned epoch(s)")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args()
    if not (1 <= args.threads <= 8 and 0 <= args.workers <= 8 and args.checkpoint_every >= 0
            and args.auxiliary_weight >= 0):
        raise SystemExit("thread/worker limits exceed server policy")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA runtime is not usable in this environment")
    device = torch.device("cuda:0")
    root, hierarchy_dir, out = args.root.resolve(), args.hierarchy_dir.resolve(), args.output_dir.resolve()
    auxiliary_spec_path = args.auxiliary_spec.resolve()
    auxiliary_spec = json.loads(auxiliary_spec_path.read_text(encoding="utf-8"))
    if auxiliary_spec.get("development_years") != list(YEARS) or auxiliary_spec.get("year_2021_accessed") is not False \
            or auxiliary_spec.get("year_2022_accessed") is not False:
        raise SystemExit("auxiliary specification violates the development-only gate")
    positive_weights = {
        name: float(auxiliary_spec[name]["positive_weight_capped"])
        for name in ("high_cost", "prolonged_los", "death")
    }
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_unified_hierarchy(root, hierarchy_dir)
    config = ClaimsTransformerConfig(
        unified_vocab_size=bundle["unified_vocab_size"],
        category_vocab_size=bundle["category_vocab_size"],
        domain_vocab_size=bundle["domain_vocab_size"],
        static_dim=admission_static_dimension(), admission_time_static_dim=admission_static_dimension(),
        d_model=256, nhead=8, num_layers=4, dim_feedforward=1024,
        dropout=0.10, max_encounters=9, year_vocab_size=4,
        auxiliary_uses_masked_joint=True,
    )
    model = ClaimsTransformer(config, bundle["token_to_category"], bundle["token_to_domain"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=(0.9, 0.98))
    paths = [root / f"data/nrd/year={y}/admissions_model.parquet" for y in YEARS]
    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
    planned_updates = planned_optimizer_updates(total_rows, args.batch_size, args.grad_accum, args.epochs)
    if args.max_steps:
        planned_updates = min(planned_updates, args.max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, planned_updates))
    scaler = torch.amp.GradScaler("cuda")
    global_step = 0
    completed_epochs = 0
    start_epoch = 0
    resume_after_batch = 0
    resumed_from = None
    rows_processed = 0
    tokens_processed = 0
    masked_tokens_processed = 0
    auxiliary_early_tokens_processed = 0
    prior_elapsed_seconds = 0.0
    auxiliary_spec_sha256 = sha256(auxiliary_spec_path)
    current_training_contract = training_contract(args)
    if args.resume:
        resume_path = args.resume.resolve()
        # Load on CPU so the saved CPU RNG state remains a CPU ByteTensor.
        # Model and optimizer states are moved to the parameter device by their
        # respective load_state_dict implementations below.
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("config") != config.to_dict():
            raise RuntimeError("resume checkpoint model configuration mismatch")
        if payload.get("auxiliary_spec_sha256") != auxiliary_spec_sha256:
            raise RuntimeError("resume checkpoint auxiliary specification mismatch")
        if payload.get("training_contract") != current_training_contract:
            raise RuntimeError("resume checkpoint training contract mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        global_step = int(payload["step"])
        start_epoch = int(payload["epoch"])
        resume_after_batch = int(payload["batch_index"])
        resumed_from = str(resume_path)
        completed_epochs = start_epoch
        rows_processed = int(payload.get("rows_processed", 0))
        tokens_processed = int(payload.get("tokens_processed", 0))
        masked_tokens_processed = int(payload.get("masked_tokens_processed", 0))
        auxiliary_early_tokens_processed = int(payload.get("auxiliary_early_tokens_processed", 0))
        prior_elapsed_seconds = float(payload.get("elapsed_seconds", 0.0))
        restore_rng_state(payload.get("rng_state", {}))
    logs = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    stop = bool(args.max_steps and global_step >= args.max_steps)
    epoch_auxiliary_early_tokens = 0
    checkpoint_path = out / "last_checkpoint.pt"
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        dataset = AdmissionStream(paths, bundle["dx_size"], args.max_tokens, epoch, auxiliary_spec)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.workers,
            collate_fn=MLMCollator(bundle["unified_vocab_size"], bundle["token_to_category"], SEED + epoch * 1009),
            pin_memory=True, persistent_workers=False,
            generator=torch.Generator().manual_seed(SEED + epoch * 7919),
        )
        running_loss = 0.0
        running_batches = 0
        running_components: dict[str, float] = {}
        epoch_tokens = 0
        epoch_masked = 0
        epoch_auxiliary_early_tokens = 0
        epoch_auxiliary_valid = {name: 0 for name in ("high_cost", "prolonged_los", "death")}
        epoch_auxiliary_positive = {name: 0 for name in epoch_auxiliary_valid}
        batch_count = 0
        processed_batches_in_epoch = 0

        def apply_optimizer_update() -> None:
            nonlocal global_step, running_loss, running_batches, stop
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            if global_step % 100 == 0:
                entry = {"step": global_step, "epoch": epoch, "mean_loss_last_100_updates": running_loss / max(1, running_batches),
                         "learning_rate": scheduler.get_last_lr()[0], "tokens": epoch_tokens,
                         "masked_tokens": epoch_masked, "auxiliary_early_tokens": epoch_auxiliary_early_tokens,
                         "mean_components": {key: value / max(1, running_batches)
                                             for key, value in sorted(running_components.items())},
                         "auxiliary_valid": dict(epoch_auxiliary_valid),
                         "auxiliary_positive": dict(epoch_auxiliary_positive),
                         "elapsed_seconds": time.time() - started}
                logs.append(entry); print(json.dumps(entry), flush=True)
                running_loss = 0.0; running_batches = 0
                for key in running_components:
                    running_components[key] = 0.0
            if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                temporary = checkpoint_path.with_suffix(".pt.tmp")
                torch.save({"model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                            "config": config.to_dict(), "auxiliary_spec_sha256": auxiliary_spec_sha256,
                            "training_contract": current_training_contract, "rng_state": capture_rng_state(),
                            "step": global_step, "epoch": epoch, "batch_index": batch_count,
                            "rows_processed": rows_processed, "tokens_processed": tokens_processed,
                            "masked_tokens_processed": masked_tokens_processed,
                            "auxiliary_early_tokens_processed": auxiliary_early_tokens_processed,
                            "elapsed_seconds": prior_elapsed_seconds + time.time() - started}, temporary)
                os.replace(temporary, checkpoint_path)
            if args.max_steps and global_step >= args.max_steps:
                stop = True

        for batch_index, batch in enumerate(loader, start=1):
            batch_count = batch_index
            if epoch == start_epoch and batch_index <= resume_after_batch:
                continue
            processed_batches_in_epoch += 1
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch["model"].items()}
            labels_code = batch["code_labels"].to(device, non_blocking=True)
            labels_category = batch["category_labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
                labels_auxiliary = {k: v.to(device, non_blocking=True) for k, v in batch["auxiliary_labels"].items()}
                masks_auxiliary = {k: v.to(device, non_blocking=True) for k, v in batch["auxiliary_masks"].items()}
                losses = multitask_pretraining_loss(
                    outputs, labels_code, labels_category, labels_auxiliary, masks_auxiliary,
                    positive_weights=positive_weights, auxiliary_weight=args.auxiliary_weight,
                )
                loss = losses["loss"] / args.grad_accum
            scaler.scale(loss).backward()
            running_loss += float(losses["loss"].detach())
            for name, value in losses.items():
                if name.endswith("loss"):
                    running_components[name] = running_components.get(name, 0.0) + float(value.detach())
            running_batches += 1
            epoch_tokens += batch["tokens"]
            epoch_masked += batch["masked_tokens"]
            epoch_auxiliary_early_tokens += batch["auxiliary_early_tokens"]
            rows_processed += int(inputs["token_ids"].shape[0])
            tokens_processed += int(batch["tokens"])
            masked_tokens_processed += int(batch["masked_tokens"])
            auxiliary_early_tokens_processed += int(batch["auxiliary_early_tokens"])
            for name, valid in batch["auxiliary_masks"].items():
                epoch_auxiliary_valid[name] += int(valid.sum())
                epoch_auxiliary_positive[name] += int(batch["auxiliary_labels"][name][valid].sum())
            if processed_batches_in_epoch % args.grad_accum == 0:
                apply_optimizer_update()
                if stop:
                    break
        flush_partial_gradient(processed_batches_in_epoch, args.grad_accum, apply_optimizer_update)
        if stop:
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({"model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                        "config": config.to_dict(), "auxiliary_spec_sha256": auxiliary_spec_sha256,
                        "training_contract": current_training_contract, "rng_state": capture_rng_state(),
                        "step": global_step, "epoch": epoch, "batch_index": batch_count,
                        "rows_processed": rows_processed, "tokens_processed": tokens_processed,
                        "masked_tokens_processed": masked_tokens_processed,
                        "auxiliary_early_tokens_processed": auxiliary_early_tokens_processed,
                        "elapsed_seconds": prior_elapsed_seconds + time.time() - started}, temporary)
            os.replace(temporary, checkpoint_path)
            break
        completed_epochs += 1
        resume_after_batch = 0
    torch.cuda.synchronize(device)
    elapsed_seconds = prior_elapsed_seconds + time.time() - started
    final_path = out / "pretrained_encoder_final.pt"
    torch.save({"model_state": model.state_dict(), "config": config.to_dict(), "step": global_step,
                "years": list(YEARS), "rows_available": total_rows}, final_path)
    manifest = {
        "status": "PASS" if completed_epochs == args.epochs else "PASS_BOUNDED_RUN",
        "development_years_only": list(YEARS), "all_admission_rows_available": total_rows,
        "optimizer_updates_completed": global_step, "optimizer_updates_planned": planned_updates,
        "epochs_completed": completed_epochs,
        "throughput": {"rows_processed": rows_processed, "tokens_processed": tokens_processed,
                       "masked_tokens_processed": masked_tokens_processed,
                       "auxiliary_early_tokens_processed": auxiliary_early_tokens_processed,
                       "elapsed_seconds": elapsed_seconds,
                       "rows_per_second": rows_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                       "tokens_per_second": tokens_processed / elapsed_seconds if elapsed_seconds > 0 else 0.0},
        "objectives": (["masked_code", "masked_hierarchy_category", "high_cost", "prolonged_los", "death"]
                       if args.auxiliary_weight > 0 else ["masked_code", "masked_hierarchy_category"]),
        "auxiliary_role": "admission-time masked representation learning; not prospective bedside validation",
        "auxiliary_spec": {"path": str(auxiliary_spec_path), "sha256": auxiliary_spec_sha256},
        "positive_weights": positive_weights, "auxiliary_weight": args.auxiliary_weight,
        "auxiliary_early_tokens_last_epoch": epoch_auxiliary_early_tokens,
        "rolling_checkpoint": str(checkpoint_path), "resumed_from": resumed_from,
        "config": config.to_dict(), "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device))},
        "final_checkpoint": {"bytes": final_path.stat().st_size,
        "sha256": sha256(final_path)}, "logs": logs, "year_2021_accessed": False,
        "year_2022_accessed": False,
    }
    (out / "pretraining_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
