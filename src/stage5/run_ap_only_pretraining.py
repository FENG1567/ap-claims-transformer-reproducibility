"""Build and run the exposure-matched AP-only MLM pretraining comparator.

Only primary-analysis-eligible 2018--2020 AP development episodes are copied
into a private derived root.  The unchanged all-admission pretraining program
is then reused with auxiliary_weight=0, so architecture, masking, optimizer and
checkpoint behavior remain identical.  The number of complete AP epochs is
chosen to match the joint model's optimizer-update exposure within 1%.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


YEARS = (2018, 2019, 2020)
EXPECTED_AP_ROWS = {2018: 135700, 2019: 138134, 2020: 130694}
PRETRAIN_COLUMNS = (
    "dx_tokens", "pr_tokens", "prday", "LOS", "AGE", "AWEEKEND", "ELECTIVE", "FEMALE",
    "HCUP_ED", "RESIDENT", "PAY1", "ZIPINC_QRTL", "PL_NCHS", "HOSP_BEDSIZE", "H_CONTRL",
    "HOSP_URCAT4", "HOSP_UR_TEACH", "DMONTH", "cost_2021_usd", "DIED",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def updates_for_rows(rows: int, batch_size: int, grad_accum: int) -> int:
    return math.ceil(math.ceil(rows/batch_size)/grad_accum)


def exposure_plan(ap_rows: int, all_rows: int, batch_size: int = 128,
                  grad_accum: int = 8) -> dict[str, Any]:
    target = updates_for_rows(all_rows, batch_size, grad_accum)
    per_epoch = updates_for_rows(ap_rows, batch_size, grad_accum)
    ratio = target/per_epoch
    candidates = {max(1, math.floor(ratio)), max(1, math.ceil(ratio))}
    # An exact tie chooses the smaller exposure rather than adding avoidable
    # repeated AP passes.
    epochs = min(candidates, key=lambda value: (abs(value*per_epoch-target), value))
    achieved = per_epoch*epochs
    relative_difference = abs(achieved-target)/target
    if relative_difference > 0.01:
        raise RuntimeError("Complete-epoch AP-only exposure cannot match the target within 1%")
    return {"target_joint_updates": target, "ap_updates_per_epoch": per_epoch,
            "ap_epochs": epochs, "ap_planned_updates": achieved,
            "relative_update_difference": relative_difference, "tolerance": 0.01,
            "batch_size": batch_size, "grad_accum": grad_accum,
            "effective_batch_size": batch_size*grad_accum}


def prepare_ap_root(source_root: Path, derived_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq
    nrd_out = derived_root / "data" / "nrd"
    nrd_out.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, Any] = {}; outputs: dict[str, Any] = {}
    for name in ("diagnosis_vocabulary_state.json", "procedure_vocabulary_state.json"):
        source = source_root / "data" / "nrd" / name
        destination = nrd_out / name
        if not destination.exists() or sha256(destination) != sha256(source):
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary); temporary.replace(destination)
        outputs[str(destination.relative_to(derived_root))] = {
            "bytes": destination.stat().st_size, "sha256": sha256(destination)}
    for year in YEARS:
        source = source_root / "data" / "nrd" / f"year={year}" / "ap_episodes.parquet"
        year_dir = nrd_out / f"year={year}"; year_dir.mkdir(parents=True, exist_ok=True)
        destination = year_dir / "admissions_model.parquet"
        table = pq.read_table(source, columns=list(PRETRAIN_COLUMNS),
                              filters=[("primary_analysis_eligible", "=", True),
                                       ("analysis_partition", "=", "development")])
        if table.num_rows != EXPECTED_AP_ROWS[year]:
            raise RuntimeError(f"AP-only row lock mismatch for {year}: {table.num_rows}")
        temporary = destination.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd", compression_level=6,
                       row_group_size=100_000)
        temporary.replace(destination)
        inputs[str(source)] = {"bytes": source.stat().st_size}
        outputs[str(destination.relative_to(derived_root))] = {
            "rows": table.num_rows, "bytes": destination.stat().st_size,
            "sha256": sha256(destination), "columns": table.column_names}
    manifest = {"status": "PASS_AP_DEVELOPMENT_ONLY", "development_years": list(YEARS),
                "selection": "primary_analysis_eligible=true and analysis_partition=development",
                "row_counts": EXPECTED_AP_ROWS, "total_rows": sum(EXPECTED_AP_ROWS.values()),
                "inputs": inputs, "outputs": outputs, "year_2021_accessed": False,
                "year_2022_accessed": False}
    atomic_json(derived_root / "ap_only_data_manifest.json", manifest)
    return manifest


def finalize_manifest(output_dir: Path, data_manifest: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "pretraining_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "PASS" or manifest.get("year_2021_accessed") is not False
            or manifest.get("year_2022_accessed") is not False
            or manifest.get("objectives") != ["masked_code", "masked_hierarchy_category"]
            or int(manifest.get("optimizer_updates_completed", -1)) != plan["ap_planned_updates"]
            or int(manifest.get("all_admission_rows_available", -1)) != data_manifest["total_rows"]):
        raise RuntimeError("AP-only pretraining did not satisfy the locked exposure/data contract")
    checkpoint = output_dir / "pretrained_encoder_final.pt"
    final = manifest.get("final_checkpoint", {})
    if final.get("sha256") != sha256(checkpoint) or int(final.get("bytes", -1)) != checkpoint.stat().st_size:
        raise RuntimeError("AP-only final checkpoint integrity mismatch")
    manifest.update({"pretraining_population": "AP_development_2018_2020",
                     "population_manifest": str((Path(data_manifest["derived_root"])
                                                  / "ap_only_data_manifest.json").resolve()),
                     "exposure_matching": plan,
                     "comparison_role": "AP-only MLM exposure-matched comparator; not the all-admission model",
                     })
    atomic_json(path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--derived-root", type=Path, required=True)
    parser.add_argument("--hierarchy-dir", type=Path, required=True)
    parser.add_argument("--auxiliary-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrain-script", type=Path,
                        default=Path(__file__).with_name("pretrain_claims_transformer.py"))
    args = parser.parse_args()
    source_root, derived_root, output_dir = (args.source_root.resolve(), args.derived_root.resolve(),
                                             args.output_dir.resolve())
    data_manifest = prepare_ap_root(source_root, derived_root)
    data_manifest["derived_root"] = str(derived_root)
    auxiliary = json.loads(args.auxiliary_spec.read_text(encoding="utf-8"))
    plan = exposure_plan(data_manifest["total_rows"], int(auxiliary["all_admission_rows"]))
    existing = output_dir / "pretraining_manifest.json"
    if existing.exists():
        manifest = json.loads(existing.read_text(encoding="utf-8"))
        if manifest.get("pretraining_population") == "AP_development_2018_2020":
            finalize_manifest(output_dir, data_manifest, plan)
            return
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(args.pretrain_script.resolve()), "--root", str(derived_root),
               "--hierarchy-dir", str(args.hierarchy_dir.resolve()), "--output-dir", str(output_dir),
               "--auxiliary-spec", str(args.auxiliary_spec.resolve()), "--epochs", str(plan["ap_epochs"]),
               "--batch-size", str(plan["batch_size"]), "--grad-accum", str(plan["grad_accum"]),
               "--max-tokens", "96", "--workers", "8", "--threads", "8", "--learning-rate", "0.0002",
               "--auxiliary-weight", "0", "--max-steps", "0", "--checkpoint-every", "1000"]
    checkpoint = output_dir / "last_checkpoint.pt"
    if checkpoint.exists(): command.extend(("--resume", str(checkpoint)))
    with (output_dir/"controller.log").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"command": command, "exposure_plan": plan}, ensure_ascii=False) + "\n")
        log.flush(); result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"AP-only pretraining failed with exit {result.returncode}")
    finalize_manifest(output_dir, data_manifest, plan)


if __name__ == "__main__":
    main()
