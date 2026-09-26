from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPACT = Path(os.environ.get("AP_CLAIMS_WORKDIR", str(REPOSITORY_ROOT / "workdir"))) / "compact"
STAGE2 = REPOSITORY_ROOT / "configs" / "stage2_lock"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "workdir" / "stage3_build" / "token_code_sets.json"
YEARS = [2018, 2019, 2020]
RESERVED = {"[PAD]", "[MASK]", "[OOV]", "[MISSING]"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_code(value: object) -> str:
    return str(value).strip().upper().replace(".", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-root", type=Path, default=DEFAULT_COMPACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dx_path = args.compact_root / "diagnosis_vocabulary_state.json"
    pr_path = args.compact_root / "procedure_vocabulary_state.json"
    dx = json.loads(dx_path.read_text(encoding="utf-8"))["token_to_id"]
    pr = json.loads(pr_path.read_text(encoding="utf-8"))["token_to_id"]
    diagnosis_vocab = {normalize_code(code): int(token) for code, token in dx.items() if code not in RESERVED}
    procedure_vocab = {normalize_code(code): int(token) for code, token in pr.items() if code not in RESERVED}

    annual_sets = pd.read_parquet(
        STAGE2 / "planned_readmission" / "annual_algorithm_code_sets.parquet"
    )
    annual_map = pd.read_parquet(
        STAGE2 / "planned_readmission" / "annual_ccs_mapping.parquet"
    )
    labels = pd.read_csv(STAGE2 / "ontology" / "icd_ccsr_labels.csv", dtype=str)
    payload: dict[str, object] = {
        "status": "LOCKED_BEFORE_2022",
        "years": YEARS,
        "reserved_token_ids": {key: int(dx[key]) for key in sorted(RESERVED)},
        "diagnosis_vocabulary_sha256": sha256(dx_path),
        "procedure_vocabulary_sha256": sha256(pr_path),
        "annual": {},
        "labels": {},
    }
    annual: dict[str, object] = {}
    for year in YEARS:
        year_sets: dict[str, object] = {}
        for table in ("PR.1", "PR.2", "PR.3", "PR.4"):
            source = {
                normalize_code(code)
                for code in annual_sets.loc[
                    (annual_sets.calendar_year == year) & (annual_sets.table == table),
                    "code",
                ].astype(str)
            }
            vocab = procedure_vocab if table in {"PR.1", "PR.3"} else diagnosis_vocab
            observed = sorted(vocab[code] for code in source if code in vocab)
            year_sets[table] = {
                "token_ids": observed,
                "source_code_count": len(source),
                "observed_token_count": len(observed),
            }
        cm_mapped = {
            normalize_code(code)
            for code in annual_map.loc[
                (annual_map.calendar_year == year) & (annual_map.code_system == "CM"),
                "code",
            ].astype(str)
        }
        pcs_mapped = {
            normalize_code(code)
            for code in annual_map.loc[
                (annual_map.calendar_year == year) & (annual_map.code_system == "PCS"),
                "code",
            ].astype(str)
        }
        year_sets["mapped_CM_token_ids"] = sorted(
            diagnosis_vocab[code] for code in cm_mapped if code in diagnosis_vocab
        )
        year_sets["mapped_PCS_token_ids"] = sorted(
            procedure_vocab[code] for code in pcs_mapped if code in procedure_vocab
        )
        annual[str(year)] = year_sets
    payload["annual"] = annual

    label_sets: dict[str, object] = {}
    for node, group in labels.groupby("label_node", sort=False):
        codes = {normalize_code(code) for code in group.code.astype(str)}
        label_sets[str(node)] = {
            "token_ids": sorted(diagnosis_vocab[code] for code in codes if code in diagnosis_vocab),
            "source_code_count": len(codes),
        }
    label_sets["principal_or_any_AP_prefix"] = {
        "token_ids": sorted(
            token for code, token in diagnosis_vocab.items() if code.startswith("K85")
        )
    }
    payload["labels"] = label_sets
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "diagnosis_vocab": len(diagnosis_vocab),
                "procedure_vocab": len(procedure_vocab),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
