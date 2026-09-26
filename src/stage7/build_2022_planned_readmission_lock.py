#!/usr/bin/env python3
"""Freeze the CMS 2024 planned-readmission code sets for NRD calendar year 2022.

This pre-test builder consumes only public CMS/QualityNet workbooks and the
official FY2022/FY2023 ICD-10 catalog.  It never accepts an NRD data path.  The
2018--2021 Stage 2 lock is deliberately left untouched; the output is a new,
immutable, calendar-year-2022 lock used by the Stage 7 ETL specification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from work.stage2 import build_planned_readmission_lock as stage2


REFERENCE = ROOT / "work" / "reference"
DEFAULT_SPECIFICATION = (
    REFERENCE
    / "CMS_2024_HWR_methodology"
    / "2024_Rdmsn_MeasureMethodology"
    / "2024_HWR_v1.0.xlsx"
)
DEFAULT_MAPPING = (
    REFERENCE
    / "CMS_2024_HWR_resources"
    / "YaleMod_CCS_PCS&CM_Map_for_2024_Reporting_External_Final_10.23.23.xlsx"
)
DEFAULT_CATALOG = REFERENCE / "official_icd10_catalog_fy2018_2023.parquet"
DEFAULT_METHODOLOGY_ZIP = REFERENCE / "2024_Rdmsn_MeasureMethodology.zip"
DEFAULT_RESOURCES_ZIP = REFERENCE / "2024_ArchivedResources_Rdmsn.zip"
DEFAULT_OUTPUT = ROOT / "outputs" / "stage7_pre2022_lock" / "planned_readmission_2022"

SOURCE = stage2.AnnualSource(
    calendar_year=2022,
    reporting_year=2024,
    fiscal_years=(2022, 2023),
    specification=DEFAULT_SPECIFICATION,
    mapping=DEFAULT_MAPPING,
)
SCHEMA_VERSION = "stage7_pra_calendar_2022_v1"
QUALITYNET_PAGE = "https://qualitynet.cms.gov/inpatient/measures/readmission/resources"
QUALITYNET_FILES = {
    "methodology": {
        "file_id": "682b4b3b662c6b68b52cc8fb",
        "filename": "2024_Rdmsn_MeasureMethodology.zip",
        "api_url": "https://qualitynet.cms.gov/publicgateway/public/files/682b4b3b662c6b68b52cc8fb",
    },
    "resources": {
        "file_id": "685314437ed210cf2a298433",
        "filename": "2024_ArchivedResources_Rdmsn.zip",
        "api_url": "https://qualitynet.cms.gov/publicgateway/public/files/685314437ed210cf2a298433",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing required public reference: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def artifact_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing generated 2022 PRA artifact: {path}")
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_valid_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_parquet(path)
    required = {"fiscal_year", "system", "code", "billable"}
    if missing := required - set(catalog.columns):
        raise RuntimeError(f"ICD catalog lacks required fields: {sorted(missing)}")
    catalog = catalog[catalog["billable"].fillna(False)].copy()
    catalog["code_system"] = catalog["system"].astype(str)
    catalog["code"] = catalog["code"].map(stage2.normalize_code)
    return catalog[["fiscal_year", "code_system", "code"]].drop_duplicates()


def expand_code_sets(
    source: stage2.AnnualSource,
    specifications: pd.DataFrame,
    mappings: pd.DataFrame,
    valid_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    valid = valid_catalog.loc[
        valid_catalog["fiscal_year"].isin(source.fiscal_years),
        ["code_system", "code"],
    ].drop_duplicates()
    valid_sets = {
        system: set(group["code"])
        for system, group in valid.groupby("code_system", sort=False)
    }
    mapping_sets = {
        system: set(group["code"])
        for system, group in mappings.groupby("code_system", sort=False)
    }
    active = specifications[specifications["active"]].copy()
    output: list[pd.DataFrame] = []
    table_qc: dict[str, Any] = {}
    for table, metadata in stage2.TABLES.items():
        system = str(metadata["system"])
        table_spec = active[active["table"] == table]
        categories = set(
            table_spec.loc[table_spec["record_type"] == "CCS", "value"].astype(str)
        )
        direct_type = "ICD-10-CM" if system == "CM" else "ICD-10-PCS"
        direct = set(
            table_spec.loc[table_spec["record_type"] == direct_type, "value"].astype(str)
        )
        expanded = set(
            mappings.loc[
                (mappings["code_system"] == system)
                & mappings["ccs_category"].isin(categories),
                "code",
            ]
        )
        valid_codes = (expanded | direct) & valid_sets.get(system, set())
        invalid_direct = direct - valid_sets.get(system, set())
        output.append(
            pd.DataFrame(
                {
                    "calendar_year": source.calendar_year,
                    "reporting_year": source.reporting_year,
                    "table": table,
                    "code_system": system,
                    "code": sorted(valid_codes),
                }
            )
        )
        table_qc[table] = {
            "active_ccs_categories": len(categories),
            "active_direct_codes": len(direct),
            "expanded_category_codes": len(expanded),
            "final_valid_codes": len(valid_codes),
            "direct_codes_invalid_for_calendar_year": len(invalid_direct),
            "direct_invalid_examples": sorted(invalid_direct)[:20],
        }
    code_sets = pd.concat(output, ignore_index=True)
    if code_sets.duplicated(["calendar_year", "table", "code_system", "code"]).any():
        raise RuntimeError("Expanded 2022 PRA code sets contain duplicates")
    qc = {
        "calendar_year": source.calendar_year,
        "reporting_year": source.reporting_year,
        "fiscal_years": list(source.fiscal_years),
        "status_rule": "active unless normalized status starts with remove",
        "valid_catalog_codes": {
            system: len(codes) for system, codes in sorted(valid_sets.items())
        },
        "mapping_codes": {
            system: len(codes) for system, codes in sorted(mapping_sets.items())
        },
        "valid_codes_missing_from_mapping": {
            system: len(valid_sets.get(system, set()) - mapping_sets.get(system, set()))
            for system in ("CM", "PCS")
        },
        "tables": table_qc,
    }
    if any(qc["valid_codes_missing_from_mapping"].values()):
        raise RuntimeError("The 2024 Yale mapping does not cover the FY2022/FY2023 catalog")
    if any(not details["final_valid_codes"] for details in table_qc.values()):
        raise RuntimeError("One or more 2022 PRA tables expanded to an empty code set")
    return code_sets, qc


def validate_lock(path: Path) -> dict[str, Any]:
    path = path.resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError("Missing 2022 PRA lock or SHA256 sidecar")
    if sidecar.read_text(encoding="ascii").strip() != f"{sha256(path)}  {path.name}":
        raise RuntimeError("2022 PRA lock SHA256 sidecar mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("status") != "PASS_2022_PRA_LOCKED_PRE_TEST_ACCESS"
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("calendar_year") != 2022
        or value.get("reporting_year") != 2024
        or value.get("fiscal_years") != [2022, 2023]
        or value.get("nrd_2022_accessed") is not False
    ):
        raise RuntimeError("Invalid 2022 PRA lock seal")
    for entry in value.get("sources", {}).values():
        target = Path(entry["path"])
        if not target.is_file() or target.stat().st_size != int(entry["bytes"]) or sha256(target) != entry["sha256"]:
            raise RuntimeError(f"2022 PRA lock identity mismatch: {target}")
    for entry in value.get("artifacts", {}).values():
        target = path.parent / entry["file"]
        if not target.is_file() or target.stat().st_size != int(entry["bytes"]) or sha256(target) != entry["sha256"]:
            raise RuntimeError(f"2022 PRA artifact identity mismatch: {target}")
    return value


def build_lock(
    output_dir: Path,
    *,
    specification: Path = DEFAULT_SPECIFICATION,
    mapping: Path = DEFAULT_MAPPING,
    catalog: Path = DEFAULT_CATALOG,
    methodology_zip: Path = DEFAULT_METHODOLOGY_ZIP,
    resources_zip: Path = DEFAULT_RESOURCES_ZIP,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    partials = list(output_dir.parent.glob(output_dir.name + ".partial*")) if output_dir.parent.exists() else []
    if output_dir.exists() or partials:
        raise RuntimeError("2022 PRA output or partial output already exists and is immutable")
    required = [specification, mapping, catalog, methodology_zip, resources_zip, Path(stage2.__file__), Path(__file__)]
    for path in required:
        identity(path)
    source = stage2.AnnualSource(2022, 2024, (2022, 2023), specification, mapping)
    specifications = stage2.read_specification(source)
    mappings = stage2.read_mapping(source)
    code_sets, qc = expand_code_sets(source, specifications, mappings, load_valid_catalog(catalog))

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".partial-", dir=output_dir.parent))
    try:
        artifacts = {
            "specification_rows": temporary / "annual_spec_rows_2022.csv",
            "ccs_mapping": temporary / "annual_ccs_mapping_2022.parquet",
            "algorithm_code_sets": temporary / "annual_algorithm_code_sets_2022.parquet",
            "algorithm_rules": temporary / "algorithm_rules_2022.yml",
        }
        specifications.to_csv(artifacts["specification_rows"], index=False, lineterminator="\n")
        mappings.to_parquet(artifacts["ccs_mapping"], index=False, compression="zstd")
        code_sets.to_parquet(artifacts["algorithm_code_sets"], index=False, compression="zstd")
        artifacts["algorithm_rules"].write_text(
            "version: CMS_PRA_v4.0_calendar_2022_reporting_2024\n"
            "status: LOCKED_PRE_2022_TEST_ACCESS\n"
            "planned_if:\n"
            "  - any_procedure_in_PR1\n"
            "  - principal_diagnosis_in_PR2\n"
            "  - any_procedure_in_PR3 AND principal_diagnosis_not_in_PR4\n"
            "unknown_policy: exclude_when_unknown_precedes_or_ties_first_unplanned_readmission\n"
            "fiscal_years: [2022, 2023]\n",
            encoding="utf-8",
        )
        source_paths = {
            "cms_hwr_2024_specification": specification,
            "yale_modified_ccs_2024_mapping": mapping,
            "official_icd10_fy2018_2023_catalog": catalog,
            "qualitynet_2024_methodology_archive": methodology_zip,
            "qualitynet_2024_resources_archive": resources_zip,
            "stage2_parser_code": Path(stage2.__file__),
            "stage7_builder_code": Path(__file__),
        }
        lock_path = temporary / "planned_readmission_2022_lock.json"
        lock = {
            "status": "PASS_2022_PRA_LOCKED_PRE_TEST_ACCESS",
            "schema_version": SCHEMA_VERSION,
            "algorithm": "CMS Planned Readmission Algorithm Version 4.0",
            "calendar_year": 2022,
            "reporting_year": 2024,
            "fiscal_years": [2022, 2023],
            "nrd_2022_accessed": False,
            "qualitynet_page": QUALITYNET_PAGE,
            "qualitynet_files": QUALITYNET_FILES,
            "qc": qc,
            "sources": {name: identity(path) for name, path in source_paths.items()},
            "artifacts": {name: artifact_identity(path) for name, path in artifacts.items()},
        }
        lock_path.write_text(stable_json(lock) + "\n", encoding="utf-8")
        digest = sha256(lock_path)
        (temporary / "planned_readmission_2022_lock.json.sha256").write_text(
            f"{digest}  planned_readmission_2022_lock.json\n", encoding="ascii"
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validated = validate_lock(output_dir / "planned_readmission_2022_lock.json")
    return {
        "status": validated["status"],
        "specification_rows": len(specifications),
        "mapping_rows": len(mappings),
        "algorithm_code_rows": len(code_sets),
        "lock_sha256": sha256(output_dir / "planned_readmission_2022_lock.json"),
        "output": str(output_dir),
        "qc": qc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--specification", type=Path, default=DEFAULT_SPECIFICATION)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--methodology-zip", type=Path, default=DEFAULT_METHODOLOGY_ZIP)
    parser.add_argument("--resources-zip", type=Path, default=DEFAULT_RESOURCES_ZIP)
    args = parser.parse_args()
    result = build_lock(
        args.output_dir,
        specification=args.specification,
        mapping=args.mapping,
        catalog=args.catalog,
        methodology_zip=args.methodology_zip,
        resources_zip=args.resources_zip,
    )
    print(stable_json(result))


if __name__ == "__main__":
    main()
