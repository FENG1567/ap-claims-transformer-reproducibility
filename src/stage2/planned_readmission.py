from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def normalize_code(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper().replace(".", "")


@dataclass(frozen=True)
class PlannedReadmissionResult:
    status: str
    planned: bool | None
    pr1: bool
    pr2: bool
    pr3: bool
    pr4: bool
    unknown_principal_diagnosis: bool
    unknown_procedure_count: int


class PlannedReadmissionClassifier:
    def __init__(self, code_sets_path: Path, mapping_path: Path):
        code_sets = pd.read_parquet(code_sets_path)
        mappings = pd.read_parquet(mapping_path)
        self._years = sorted(int(year) for year in code_sets.calendar_year.unique())
        self._sets: dict[tuple[int, str], set[str]] = {
            (int(year), str(table)): set(group.code.astype(str))
            for (year, table), group in code_sets.groupby(
                ["calendar_year", "table"], sort=False
            )
        }
        self._mapped: dict[tuple[int, str], set[str]] = {
            (int(year), str(system)): set(group.code.astype(str))
            for (year, system), group in mappings.groupby(
                ["calendar_year", "code_system"], sort=False
            )
        }
        self._direct: dict[tuple[int, str], set[str]] = {}

    @classmethod
    def from_stage2_lock(cls, stage2_dir: Path) -> "PlannedReadmissionClassifier":
        directory = stage2_dir / "planned_readmission"
        return cls(
            directory / "annual_algorithm_code_sets.parquet",
            directory / "annual_ccs_mapping.parquet",
        )

    def classify(
        self,
        calendar_year: int,
        principal_diagnosis: object,
        procedures: list[object] | tuple[object, ...],
    ) -> PlannedReadmissionResult:
        if calendar_year not in self._years:
            raise ValueError(f"calendar_year must be one of {self._years}")
        diagnosis = normalize_code(principal_diagnosis)
        procedure_codes = [normalize_code(code) for code in procedures]
        procedure_codes = [code for code in procedure_codes if code]

        pr1_set = self._sets[(calendar_year, "PR.1")]
        pr2_set = self._sets[(calendar_year, "PR.2")]
        pr3_set = self._sets[(calendar_year, "PR.3")]
        pr4_set = self._sets[(calendar_year, "PR.4")]
        cm_map = self._mapped[(calendar_year, "CM")]
        pcs_map = self._mapped[(calendar_year, "PCS")]

        unknown_diagnosis = bool(diagnosis) and diagnosis not in cm_map
        unknown_procedures = sum(code not in pcs_map for code in procedure_codes)
        pr1 = any(code in pr1_set for code in procedure_codes)
        pr2 = diagnosis in pr2_set
        pr3 = any(code in pr3_set for code in procedure_codes)
        pr4 = diagnosis in pr4_set

        if not diagnosis or unknown_diagnosis or unknown_procedures:
            return PlannedReadmissionResult(
                status="algorithm_unknown",
                planned=None,
                pr1=pr1,
                pr2=pr2,
                pr3=pr3,
                pr4=pr4,
                unknown_principal_diagnosis=not diagnosis or unknown_diagnosis,
                unknown_procedure_count=unknown_procedures,
            )
        planned = pr1 or pr2 or (pr3 and not pr4)
        return PlannedReadmissionResult(
            status="planned" if planned else "unplanned",
            planned=planned,
            pr1=pr1,
            pr2=pr2,
            pr3=pr3,
            pr4=pr4,
            unknown_principal_diagnosis=False,
            unknown_procedure_count=0,
        )
