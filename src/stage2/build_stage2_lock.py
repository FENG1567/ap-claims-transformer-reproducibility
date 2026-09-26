from __future__ import annotations

import hashlib
import csv
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[2]
REFERENCE = ROOT / "work" / "reference"
OUT = ROOT / "outputs" / "stage2_lock"
YEARS = [2018, 2019, 2020, 2021, 2022]

CATALOG = REFERENCE / "official_icd10_catalog_fy2018_2023.parquet"
BILIARY = REFERENCE / "nrd_biliary_codebook_v1.csv"
DX_CCSR = REFERENCE / "DXCCSR_v2021-2.zip"
PR_CCSR = REFERENCE / "PRCCSR_v2021-1.zip"
CMS_PLANNED = REFERENCE / "cms_planned_readmission_algorithm.zip"
PLANNED_DIR = OUT / "planned_readmission"
PLANNED_LOCK = PLANNED_DIR / "planned_readmission_lock.json"

BILIARY_CAUSE_CATEGORIES = {
    "acute_cholecystitis",
    "biliary_obstruction",
    "cholangitis",
    "choledocholithiasis",
    "gallstone_broad",
}
BILIARY_PROCEDURE_CATEGORIES = {
    "cholecystectomy_partial",
    "cholecystectomy_total",
    "ercp_clearance_proxy",
    "ercp_diagnostic",
    "ercp_therapeutic",
    "percutaneous_biliary_drainage",
    "percutaneous_cholecystostomy",
    "surgical_bile_duct_exploration",
    "surgical_cbd_clearance",
}
ORGAN_PREFIXES = {
    "sepsis": ["A40", "A41", "R652"],
    "acute_respiratory_failure": ["J960", "J962"],
    "acute_kidney_failure": ["N17"],
    "shock": ["R57"],
    "disseminated_intravascular_coagulation": ["D65"],
    "acute_hepatic_failure": ["K720", "K711"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().strip("'").strip()


def read_zip_csv(
    path: Path, member_contains: str, repair_trailing_unquoted_commas: bool = False
) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        matches = [n for n in archive.namelist() if member_contains in n and n.lower().endswith(".csv")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {member_contains} CSV in {path}, found {matches}")
        with archive.open(matches[0]) as handle:
            if not repair_trailing_unquoted_commas:
                frame = pd.read_csv(handle, dtype=str, low_memory=False)
            else:
                text = io.TextIOWrapper(handle, encoding="utf-8-sig", newline="")
                reader = csv.reader(text)
                header = next(reader)
                records = []
                for line_number, row in enumerate(reader, start=2):
                    if len(row) < len(header):
                        raise RuntimeError(
                            f"Short PRCCSR row at line {line_number}: {len(row)} fields"
                        )
                    if len(row) > len(header):
                        row = row[: len(header) - 1] + [",".join(row[len(header) - 1 :])]
                    records.append(row)
                frame = pd.DataFrame(records, columns=header)
    frame.columns = [clean(c) for c in frame.columns]
    return frame.map(clean)


def diagnosis_ccsr_map() -> dict[str, dict[str, str]]:
    frame = read_zip_csv(DX_CCSR, "DXCCSR_v2021-2")
    code_col = next(c for c in frame.columns if "ICD-10-CM CODE" == c)
    default_col = next(c for c in frame.columns if c.startswith("Default CCSR CATEGORY IP"))
    default_desc_col = next(
        c for c in frame.columns if c.startswith("Default CCSR CATEGORY DESCRIPTION IP")
    )
    cat_cols = [c for c in frame.columns if c.startswith("CCSR CATEGORY ") and "DESCRIPTION" not in c]
    mapping: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        code = clean(row[code_col]).replace(".", "")
        cats = sorted({clean(row[c]) for c in cat_cols if clean(row[c])})
        mapping[code] = {
            "ccsr_default": clean(row[default_col]),
            "ccsr_default_description": clean(row[default_desc_col]),
            "ccsr_all": ";".join(cats),
        }
    return mapping


def procedure_ccsr_map() -> dict[str, dict[str, str]]:
    frame = read_zip_csv(PR_CCSR, "PRCCSR_v2021-1", repair_trailing_unquoted_commas=True)
    code_col = next(c for c in frame.columns if c.startswith("ICD-10-PCS"))
    cat_col = next(c for c in frame.columns if c == "PRCCSR")
    desc_col = next(c for c in frame.columns if c == "PRCCSR DESCRIPTION")
    return {
        clean(row[code_col]).replace(".", ""): {
            "ccsr_default": clean(row[cat_col]),
            "ccsr_default_description": clean(row[desc_col]),
            "ccsr_all": clean(row[cat_col]),
        }
        for _, row in frame.iterrows()
    }


def validate_codebook(catalog: pd.DataFrame, codebook: pd.DataFrame) -> list[dict[str, object]]:
    valid = {
        (str(row.system), int(row.fiscal_year), str(row.code))
        for row in catalog.itertuples()
        if bool(row.billable) and int(row.fiscal_year) in YEARS
    }
    failures: list[dict[str, object]] = []
    for row in codebook.itertuples():
        claimed = [int(x) for x in str(row.fiscal_years_available).split(";") if x]
        for year in sorted(set(claimed) & set(YEARS)):
            if (str(row.system), year, str(row.code)) not in valid:
                failures.append({"system": row.system, "year": year, "code": row.code})
    return failures


def add_union_rows(
    rows: list[dict[str, object]],
    frame: pd.DataFrame,
    label: str,
    subtype: str,
    priority: int,
    include_for: str,
    ccsr: dict[str, dict[str, str]],
) -> None:
    for code, group in frame.groupby("code", sort=True):
        latest = group.sort_values("fiscal_year").iloc[-1]
        rows.append(
            {
                "label_node": label,
                "subtype": subtype,
                "ontology_role": "cause_label",
                "priority": priority,
                "code_system": latest["system"],
                "code": code,
                "description": latest["long_description"],
                "valid_years": ";".join(str(x) for x in sorted(group["fiscal_year"].unique())),
                "include_for": include_for,
                **ccsr.get(code, {"ccsr_default": "", "ccsr_default_description": "", "ccsr_all": ""}),
                "verification": "present_as_billable_in_official_annual_catalog",
            }
        )


def build_ontology() -> tuple[pd.DataFrame, dict[str, object]]:
    catalog = pd.read_parquet(CATALOG)
    catalog = catalog[
        catalog["fiscal_year"].isin(YEARS) & catalog["billable"].fillna(False)
    ].copy()
    catalog["code"] = catalog["code"].astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
    codebook = pd.read_csv(BILIARY, dtype=str).fillna("")
    failures = validate_codebook(catalog, codebook)
    if failures:
        raise RuntimeError(f"Codebook claims codes absent from official annual catalog: {failures[:20]}")
    dx_map = diagnosis_ccsr_map()
    pr_map = procedure_ccsr_map()
    rows: list[dict[str, object]] = []

    ap = catalog[(catalog.system == "CM") & catalog.code.str.startswith("K85")]
    add_union_rows(rows, ap, "readmission.ap_specific", "acute_pancreatitis", 1, "principal_and_sensitivity", dx_map)

    biliary_codes = set(
        codebook.loc[
            (codebook.system == "CM") & codebook.category.isin(BILIARY_CAUSE_CATEGORIES), "code"
        ]
    )
    biliary = catalog[(catalog.system == "CM") & catalog.code.isin(biliary_codes)]
    for subtype, group in biliary.groupby(
        biliary.code.map(
            codebook[codebook.system == "CM"].drop_duplicates("code").set_index("code")["category"]
        ),
        sort=True,
    ):
        add_union_rows(rows, group, "readmission.biliary_event", str(subtype), 2, "principal_and_sensitivity", dx_map)

    for subtype, prefixes in ORGAN_PREFIXES.items():
        selected = catalog[
            (catalog.system == "CM")
            & catalog.code.map(lambda code: any(code.startswith(prefix) for prefix in prefixes))
        ]
        add_union_rows(
            rows,
            selected,
            "readmission.sepsis_or_acute_organ_dysfunction",
            subtype,
            3,
            "principal_and_sensitivity",
            dx_map,
        )

    procedure_codes = codebook[
        (codebook.system == "PCS") & codebook.category.isin(BILIARY_PROCEDURE_CATEGORIES)
    ]
    catalog_lookup = (
        catalog[catalog.system == "PCS"]
        .sort_values("fiscal_year")
        .groupby("code", sort=False)
        .last()
    )
    for row in procedure_codes.drop_duplicates("code").sort_values(["category", "code"]).itertuples():
        cat_row = catalog_lookup.loc[row.code]
        years = sorted(
            catalog.loc[(catalog.system == "PCS") & (catalog.code == row.code), "fiscal_year"].unique()
        )
        rows.append(
            {
                "label_node": "context.biliary_procedure",
                "subtype": row.category,
                "ontology_role": "procedure_context_only",
                "priority": "",
                "code_system": "PCS",
                "code": row.code,
                "description": cat_row.long_description,
                "valid_years": ";".join(str(x) for x in years),
                "include_for": "feature_and_secondary_overlap_only",
                **pr_map.get(row.code, {"ccsr_default": "", "ccsr_default_description": "", "ccsr_all": ""}),
                "verification": "present_as_billable_in_official_annual_catalog",
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["ontology_role", "priority", "label_node", "subtype", "code_system", "code"],
        na_position="last",
    )
    qc = {
        "rows": len(result),
        "unique_codes": int(result[["code_system", "code"]].drop_duplicates().shape[0]),
        "by_label": result.groupby("label_node").size().to_dict(),
        "missing_default_ccsr": int((result.ccsr_default == "").sum()),
        "codebook_claim_failures": failures,
    }
    return result, qc


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    required = [CATALOG, BILIARY, DX_CCSR, PR_CCSR, CMS_PLANNED, PLANNED_LOCK]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing Stage 2 references: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    ontology_dir = OUT / "ontology"
    ontology_dir.mkdir(parents=True, exist_ok=True)

    labels, ontology_qc = build_ontology()
    planned_lock = json.loads(PLANNED_LOCK.read_text(encoding="utf-8"))
    if planned_lock.get("status") != "PASS" or planned_lock.get("years_locked") != [
        2018,
        2019,
        2020,
        2021,
    ]:
        raise RuntimeError("Planned-readmission lock is not PASS for 2018-2021")
    labels_path = ontology_dir / "icd_ccsr_labels.csv"
    labels.to_csv(labels_path, index=False, lineterminator="\n")

    write_text(
        ontology_dir / "hierarchy.yml",
        """
version: 1.0.0
status: LOCKED_PRE_2022
root: thirty_day_outcome
nodes:
  - id: no_unplanned_readmission
    parent: thirty_day_outcome
    mutually_exclusive_primary: true
  - id: readmission
    parent: thirty_day_outcome
    mutually_exclusive_primary: true
  - id: readmission.ap_specific
    parent: readmission
    mutually_exclusive_primary: true
  - id: readmission.biliary_event
    parent: readmission
    mutually_exclusive_primary: true
  - id: readmission.sepsis_or_acute_organ_dysfunction
    parent: readmission
    mutually_exclusive_primary: true
  - id: readmission.other
    parent: readmission
    mutually_exclusive_primary: true
secondary_multilabel:
  source: all_diagnosis_positions_of_first_qualifying_readmission
  overlap_allowed: true
  never_substitutes_for: principal_diagnosis_mutually_exclusive_primary
closure_rules:
  - readmission_equals_union_of_four_child_causes
  - every_patient_has_exactly_one_primary_leaf
  - every_predicted_risk_set_is_ancestor_closed
""",
    )

    write_text(
        ontology_dir / "priority_rules.yml",
        """
version: 1.0.0
status: LOCKED_PRE_2022
index_episode:
  adult: AGE >= 18
  principal_ap: normalized_I10_DX1 starts_with K85
  alive_discharge: DIED == 0
  observation_window: index_episode_discharge_day <= 335
  cross_year_linkage: forbidden
episode_collapse:
  order: [NRD_VisitLink, NRD_DaysToEvent, KEY_NRD]
  rule: merge overlapping or contiguous transfer-linked records before outcome search
  exact_transfer_value_sets: must_be_resolved_from_year_specific_HCUP_labels_before_stage4
  conservative_fail_rule: unresolved_transfer_labels_blocks_stage4
candidate_readmission:
  interval: 1_through_30_days_after_episode_discharge
  use_first_qualifying_episode_only: true
  planned_status_primary: CMS_Planned_Readmission_Algorithm_v4_annual_alignment
  elective_proxy_is_not_equivalent: true
  unknown_mapping_primary: exclude_and_report
  unknown_mapping_sensitivity: [all_unknown_unplanned, all_unknown_planned]
primary_cause_priority:
  - rank: 1
    node: readmission.ap_specific
    rule: principal_diagnosis_in_locked_code_table
  - rank: 2
    node: readmission.biliary_event
    rule: principal_diagnosis_in_locked_code_table
  - rank: 3
    node: readmission.sepsis_or_acute_organ_dysfunction
    rule: principal_diagnosis_in_locked_code_table
  - rank: 4
    node: readmission.other
    rule: any_other_principal_diagnosis
  - rank: 5
    node: no_unplanned_readmission
    rule: no_qualifying_readmission_within_window
""",
    )

    timing = pd.DataFrame(
        [
            ["any_30d_readmission", "co-primary", "index episode discharge", "days 1-30", "all data available by discharge", "future encounters; post-discharge data", "in-hospital death excludes index; transfer chain collapsed", "exclude discharge after day 335; no cross-year linkage"],
            ["ap_specific_30d_readmission", "co-primary", "index episode discharge", "first qualifying readmission days 1-30", "all data available by discharge", "future encounters; readmission diagnosis", "same as any readmission", "same as any readmission"],
            ["biliary_30d_readmission", "secondary hierarchical", "index episode discharge", "first qualifying readmission days 1-30", "all data available by discharge", "future encounters; readmission diagnosis", "same as any readmission", "same as any readmission"],
            ["sepsis_or_organ_30d_readmission", "secondary hierarchical", "index episode discharge", "first qualifying readmission days 1-30", "all data available by discharge", "future encounters; readmission diagnosis", "same as any readmission", "same as any readmission"],
            ["high_cost", "secondary auxiliary", "admission day 0", "complete index stay", "demographics, prior history, admission source, early diagnosis/procedure only", "TOTCHG; final CCR cost; LOS; discharge disposition; DIED", "death remains an observed cost outcome", "no right censoring after completed discharge"],
            ["prolonged_LOS", "secondary auxiliary", "admission day 0", "complete index stay", "demographics, prior history, admission source, early diagnosis/procedure only", "LOS; late procedures; discharge disposition; DIED; TOTCHG", "in-hospital death reported separately", "completed stays only"],
            ["in_hospital_death", "secondary auxiliary", "admission day 0", "through discharge", "demographics, prior history, admission source, early diagnosis/procedure only", "DIED; LOS; discharge disposition; late procedures; TOTCHG", "none", "completed stays only"],
        ],
        columns=["outcome", "role", "prediction_anchor", "label_window", "allowed_feature_time", "forbidden_post_anchor_information", "competing_event_rule", "right_censoring_rule"],
    )
    timing.to_csv(OUT / "outcome_timing_matrix.csv", index=False, lineterminator="\n")

    write_text(
        OUT / "leakage_blacklist.yml",
        """
version: 1.0.0
status: LOCKED_PRE_2022
global_forbidden:
  - direct_or_derived_future_encounter_labels
  - cross_year_patient_linkage
  - 2022_statistics_in_vocabulary_preprocessing_model_selection_calibration_or_thresholds
  - 2021B_in_model_or_threshold_selection
  - identifiers_except_year_salted_nonreversible_group_hashes
readmission_at_discharge_forbidden:
  - next_admission_date_or_diagnoses_or_procedures
  - label_category_or_time_to_readmission
  - statistics_fit_on_2021B_or_2022
admission_time_auxiliary_forbidden:
  - LOS
  - TOTCHG
  - CCR_adjusted_cost
  - DIED
  - DISPUNIFORM
  - discharge_destination
  - procedures_after_locked_early_window
  - diagnoses_documented_only_after_locked_early_window
split_controls:
  training_and_pretraining: 2018-2020
  model_selection: 2021A_patient_hash_split
  conformal_calibration_only: 2021B_patient_hash_split
  temporal_test_once: 2022
  patient_overlap_across_2021A_2021B: forbidden
""",
    )

    write_text(
        OUT / "decision_log.md",
        f"""
# Stage 2 decision log

Frozen on {date.today().isoformat()} before any 2022 outcome extraction.

1. The active diagnosis and procedure vocabularies are updated only by 2018–2020. Codes first encountered in 2021A or 2021B map to OOV. This is stricter than the original instruction that allowed 2018–2021 vocabulary construction and prevents conformal-calibration information from shaping the representation.
2. AHRQ HCUP states that NRD does not itself distinguish planned from unplanned readmissions, so `ELECTIVE != 1` is not used as the formal definition. The official QualityNet HWR supplements and Yale/CORE-modified CCS maps implement CMS Planned Readmission Algorithm v4.0. Specifications are aligned to each NRD calendar year's two fiscal-year code sets: 2018→2020 reporting specification, 2019→2021, 2020→2022, and 2021→2023. A readmission is planned if any procedure is in PR.1, the principal diagnosis is in PR.2, or any procedure is in PR.3 while the principal diagnosis is not in PR.4. Codes absent from both the year-specific map and direct-code exceptions produce `algorithm_unknown`; they are excluded in the primary analysis and assigned to both extremes in sensitivity bounds.
3. CCSR hierarchy is frozen to Diagnosis CCSR v2021.2 and Procedure CCSR v2021.1, both downloaded from the official AHRQ HCUP archive. A 2021 mapping is used for 2018–2021 development so that the 2022 test-year taxonomy cannot influence representation design. Unmapped historical or new codes receive an explicit missing-hierarchy/OOV parent rather than being treated as clinically negative.
4. Primary mutually exclusive readmission cause uses the principal diagnosis of the first qualifying readmission. An all-diagnosis-position, overlapping multi-label version is sensitivity analysis only.
5. The organ-complication node is deliberately narrow and reproducible: sepsis/severe sepsis, acute respiratory failure, acute kidney failure, shock, disseminated intravascular coagulation, and acute hepatic failure. It is named `sepsis_or_acute_organ_dysfunction`, not a claim of AP causality.
6. Cause priority is AP-specific, biliary, sepsis/acute organ dysfunction, then other. This preserves the clinically central AP recurrence endpoint while keeping every readmission in exactly one primary leaf.
7. 2021A/2021B are split at patient level by a deterministic salted hash after labels are constructed. 2021B is calibration-only and may not select models, thresholds, features, ontologies, vocabularies, or subgroup definitions.
8. The 2022 set remains sealed until Stage 7 verifies all analysis, baseline, model, and code hashes. Structural schema checks do not authorize reading or aggregating 2022 labels.
""",
    )

    write_text(
        OUT / "sap_v1.md",
        """
# Statistical Analysis Plan v1.0

## Objective and estimands

The study evaluates whether a large-scale pretrained claims representation model improves temporally external prediction and calibrated uncertainty over a strong LightGBM baseline after an acute-pancreatitis (AP) hospitalization. The co-primary prediction targets are (i) any qualifying 30-day readmission and (ii) AP-specific 30-day readmission. Cause-specific secondary leaves are biliary events, sepsis or acute organ dysfunction, and other causes.

The primary model contrast is the paired patient-level difference in 2022 AUPRC between the frozen Transformer and frozen LightGBM for each co-primary target. AUROC, Brier score, calibration intercept and slope, log loss, decision-curve net benefit, fixed-capacity detection, conformal coverage, prediction-set size, single-label rate, and abstention rate are required companion measures. An effect is not called clinically useful from discrimination alone.

## Data partitions and nonreuse

- 2018–2020: all-admission self-supervised pretraining; AP supervised development.
- 2021A: patient-disjoint model selection, early stopping, hyperparameters, probability calibration choice, and fixed operating point.
- 2021B: split-conformal nonconformity calibration only; no model, threshold, feature, ontology, vocabulary, or subgroup selection.
- 2022: one-time temporal test after hash verification of every upstream lock.
- MIMIC-IV 3.1: transportability analysis using common variables only; its endpoint is `same-system readmission`.

Patients are linkable only within an NRD calendar year. No cross-year patient identity is inferred. The 2021A/2021B split is patient-level and deterministic; balance checks are diagnostic and cannot trigger outcome-guided reassignment.

## Cohort and episodes

The primary index cohort comprises age ≥18 years, principal ICD-10-CM diagnosis beginning K85, alive discharge, valid `NRD_VisitLink` and `NRD_DaysToEvent`, and a full observable 30-day window within the calendar year. AP in any diagnosis position is a sensitivity cohort. Overlapping or contiguous transfer-linked records are collapsed before outcome search. The exact year-specific transfer value labels are a Stage 4 implementation gate and cannot be guessed.

The candidate outcome is the first qualifying inpatient episode beginning 1–30 days after index-episode discharge. Planned admissions are removed using the frozen, annual-aligned CMS Planned Readmission Algorithm v4.0. The algorithm uses any procedure position for PR.1/PR.3 and the principal diagnosis only for PR.2/PR.4. Any `algorithm_unknown` episode is excluded from the primary analysis and included in worst-case sensitivity bounds; `ELECTIVE` is reported only as a separate proxy sensitivity.

## Label hierarchy

Every index episode receives one primary leaf: no qualifying readmission; AP-specific; biliary; sepsis or acute organ dysfunction; or other. Cause classification is based on the next admission's principal diagnosis and the locked code table. Priority resolves overlaps in that order. The secondary multi-label sensitivity analysis scans all diagnosis positions and explicitly permits overlap. Parent `readmission` must equal the union of its four primary cause leaves.

## Features and timing

Readmission prediction is anchored at index discharge and may use only information available by that time, with causal masking across prior encounters. Cost, prolonged LOS, and in-hospital death are admission-time auxiliary tasks and exclude final LOS, total charges, final cost, death, discharge disposition, and any late diagnosis/procedure. All preprocessing, imputation, feature filtering, scaling, calibration, and representation learning are fit on their authorized partitions only.

## Baselines and model selection

Required baselines are a prespecified structured logistic model, sparse elastic-net logistic regression, and LightGBM as the strong primary comparator. Class-prevalence-changing resampling such as SMOTE is prohibited for probability models. LightGBM receives bounded tuning on 2018–2020 with selection on 2021A. The Transformer must include parameter-matched and component-removal ablations: no all-admission pretraining, AP-only, no hierarchy, no PRDAY, no prior encounters, no hospital/socioeconomic context, no year/version token, and common-variable-only.

## Statistical inference

All 2022 model contrasts use paired patient bootstrap resampling (1,000 replicates) with percentile 95% confidence intervals; the patient is the resampling unit. Hospital-cluster bootstrap is a prespecified sensitivity analysis. Both unweighted and `DISCWT`-weighted estimates are reported and never pooled. AUPRC is accompanied by outcome prevalence. Secondary comparisons use Benjamini–Hochberg FDR at q=0.05. Effect sizes and uncertainty take precedence over p-values.

## Conformal prediction and abstention

Split conformal calibration uses 2021B only. The primary nominal coverage is 90%; 80% and 95% are prespecified sensitivity levels. Hierarchical risk sets must be ancestor-closed. A prediction abstains when the locked risk set is not a singleton or meets the frozen conflict/uncertainty rule. Report marginal coverage, subgroup coverage with confidence intervals, mean/median set size, singleton/empty/full-set rates, abstention rate, and outcome/error rates among retained and abstained patients.

Mondrian groups are sex, age group, primary payer, and ZIP income quartile. A group requires ≥100 target events for confirmatory coverage statements; 50–99 events are exploratory; <50 events receive descriptive estimates only. The same gate applies to intersections. No subgroup is dropped because its coverage is poor.

## Drift

Compare 2018–2020, 2021A, 2021B, and 2022 in covariate frequencies, code and procedure distributions, OOV rates, hospital structure, payer, ZIP income, cost, label prevalence, calibration, and conformal coverage. Separate covariate, label, calibration, and coverage drift. Drift analyses explain performance but cannot revise the locked primary test.

## Missingness and sensitivity analyses

Missing administrative values are represented explicitly and are not converted to clinical absence. Prespecified sensitivities are: AP in any diagnosis position; planned-status exclusion versus no exclusion; exact 30-day observability versus December exclusion; alternative transfer-contiguity tolerance; time-ordered versus hash 2021 split; weighted versus unweighted estimates; and common-variable-only modeling. Any analysis motivated after 2022 inspection is labeled post hoc and retained in a registry.

## Reporting and reproducibility

Report per TRIPOD+AI and use PROBAST+AI as a risk-of-bias audit. All metrics must be reproducible from frozen prediction tables. Code, configuration, ontology, synthetic examples, and aggregate results may be shared; patient-level data, embeddings, and model weights remain restricted unless the HCUP and PhysioNet agreements explicitly permit release.
""",
    )

    core_artifacts = [
        labels_path,
        ontology_dir / "hierarchy.yml",
        ontology_dir / "priority_rules.yml",
        OUT / "sap_v1.md",
        OUT / "outcome_timing_matrix.csv",
        OUT / "leakage_blacklist.yml",
        OUT / "decision_log.md",
        PLANNED_DIR / "planned_readmission_lock.json",
        PLANNED_DIR / "annual_spec_rows.csv",
        PLANNED_DIR / "annual_ccs_mapping.parquet",
        PLANNED_DIR / "annual_algorithm_code_sets.parquet",
        PLANNED_DIR / "algorithm_rules.yml",
    ]
    sources = {
        str(path.relative_to(ROOT)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in required
    }
    lock = {
        "lock_version": "1.0.0",
        "status": "LOCKED_PRE_2022",
        "created_date": date.today().isoformat(),
        "sealed_test_year": 2022,
        "development_years": [2018, 2019, 2020],
        "model_selection_partition": "2021A",
        "conformal_only_partition": "2021B",
        "active_vocabulary_update_years": [2018, 2019, 2020],
        "ccsr_versions": {"diagnosis": "v2021.2", "procedure": "v2021.1"},
        "planned_readmission_gate": "PASS_CMS_PRA_V4_ANNUAL_ALIGNMENT_2018_2021",
        "planned_readmission_lock_sha256": sha256(PLANNED_LOCK),
        "fallback_endpoint_name": "30-day nonelective readmission",
        "ontology_qc": ontology_qc,
        "sources": sources,
        "artifacts": {
            str(path.relative_to(OUT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in core_artifacts
        },
    }
    (OUT / "analysis_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_text(
        OUT / "stage2_gate.md",
        f"""
# Stage 2 gate

- Status: **PASS**
- Candidate: `analysis_lock.json` SHA256 `{sha256(OUT / 'analysis_lock.json')}`
- Ontology rows: {ontology_qc['rows']}; unique ICD codes: {ontology_qc['unique_codes']}
- Official-codebook claim failures: {len(ontology_qc['codebook_claim_failures'])}
- Missing default CCSR mappings: {ontology_qc['missing_default_ccsr']} (explicit hierarchy-missing token required)
- Planned-readmission lock: CMS PRA v4.0, annual-aligned 2018–2021, SHA256 `{sha256(PLANNED_LOCK)}`
- 2022 outcome access: none

The ICD/CCSR ontology, hierarchy, timing matrix, leakage blacklist, SAP, decision log, and annual-aligned CMS Planned Readmission Algorithm v4.0 code sets are frozen. All valid 2018–2021 ICD-10-CM/PCS catalog codes are covered by the corresponding Yale/CORE-modified CCS maps. The main outcome may be named 30-day unplanned readmission when the implementation tests and observed-code mapping gate pass. The 2022 test outcome remains sealed; its predeclared annual code alignment is implemented only at Stage 7.
""",
    )
    print(json.dumps({"status": "PASS", "out": str(OUT), "qc": ontology_qc}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
