# Immutable input provenance

## Formal result inputs (read-only)

| Input directory | Role in candidate package | Permitted content |
|---|---|---|
| `outputs/ijmi_revision_v4_20260921` | Frozen 2022 temporal reanalysis | Aggregate 2022 cohort, absolute performance, paired inference, subgroup, threshold, ablation and calibration-curve data |
| `outputs/ijmi_revision_v4_20260921_calibration_ci_v1` | 2022 calibration intervals | Aggregate calibration intercept/slope bootstrap intervals |
| `outputs/ijmi_common_baselines_v2_20260921` | Fair common-variable comparator provenance | Training-period-only baseline provenance and aggregate artefacts |
| `outputs/ijmi_revision_v4_mimic_20260921_v4` | Restricted same-system MIMIC transfer | Aggregate, suppression-audited MIMIC outcomes only |
| `outputs/ijmi_mimic_common_baseline_comparison_v1_20260921` | Frozen MIMIC four-model comparison | Aggregate paired bootstrap differences and model-binding audit |

## Read-only legacy context

`outputs/paper_v3_upgrade_20260921/release_package_final_v2` is used only to retain non-numeric cohort/method wording where it does not conflict with the formal reanalyses. Its earlier figures, numbers, conclusions, and source tables are not reused unless independently traced to the formal input set above.

## Exclusions

- No MIMIC row-level files, patient-level identifiers, or unsuppressed small cells.
- No partial, failed, historical runtime, or old execution-lock output as a numerical source.
- No data or results from 2022 or MIMIC are used to update a fitted model, feature, threshold, calibration rule, ontology, mapping, or correction rule.

## Candidate identity

Candidate root: `outputs/ijmi_submission_revision_v4_20260921_candidate_v1`.

The final `MANIFEST_SHA256.tsv` binds every released artifact to this candidate.
