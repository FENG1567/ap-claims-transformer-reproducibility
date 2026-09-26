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
