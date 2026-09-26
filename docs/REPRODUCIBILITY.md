# Reproducibility workflow

## Public checks (no controlled data)

1. Create the environment from `environment.yml` or install `requirements.txt`.
2. Run `python scripts/verify_release.py`.
3. Run `pytest -q tests/test_release_smoke.py`.

These steps validate the release inventory.

## Full scientific workflow

The scripts preserve the original stage structure.

1. `src/local_pipeline/`: ingest licensed claims files, sanitize admissions, derive exact flags, construct AP episodes, and build token/code sets.
2. `src/stage2/`: freeze ontology, leakage blacklist, planned-readmission rules, endpoint timing, and SAP.
3. `src/stage4/`: build history features and train structured baselines.
4. `src/stage5/`: pretrain the claims Transformer, fine-tune frozen candidates, select on 2021A, and freeze calibration/operating points.
5. `src/stage6/`: predict 2021B and fit conformal thresholds only.
6. `src/stage7/`: freeze the pre-2022 lock, construct/evaluate the 2022 cohort once, and produce the public aggregate export. The final release retains documented technical-amendment code because the corrected 2022 evaluation is explicitly reported as a deterministic post-unblinding amendment.
7. `src/mimic_transfer/`: build a hash-bound execution specification, run `--validate-only` before opening MIMIC-IV, then execute the formal v7 transfer using the v2 adapter and 1,000 subject-cluster bootstrap replicates.
8. `src/revision_analyses/`: run additive 2022 calibration, baseline-comparison, and transfer analyses without changing the frozen model.

Each original script exposes its command-line contract with `--help`. Paths in the public release must be supplied by the runner; no repository file contains the original workstation or server location.

## Frozen scientific rules

- 2018-2020 development; 2021A selection/calibration; 2021B conformal only; 2022 and MIMIC evaluation only.
- No 2022 or MIMIC tuning, feature selection, vocabulary growth, recalibration, threshold optimization, ontology change, endpoint remapping, or repair-rule selection.
- The primary 2022 comparison uses the survey-weighted AUPRC and paired 1,000-replicate patient- and hospital-cluster bootstrap intervals.
- The formal MIMIC transfer uses 1,000 subject-cluster bootstrap replicates.
- Five-year planned-readmission consensus, `algorithm_unknown` precedence, event-count gates, leakage checks, OOV retention, and public suppression are mandatory.

## Determinism and expected differences

Random seeds are fixed inside the formal analysis scripts and manifests. GPU kernels, library builds, and graphics backends can still introduce small floating-point or rendering differences. Verify scientific outputs at the reported numeric precision and use `manifests/SHA256SUMS.txt` only for the released files themselves.

