# IJMI revision v4 analysis workspace

This directory contains additive, post-unblinding analyses requested during
pre-submission review.  The scripts do not overwrite the frozen Stage 4-7 or
MIMIC artifacts.  They only read authenticated outputs and create a new,
versioned results directory.

Scientific boundaries:

- 2018-2020 remain development data.
- 2021A remains the model-selection, calibration, and threshold partition.
- 2021B remains conformal-calibration only.
- 2022 and MIMIC remain evaluation-only and cannot select or modify models,
  features, thresholds, ontology, endpoint mappings, or repair rules.
- The 2022 correction is described as a deterministic post-unblinding
  technical amendment, not as an untouched sealed test.
- MIMIC is a restricted same-system transfer evaluation, not complete external
  validation of all-cause readmission.

