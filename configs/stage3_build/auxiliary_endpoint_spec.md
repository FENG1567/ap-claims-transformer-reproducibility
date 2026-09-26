# Auxiliary endpoint and cost specification

Status: `LOCKED_BEFORE_2022`


Access date: 2026-09-12

## Cost construction

HCUP states that discharge-level inpatient cost is estimated by multiplying
`TOTCHG` by the year-matched hospital cost-to-charge ratio. Therefore:

```text
nominal_cost = TOTCHG * CCR_NRD
```

The join is within year on `HOSP_NRD`. Invalid/nonpositive charges, absent CCR,
or nonpositive CCR produce a missing cost label; they are not imputed for the
high-cost outcome. Linkage coverage and excluded counts are reported by year.

Nominal costs are converted to constant 2021 US dollars using the annual
average CPI-U all-items series `CUUR0000SA0`:

```text
cost_2021_usd = nominal_cost * 270.970 / annual_CPI_U
```

Locked annual averages: 2018 `251.107`; 2019 `255.657`; 2020 `258.811`; 2021
`270.970`; 2022 `292.655`.

Authoritative sources:

- HCUP inpatient CCR documentation:
  https://hcup-us.ahrq.gov/db/ccr/ip-ccr/ip-ccr.jsp
- US Bureau of Labor Statistics public API, series `CUUR0000SA0`:
  https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0?startyear=2018&endyear=2022&annualaverage=true

## Locked labels

- `high_cost`: `cost_2021_usd` above the `DISCWT`-weighted 90th percentile
  estimated once among eligible 2018–2020 principal-AP index stays with valid
  cost. The resulting dollar cutoff is frozen before 2021 evaluation and 2022
  access. An unweighted development-set 90th percentile is a sensitivity
  definition only.
- `prolonged_LOS`: `LOS > 7` days. A secondary sensitivity uses the fixed
  2018–2020 `DISCWT`-weighted 90th percentile of LOS. Negative or missing-coded
  LOS is excluded for this task.
- `in_hospital_death`: `DIED=1`; `DIED=0` is the negative class. Missing or
  invalid DIED is excluded for this task.

The high-cost cutoff and LOS sensitivity cutoff must be written with their
training sample denominators, event rates, and SHA256-bound input manifest.
Neither cutoff may be recomputed from 2021A, 2021B, or 2022.

## Time-zero feature restriction

These auxiliary outcomes are prediction/representation-learning tasks anchored
at admission. Because NRD diagnosis codes do not contain a diagnosis timestamp
and the principal diagnosis is assigned after the stay, no current-stay
diagnosis code is treated as prospectively known at admission. The admissible
feature set is limited to:

- demographics, payer, ZIP-income quartile, resident status;
- admission month, weekend/elective and emergency-department indicators;
- hospital structural variables;
- prior encounters ending before the current admission;
- procedures with `PRDAY <= 0` only, with invalid/missing timing kept separate.

Current-stay diagnoses, final LOS, charges/cost, death, discharge disposition,
and procedures after day 0 are masked. Results from these tasks are described
as auxiliary representation-learning performance, not as validated bedside
admission models. Readmission models remain anchored at AP index discharge and
may use information available by discharge.
