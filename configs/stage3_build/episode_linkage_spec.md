# NRD episode-linkage and 30-day observability implementation specification

Status: `LOCKED_BEFORE_2022`


Access date: 2026-09-12

## Authoritative source findings

The HCUP NRD data-element documentation states that identifiable transfers and
same-day stays have already been collapsed into one combined NRD record. The
`SAMEDAYEVENT` values describe the combined record:

- `0`: not a transfer or other same-day stay;
- `1`: transfer involving two discharges from different hospitals;
- `2`: same-day stay involving two discharges from different hospitals;
- `3`: same-day stay involving two discharges at the same hospital;
- `4`: same-day stay involving three or more discharges.

`REHABTRANSFER=1` identifies an already-combined record involving
rehabilitation, evaluation, or other aftercare. Beginning in data year 2018,
HCUP identifies this from specified rehabilitation discharge dispositions or
principal CCSR `FAC010` in a later component record.

`DISPUNIFORM=2` means transfer to a short-term hospital. `DISPUNIFORM=20` means
in-hospital death. The other uniform values used by this project are `1`
(routine), `5` (other facility), `6` (home health), `7` (against medical
advice), `21` (court/law enforcement), and `99` (alive, destination unknown).

`NRD_DaysToEvent` is admission timing relative to a patient-specific random
start date and is comparable only within the same `NRD_VisitLink`. HCUP's
documented discharge-to-next-admission interval is:

```text
gap_days = next.NRD_DaysToEvent - index.NRD_DaysToEvent - index.LOS
```

Sources:

- https://hcup-us.ahrq.gov/db/vars/samedayevent/nrdnote.jsp
- https://hcup-us.ahrq.gov/db/vars/rehabtransfer/nrdnote.jsp
- https://hcup-us.ahrq.gov/db/vars/dispuniform/nrdnote.jsp
- https://hcup-us.ahrq.gov/db/vars/nrd_daystoevent/nrdnote.jsp

## Locked operational rules

1. Do not join separate rows merely because `SAMEDAYEVENT` or
   `REHABTRANSFER` is nonzero. These variables describe combinations already
   performed by HCUP. This corrects the ambiguous phrase "collapse transfer
   chains" in SAP v1 without examining 2022 outcomes.
2. Sort records only within `year + patient_hash` by `NRD_DaysToEvent`, then
   `encounter_hash` for deterministic ties. Never compare
   `NRD_DaysToEvent` across patients or years.
3. A candidate readmission must be the first later row with `1 <= gap_days <=
   30`. `gap_days=0` is a same-day/contiguous event and is not a qualifying
   readmission. `gap_days<0` is an overlap or inconsistent sequence; flag it,
   exclude the affected index from the primary analysis, and report a
   sensitivity analysis.
4. Exclude an index with `DISPUNIFORM=2` from the primary cohort because its
   terminal acute-care discharge is not reliably observed when a receiving
   record was not successfully combined. Retain it in a prespecified
   sensitivity analysis.
5. Require alive discharge (`DIED=0` and `DISPUNIFORM != 20`), valid patient
   and timing identifiers, nonnegative LOS, and age at least 18 years.
6. Preserve `SAMEDAYEVENT` and `REHABTRANSFER` as index features/context flags;
   they are not labels. Report their frequencies by year.

## Year-end observability amendment

NRD supplies admission month but not the exact calendar admission day.
Therefore an exact proof of a full 30-day calendar-year follow-up window is not
identifiable for every index. The primary implementation uses the conventional
and reproducible rule `DMONTH < 12` and labels it "December-index exclusion",
not "exact 30-day observability".

The Stage 2 placeholder `index_episode_discharge_day <= 335` must not be
implemented: `NRD_DaysToEvent` has a different random origin for every patient,
so an absolute cutoff such as 335 is invalid. This amendment supersedes that
placeholder before any 2022 outcome is inspected.

A prespecified conservative sensitivity analysis requires that the latest
possible discharge, calculated from the last calendar day of the admission
month plus LOS, still leaves 30 days before year end:

```text
last_day_of_admission_month + LOS + 30 <= last_day_of_year
```

This rule guarantees, rather than estimates, the observation window. The
primary/sensitivity distinction is frozen here before any 2022 access. The
limitation that late-November admissions may have incomplete follow-up under
the primary rule must be reported explicitly.
