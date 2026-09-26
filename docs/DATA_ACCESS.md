# Data access

## NRD

The Nationwide Readmissions Database (NRD) is distributed by the Healthcare Cost and Utilization Project (HCUP). It cannot be redistributed through this repository. Reproduction of the NRD analyses requires the researcher to obtain the relevant 2018-2022 NRD releases and comply with the HCUP Data Use Agreement and training requirements.

The public repository contains no NRD patient, encounter, hospital, prediction, or bootstrap-replicate records. Frozen public method files and non-patient code mappings are provided only when they do not disclose protected records.

## MIMIC-IV

MIMIC-IV is available through PhysioNet under credentialed access and a data-use agreement. Researchers must obtain access independently. The formal transfer used MIMIC-IV 3.1 and these archive members:

- `mimic-iv-3.1/hosp/admissions.csv.gz`
- `mimic-iv-3.1/hosp/patients.csv.gz`
- `mimic-iv-3.1/hosp/diagnoses_icd.csv.gz`
- `mimic-iv-3.1/hosp/procedures_icd.csv.gz`

Optional quality-assurance inputs were `drgcodes.csv.gz`, `services.csv.gz`, `transfers.csv.gz`, and `icustays.csv.gz`. Do not upload any MIMIC-IV files to GitHub.

## Public repository content

`results/public_source_data/` contains only disclosure-safe aggregate tables used by the final manuscript figures and tables. The release verifier rejects obvious direct-identifier columns and checks the locked small-cell suppression rule where count fields are present.

## Data Availability text

The NRD data are available from HCUP to qualified users under its Data Use Agreement. MIMIC-IV 3.1 is available to credentialed users through PhysioNet. Neither controlled-access dataset can be redistributed by the authors. Analysis code, frozen non-patient configuration files, disclosure-safe aggregate source data, and release manifests are included in this repository. Exact model reruns additionally require the non-redistributed frozen model bundle described in `docs/RESTRICTED_DATA.md`.

