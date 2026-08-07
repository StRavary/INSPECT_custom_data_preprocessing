# INSPECT EHR Feature Extraction — Access Guide

A guide to getting features and labels out of the INSPECT EHR data.

If you only read one section, read **§3 Anchoring** and **§6 Caveats** — those
are where results go silently wrong.

---

## 1. What the dataset is

INSPECT pairs CTPA imaging with structured EHR for a Stanford cohort.
**22,457 studies across 18,738 patients**, split patient-level 81.5 / 4.7 / 13.8
(train / valid / test). No patient appears in more than one split.

Each row of the cohort is one CTPA study. Labels attach to that study; EHR
features are computed from the patient's record *before* it.

### Where things live

| what                           | path                                                  | size    |
|--------------------------------|-------------------------------------------------------|---------|
| cohort master file             | `DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv`    | 6 MB    |
| labels (incl. survival)        | `DATA_RAW/LABELS/labels_20250611.tsv`                 | 4 MB    |
| canonical splits               | `DATA_RAW/LABELS/splits_20250611.tsv`                 | 0.5 MB  |
| FEMR database (processed)      | `DATA_RAW/EHR_FEMR_DB/extract/`                       | 21 GB   |
| raw OMOP CSVs                  | `DATA_RAW/EHR_CSV/`                                   | 42 GB   |

Paths are relative to the repository's parent directory.

---

## 2. Two routes into the data

**Route A — the FEMR database (`DATA_RAW/EHR_FEMR_DB/extract/`).**
Pre-processed, indexed by patient, with the OMOP concept hierarchy materialised.
This is what the published INSPECT baselines use, so results from it are
comparable to the paper. Access it through `femr.datasets.PatientDatabase`;
the 21 GB `event_metadata` file is an internal binary store and is not readable
directly.

**Route B — the raw OMOP CSVs (`DATA_RAW/EHR_CSV/`).**
Complete and unreduced. Use this when you need measurement *values*, true
sampling frequency, or control over ontology expansion depth. The extractor
(`route_b_labs.py`) queries OMOP CSVs via DuckDB — large files are streamed
out-of-core and never loaded into RAM. Supports labs (numeric values from
`measurement.csv`), diagnoses, drugs, procedures, observations, and visits
(event counts), plus optional concept ancestor rollup and LOS features.

| table                | size    | est. rows |
|----------------------|---------|-----------|
| measurement          | 22.7 GB | ~143M     |
| observation          | 2.9 GB  | ~27M      |
| condition_occurrence | 1.4 GB  | ~12.5M    |
| drug_exposure        | 2.1 GB  | ~11.8M    |
| procedure_occurrence | 1.0 GB  | ~8.7M     |
| visit_occurrence     | 0.7 GB  | ~4.3M     |
| concept_ancestor     | 1.7 GB  | ~79M      |
| concept              | 1.1 GB  | ~9M       |

**Route A is de-duplicated relative to Route B** — see §6.1. If you need time
series or actual measurement values, you need Route B.

---

## 3. Anchoring — read this one

Every feature is computed backwards from an **anchor time**. Which anchor
depends on whether the task is diagnostic or prognostic.

| kind                | anchor                | tasks                                                                |
|---------------------|-----------------------|----------------------------------------------------------------------|
| **dx** (diagnostic) | `StudyTime − 1 day`   | `pe_positive_nlp`, `pe_positive`, `pe_acute`, `pe_subsegmentalonly` |
| **px** (prognostic) | `StudyTime`           | all mortality, all readmission, `12_month_PH`, and the five radiographic findings |

The criterion is **whether the label describes this CTPA or a future event**,
not whether the outcome is a diagnosis.

PE is the only diagnostic task: the label *is* the finding read off this scan,
so anything recorded on the study date can encode the answer. Backing off one
day removes that leakage path.

`12_month_PH` is a diagnosis, but you are forecasting its occurrence from prior
data, so there is no same-day contamination and the anchor stays at
`StudyTime`. This matches `ehr/2_generate_labels_and_features.py`, which applies
the offset only in its PE branch.

---

## 4. Quickstart

### Option A — the Streamlit interface

```bash
cd Custom
../.venv_legacy/bin/python -m streamlit run ./app_feature_extraction.py
```

The app has five tabs:

| tab                  | purpose                                                                                                   |
|----------------------|-----------------------------------------------------------------------------------------------------------|
| **0 · Load**         | Lists all saved extractions. Load instantly — no reconfiguration needed. Start here.                      |
| **1 · Data sources** | Validate file paths: cohort, OMOP CSVs (`measurement`, `concept`, `person`), and labels.                  |
| **2 · Extract**      | Choose feature types, set per-type lookback windows, run DuckDB extraction. Logs stream live.             |
| **3 · Describe**     | GDC-style cohort builder — filter by label, survival event, sex, age band; browse individual cases.       |
| **4 · Export**       | Write split-ready files to disk (`X.npy`, `y.npy`, `metadata.csv`, `feature_names.csv`, load scripts).   |

### Option B — the Python API (Route A)

```python
from Custom.temporal_features import (
    TemporalFeatureExtractor, VITALS_LABS, DIAG_PROC, DRUGS,
)

ex = TemporalFeatureExtractor()          # paths auto-resolve
ds = ex.build(
    task="12_month_PH",
    groups=[VITALS_LABS(bins=[2, 30, 365]),      # window edges in days
            DIAG_PROC(bins=[365, 1825]),
            DRUGS(bins=[30, 365])],
)

ds.X            # scipy sparse CSR, one row per study
ds.columns      # column names aligned to X
ds.y            # binary label
ds.tte, ds.event   # survival: days to event, 1 = observed / 0 = censored
ds.split        # 'train' / 'valid' / 'test'
ds.to_frame()   # metadata as a DataFrame
print(ds.describe())
```

### Option C — the Python API (Route B)

```python
from Custom.route_b_labs import LabExtractor

ex = LabExtractor()   # paths auto-resolve; person.csv appended automatically
ds = ex.build(
    task="12_month_PH",
    windows_days=[2, 7, 30, 365],   # cumulative window edges for labs (days)
    loinc_codes=None,               # None = all LOINC-coded measurements

    # which feature types to extract
    feature_types=["labs", "diagnoses", "drugs", "procedures",
                   "observations", "visits"],

    # per-type lookback windows for count features (days)
    count_window_days={
        "diagnoses":    365,
        "drugs":         60,
        "procedures":   365,
        "observations": 365,
        "visits":       365,
    },

    min_studies_per_lab=50,   # drop rare labs / count codes

    # concept ancestor rollup (requires concept_ancestor.csv, ~1.7 GB)
    use_concept_ancestor=False,
    ancestor_levels=2,           # only used when use_concept_ancestor=True
    min_studies_ancestor=100,    # separate threshold for ancestor columns
    max_ancestor_features=2000,  # hard cap per event table
)

ds.X            # numpy float32 array, one row per study; NaN = not measured
ds.columns      # feature names aligned to X
ds.y            # binary label
ds.tte, ds.event
ds.split
ds.to_frame()
print(ds.describe())
```

Requires `pip install duckdb` in the venv (not bundled by default).

### Available tasks

**Binary + survival**: `12_month_PH`, `{1,6,12}_month_mortality`,
`{1,6,12}_month_readmission`
**Binary only**: `pe_positive_nlp`, `pe_positive`, `pe_acute`, `pe_subsegmentalonly`
**Survival available**: also `Atelectasis`, `Cardiomegaly`, `Consolidation`,
`Edema`, `Pleural_Effusion`

Survival columns (`tte_*` / `is_censored_*`) come from `labels_20250611.tsv` and
are **not re-derived**. `tte` is converted from minutes to days. Censoring is
genuine per-patient right-censoring, not one administrative horizon.

---

## 5. What the columns mean

### 5.1 Route A columns (FEMR count features)

Each column is one **(code, time-window)** pair. The value is the count of
qualifying events for that code inside that window, counted backwards from the
anchor.

| pattern                                 | meaning                                                    |
|-----------------------------------------|------------------------------------------------------------|
| `age`                                   | age at anchor, **z-scored** across all labels — not years  |
| `<group>:<CODE>_<window>`               | count of CODE within that window                           |
| `<group>:<CODE> <value>_<window>`       | string-valued events, one column per (code, value)         |
| `<group>:<CODE> [lo, hi)_<window>`      | numeric-valued events bucketed by decile                   |

`<window>` is a `timedelta` such as `365 days, 0:00:00`, meaning the window
running from the anchor back to 365 days before it. Windows are non-overlapping
and consecutive: bins `[2, 30]` give `[T0, T0−2d]` and `[T0−2d, T0−30d]`.

Internally the layout is `column_idx = code_col + bin_idx × num_codes` — all
codes for the first window, then all codes for the second, and so on.

#### Ontology expansion — both leaf *and* ancestors

Every group runs with `is_ontology_expansion=True`, so each event increments
**its own code and every ancestor** in the OMOP hierarchy. FEMR's docstring: for
`A → B → C`, two occurrences of C also count as two of B and two of A.

Three consequences:

1. Both leaf and parent codes appear as columns — it is not one or the other.
2. **A parent column is the sum over its descendants**, so columns are strongly
   collinear. Do not read individual numbers as independent effects; use
   permutation or grouped importance.
3. The feature space is much wider than the number of distinct raw codes.

### 5.2 Route B columns (OMOP feature types)

Route B can extract six complementary feature types from different OMOP tables.
Each type has its own column naming convention.

#### 5.2.1 Lab features (`labs:`) — `measurement.csv`

Each column is one **(LOINC code, aggregate, time-window)** triple. The value is
a real-valued measurement from `measurement.value_as_number`, summarised over
the window. NaN means the lab was not measured at all in that window — this is
different from zero, which means "not observed" for count features.

Column names follow the pattern `labs:LOINC/<code>_<agg>_<N>d` where:

| `<agg>`                             | meaning                                                               |
|-------------------------------------|-----------------------------------------------------------------------|
| `last`                              | value of the most recent measurement in the window (before anchor)    |
| `min`                               | minimum value across all measurements in the window                   |
| `max`                               | maximum value                                                         |
| `mean`                              | arithmetic mean                                                       |
| `n`                                 | count of distinct measurement days                                    |
| `days_since`                        | calendar days between the most recent measurement and the anchor      |

For example, `labs:LOINC/2160-0_last_30d` is the most recent creatinine
reading in the 30 days before the anchor. `labs:LOINC/2160-0_n_30d` is how
many times creatinine was measured in that period.

Windows are **cumulative**, not nested: `_2d` covers the last 2 days, `_30d`
covers the last 30 days (including those 2). A measurement at day −5 appears
in `_30d` and `_365d` but not `_2d`.

Route B extracts only LOINC-coded entries from `measurement.csv`. Vital signs
recorded under other vocabularies are not included unless you pass their LOINC
equivalents to `loinc_codes`.

#### 5.2.2 Count features — diagnoses, drugs, procedures, observations, visits

The remaining five feature types all follow the same pattern: a count of
**distinct event days** within the lookback window, drawn from a single OMOP
table. The value is 0 (not observed) rather than NaN (not measured), because
absence is unambiguous for coded events.

| feature_type   | OMOP table            | col prefix | vocabularies                          |
|----------------|-----------------------|------------|---------------------------------------|
| `diagnoses`    | `condition_occurrence`| `diag`     | ICD-10-CM, ICD-9-CM, SNOMED           |
| `drugs`        | `drug_exposure`       | `drug`     | RxNorm, RxNorm Extension              |
| `procedures`   | `procedure_occurrence`| `proc`     | CPT4, ICD-10-PCS, HCPCS, ICD-9-Proc  |
| `observations` | `observation`         | `obs`      | OMOP Observation domain               |
| `visits`       | `visit_occurrence`    | `visit`    | OMOP Visit domain                     |

Column name pattern: `{prefix}:{vocabulary}/{concept_code}_{N}d`

Examples:
- `diag:ICD10CM/I26.90_365d` — days with a PE diagnosis in the past year
- `drug:RxNorm/11289_60d` — days apixaban was dispensed in the past 60 days
- `proc:CPT4/71275_365d` — days a CTPA was performed in the past year
- `visit:Visit/9201_365d` — inpatient visit days in the past year

Each feature type has its own independently configurable lookback window
(`count_window_days` dict), so you can look back 365 days for diagnoses but
only 60 days for drugs.

#### 5.2.3 Length-of-stay (LOS) features — `visit_occurrence`

When `visits` is selected, two LOS columns are automatically appended alongside
the visit-count columns:

| column pattern                  | meaning                                                         |
|---------------------------------|-----------------------------------------------------------------|
| `visit:LOS/total_{N}d`          | sum of `(visit_end_date − visit_start_date)` across all visits in window |
| `visit:LOS/max_{N}d`            | maximum single-visit length of stay in window                   |

Values are in days. These capture hospitalisation burden independently of visit
count: a patient with one 30-day admission differs from one with 30 day visits.

#### 5.2.4 Concept ancestor rollup (`{prefix}_anc:`) — optional

When `use_concept_ancestor=True`, each event is also rolled up to its ancestor
concepts in the OMOP hierarchy using `concept_ancestor.csv`. This creates
additional columns grouped at a broader clinical level — e.g., all specific PE
ICD codes roll up to the parent VTE category.

Column name pattern: `{prefix}_anc:{vocabulary}/{concept_code}_{N}d`

Examples:
- `diag_anc:SNOMED/59282003_365d` — events under the SNOMED pulmonary embolism ancestor
- `drug_anc:ATC/B01AF_60d` — days any direct oral anticoagulant was dispensed

Because ancestor codes are far more numerous than direct codes, use stricter
filtering: `min_studies_ancestor` (default 100) and `max_ancestor_features`
(default 2000 per table). Setting `ancestor_levels=2` (parent and grandparent
only) dramatically reduces the ancestor pair count relative to unlimited depth.

### 5.3 Demographic columns (both routes)

Nine demographic columns sourced from `person.csv` are appended to **every**
extraction automatically when `person.csv` is configured. They carry the prefix
`demo:`.

| column                                     | type     | encoding                                                      |
|--------------------------------------------|----------|--------------------------------------------------------------|
| `demo:age_years`   | float    | true age at anchor in years — **not z-scored**, so you can choose your own normalisation. NaN if birth date is missing. |
| `demo:is_female`   | binary   | 1 = female (OMOP gender_concept_id 8532), 0 = male (8507), NaN = other or not recorded |
| `demo:sex_unknown` | binary   | 1 if gender_concept_id is 0 or the patient is absent from person.csv |
| `demo:is_hispanic` | binary   | 1 = Hispanic or Latino (OMOP 38003563), 0 = Not Hispanic or Latino (38003564), NaN = not recorded |
| `demo:race_white`  | binary   | 1 if race_concept_id = 8527 (White) |
| `demo:race_black`  | binary   | 1 if race_concept_id = 8516 (Black or African American) |
| `demo:race_asian`  | binary   | 1 if race_concept_id = 8515 (Asian) |
| `demo:race_other`  | binary   | 1 if race is recorded but is not white, black, or Asian |
| `demo:race_unknown`| binary   | 1 if race_concept_id = 0 or the patient is absent from person.csv |

**Sparse matrix note (Route A).** Scipy sparse matrices cannot store NaN, so
before stacking the demographic columns `append_demographics()` fills age NaN
with the cohort median age, and fills is_female / is_hispanic NaN with 0. Use
the `demo:sex_unknown` and `demo:race_unknown` columns to recover the
missing-data signal in models — a `sex_unknown=1` row is distinct from a
`is_female=0` row even though both have `is_female=0` in the sparse matrix.

**Dense matrix note (Route B).** NaN is preserved. You should impute before
training; `sklearn.impute.SimpleImputer(strategy="median")` is a reasonable
starting point. The exported `load_survival.py` includes a commented example.

If `person.csv` is not found or not configured in Data sources, the step is
silently skipped (logged) and no `demo:` columns are added.

---

## 5.4 Full column reference

A typical extraction produces tens of thousands of columns. The tables below
give the vocabulary format, clinical meaning, and key examples for every column
type you will encounter. Use `fm.human_columns(concept_map)` to resolve raw
codes to readable names in bulk (see §5.5).

---

### 5.4.1 The `age` column (Route A only)

| column | source               | value                                                                  |
|--------|----------------------|------------------------------------------------------------------------|
| `age`  | FEMR `AgeFeaturizer` | patient age at anchor, **z-scored** across all labelled studies — not in years |

This is the only column not associated with a time window. Route B does not
produce it; use `demo:age_years` for true years on either route.

---

### 5.4.2 `vitals_labs` group — OMOP tables: `measurement`, `observation`

Column prefix: `vitals_labs:LOINC/<code>_<window>` (most entries) or
`vitals_labs:SNOMED/<code>_<window>` (some observation concepts).

Values are **counts of distinct days** on which that code appeared with a
changed value (Route A delta-encoding — see §6.1).

#### Vital signs (LOINC-coded, from `measurement`)

| LOINC code  | what it measures                      |
|-------------|---------------------------------------|
| `8867-4`    | Heart rate (bpm)                      |
| `8480-6`    | Systolic blood pressure (mmHg)        |
| `8462-4`    | Diastolic blood pressure (mmHg)       |
| `59408-5`   | Oxygen saturation, pulse ox (%)       |
| `9279-1`    | Respiratory rate (breaths/min)        |
| `8310-5`    | Body temperature (°C or °F)           |
| `29463-7`   | Body weight (kg)                      |
| `8302-2`    | Body height (cm)                      |
| `39156-5`   | BMI (kg/m²)                           |

#### Renal function

| LOINC code       | what it measures                |
|------------------|---------------------------------|
| `2160-0`         | Creatinine, serum/plasma        |
| `3094-0`         | BUN (blood urea nitrogen)       |
| `33914-3`        | eGFR (CKD-EPI or MDRD equation) |

#### Complete blood count (CBC)

| LOINC code       | what it measures                                             |
|------------------|--------------------------------------------------------------|
| `718-7`          | Haemoglobin (g/dL) — anaemia, O₂-carrying capacity           |
| `4544-3`         | Haematocrit (%)                                              |
| `6690-2`         | WBC / leukocyte count — infection, inflammation              |
| `26515-7`        | Platelet count — thrombocytopaenia, coagulopathy             |
| `769-0`          | Neutrophils (absolute, ×10⁹/L)                               |
| `731-0`          | Lymphocytes (absolute)                                       |
| `787-2`          | MCV — macrocytosis (B12/folate), microcytosis (iron)         |
| `785-6`          | MCH                                                          |
| `786-4`          | MCHC                                                         |

#### Coagulation — especially relevant for PE diagnosis and anticoagulation

| LOINC code      | what it measures                                             |
|-----------------|--------------------------------------------------------------|
| `5902-2`        | Prothrombin time (PT, seconds)                               |
| `6301-6`        | INR — anticoagulation intensity for warfarin                 |
| `14804-9`       | INR (alternate LOINC — both may appear)                      |
| `3173-2`        | aPTT — heparin monitoring                                    |
| `3255-7`        | Fibrinogen (mg/dL) — DIC, acute phase                        |
| `48065-7`       | D-dimer (fibrin degradation product) — PE screening biomarker|
| `25315-4`       | D-dimer (alternate LOINC)                                    |
| `48066-5`       | D-dimer (quantitative, alternate)                            |

#### Cardiac biomarkers — right heart strain in PE

| LOINC code   | what it measures                                             |
|--------------|--------------------------------------------------------------|
| `10839-9`    | Troponin I (conventional) — myocardial injury                |
| `6598-7`     | Troponin T                                                   |
| `67151-1`    | Troponin I (high-sensitivity)                                |
| `89579-7`    | Troponin I (high-sensitivity, alternate)                     |
| `33762-6`    | NT-proBNP — right ventricular strain, heart failure          |
| `42637-9`    | BNP (B-type natriuretic peptide)                             |
| `2532-0`     | LDH (lactate dehydrogenase) — tissue injury                  |

#### Metabolic / comprehensive metabolic panel

| LOINC code   | what it measures                                             |
|--------------|--------------------------------------------------------------|
| `2951-2`     | Sodium (mEq/L)                                               |
| `2823-3`     | Potassium (mEq/L)                                            |
| `2075-0`     | Chloride (mEq/L)                                             |
| `2028-9`     | CO₂ / bicarbonate — acid-base status                         |
| `17861-6`    | Calcium (total, mg/dL)                                       |
| `2339-0`     | Glucose (mg/dL)                                              |
| `1742-6`     | ALT (alanine aminotransferase) — hepatocellular injury       |
| `1920-8`     | AST (aspartate aminotransferase)                             |
| `6768-6`     | Alkaline phosphatase — cholestasis, bone disease             |
| `1975-2`     | Total bilirubin                                              |
| `1751-7`     | Albumin (serum) — nutritional status, liver function         |

#### Inflammatory markers

| LOINC code   | what it measures                                             |
|--------------|--------------------------------------------------------------|
| `1988-5`     | C-reactive protein (CRP) — acute phase reactant              |
| `30341-2`    | ESR (erythrocyte sedimentation rate)                         |
| `33959-8`    | Procalcitonin (PCT) — bacterial infection severity           |

#### Iron studies

| LOINC code   | what it measures                                             |
|--------------|--------------------------------------------------------------|
| `2276-4`     | Ferritin — iron stores, acute phase reactant                 |
| `2498-4`     | Iron (serum)                                                 |
| `2502-3`     | TIBC (total iron binding capacity)                           |

#### Diabetes / endocrine

| LOINC code   | what it measures                                             |
|--------------|--------------------------------------------------------------|
| `4548-4`     | HbA1c (%) — glycaemic control, DM diagnosis                  |
| `1556-0`     | Glucose (fasting)                                            |
| `3016-3`     | TSH — thyroid function                                       |
| `3053-6`     | Free T4                                                      |

#### Lipids

| LOINC code   | what it measures                                             |
|--------------|--------------------------------------------------------------|
| `2093-3`     | Total cholesterol                                            |
| `2089-1`     | LDL cholesterol                                              |
| `2085-9`     | HDL cholesterol                                              |
| `2571-8`     | Triglycerides                                                |

#### Blood gas / pulmonary

| LOINC code   | what it measures                                             |
|--------------|--------------------------------------------------------------|
| `11558-4`    | pH (arterial blood gas)                                      |
| `2703-7`     | PaO₂ (partial pressure of oxygen)                            |
| `2019-8`     | PaCO₂ (partial pressure of CO₂)                              |
| `19994-3`    | PaO₂/FiO₂ ratio — lung injury index                          |
| `2708-6`     | O₂ saturation (arterial, blood gas)                          |

---

### 5.4.3 `diag` / `diag_proc` group — OMOP tables: `condition_occurrence`

Column prefix: `diag:ICD10CM/<code>_<window>` (most entries) or
`diag:SNOMED/<code>_<window>`.

Values are counts of days the diagnosis was recorded. Because of ontology
expansion, a code for "submassive PE" (I26.02) also increments every parent
ICD-10 chapter (I26.0, I26, I00-I99).

#### Pulmonary embolism

| ICD-10-CM code | meaning                                                        |
|----------------|----------------------------------------------------------------|
| `I26.01`       | Saddle embolus of pulmonary artery with acute cor pulmonale      |
| `I26.02`       | Saddle embolus without acute cor pulmonale                     |
| `I26.09`       | Other PE with acute cor pulmonale                              |
| `I26.90`       | Unspecified PE without acute cor pulmonale                       |
| `I26.92`       | Subsegmental PE without acute cor pulmonale                    |
| `I26.93`       | Single subsegmental PE                                       |
| `I26.94`       | Multiple subsegmental PE                                       |
| `Z86.711`      | Personal history of PE                                       |
| `Z86.718`      | Personal history of other VTE                                |

#### Pulmonary hypertension

| ICD-10-CM code | meaning                                                        |
|----------------|----------------------------------------------------------------|
| `I27.0`        | Primary pulmonary hypertension (Group 1)                       |
| `I27.20`       | Pulmonary hypertension, unspecified                            |
| `I27.21`       | Secondary pulmonary arterial hypertension                      |
| `I27.22`       | PH due to left heart disease (Group 2)                         |
| `I27.23`       | PH due to lung disease / hypoxia (Group 3)                     |
| `I27.24`       | Chronic thromboembolic PH — CTEPH (Group 4)                    |
| `I27.29`       | Other secondary PH                                             |

#### Deep vein thrombosis (DVT) — frequent PE precursor

| ICD-10-CM code | meaning                                                               |
|----------------|-----------------------------------------------------------------------|
| `I82.401`      | Acute DVT of unspecified deep vein of right lower extremity           |
| `I82.411`      | Acute DVT of right femoral vein                                       |
| `I82.431`      | Acute DVT of right tibial vein                                        |
| `I82.4Y1`      | Acute DVT of other specified deep vein, right lower extremity         |
| `I82.A11`      | Acute DVT of right axillary vein (upper extremity DVT)                |
| `I82.B11`      | Chronic DVT of right axillary vein                                    |

#### Cardiac conditions

| ICD-10-CM code | meaning                                                             |
|----------------|---------------------------------------------------------------------|
| `I50.20`       | Unspecified systolic (congestive) heart failure                     |
| `I50.22`       | Chronic systolic heart failure                                      |
| `I50.30`       | Unspecified diastolic heart failure                                 |
| `I50.9`        | Heart failure, unspecified                                          |
| `I21.3`        | STEMI, unspecified site                                             |
| `I21.4`        | NSTEMI                                                              |
| `I48.0`        | Paroxysmal atrial fibrillation                                      |
| `I48.11`       | Longstanding persistent AF                                          |
| `I48.19`       | Other persistent AF                                                 |
| `I48.20`       | Unspecified chronic AF                                              |
| `I10`          | Essential (primary) hypertension                                    |
| `I46.9`        | Cardiac arrest, cause unspecified                                   |

#### Lung and respiratory conditions

| ICD-10-CM code | meaning                                                             |
|----------------|---------------------------------------------------------------------|
| `J18.9`        | Pneumonia, unspecified organism                                     |
| `J44.0`        | COPD with acute lower respiratory infection                         |
| `J44.1`        | COPD with acute exacerbation                                        |
| `J44.9`        | COPD, unspecified                                                 |
| `J45.20`       | Mild intermittent asthma, uncomplicated                             |
| `J81.0`        | Acute pulmonary oedema                                              |
| `J90`          | Pleural effusion                                                    |
| `J84.10`       | Pulmonary fibrosis, unspecified                                     |
| `J93.11`       | Primary spontaneous pneumothorax                                    |

#### Metabolic / chronic conditions (PE risk factors)

| ICD-10-CM code | meaning                                                             |
|----------------|---------------------------------------------------------------------|
| `E11.9`        | Type 2 diabetes mellitus without complications                      |
| `E66.9`        | Obesity, unspecified                                                |
| `N18.3`        | CKD stage 3                                                         |
| `N18.4`        | CKD stage 4                                                         |
| `N18.5`        | CKD stage 5                                                         |
| `N18.6`        | End-stage renal disease                                             |
| `C34.10`       | Malignant neoplasm of upper lobe, unspecified bronchus/lung         |
| `C80.1`        | Malignant neoplasm, unspecified, primary                            |

---

### 5.4.4 `proc` / `diag_proc` group — OMOP tables: `procedure_occurrence`, `device_exposure`

Column prefix: `proc:ICD10PCS/<code>_<window>`, `proc:CPT4/<code>_<window>`,
`proc:HCPCS/<code>_<window>`, or `proc:SNOMED/<code>_<window>`.

Values are counts of days a procedure was recorded.

#### Imaging and diagnostic procedures (CPT4)

| CPT4 code | meaning                                |
|-----------|----------------------------------------|
| `71250`   | CT thorax without contrast             |
| `71260`   | CT thorax with contrast                |
| `71270`   | CT thorax without and with contrast    |
| `71275`   | CT angiography, thorax (CTPA)          |
| `93306`   | Echocardiography with Doppler          |
| `75741`   | Pulmonary angiography                  |
| `78582`   | Ventilation-perfusion (V/Q) scan       |
| `93000`   | 12-lead ECG with interpretation        |

#### PE treatment procedures (ICD-10-PCS)

| ICD-10-PCS code | meaning                             |
|-----------------|-------------------------------------|
| `05H633Z`       | Inferior vena cava filter insertion |
| `06BQ0ZZ`       | Excision of pulmonary trunk         |

#### General procedures

Procedure columns span a wide range from lab draw (pathology) to surgical
procedures. The `diag_proc` group combines these with diagnoses, so the same
time window covers both a cancer diagnosis and a chemotherapy procedure.

#### Devices (`device_exposure`)

Device exposure concept IDs appear as `proc:Device/<concept_id>_<window>`. They
represent implanted or applied devices (IVC filters, pacemakers, oxygen
delivery). These are less commonly important features but are included by default
in `diag_proc` and `proc`.

---

### 5.4.5 `drugs` group — OMOP table: `drug_exposure`

Column prefix: `drugs:RxNorm/<ingredient_concept_id>_<window>`.

Values are counts of days the drug was administered or dispensed. FEMR maps
all drug records to the RxNorm ingredient level (the active molecule), not the
branded product or dose form, so `drugs:RxNorm/1049502_<window>` covers all
formulations of warfarin regardless of dose or brand.

Because of ontology expansion, an ingredient-level event also increments its
ATC class ancestors.

#### Anticoagulants — drugs most directly tied to PE management

| RxNorm ingredient | drug                                                                   |
|-------------------|------------------------------------------------------------------------|
| `11289`           | Warfarin (Coumadin) — vitamin K antagonist                             |
| `5224`            | Heparin (unfractionated) — IV anticoagulation                          |
| `67108`           | Enoxaparin (Lovenox) — LMWH, bridging and treatment                    |
| `321208`          | Fondaparinux (Arixtra) — factor Xa inhibitor                           |
| `1364435`         | Apixaban (Eliquis) — DOAC, direct factor Xa                            |
| `1114195`         | Rivaroxaban (Xarelto) — DOAC, direct factor Xa                         |
| `1037042`         | Dabigatran (Pradaxa) — DOAC, direct thrombin inhibitor                 |
| `1599538`         | Edoxaban (Savaysa) — DOAC                                              |
| `1736754`         | Betrixaban — DOAC                                                      |

#### Pulmonary vasodilators — drugs for PH management

| RxNorm ingredient     | drug                                                                   |
|-----------------------|------------------------------------------------------------------------|
| `36117`               | Sildenafil (Revatio / Viagra) — PDE-5 inhibitor                        |
| `187899`              | Tadalafil (Ad circa / Cialis) — PDE-5 inhibitor                        |
| `1546954`             | Riociguat (Adempas) — soluble guanylate cyclase stimulator             |
| `41140`               | Bosentan (Tracleer) — endothelin receptor antagonist                   |
| `1745276`             | Macitentan (Opsumit) — ERA                                             |
| `240131`              | Ambrisentan (Letairis) — ERA                                           |
| `3489`                | Epoprostenol (Flolan) — prostacyclin analogue                          |
| `77492`               | Treprostinil (Remodulin) — prostacyclin analogue                       |
| `2200691`             | Selexipag (Uptravi) — prostacyclin receptor agonist                    |

#### Diuretics — right heart failure management

| RxNorm ingredient | drug                                                                   |
|-------------------|------------------------------------------------------------------------|
| `4603`              | Furosemide (Lasix) — loop diuretic                                   |
| `1808`              | Bumetanide (Bumex) — loop diuretic                                   |
| `38413`             | Torsemide — loop diuretic                                            |
| `3289`              | Spironolactone — aldosterone antagonist                              |

#### Cardiovascular

| RxNorm ingredient | drug                                                                   |
|-------------------|------------------------------------------------------------------------|
| `41493`           | Metoprolol — beta-blocker                                              |
| `20352`           | Carvedilol — beta-blocker                                              |
| `83367`           | Atorvastatin — statin                                                  |
| `301542`          | Rosuvastatin — statin                                                  |
| `36567`           | Simvastatin — statin                                                   |
| `29046`           | Lisinopril — ACE inhibitor                                             |
| `52175`           | Losartan — ARB                                                         |
| `1191`            | Aspirin — antiplatelet                                                 |
| `32592`           | Clopidogrel — P2Y12 antiplatelet                                       |

#### Thrombolytics

| RxNorm ingredient | drug                                                                   |
|-------------------|------------------------------------------------------------------------|
| `688965`          | Alteplase (tPA, Activase) — systemic thrombolysis for massive PE       |
| `1347111`         | Tenecteplase (TNKase)                                                  |

---

### 5.4.6 `visits` group — OMOP tables: `visit_occurrence`, `visit_detail`

Column prefix: `visits:Visit/<concept_id>_<window>`.

Values are counts of distinct visit episodes within the window. These are
typically the broadest features: "was the patient hospitalised in the last year".

| OMOP Visit concept   | meaning                                                      |
|----------------------|--------------------------------------------------------------|
| `9201`               | Inpatient Visit (IP) — overnight hospital admission          |
| `9202`               | Outpatient Visit (OP) — ambulatory, clinic, or day procedure |
| `9203`               | Emergency Room Visit (ER)                                    |
| `8883`               | Observation Room — less-than-24h monitoring                  |
| `262`                | Emergency Room and Inpatient Visit (ER → admitted)           |
| `32037`              | Intensive Care Unit (ICU) stay                               |
| `32044`              | Home Visit                                                   |

Visit counts are often among the strongest predictors because frequent
hospitalisation correlates directly with disease severity.

---

### 5.5 Decoding column names programmatically

```python
from Custom.temporal_features import load_concept_map

# Load once per session — concept.csv is 1.1 GB but pandas reads it in ~30s
concept_map = load_concept_map("path/to/DATA_RAW/EHR_CSV/concept.csv")

# Get all human-readable names for a loaded feature matrix
human = fm.human_columns(concept_map)
# e.g. 'vitals_labs:Creatinine [Mass/vol] in Serum [LOINC/2160-0]_365d'

# Find every creatinine column
cr_cols = [c for c in fm.columns if "LOINC/2160-0" in c]

# Find every PE diagnosis column
pe_cols = [c for c in fm.columns if "ICD10CM/I26" in c]

# Find every anticoagulant column (warfarin)
warf_cols = [c for c in fm.columns if "RxNorm/11289" in c]

# Map raw name → human name for a specific column
col = "diag:ICD10CM/I26.90_365 days, 0:00:00"
human_name = concept_map.get("ICD10CM/I26.90", col)
# → 'Pulmonary embolism without acute cor pulmonale'

# Route B: find all 30-day lab columns for troponin I
trop_30d = [c for c in fm.columns if "LOINC/10839-9" in c and "_30d" in c]
# → ['labs:LOINC/10839-9_last_30d', 'labs:LOINC/10839-9_min_30d', ...]
```

The column format for each group is:

| group                         | format                                  | example raw name                                 |
|-------------------------------|-----------------------------------------|--------------------------------------------------|
| Route A vitals/labs           | `<group>:LOINC/<code>_<timedelta>`      | `vitals_labs:LOINC/2160-0_365 days, 0:00:00`     |
| Route A diagnosis             | `diag:ICD10CM/<code>_<timedelta>`       | `diag:ICD10CM/I26.90_365 days, 0:00:00`          |
| Route A procedure             | `proc:CPT4/<code>_<timedelta>`          | `proc:CPT4/71275_365 days, 0:00:00`              |
| Route A drug                  | `drugs:RxNorm/<id>_<timedelta>`         | `drugs:RxNorm/11289_30 days, 0:00:00`            |
| Route A visit                 | `visits:Visit/<id>_<timedelta>`         | `visits:Visit/9201_365 days, 0:00:00`            |
| Route B lab                   | `labs:LOINC/<code>_<agg>_<N>d`          | `labs:LOINC/2160-0_last_30d`                     |
| Route B diagnosis count       | `diag:<vocab>/<code>_<N>d`              | `diag:ICD10CM/I26.90_365d`                       |
| Route B drug count            | `drug:<vocab>/<code>_<N>d`              | `drug:RxNorm/11289_60d`                          |
| Route B procedure count       | `proc:<vocab>/<code>_<N>d`              | `proc:CPT4/71275_365d`                           |
| Route B observation count     | `obs:<vocab>/<code>_<N>d`               | `obs:SNOMED/271649006_365d`                      |
| Route B visit count           | `visit:<vocab>/<code>_<N>d`             | `visit:Visit/9201_365d`                          |
| Route B LOS (total)           | `visit:LOS/total_<N>d`                  | `visit:LOS/total_365d`                           |
| Route B LOS (max)             | `visit:LOS/max_<N>d`                    | `visit:LOS/max_365d`                             |
| Route B ancestor rollup       | `{prefix}_anc:<vocab>/<code>_<N>d`      | `diag_anc:SNOMED/59282003_365d`                  |
| Demographics                  | `demo:<name>`                           | `demo:age_years`                                 |

Note that Route A window labels use Python `timedelta` format (`365 days, 0:00:00`),
not the short `_365d` notation used by Route B. The `human_columns()` method
converts them to the short form for display.

---

## 6. Caveats that will hinder you

### 6.1 The FEMR extract is de-duplicated

Building it dropped **72,508,452** events via `delta_encode` and 130,015 via
`remove_nones`. `delta_encode` removes *sequential duplicates of the same
(concept_id, date) with the same value*.

So counts from Route A mean **"days on which this code appeared with a changed
value"**, not raw record counts. Distinct days and distinct values survive;
within-day repetition does not.

For count features this is arguably a feature — it makes counts insensitive to
charting frequency. **For time series it is fatal.** An hourly blood pressure
record collapsed to one reading per changed value cannot support trajectory
modelling. Use Route B for anything temporal.

### 6.2 Windows silently discard older events (Route A)

`CountFeaturizer` emits `len(time_bins)` bins but tracks `len + 1`. Events older
than the last bin edge land in a bucket that is never written out.

`bins=[1825]` therefore **discards everything beyond five years without warning**.
`temporal_features.py` appends a 100-year catch-all bin by default; pass
`catch_all=False` only if you mean to truncate.

### 6.3 Unsorted bins mislabel columns (Route A)

`featurize()` sorts `time_bins` internally, but `get_column_name()` indexes the
original list. Unsorted bins produce mislabelled columns. The API always sorts
first — if you call FEMR directly, sort them yourself.

### 6.4 Missingness is informative

"Lab not drawn" is a clinical decision, not absent data. Encode presence
explicitly rather than imputing. Measurements cluster during instability and
vanish otherwise, so the sampling process is not missing-at-random. Route B's
`demo:n_Nd` columns capture measurement frequency; the NaN in value columns
captures absence.

### 6.5 `concept_id = 0`

In OMOP this means "no matching concept". Route B filters it automatically
(`measurement_concept_id != 0` in the DuckDB query). If you write your own SQL,
add this filter or it becomes a large junk column.

### 6.6 Route B windows are cumulative, not nested

Route A's bins are non-overlapping (bin `[2, 30]` means days 2–30 only). Route B
windows are cumulative: `_30d` includes everything from days 1–30, so a
measurement at day 5 contributes to both `_7d` and `_30d`. This means the `last`
and `min`/`max` columns for a wider window subsume the narrower window's values,
creating some collinearity. Plan accordingly when selecting features.

### 6.7 `demo:age_years` replaces `age` (Route A only)

Route A already includes an `age` column from FEMR's `AgeFeaturizer`, which is
z-scored across all labels. The new `demo:age_years` column is the raw age in
years, not z-scored. Both appear in the matrix after demographics are appended.
The z-scored `age` column is absent from Route B.

---

## 7. Context descriptors

For studying **why a model performs well in one context and worse in another**,
the unit of analysis is the slice, not the patient. `ContextDescriber` produces
one row per slice with properties that might explain performance variation.

```python
from Custom.context_descriptors import ContextDescriber

cd = ContextDescriber(ds, demographics="auto",
                      key_columns=["labs:LOINC/2160-0_30 days, 0:00:00"])
table = cd.describe(by=["split", "sex", "age_band", "density_quartile"])
table = cd.attach_performance(table,
                              {"train": 0.91, "valid": 0.86, "test": 0.85},
                              key="split", name="auroc")
```

Descriptors produced per slice:

| group              | columns                                                                                           |
|--------------------|---------------------------------------------------------------------------------------------------|
| composition        | `n_studies`, `n_patients`, `event_rate`, `n_positive`                                               |
| survival           | `censoring_rate`, `median_tte_days`                                                                 |
| demographics       | `age_mean`, `age_iqr`, `frac_female`, `n_care_sites`, `top_site_share`                              |
| density            | `mean_distinct_codes`, `median_distinct_codes`, `mean_total_count`                                |
| vocabulary         | `vocab_share_ICD10CM`, `vocab_share_LOINC`, … (shares sum to 1)                                   |
| missingness        | `missing_<column>` for each nominated key column                                                  |
| history proxy      | `frac_with_oldest_window_events`                                                                  |
| vocabulary         | `vocab_share_ICD10CM`, `vocab_share_LOINC`, … (shares sum to 1) |
| missingness        | `missing_<column>` for each nominated key column |
| history proxy      | `frac_with_oldest_window_events` |

Built-in slicers: `all`, `split`, `density_quartile`, `age_band`, `sex`,
`race_concept_id`, `ethnicity_concept_id`, `care_site_id`. Pass
`{name: array}` for custom slices. Demographics require `person.csv`
(auto-located).

**What is not computable here.** Anything needing the raw event stream — true
history length, sampling intervals, measurement values — has already been
aggregated away by the count matrix. `frac_with_oldest_window_events` and
`mean_distinct_codes` are labelled proxies, not measurements. Use Route B for
genuine temporal descriptors.

---

## 8. Reproducibility rules

**Keep the FEMR path frozen.** The published INSPECT baselines are reproduced
through Route A to within +0.013 mean AUROC across eight tasks. Any change to
the feature source breaks that comparability. Build Route B work as a clearly
labelled parallel track.

**Features must be strictly pre-anchor.** The extractor enforces this, but if
you write your own SQL, `e.dt < a.anchor` is not optional.

**Splits are patient-level and canonical.** They originate from a frozen FEMR
table (`splits_omop_2023_03_05`) and propagate to every arm — EHR, imaging and
reports. Do not regenerate them; zero patients currently span splits and it
should stay that way.

**Validation is small.** At 4.7% of the cohort — ~1,056 studies, ~150 positives
for a 14%-prevalence task — model selection on it is noisy. Prefer
selection-robust summaries (repeated seeds, mean ± sd) over single
best-validation values.

---

## 9. Open questions

Things not yet settled, flagged so you do not assume they are:

* **Radiographic findings anchoring.** Atelectasis, Cardiomegaly, Consolidation,
  Edema and Pleural Effusion are currently `px`. If any are read off the index
  CTPA report the way PE is, they need `dx` treatment. Depends on how the AIMI
  labels were generated.
* **Competing risks.** Death competes with readmission and PH. `tte_mortality`
  is available alongside the others, so cause-specific or Fine–Gray models are
  possible, but nothing currently handles this.
* **Discrete-time survival** needs person-period expansion; the API emits one
  row per study only.
* **Route B vital signs.** Vital signs (BP, HR, SpO₂, temperature) are often
  recorded in OMOP under LOINC codes but may also appear under SNOMED or local
  vocabulary IDs depending on the source system. The current Route B extractor
  filters to `vocabulary_id = 'LOINC'` only. Pass specific LOINC codes for
  vitals (e.g. 8480-6 for systolic BP) to include them explicitly.
* **Route B slope features.** The extractor currently computes last, min, max,
  mean, n, and days_since. Linear regression slope over the window (a common
  strong predictor for trending labs like troponin or creatinine) is not yet
  implemented.
* **care_site_id as a feature.** Available in `person.csv` and used for
  slicing in the Describe tab, but not currently added to the demographic
  feature columns. Care site captures both institutional differences and
  patient-mix confounding.

---

## 10. File map

| file                               | purpose |
|------------------------------------|---------|
| `Custom/app_feature_extraction.py` | Streamlit interface — 7-tab UI over Route A, Route B, Cohort Builder, and Export |
| `Custom/temporal_features.py`      | Route A extraction API (`TemporalFeatureExtractor`, `FeatureMatrix`) |
| `Custom/route_b_labs.py`   | Route B extraction API (`LabExtractor`, `LabFeatureMatrix`); also contains `load_demographic_features()` and `append_demographics()` used by both routes |
| `Custom/_femr_worker.py`           | Subprocess worker for Route A (avoids Streamlit multiprocessing deadlock) |
| `Custom/_route_b_worker.py`        | Subprocess worker for Route B |
| `Custom/context_descriptors.py`    | Per-slice descriptor tables (`ContextDescriber`) |
| `Custom/x_generate_binned_features.py` | Earlier CLI version (superseded) |
| `Custom/extract_temporal_features.py` | Long-format event extraction with signed day windows |
| `Custom/9a_run_baseline_benchmark.py` | Generates `labeled_patients.csv` per task |
| `Custom/INSPECT_Baseline_Reconstruction.md` | Full methods and findings record |
