# INSPECT EHR Feature Extraction App — Overview

The app (`Custom/app_feature_extraction.py`) is a Streamlit interface for extracting structured EHR features from the INSPECT dataset and exporting them as machine-learning-ready arrays. It queries raw OMOP CSVs via DuckDB (no large files loaded into RAM) and is designed to let lab members run and share reproducible feature extractions without writing code, then immediately inspect the resulting cohort before any modelling.

Run it with:

```bash
streamlit run Custom/app_feature_extraction.py
```

---

## What problem it solves

The INSPECT dataset contains two complementary EHR data sources: a pre-processed FEMR database of patient timelines, and a set of raw OMOP CSV files (measurement, condition, drug, etc.). Turning either source into a feature matrix for a specific prediction task — selecting patients, anchoring in time, binning history into windows, and aligning with survival outcomes — requires a non-trivial pipeline. The app wraps that pipeline so that changing a task or a time window is a matter of adjusting a slider, not rewriting code.

---

## The five tabs

### Tab 0 · Load

This is the starting point for anyone who wants to work with a feature matrix that has already been extracted. The tab lists every cached extraction — both Route A (FEMR) and Route B (lab values) — sorted by date, each with a coloured badge indicating its route. Clicking **Load** pulls the extraction from disk into the session without running anything; from that point the Describe and Export tabs work immediately. Clicking **Delete** removes the cache file permanently.

A quick peek panel shows the dimensions, task, number of studies, and label prevalence of the loaded extraction at a glance. This tab is completely independent of the Data Sources and Configure tabs, so a lab-mate can load and explore an extraction made by someone else without configuring any file paths.

---

### Tab 1 · Data Sources

Configures the file paths the two extraction routes need. Each path has a text input pre-filled with the canonical INSPECT location, a **Validate** button that checks the file exists and parses its header, and a status indicator. The sources are:

**Cohort master file** — the CSV or TSV containing one row per CTPA study. Required columns are `StudyTime`, `impression_id`, and a patient ID column (`patient_id`, `PatientID`, or `person_id`). Task label columns and optional survival columns (`tte_*`, `is_censored_*`) also live here. A `split` column is accepted if present but is no longer required.

**FEMR database** — the `extract/` directory produced by FEMR's preprocessing pipeline. Required for Route A only.

**Labels TSV** — the survival outcomes file (`labels_20250611.tsv`). Optional if the cohort file already carries `tte_*` columns; otherwise required for survival export.

**person.csv** — the OMOP `person` table. Optional; when provided, demographic features (age, sex, race, ethnicity) are automatically appended to both Route A and Route B feature matrices.

**measurement.csv** — the raw OMOP measurement table (~22 GB). Required for Route B only.

**concept.csv** — the OMOP concept vocabulary. Optional; used to resolve raw LOINC / ICD-10 / RxNorm codes to human-readable names in the Export tab.

---

### Tab 2 · Extract

Configures and runs a DuckDB-based extraction from the raw OMOP CSVs. All parameters are set here; clicking **Run extraction** spawns a subprocess and streams the log live.

**Task and anchor type** — the prediction target (must match a column in the cohort file) and anchor direction. `dx` anchors one day before study time (diagnostic tasks); `px` anchors at study time (prognostic tasks). See §3 of the EHR guide for details.

**Feature types** — six types can be selected independently via checkboxes:

| feature type    | OMOP table              | value type | col prefix |
|-----------------|-------------------------|------------|------------|
| Labs            | `measurement.csv`       | float (real measurement values)   | `labs:`    |
| Diagnoses       | `condition_occurrence`  | int (distinct event days)         | `diag:`    |
| Drugs           | `drug_exposure`         | int (distinct event days)         | `drug:`    |
| Procedures      | `procedure_occurrence`  | int (distinct event days)         | `proc:`    |
| Observations    | `observation`           | int (distinct event days)         | `obs:`     |
| Visits          | `visit_occurrence`      | int (distinct visit days + LOS)   | `visit:`   |

**Lab windows** — cumulative lookback windows for lab features (e.g., `[2, 7, 30, 365]` days). For each lab × window, six aggregate columns are produced: `last`, `min`, `max`, `mean`, `n`, and `days_since`. Windows are cumulative: a measurement at day −5 appears in `_30d` and `_365d` but not `_2d`.

**LOINC codes** — by default all LOINC-coded measurements are included. A text box restricts extraction to a specific panel (one code per line or comma-separated).

**Count windows** — each non-lab feature type has its own independent lookback window in days (e.g., diagnoses in the past year, drugs in the past 60 days). Only active feature types show a window input.

**Length-of-stay** — when Visits is selected, two LOS columns are automatically added alongside the visit-count columns: `visit:LOS/total_{N}d` (total inpatient days) and `visit:LOS/max_{N}d` (longest single admission) within the window.

**Min studies per feature** — labs or codes present in fewer than this many studies are dropped. Keeps the matrix width manageable and avoids near-empty columns.

**Concept ancestor rollup** (optional, collapsed by default) — when enabled, events are also rolled up to their OMOP ancestor concepts via `concept_ancestor.csv` (~1.7 GB). Ancestor columns are prefixed `{type}_anc:` (e.g. `diag_anc:`, `drug_anc:`). Three sub-options control the scope:

- *Limit ancestor levels* (default on) — restrict how many levels to climb; 2 gives parent + grandparent
- *Min studies (ancestor)* — coverage threshold for ancestor codes (default 100, intentionally higher than the direct-code threshold)
- *Max ancestor features* — hard cap per table ordered by descending coverage (default 2000)

The extraction is identified by a SHA-256 hash of its full specification; re-running with identical settings loads from cache instantly.

---

### Demographics (both routes)

When `person.csv` is configured, nine demographic columns are appended to the feature matrix automatically at the end of extraction:

| column | meaning |
|---|---|
| `demo:age_years` | true age in years at anchor (float) |
| `demo:is_female` | 1 if sex concept = female (8532) |
| `demo:sex_unknown` | 1 if sex concept is missing or unclassifiable |
| `demo:is_hispanic` | 1 if ethnicity concept = Hispanic (38003563) |
| `demo:race_white` | 1 if race concept = white (8527) |
| `demo:race_black` | 1 if race concept = Black/African American (8516) |
| `demo:race_asian` | 1 if race concept = Asian (8515) |
| `demo:race_other` | 1 for any other non-missing race code |
| `demo:race_unknown` | 1 if race is missing |

For Route A (sparse), age NaN is filled with the cohort median and binary NaN is filled with 0; explicit `_unknown` binary columns preserve the missing-data signal. For Route B (dense), NaN is kept as NaN and imputed at modelling time alongside the lab values.

---

### Tab 3 · Describe

A GDC-style cohort builder for inspecting a loaded feature matrix without running any models. The tab lets you define a cohort filter and one or two stratification variables, then shows a summary table and charts for the resulting slice.

**Filters** — applied row-wise before any summary is computed. You can filter on task label (`y = 1`), anchor year range, or any demographic variable (sex, race, ethnicity, age band). Filters stack (AND logic).

**Stratify by** — breaks the cohort into subgroups. Available stratifiers include `sex`, `race_concept_id`, `ethnicity_concept_id`, `age_band` (decade buckets), `density_quartile` (by EHR record richness), and a free-text vocabulary slicer (e.g., "show only studies where the patient has at least one ICD-10 code matching `I26`").

**Summary table** — shows n, label prevalence, and data density per subgroup. Useful for checking whether a task is well-powered across subgroups before modelling.

**Feature density** — shows what fraction of columns are non-zero (Route A) or observed (Route B) per study, as a histogram. A very sparse matrix suggests the time windows may be too short or the cohort too selective.

---

### Tab 4 · Export

Writes the loaded feature matrix to a directory of flat files ready for modelling. All studies are exported together — train/test splitting is left to the downstream pipeline.

**Export directory** — the path where files will be written.

**Drop all-missing columns** — removes features that carry no information for any study (all-zero for Route A, all-NaN for Route B). Recommended; reduces file size substantially.

**Resolve human names** — when `concept.csv` is configured, writes `feature_names.csv` with a human-readable column name alongside the raw code-based name.

**Output files produced:**

| file                | description                                                                                      |
|---------------------|--------------------------------------------------------------------------------------------------|
| `metadata.csv`      | One row per study: `impression_id`, `patient_id`, `anchor_time`, `y`, `tte_days`, `event`        |
| `X.npy`             | float32 feature matrix, shape `(n_studies, n_features)`. Lab values: NaN = not measured; count features: 0 = not observed. Rows align with `metadata.csv` |
| `X_mask.npy`        | uint8 observation mask, same shape. 1 = observed, 0 = missing. For imputation-aware models       |
| `y.npy`             | Binary label array, length `n_studies`                                                           |
| `tte.npy` / `event.npy` | Time-to-event (days) and event indicator; written only when survival columns are present      |
| `feature_names.csv` | Two columns: `raw` (e.g. `drug:RxNorm/11289_60d`) and `human` (e.g. `drug:apixaban [RxNorm/11289]_60d`). Index aligns with `X.npy` columns |
| `load_survival.py`  | Ready-to-run script: loads all arrays, applies train/test split, fits a survival model           |
| `load_multimodal.py`| Script showing how to join EHR features with imaging or NLP outputs using `impression_id`        |

For legacy Route A (sparse) extractions, `X.npy` is replaced by `X_sparse.npz` (scipy CSR format).

---

## Caching and reproducibility

Every extraction is identified by a SHA-256 hash of its full specification (task, anchor, groups, windows, file paths). The hash is stored alongside the `.pkl` in `DATA_PROCESSED/femr_cache/`. Re-running with identical settings skips the extraction entirely and loads from cache. The accompanying `<hash>_spec.json` records all parameters and the route, so you can always reconstruct exactly what produced a given cache file.

---

## Typical workflow for a new lab-mate

1. Open **Tab 1 · Data Sources** and validate each path (most will be pre-filled correctly for the INSPECT server).
2. Open **Tab 2 · Extract**, select the task and anchor type, check the feature types you want, set lookback windows, and click **Run extraction**. Extraction time ranges from a few minutes (targeted lab panel only) to tens of minutes (all feature types). Labs-only with `windows_days=[365]` is usually the fastest starting point.
3. The extraction is now cached. Future sessions can skip steps 1–2 entirely and go straight to **Tab 0 · Load**.
4. Use **Tab 3 · Describe** to sanity-check the cohort — check prevalence, demographic balance, and data density.
5. Use **Tab 4 · Export** to write the arrays to a folder, then use the generated `load_survival.py` as a starting point for your model.
