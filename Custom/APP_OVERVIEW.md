# INSPECT EHR Feature Extraction App — Overview

The app (`Custom/app_feature_extraction.py`) is a Streamlit interface for extracting structured EHR features from the INSPECT dataset and exporting them as machine-learning-ready arrays. It is designed to let lab members run and share reproducible feature extractions without writing code, and to immediately inspect the resulting cohort before any modelling.

Run it with:

```bash
streamlit run Custom/app_feature_extraction.py
```

---

## What problem it solves

The INSPECT dataset contains two complementary EHR data sources: a pre-processed FEMR database of patient timelines, and a set of raw OMOP CSV files (measurement, condition, drug, etc.). Turning either source into a feature matrix for a specific prediction task — selecting patients, anchoring in time, binning history into windows, and aligning with survival outcomes — requires a non-trivial pipeline. The app wraps that pipeline so that changing a task or a time window is a matter of adjusting a slider, not rewriting code.

---

## The seven tabs

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

### Tab 2 · Configure

Sets parameters that are shared between Route A and Route B:

**Task** — the prediction target. This must match a column name in the cohort file (e.g., `12_month_mortality`, `30_day_PE_recurrence`). The dropdown is populated from the cohort file automatically.

**Anchor type** — controls where in time the feature window ends. `dx` anchors one day before the study time (information available at time of ordering); `px` anchors at the study time itself (information available after reading the scan). Most PE-outcome tasks should use `dx`.

---

### Tab 3 · Route A · FEMR

Extracts count-based features from the FEMR patient timeline database. This is the same representation used in the published INSPECT baselines.

**Feature groups** — each group covers a set of OMOP tables and can be toggled on or off. The available groups are:

- `vitals_labs` — measurements and observations (LOINC-coded labs and vitals)
- `diag_proc` — diagnoses and procedures combined (ICD-10-CM, ICD-10-PCS, CPT4)
- `diag` — diagnoses only (condition_occurrence)
- `proc` — procedures and device exposures only
- `drugs` — drug exposures (RxNorm ingredient-level)
- `visits` — inpatient, outpatient, and emergency visits

**Time windows (bins)** — each group can have independent time windows. A `[30, 365]` setting produces two non-overlapping bins: days 1–30 before anchor and days 31–365. Multiple windows allow the model to distinguish recent from distant history.

**Catch-all** — when enabled, any code that appears fewer than a threshold number of times is lumped into a single `other` column per vocabulary, preventing rare codes from creating millions of near-zero columns.

**Number of threads** — controls FEMR's internal parallelism.

Values in Route A are counts of distinct days on which a code appeared with a changed value (delta-encoding removes repeated identical measurements). The result is a sparse CSR matrix; most entries are zero.

Clicking **Run extraction** spawns a subprocess (to avoid a Streamlit–multiprocessing deadlock), streams the log in real time, and caches the result to disk. A spec hash is computed from all parameters so that re-running with identical settings loads from cache instantly.

---

### Tab 4 · Route B · Labs

Extracts numerical lab-value features directly from `measurement.csv` using DuckDB, which scans the 22 GB file out-of-core without loading it into memory.

**Time windows** — unlike Route A's non-overlapping bins, Route B windows are cumulative: a `30d` window collects measurements from days 1–30, a `365d` window from days 1–365. This means that for each lab, a single DuckDB scan is performed up to the maximum window, and the per-window statistics are computed in pandas afterwards.

**LOINC codes** — by default, every LOINC-coded measurement in `concept.csv` is included. A text box allows restricting to a specific list of codes (one per line or comma-separated), which is useful for extracting a targeted panel (e.g., D-dimer, troponin I, BNP) quickly.

**Minimum studies per lab** — labs measured in fewer than this many studies are excluded. Raising this threshold keeps only the most commonly ordered labs and reduces the feature matrix width.

For each lab × window combination, Route B produces six aggregate columns: `last` (most recent value), `min`, `max`, `mean`, `n` (number of measurements), and `days_since` (days between the most recent measurement and the anchor). The result is a dense float32 matrix with NaN where a lab was not measured for a study. Column names follow the pattern `labs:LOINC/<code>_<agg>_<N>d`.

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

### Tab 5 · Describe

A GDC-style cohort builder for inspecting a loaded feature matrix without running any models. The tab lets you define a cohort filter and one or two stratification variables, then shows a summary table and charts for the resulting slice.

**Filters** — applied row-wise before any summary is computed. You can filter on task label (`y = 1`), anchor year range, or any demographic variable (sex, race, ethnicity, age band). Filters stack (AND logic).

**Stratify by** — breaks the cohort into subgroups. Available stratifiers include `sex`, `race_concept_id`, `ethnicity_concept_id`, `age_band` (decade buckets), `density_quartile` (by EHR record richness), and a free-text vocabulary slicer (e.g., "show only studies where the patient has at least one ICD-10 code matching `I26`").

**Summary table** — shows n, label prevalence, and data density per subgroup. Useful for checking whether a task is well-powered across subgroups before modelling.

**Feature density** — shows what fraction of columns are non-zero (Route A) or observed (Route B) per study, as a histogram. A very sparse matrix suggests the time windows may be too short or the cohort too selective.

---

### Tab 6 · Export

Writes the loaded feature matrix to a directory of flat files ready for modelling. All studies are exported together — train/test splitting is left to the downstream pipeline.

**Export directory** — the path where files will be written.

**Drop all-missing columns** — removes features that carry no information for any study (all-zero for Route A, all-NaN for Route B). Recommended; reduces file size substantially.

**Resolve human names** — when `concept.csv` is configured, writes `feature_names.csv` with a human-readable column name alongside the raw code-based name.

**Output files produced:**

For Route B (dense):
- `X.npy` — float32 array, shape (n_studies, n_features), NaN = not measured
- `X_mask.npy` — uint8 array, 1 = value observed, 0 = missing
- `y.npy` — binary task label
- `tte.npy` / `event.npy` — survival outcome arrays (if available)
- `metadata.csv` — impression_id, patient_id, anchor_time, y, and survival columns per row
- `feature_names.csv` — raw and human-readable name for each column
- `load_survival.py` — ready-to-run script showing how to load the arrays and apply a train/test split for lifelines or scikit-survival
- `load_multimodal.py` — script showing how to join EHR features with imaging or NLP outputs using `impression_id`

For Route A (sparse), `X.npy` is replaced by `X_sparse.npz` (scipy CSR format).

---

## Caching and reproducibility

Every extraction is identified by a SHA-256 hash of its full specification (task, anchor, groups, windows, file paths). The hash is stored alongside the `.pkl` in `DATA_PROCESSED/femr_cache/`. Re-running with identical settings skips the extraction entirely and loads from cache. The accompanying `<hash>_spec.json` records all parameters and the route, so you can always reconstruct exactly what produced a given cache file.

---

## Typical workflow for a new lab-mate

1. Open **Tab 1 · Data Sources** and validate each path (most will be pre-filled correctly for the INSPECT server).
2. Open **Tab 2 · Configure**, select the task you care about, and confirm the anchor type.
3. Run **Tab 3 · Route A** or **Tab 4 · Route B** (or both) and wait for the log to finish. Route A typically takes 5–30 minutes depending on thread count; Route B depends on which LOINC codes are requested and can range from a few minutes (targeted panel) to an hour (all labs).
4. The extraction is now cached. Future sessions can skip steps 1–3 entirely and go straight to **Tab 0 · Load**.
5. Use **Tab 5 · Describe** to sanity-check the cohort — check prevalence, demographic balance, and data density.
6. Use **Tab 6 · Export** to write the arrays to a folder, then use the generated `load_survival.py` as a starting point for your model.
