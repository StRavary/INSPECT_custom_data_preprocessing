# INSPECT EHR Feature Extraction — Access Guide

A practical guide to getting features and labels out of the INSPECT EHR data.
Written for someone who has not worked with this dataset before.

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

| what | path | size |
|---|---|---|
| cohort master file | `DATA_PROCESSED/cohort_0.2.0_master_file_anon.csv` | 6 MB |
| labels (incl. survival) | `DATA_RAW/LABELS/labels_20250611.tsv` | 4 MB |
| canonical splits | `DATA_RAW/LABELS/splits_20250611.tsv` | 0.5 MB |
| FEMR database (processed) | `DATA_RAW/EHR_FEMR_DB/extract/` | 21 GB |
| raw OMOP CSVs | `DATA_RAW/EHR_CSV/` | 42 GB |

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
sampling frequency, or control over ontology expansion depth. Query with DuckDB
rather than pandas — it reads CSV/Parquet out-of-core, so 42 GB is comfortable.

| table | size | est. rows |
|---|---|---|
| measurement | 22.7 GB | ~143M |
| observation | 2.9 GB | ~27M |
| condition_occurrence | 1.4 GB | ~12.5M |
| drug_exposure | 2.1 GB | ~11.8M |
| procedure_occurrence | 1.0 GB | ~8.7M |
| visit_occurrence | 0.7 GB | ~4.3M |
| concept_ancestor | 1.7 GB | ~79M |
| concept | 1.1 GB | ~9M |

**Route A is de-duplicated relative to Route B** — see §6.1. If you need time
series, you need Route B.

---

## 3. Anchoring — read this one

Every feature is computed backwards from an **anchor time**. Which anchor
depends on whether the task is diagnostic or prognostic.

| kind | anchor | tasks |
|---|---|---|
| **dx** (diagnostic) | `StudyTime − 1 day` | `pe_positive_nlp`, `pe_positive`, `pe_acute`, `pe_subsegmentalonly` |
| **px** (prognostic) | `StudyTime` | all mortality, all readmission, `12_month_PH`, and the five radiographic findings |

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

### Option A — the interface

```bash
cd Custom
../.venv_legacy/bin/python -m streamlit run ./app_feature_extraction.py
```

Three tabs. **Configure** validates a spec and shows exactly which windows and
anchor will be used — instant, no data access, and the right place to start if
you are new to the dataset. **Extract** runs the job once and caches it.
**Describe** explores slices of the result.

The Configure tab emits a YAML spec plus the equivalent Python call, so anything
you arrive at in the UI is reproducible from a script without it.

### Option B — the Python API

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

Anchoring is chosen automatically from the task name; `anchor="dx"|"px"`
overrides. Unknown task names raise rather than guessing.

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

Each column is one **(code, time-window)** pair. The value is the count of
qualifying events for that code inside that window, counted backwards from the
anchor.

| pattern | meaning |
|---|---|
| `age` | age at anchor, **z-scored** across all labels — not years |
| `<group>:<CODE>_<window>` | count of CODE in that window |
| `<group>:<CODE> <value>_<window>` | string-valued events, one column per (code, value) |
| `<group>:<CODE> [lo, hi)_<window>` | numeric-valued events bucketed by decile |

`<window>` is a `timedelta` such as `365 days, 0:00:00`, meaning anchor back to
365 days before it. Windows are consecutive and non-overlapping: bins `[2, 30]`
give `[T0, T0−2d]` and `[T0−2d, T0−30d]`.

Internally the layout is `column_idx = code_col + bin_idx × num_codes` — all
codes for the first window, then all codes for the second, and so on.

### Ontology expansion — both leaf *and* ancestors

Every group runs with `is_ontology_expansion=True`, so each event increments
**its own code and every ancestor** in the OMOP hierarchy. FEMR's docstring: for
`A → B → C`, two occurrences of C also count as two of B and two of A.

Three consequences:

1. Both leaf and parent codes appear as columns — it is not one or the other.
2. **A parent column is the sum over its descendants**, so columns are strongly
   collinear. Do not read individual coefficients as independent effects; use
   permutation or grouped importance.
3. The feature space is much wider than the number of distinct raw codes.

Route B lets you cap this: `concept_ancestor.min_levels_of_separation <= 2`
expands only two levels. FEMR's API is all-or-nothing.

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

### 6.2 Windows silently discard older events

`CountFeaturizer` emits `len(time_bins)` bins but tracks `len + 1`. Events older
than the last bin edge land in a bucket that is never written out.

`bins=[1825]` therefore **discards everything beyond five years without warning**.
`temporal_features.py` appends a 100-year catch-all bin by default; pass
`catch_all=False` only if you mean to truncate.

### 6.3 Unsorted bins mislabel columns

`featurize()` sorts `time_bins` internally, but `get_column_name()` indexes the
original list. Unsorted bins produce mislabelled columns. The API always sorts
first — if you call FEMR directly, sort them yourself.

### 6.4 Missingness is informative

"Lab not drawn" is a clinical decision, not absent data. Encode presence
explicitly rather than imputing. Measurements cluster during instability and
vanish otherwise, so the sampling process is not missing-at-random.

### 6.5 `concept_id = 0`

In OMOP this means "no matching concept". Filter it in Route B or it becomes a
large junk column.

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

| group | columns |
|---|---|
| composition | `n_studies`, `n_patients`, `studies_per_patient`, `event_rate`, `n_positive` |
| survival | `censoring_rate`, `median_tte_days` |
| demographics | `age_mean`, `age_iqr`, `frac_female`, `n_care_sites`, `top_site_share` |
| density | `mean_distinct_codes`, `median_distinct_codes`, `mean_total_count` |
| vocabulary | `vocab_share_ICD10CM`, `vocab_share_LOINC`, … (shares sum to 1) |
| missingness | `missing_<column>` for each nominated key column |
| history proxy | `frac_with_oldest_window_events` |

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
* **Measurement values are unused.** `measurement.value_as_number` across ~143M
  rows is untouched. Last-value, min/max, slope and time-since-last are the
  standard strong clinical features and none exist yet.
* **Demographics are barely used.** Only z-scored age reaches the models. Sex,
  race, ethnicity and care site are all available in `person.csv`.

---

## 10. File map

| file | purpose |
|---|---|
| `Custom/app_feature_extraction.py` | Streamlit interface over both modules |
| `Custom/temporal_features.py` | patient-level feature extraction API |
| `Custom/context_descriptors.py` | per-slice descriptor tables |
| `Custom/x_generate_binned_features.py` | earlier CLI version (superseded) |
| `Custom/extract_temporal_features.py` | long-format event extraction with signed day windows |
| `Custom/9a_run_baseline_benchmark.py` | generates `labeled_patients.csv` per task |
| `Custom/INSPECT_Baseline_Reconstruction.md` | full methods and findings record |
