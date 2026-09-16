# INSPECT EHR Feature Extraction — Quick Start

A Streamlit app that turns raw OMOP CSVs into ML-ready feature matrices —
no code required. It scans OMOP tables out-of-core via DuckDB (nothing
large loads into RAM), so it's safe to run against the full ~22 GB
`measurement.csv` directly.

```bash
pip install -r Custom/appd_extract_requirements.txt
streamlit run Custom/app_feature_extraction.py
```

For column-level definitions and task semantics, see
`appd_EHR_FEATURE_EXTRACTION_GUIDE.md`. This page is just the fast on-ramp.

---

## 5-minute path to your first matrix

1. **Tab 0 · Load** — check if someone already extracted what you need.
   Cached extractions load instantly, no config required. If yes, skip to
   step 4.
2. **Tab 1 · Data sources** — confirm the four paths validate (green ✅):
   cohort file, OMOP CSVs directory, labels TSV (optional), `person.csv`
   (optional). Defaults are pre-filled for the standard INSPECT layout.
3. **Tab 2 · Extract** — pick a task, check **Labs** only, leave windows at
   the default `2, 7, 30, 365` (for faster and lighter extract, remove some time windows), and click **▶ Run extraction**. This is the
   cheapest extraction to start with (a few minutes) — add diagnoses/drugs/
   procedures/observations/visits once you know labs alone aren't enough or extract the features seperately to extract "submodalites".
   The log streams live; once it finishes the result is cached until deleted.
4. **Tab 3 · Describe** — filter the cohort (label, sex, age band, survival
   status) and sanity-check label prevalence and patient counts before you
   trust anything downstream.
5. **Tab 4 · Export** — write `X.npy` / `y.npy` / `metadata.csv` to a
   folder, or export a long-format CSV if you'd rather work in
   pandas/Excel/R. Every export ships a ready-to-run `load_survival.py`.
6. **Tab 5 · Timeline** — explore event density relative to CTPA or hospital
   admission. Two views: a **Cohort trajectory heatmap** (panel × time bin,
   black→red→pale-yellow colorscale, zero anchored to black) and a **Bubble
   timeline** (same data, bubble area proportional to the metric). Switch
   between CTPA-anchored mode (x-axis counts backwards to T0) and
   **Admission-anchored** mode (x-axis counts forward from day of admission
   as "Day 0", "Day 1", …). Toggle **"Hide 'Other …' panels"** (on by
   default) to suppress uninformative catch-all categories such as
   "Other drug" or "Other visit".

---

## Feature types (Tab 2)

| type | OMOP table | column prefix | value | default window |
|---|---|---|---|---|
| Labs | `measurement.csv` | `labs:` | real value (last/min/max/mean/n/days_since) | user-set, cumulative |
| Diagnoses | `condition_occurrence.csv` | `diag:` | distinct event days | 365 d |
| Drugs | `drug_exposure.csv` | `drug:` | distinct event days | 60 d |
| Procedures | `procedure_occurrence.csv` | `proc:` | distinct event days | 365 d |
| Observations | `observation.csv` | `obs:` | distinct event days | 365 d |
| Visits | `visit_occurrence.csv` | `visit:` | distinct visit days + LOS | 365 d |

Each type has its own independent lookback window. Enabling **person.csv**
in Tab 1 auto-appends 9 `demo:*` columns (age, sex, race, ethnicity) to
whatever you extract — no extra step needed.

---

## Things to consider!

- **Windows are cumulative, not bucketed.** `windows_days=[2, 30, 365]`
  means a measurement 5 days before anchor lands in `_30d` and `_365d`,
  not just one bucket.
- **NaN ≠ 0.** Labs use `NaN` for "not measured"; diagnoses/drugs/
  procedures/observations/visits use `0` for "not observed." Don't impute
  count columns with the same strategy you'd use for labs. The matrix
  viewer and coverage tab honour this distinction: a count feature with
  value `0` is treated as "not observed" (coverage = 0), not as a valid
  measurement — this prevents count features from appearing at 100%
  coverage when most patients simply had zero events.
- **`dx` vs `px` anchor.** Diagnostic tasks (e.g. `pe_positive`) anchor one
  day *before* `StudyTime`; prognostic tasks (e.g. `12_month_PH`,
  mortality/readmission) anchor *at* `StudyTime`. "auto" gets this right
  for you — only override it if you know why.
- **`Min studies per feature`** (default 50) drops any lab/code seen in
  fewer studies than that. If a feature you expect is missing from the
  matrix, check this threshold before assuming the extraction is broken.
- **Concept ancestor rollup is expensive.** It's off by default and only
  worth turning on if sparse, highly specific codes are hurting your model
  — start without it.
- **Re-running an identical config is free.** Every extraction is
  fingerprinted (MD5 hash of task + windows + feature types + paths, etc.)
  and cached under `DATA_PROCESSED/femr_cache/`. Changing one slider
  produces a new cache entry; it never overwrites an old one.
- **Admission trajectory re-run is free.** "Build admission trajectory" is
  a lightweight post-processing step over an already-extracted matrix — it
  does not re-run DuckDB extraction. The two-phase admission matching (see
  `appd_EHR_FEATURE_EXTRACTION_GUIDE.md §10.2`) runs in seconds.

---

## Where things live

- Cached extractions: `DATA_PROCESSED/femr_cache/<hash>.pkl` (+ `.log` and
  `_spec.json` recording exactly what produced it, safe to `git`-ignore,
  never to hand-edit).
- Exported arrays: wherever you point Tab 4's **Export directory** (default
  `DATA_PROCESSED/exports/<task>/`).

---

## Matrix viewer (`appd_matrix_viewer.py`)

A standalone Streamlit app for visually inspecting the exported feature
matrix (`X.npy`).  Run it separately from the main extractor:

```bash
streamlit run Custom/appd_matrix_viewer.py
```

| mode | colorscale | NaN handling |
|------|-----------|-------------|
| **Z-score** | dark-blue (−3) → black (0) → red → yellow (+3) | NaN renders as black (plot background) |
| **Values** | same scale, raw values | count features: `0` converted to NaN before z-scoring so "not observed" = black; all-NaN columns hidden |
| **Coverage** | % of studies with a non-missing value | count features: `X > 0` used as observed (not the NaN-based mask), so diagnoses/drugs/procedures report genuine event coverage |

The **Coverage** tab shows what fraction of studies have a non-missing value
for each feature. For lab features this uses the binary observed/absent mask;
for count features (`diag:`, `drug:`, `proc:`, `obs:`, `visit:`) it uses
`X > 0` because these columns encode absence as `0`, not `NaN`. Without this
distinction, every count feature would appear at 100% coverage (every study
has a `0`).

Questions or something looks wrong? See `appd_EHR_FEATURE_EXTRACTION_GUIDE.md`
or check the live log in Tab 2 first — most extraction issues (missing table,
empty LOINC filter, bad path) show up there in plain English.
