# INSPECT EHR Feature Extraction — Quick Start

A Streamlit app that turns raw OMOP CSVs into ML-ready feature matrices —
no code required. It scans OMOP tables out-of-core via DuckDB (nothing
large loads into RAM), so it's safe to run against the full ~22 GB
`measurement.csv` directly.

```bash
pip install -r Custom/extract_requirements.txt
streamlit run Custom/app_feature_extraction.py
```

For column-level definitions and task semantics, see
`EHR_FEATURE_EXTRACTION_GUIDE.md`. This page is just the fast on-ramp.

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
  count columns with the same strategy you'd use for labs.
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

---

## Where things live

- Cached extractions: `DATA_PROCESSED/femr_cache/<hash>.pkl` (+ `.log` and
  `_spec.json` recording exactly what produced it, safe to `git`-ignore,
  never to hand-edit).
- Exported arrays: wherever you point Tab 4's **Export directory** (default
  `DATA_PROCESSED/exports/<task>/`).

Questions or something looks wrong? See EHR_FEATURE_EXTRACTION_GUIDE.md. or check the live log in Tab 2 first, most extraction issues (missing table, empty LOINC filter, bad path) show
up there in plain English.
