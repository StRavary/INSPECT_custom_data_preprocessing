"""
app_feature_extraction.py
-------------------------
Streamlit interface for INSPECT EHR feature extraction.
Extracts labs, diagnoses, drugs, and procedures from raw OMOP CSVs via DuckDB.

Tabs
----
  0 · Load          Browse and reload cached extractions
  1 · Data sources  Configure file paths (cohort, OMOP CSVs, labels, person.csv)
  2 · Extract       Choose feature types, set lookback windows, run extraction
  3 · Describe      Filter cohort + browse per-impression event history
  4 · Export        Write flat arrays and long-format CSV ready for modelling

Run (from the Custom/ directory):
    pip install -r extract_requirements.txt   # duckdb, streamlit, numpy, pandas, scipy
    streamlit run app_feature_extraction.py
    # or, with the bundled legacy venv (already has these installed):
    ../../.venv_legacy/bin/python -m streamlit run ./app_feature_extraction.py
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

# ---------------------------------------------------------------------------
# Constants and path defaults
# ---------------------------------------------------------------------------

DATA_ROOT = SCRIPT_DIR.parent.parent

OMOP_EXPECTED = [
    "measurement", "condition_occurrence", "drug_exposure",
    "procedure_occurrence", "concept", "person",
]

# Cached extractions are loaded with pickle.load() below (tab 0 · Load, and the
# Extract/Re-run flow). pickle.load() executes arbitrary code for a maliciously
# crafted file, so this is only safe as long as CACHE_DIR is writable solely by
# this tool/user — do not point it at shared or network storage with looser
# permissions than the rest of the repo.
CACHE_DIR             = DATA_ROOT / "DATA_PROCESSED" / "femr_cache"
ROUTE_B_WORKER_SCRIPT = SCRIPT_DIR / "appd_route_b_worker.py"

DATA_SOURCES = {
    "cohort": (
        "Cohort master file", False,
        DATA_ROOT / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv",
        "One row per CTPA study. Required columns: patient_id / PatientID, "
        "StudyTime, impression_id, plus task-label columns. "
        "A 'split' column is accepted but not required.",
    ),
    "omop": (
        "Raw OMOP CSVs directory", True,
        DATA_ROOT / "DATA_RAW" / "EHR_CSV",
        "Contains measurement.csv, condition_occurrence.csv, drug_exposure.csv, "
        "procedure_occurrence.csv, concept.csv, person.csv.",
    ),
    "labels": (
        "Labels TSV", False,
        DATA_ROOT / "DATA_RAW" / "LABELS" / "labels_20250611.tsv",
        "Survival outcomes (tte_* / is_censored_*). Optional if the cohort "
        "file already carries tte_* columns.",
    ),
    "person": (
        "person.csv (demographics)", False,
        DATA_ROOT / "DATA_RAW" / "EHR_CSV" / "person.csv",
        "Sex, race, ethnicity, birth date. Optional — enables demographic "
        "columns in the feature matrix and filters in the Describe tab.",
    ),
}

PATIENT_ID_CANDIDATES = ("patient_id", "PatientID", "person_id")


# ---------------------------------------------------------------------------
# Column glossary — plain-English descriptions surfaced two ways: as hover
# tooltips on dataframe column headers (see _glossary_column_config) and as
# an in-app "what do these columns mean?" expander (see _render_glossary),
# so a first-time user isn't sent to EHR_FEATURE_EXTRACTION_GUIDE.md just to
# find out that "tte" means "time to event".
# ---------------------------------------------------------------------------

COLUMN_GLOSSARY = {
    "impression_id": "Unique ID for one CTPA study.",
    "patient_id":     "Unique ID for the patient. One patient can have several studies.",
    "anchor_time":    "The date features are computed backwards from — StudyTime for "
                       "prognostic tasks, StudyTime − 1 day for diagnostic tasks.",
    "y":              "Binary task label for this study: 1 = positive, 0 = negative.",
    "tte_days":       "Time-to-event — days from anchor_time to the survival event, "
                       "or to censoring if none was observed.",
    "event":          "Survival event indicator: 1 = event observed, 0 = censored "
                       "(no event seen before the patient's data ends), NaN = unknown.",
    "sex":            "Patient sex, derived from OMOP gender_concept_id.",
    "age_years":      "Patient age in years at anchor_time.",
    "age_band":       "Age bucketed into ranges for cohort slicing.",
    "split":          "Canonical train / valid / test split. Patient-level — no "
                       "patient appears in more than one split.",
    "feature":        "Raw feature column name, e.g. labs:LOINC/2160-0_last_30d — "
                       "see the prefix legend below.",
    "feature_human_readable": "The same feature with OMOP codes resolved to plain "
                       "English via concept.csv.",
    "record_date":    "Date the underlying measurement/event actually happened "
                       "(not the anchor date).",
    "value":          "Observed value — a real number for labs/observations, NaN "
                       "for coded events (diagnoses, drugs, procedures, visits).",
    "days_before_anchor": "How many days before anchor_time this event occurred.",
    "days_before_ctpa":   "How many days before the CTPA study (anchor) this event occurred.",
    "source_table":   "Which OMOP table this event came from.",
    "event_type":     "Feature type this event belongs to (lab, diagnosis, drug, "
                       "procedure, observation, visit).",
    "vocabulary":     "Coding vocabulary for concept_code, e.g. LOINC, ICD10CM, RxNorm, CPT4.",
    "concept_code":   "Raw vocabulary code, e.g. '2160-0' (LOINC) or 'I26.90' (ICD-10-CM).",
    "concept_name":   "Human-readable name for the code, resolved via concept.csv.",
    "concept_id":     "OMOP's internal integer ID for a concept.",
    "event_date":     "Calendar date of the underlying event.",
    "event_datetime": "Timestamp of the underlying event (midnight if no time was recorded).",
}

# (prefix, meaning) — the column-name convention for feature columns, which
# COLUMN_GLOSSARY can't cover one-by-one since there are tens of thousands of
# them (one per code × window).
PREFIX_GLOSSARY = [
    ("labs:",  "real-valued lab result — e.g. `labs:LOINC/2160-0_last_30d` = "
               "creatinine, most recent value, 30-day window"),
    ("diag:",  "count of distinct days a diagnosis code was recorded"),
    ("drug:",  "count of distinct days a drug was administered or dispensed"),
    ("proc:",  "count of distinct days a procedure was recorded"),
    ("obs:",   "count of distinct days an OMOP \"observation\" concept was recorded "
               "(smoking status, functional status, etc.)"),
    ("visit:", "count of distinct visit days; `visit:LOS/total_Nd` / "
               "`visit:LOS/max_Nd` give length-of-stay instead"),
    ("demo:",  "demographic feature from person.csv (age, sex, race, ethnicity)"),
    ("*_anc:", "optional concept-ancestor rollup — same meaning as the base "
               "prefix, but for parent/ancestor codes (e.g. `diag_anc:`)"),
]


def _glossary_column_config(df) -> dict:
    """Hover-tooltip help text (Streamlit column_config) for any column in
    `df` that's in COLUMN_GLOSSARY. Columns not in the glossary — including
    the thousands of labs:/diag:/... feature columns — are left with no
    special config, so they just render normally."""
    import streamlit as st
    return {
        col: st.column_config.Column(help=COLUMN_GLOSSARY[col])
        for col in df.columns if col in COLUMN_GLOSSARY
    }


def _render_glossary(st, key: str) -> None:
    """Collapsed-by-default 'what do these columns mean?' expander."""
    with st.expander("ℹ️ What do these columns mean?", key=key):
        st.markdown(
            "**Study metadata**\n\n"
            + "\n".join(
                f"- `{c}` — {COLUMN_GLOSSARY[c]}"
                for c in ("impression_id", "patient_id", "anchor_time", "y",
                          "tte_days", "event", "sex", "age_years", "age_band")
                if c in COLUMN_GLOSSARY
            )
            + "\n\n**Feature column prefixes**\n\n"
            + "\n".join(f"- `{p}` — {d}" for p, d in PREFIX_GLOSSARY)
            + "\n\n**Feature column suffixes** — `_Nd` = lookback window in "
              "days before anchor. Lab columns additionally carry an "
              "aggregate tag: `_last`, `_min`, `_max`, `_mean`, `_n` (count "
              "of distinct measurement days), `_days_since` (days since the "
              "most recent measurement).\n\n"
              "Full reference: `EHR_FEATURE_EXTRACTION_GUIDE.md`."
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _line_count(path, cap: int = 5_000_000) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
            if n >= cap:
                break
    return n


def validate_source(kind: str, path) -> tuple[bool, list[str]]:
    label, is_dir, _, _ = DATA_SOURCES[kind]
    p = Path(path).expanduser()
    if not p.exists():
        return False, [f"does not exist: {p}"]
    if is_dir and not p.is_dir():
        return False, ["expected a directory"]
    if not is_dir and not p.is_file():
        return False, ["expected a file"]

    msgs: list[str] = []
    try:
        if kind == "cohort":
            n = _line_count(p) - 1
            with open(p) as f:
                header = f.readline().rstrip("\n").split(
                    "\t" if p.suffix == ".tsv" else ",")
            hset = set(header)
            pid = next((c for c in PATIENT_ID_CANDIDATES if c in hset), None)
            missing = {"StudyTime", "impression_id"} - hset
            if pid is None:
                missing.add("patient_id / PatientID")
            msgs.append(f"{n:,} rows, {len(header)} columns")
            if missing:
                return False, msgs + [f"missing required columns: {sorted(missing)}"]
            msgs.append(f"required columns present (patient id: {pid})")
            n_tte = len([c for c in header if c.startswith("tte_")])
            if n_tte:
                msgs.append(f"{n_tte} survival endpoints in-file")

        elif kind == "omop":
            found = [t for t in OMOP_EXPECTED
                     if (p / f"{t}.csv").exists()]
            total_gb = sum(
                (p / f"{t}.csv").stat().st_size
                for t in found) / 1024 ** 3
            msgs.append(f"{len(found)}/{len(OMOP_EXPECTED)} expected tables, "
                        f"{total_gb:.1f} GB")
            missing = [t for t in OMOP_EXPECTED if t not in found]
            if missing:
                msgs.append(f"missing: {', '.join(missing)}")

        elif kind == "labels":
            n = _line_count(p) - 1
            with open(p) as f:
                header = f.readline().rstrip("\n").split("\t")
            tte = [c for c in header if c.startswith("tte_")]
            msgs.append(f"{n:,} rows, {len(tte)} survival endpoints")
            if not tte:
                return False, msgs + ["no tte_* columns found"]

        elif kind == "person":
            n = _line_count(p) - 1
            with open(p) as f:
                header = f.readline().rstrip("\n").split(",")
            msgs.append(f"{n:,} rows")
            if "person_id" not in header:
                return False, msgs + ["no person_id column"]

    except (OSError, UnicodeDecodeError) as e:
        return False, [f"could not read: {e}"]

    return True, msgs


def list_dir_entries(path, dirs_only: bool = False):
    p = Path(path)
    try:
        entries = sorted(p.iterdir(), key=lambda e: e.name.lower())
    except (OSError, PermissionError):
        return [], []
    dirs  = [e.name for e in entries if e.is_dir()]
    files = [] if dirs_only else [e.name for e in entries if e.is_file()]
    return dirs, files


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _spec_hash(spec: dict) -> str:
    return hashlib.md5(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()[:12]


def _cache_pkl(spec: dict) -> Path:
    return CACHE_DIR / f"{_spec_hash(spec)}.pkl"


def _cache_log(spec: dict) -> Path:
    return CACHE_DIR / f"{_spec_hash(spec)}.log"


def _cache_spec(spec: dict) -> Path:
    return CACHE_DIR / f"{_spec_hash(spec)}_spec.json"


def _list_cached() -> list[tuple[str, Path]]:
    """Return [(spec_hash, pkl_path)] for every finished cached extraction."""
    if not CACHE_DIR.exists():
        return []
    out = []
    for p in sorted(CACHE_DIR.glob("*.pkl"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        chash     = p.stem
        spec_file = CACHE_DIR / f"{chash}_spec.json"
        # skip any legacy Route A caches
        if spec_file.exists():
            try:
                route = json.loads(spec_file.read_text()).get("route", "A")
                if route != "B":
                    continue
            except Exception:
                continue
        out.append((chash, p))
    return out


# ---------------------------------------------------------------------------
# Export directory naming — sustainable multi-extraction layout
# ---------------------------------------------------------------------------
# The export defaults used to be exports/<task>/ with fixed filenames
# (X.npy, metadata.csv, ...) regardless of *which* extraction for that task
# produced them. Two different extractions for the same task (e.g. labs-only
# vs. drugs-only) would silently overwrite each other's files if exported to
# the default path. The fix: key the export directory off the same spec hash
# already used for caching, so it's automatically unique per extraction
# config — plus a short human-readable tag so the folder name still says
# something at a glance in a file browser.

_FT_ABBR = {
    "labs": "labs", "diagnoses": "diag", "drugs": "drug",
    "procedures": "proc", "observations": "obs", "visits": "visit",
}


def _feature_summary(fm) -> str:
    """Short human-readable tag from an fm's feature types, e.g. 'labs' or
    'labs+diag+drug'. Not guaranteed unique on its own — pair with a hash."""
    fts = list(getattr(fm, "feature_types", None) or ["labs"])
    return "+".join(_FT_ABBR.get(ft, ft) for ft in fts) or "labs"


def _export_subdir(fm, fm_hash: "str | None") -> str:
    """{feature_summary}_{hash8}, or just {feature_summary} if this fm's
    cache hash isn't known (e.g. a pickle loaded before this existed) —
    still human-readable, just without the collision guarantee."""
    tag = _feature_summary(fm)
    return f"{tag}_{fm_hash[:8]}" if fm_hash else tag


def _default_concept_csv(omop_path: str) -> Path:
    return Path(omop_path) / "concept.csv"


# ---------------------------------------------------------------------------
# Concept-map cache
# ---------------------------------------------------------------------------

def _load_concept_map_cached(concept_csv: str, st):
    """Streamlit-cached wrapper around route_b_labs.load_concept_map."""
    from Custom.appd_route_b_labs import load_concept_map

    @st.cache_resource(show_spinner="Loading OMOP concept names …")
    def _load(path: str):
        return load_concept_map(path)

    return _load(concept_csv)


def _load_concept_id_map_cached(concept_csv: str, st):
    """Streamlit-cached wrapper around route_b_labs.load_concept_id_map."""
    from Custom.appd_route_b_labs import load_concept_id_map

    @st.cache_resource(show_spinner="Loading concept ID map …")
    def _load(path: str):
        return load_concept_id_map(path)

    return _load(concept_csv)


# ---------------------------------------------------------------------------
# Long-format (melted) export helper
# ---------------------------------------------------------------------------

# Hard cap on the number of (study × feature) cells the in-memory long-format
# builder (_build_long_df) will materialise. A wide extraction easily has
# 20,000+ feature columns; at 22,457 studies that's 450M+ cells if every one
# is kept — enough to exhaust RAM and freeze the machine building it. This
# applies to observed_only=True as well as False: a wide window (e.g. 365d)
# means most labs really are observed for most studies, so "observed only"
# alone can still leave tens/hundreds of millions of rows.
#
# This cap does NOT apply to _build_long_df_streamed — that path writes to
# disk one study-chunk at a time, so its peak memory is bounded by one chunk
# regardless of the total row count. Use it for extractions too large for
# the in-memory path (see the app's "Stream full CSV to disk" button).
_LONG_DF_MAX_CELLS = 20_000_000


def _prep_long_df(fm, concept_map: dict):
    """Shared setup for _build_long_df / _build_long_df_streamed: everything
    that's O(n_studies) or O(n_features) — cheap regardless of extraction
    size — and independent of which studies are being melted right now.

    Critically, this does **not** copy `fm.X`. An earlier version did
    ``fm.X[:, feat_idx]`` to drop the handful of ``demo:`` columns before
    melting — for a "labs only" extraction, `feat_idx` excludes at most 9
    columns out of possibly tens of thousands, so that line was silently
    duplicating almost the *entire* feature matrix in RAM before any
    chunking even started, on top of the copy already held by the loaded
    `fm`. `demo:` columns are excluded later instead, via a boolean mask
    applied in-place per chunk (see `_melt_batch`), which costs nothing.

    Returns a dict consumed by _melt_batch and both builders below it.
    """
    import pandas as pd
    import numpy as np
    from Custom.appd_route_b_labs import humanize_column
    import re

    col_list = list(fm.columns)
    col_arr  = np.array(col_list, dtype=object)   # aligned 1:1 with fm.X's columns
    col_idx  = {c: i for i, c in enumerate(col_list)}
    is_demo  = np.array([c.startswith("demo:") for c in col_list])

    def _get_demo(name):
        # Single-column basic indexing (`fm.X[:, i]` with a scalar i) is a
        # numpy *view*, not a copy of the matrix — only .copy() below (of
        # that one column) allocates anything, and it's O(n_studies).
        if name in col_idx:
            return fm.X[:, col_idx[name]].copy()
        return np.full(len(fm.impression_ids), np.nan, dtype=np.float32)

    is_f = _get_demo("demo:is_female")
    # Lowercase to match the GENDER-derived "female"/"male"/"unknown" labels used
    # elsewhere in the app (Describe tab, ContextDescriber) — keeps `sex` values
    # joinable/comparable across the long-format export and the cohort table.
    sex = np.where(np.isnan(is_f), "unknown",
          np.where(is_f == 1.0,    "female",  "male"))

    meta = fm.to_frame().reset_index(drop=True)   # impression_id, patient_id, …
    meta["sex"]       = sex
    meta["age_years"] = _get_demo("demo:age_years")
    if "tte_days" not in meta.columns:
        meta["tte_days"] = np.nan
        meta["event"]    = np.nan

    id_vars = ["impression_id", "patient_id", "anchor_time", "y",
               "tte_days", "event", "sex", "age_years"]
    out_cols = id_vars + ["feature", "record_date", "feature_human_readable", "value"]

    # Labs: record_date = anchor_time - days_since (already in the matrix).
    # Count features: record_date from fm.count_dates (most-recent event date).
    # Indices below are into the FULL column space (same as fm.X's columns),
    # not a demo-excluded subset — demo: columns simply never match either
    # pattern, so they naturally end up with ds_source == -1.
    _LAB_COL_PAT    = re.compile(r'^(labs:[^_]+(?:_[^_]+)*)_([^_]+)_(\d+d)$')
    _DAYS_SINCE_PAT = re.compile(r'^(labs:.+)_days_since_(\d+d)$')

    ds_local_idx: dict = {}   # (base, window) -> global column index of the days_since column
    for j, c in enumerate(col_list):
        m = _DAYS_SINCE_PAT.match(c)
        if m:
            ds_local_idx[(m.group(1), m.group(2))] = j

    n_total = len(col_list)
    ds_source = np.full(n_total, -1, dtype=np.int64)
    for j, c in enumerate(col_list):
        if c.startswith("labs:"):
            m = _LAB_COL_PAT.match(c)
            if m:
                ds_source[j] = ds_local_idx.get((m.group(1), m.group(3)), -1)

    count_dates = getattr(fm, "count_dates", None)   # None for old pickles
    count_dates_cols = set(count_dates.columns) if count_dates is not None else set()
    anchor_dt = pd.to_datetime([str(a) for a in fm.anchor_times]).to_numpy()

    # Human-readable names computed once for every non-demo feature column up
    # front — O(n_features), shared across every study-chunk that gets melted.
    name_map = {c: humanize_column(c, concept_map)
                for c in col_arr[~is_demo]}

    return dict(
        X=fm.X, meta=meta, col_arr=col_arr, is_demo=is_demo, ds_source=ds_source,
        count_dates=count_dates, count_dates_cols=count_dates_cols,
        anchor_dt=anchor_dt, name_map=name_map, id_vars=id_vars,
        out_cols=out_cols, n_imp=fm.X.shape[0],
        n_feat=n_total - int(is_demo.sum()),
    )


def _melt_batch(prep: dict, batch_rows, observed_only: bool) -> "pd.DataFrame":
    """Melt one chunk of studies to long format.

    `batch_rows` is either `None` — use the full matrix as-is, no row copy,
    for the single-shot in-memory path where the row set already *is*
    everything — or an int array of a few thousand study indices for one
    streamed chunk. Either way this is the only place that ever slices rows
    out of `fm.X`, and only by the number of rows actually requested.
    """
    import pandas as pd
    import numpy as np

    X, meta, col_arr = prep["X"], prep["meta"], prep["col_arr"]
    is_demo                 = prep["is_demo"]
    ds_source, count_dates  = prep["ds_source"], prep["count_dates"]
    count_dates_cols        = prep["count_dates_cols"]
    anchor_dt, name_map     = prep["anchor_dt"], prep["name_map"]
    id_vars, out_cols       = prep["id_vars"], prep["out_cols"]

    if batch_rows is None:
        Xb = X                              # no copy — use the matrix directly
        abs_rows = np.arange(X.shape[0])
    else:
        Xb = X[batch_rows]                  # copy, but only of this chunk
        abs_rows = np.asarray(batch_rows)

    if observed_only:
        keep_mask = ~np.isnan(Xb) & (Xb != 0.0)
    else:
        keep_mask = np.ones(Xb.shape, dtype=bool)
    keep_mask[:, is_demo] = False           # demo: columns are metadata, not features

    r_local, col_pos = np.nonzero(keep_mask)
    if r_local.size == 0:
        return pd.DataFrame(columns=out_cols)
    row_idx = abs_rows[r_local]
    n_kept  = row_idx.size

    long = pd.DataFrame({c: meta[c].to_numpy()[row_idx] for c in id_vars})
    long["feature"] = col_arr[col_pos]
    long["value"]   = Xb[r_local, col_pos]

    record_date = np.full(n_kept, "", dtype=object)
    # Group kept cells by feature column (via argsort, not a python loop over
    # cells) so each column's date logic runs once as one vectorised op,
    # touching only the cells kept for that column in this batch.
    order          = np.argsort(col_pos, kind="stable")
    col_pos_sorted = col_pos[order]
    boundaries     = np.flatnonzero(np.diff(col_pos_sorted)) + 1
    for positions in np.split(order, boundaries):
        j    = col_pos[positions[0]]
        rows = row_idx[positions]
        c    = col_arr[j]
        if ds_source[j] != -1:
            ds_vals = X[rows, ds_source[j]].astype(float)
            valid   = ~np.isnan(ds_vals)
            if valid.any():
                dates = anchor_dt[rows[valid]] - pd.to_timedelta(
                    ds_vals[valid], unit="d")
                record_date[positions[valid]] = (
                    pd.DatetimeIndex(dates).strftime("%Y-%m-%d"))
        elif c in count_dates_cols:
            col_arr_dates = count_dates[c].fillna("").astype(str).to_numpy()
            record_date[positions] = col_arr_dates[rows]
    long["record_date"] = record_date

    long["feature_human_readable"] = long["feature"].map(name_map)
    return long[out_cols]


def _build_long_df(fm, concept_map: dict, observed_only: bool = True) -> "pd.DataFrame":
    """Melt a LabFeatureMatrix to one row per (impression × observed feature),
    entirely in memory. See module docstring on `_LONG_DF_MAX_CELLS` for why
    this refuses very large extractions — use `_build_long_df_streamed` for
    those instead.

    Metadata columns (impression_id, patient_id, anchor_time, y, tte_days, event,
    sex, age_years) are repeated on every row.  Demo features (demo:*) are
    extracted into those metadata columns and excluded from the feature rows.

    Parameters
    ----------
    fm            : LabFeatureMatrix
    concept_map   : {VOCAB/code: human_name} from load_concept_map()
    observed_only : if True (default), drop rows where value is NaN or 0.

    Returns
    -------
    pd.DataFrame with columns:
        impression_id | patient_id | anchor_time | y | tte_days | event |
        sex | age_years | feature | feature_human_readable | value
    """
    import pandas as pd
    import numpy as np

    prep = _prep_long_df(fm, concept_map)
    n_imp, n_feat = prep["n_imp"], prep["n_feat"]

    # ── cheap pre-check: know the output row count before building anything.
    #    Computed directly on prep["X"] (the original fm.X, no copy) — the
    #    boolean mask allocated here is 1 byte/cell vs. float32's 4, and
    #    unlike a column-subset copy it never duplicates the underlying data.
    if observed_only:
        X = prep["X"]
        keep_mask = ~np.isnan(X) & (X != 0.0)
        keep_mask[:, prep["is_demo"]] = False
        n_kept = int(keep_mask.sum())
        del keep_mask
        if n_kept > _LONG_DF_MAX_CELLS:
            pct = 100 * n_kept / (n_imp * n_feat) if n_imp * n_feat else 0
            raise MemoryError(
                f"Long-format table would still have {n_kept:,} observed "
                f"rows ({n_imp:,} studies × {n_feat:,} features, "
                f"{pct:.0f}% observed) — refusing to build it in memory, "
                f"this is what has frozen machines before. A wide lookback "
                f"window means most labs really are observed for most "
                f"studies, so 'Observed features only' alone doesn't "
                f"shrink this enough. Use 'Stream full CSV to disk' "
                f"instead — it writes the same {n_kept:,} rows a chunk of "
                f"studies at a time with no such cap — or narrow the "
                f"extraction (fewer feature types, a narrower window, "
                f"specific LOINC codes instead of 'all LOINC').")
    else:
        n_cells = n_imp * n_feat
        if n_cells > _LONG_DF_MAX_CELLS:
            raise MemoryError(
                f"Complete long-format table would have {n_cells:,} rows "
                f"({n_imp:,} studies × {n_feat:,} features) — refusing to "
                f"build it in memory, this is what has frozen machines "
                f"before. Use 'Stream full CSV to disk' instead (no cap), "
                f"check 'Observed features only' (usually 10-100x fewer "
                f"rows), or narrow the extraction and re-run.")
        n_kept = n_cells

    if n_kept == 0:
        return pd.DataFrame(columns=prep["out_cols"])

    # batch_rows=None → _melt_batch uses prep["X"] directly, no row copy.
    long = _melt_batch(prep, None, observed_only)
    return long.sort_values(["impression_id", "feature"]).reset_index(drop=True)


# Target size, in cells (studies × features), for one chunk's dense
# (batch_rows × n_features) slice inside _melt_batch — i.e. the actual thing
# that determines a chunk's memory footprint. A *fixed row-count* chunk
# silently stops bounding anything once the extraction is wide: "all LOINC,
# labs only" with no filter easily produces 50,000-150,000+ columns, at
# which point even a modest few thousand studies per chunk is hundreds of
# MB to multiple GB — this is what was still freezing the app despite
# chunking existing. Sizing by cells instead means chunk_studies shrinks
# automatically for wide extractions instead of stopping being a bound at
# all. ~2M cells × 4 bytes (float32 slice) + 1 byte (bool mask) ≈ 10 MB of
# actual matrix data per chunk — deliberately conservative headroom, since
# the melted output rows add more on top depending on density.
_STREAM_CHUNK_CELLS = 2_000_000


def _build_long_df_streamed(
    fm, concept_map: dict, out_path, observed_only: bool = True,
    chunk_studies: "int | None" = None, on_progress=None,
) -> int:
    """Same output as _build_long_df, written straight to a CSV file on disk
    a bounded-memory chunk of studies at a time instead of held in memory
    all at once — one file, written progressively; there's no separate
    "fuse the chunks together" step because each chunk is appended directly
    to the same open file handle as it's produced.

    Peak memory scales with one chunk's cell count (~`_STREAM_CHUNK_CELLS`
    by default), not the full result, so there is no `_LONG_DF_MAX_CELLS`-
    style cap here — this is the path for extractions too large for the
    in-memory builder (e.g. the full cohort, labs only, 365-day window, no
    LOINC filter).

    The output is written grouped by chunk (i.e. by the extraction's
    original study order) rather than globally sorted by impression_id —
    sorting the whole file would need the whole file in memory, defeating
    the point. Each chunk is internally sorted by `["impression_id",
    "feature"]` for local readability.

    Parameters
    ----------
    fm, concept_map, observed_only : see _build_long_df
    out_path      : destination CSV path (parent dir created if needed)
    chunk_studies : studies processed per chunk. Default (None) auto-sizes
                    this from `_STREAM_CHUNK_CELLS // n_features`, so it
                    shrinks automatically for wide extractions instead of
                    needing to be hand-tuned per extraction. Pass an
                    explicit value to override (e.g. lower it further on a
                    memory-constrained machine).
    on_progress   : optional callable(studies_done, studies_total) invoked
                    after each chunk — wire to st.progress() for a live bar.

    Returns
    -------
    int : total number of rows written.
    """
    import gc
    import numpy as np
    from pathlib import Path as _Path

    prep  = _prep_long_df(fm, concept_map)
    n_imp, n_feat = prep["n_imp"], prep["n_feat"]

    if chunk_studies is None:
        chunk_studies = max(1, _STREAM_CHUNK_CELLS // max(n_feat, 1))

    out_path = _Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with open(out_path, "w", newline="") as fh:
        for start in range(0, n_imp, chunk_studies):
            batch_rows = np.arange(start, min(start + chunk_studies, n_imp))
            chunk_df = _melt_batch(prep, batch_rows, observed_only)
            if not chunk_df.empty:
                chunk_df = chunk_df.sort_values(["impression_id", "feature"])
            chunk_df.to_csv(fh, index=False, header=(start == 0))
            total_rows += len(chunk_df)
            del chunk_df, batch_rows
            gc.collect()   # encourage prompt release of this chunk's arrays
            if on_progress:
                on_progress(min(start + chunk_studies, n_imp), n_imp)

    return total_rows


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def _do_export(fm, export_dir: str, drop_zero: bool, concept_map: dict,
                _fm_hash: "str | None" = None) -> str:
    """Write flat (unsplit) files to export_dir. Returns a summary string.

    _fm_hash : cache hash of the extraction being exported (from
               st.session_state["fm_hash"]), used to locate and copy the
               original full spec.json as extraction_spec.json in the
               export directory. None if unknown (older pickle, or loaded
               before fm_hash tracking existed) — falls back to a partial
               manifest built from whatever fm itself still carries.

    X.npy is always dense — LabFeatureMatrix.X is a dense float32 array by
    design (unlike the retired Route A, which used scipy.sparse). For a
    very wide extraction this can be a large file, and — more importantly —
    loading it back (e.g. via the generated load_survival.py's
    `np.load(".../X.npy")`) needs that much RAM again downstream.
    """
    import pandas as pd
    import numpy as np

    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Manifest: what exactly produced these files. Independent of however
    # the export directory ends up named/moved/renamed — always copy the
    # full extraction spec if we know which cache entry this fm came from;
    # fall back to whatever fm itself still carries (older pickles predating
    # fm_hash tracking won't have a matching spec.json, but still have this
    # much on the object).
    fm_hash = _fm_hash
    spec_src = CACHE_DIR / f"{fm_hash}_spec.json" if fm_hash else None
    if spec_src and spec_src.exists():
        (out / "extraction_spec.json").write_text(spec_src.read_text())
    else:
        partial = {
            "task":               fm.task,
            "anchor":             fm.anchor_kind,
            "feature_types":      getattr(fm, "feature_types", None),
            "windows_days":       fm.windows_days,
            "count_window_days":  getattr(fm, "count_window_days", None),
            "admission_anchored": getattr(fm, "admission_anchored", False),
            "note": "Reconstructed from the loaded extraction, not the "
                    "original cache spec — fm_hash unknown or its "
                    "_spec.json was missing/deleted, so LOINC filter, "
                    "min_studies thresholds, and ancestor-rollup settings "
                    "aren't recoverable here.",
        }
        (out / "extraction_spec.json").write_text(json.dumps(partial, indent=2))

    raw_cols   = list(fm.columns)
    human_cols = (fm.human_columns(concept_map) if concept_map else raw_cols)

    # Drop all-NaN columns
    X = fm.X
    if drop_zero:
        keep       = ~np.all(np.isnan(X), axis=0)
        X          = X[:, keep]
        raw_cols   = [c for c, k in zip(raw_cols,   keep) if k]
        human_cols = [c for c, k in zip(human_cols, keep) if k]

    n_dropped = len(fm.columns) - len(raw_cols)

    pd.DataFrame({"raw": raw_cols, "human": human_cols}).to_csv(
        out / "feature_names.csv", index_label="col_index")

    meta = fm.to_frame()
    meta.to_csv(out / "metadata.csv", index=False)

    # copy=False: X is already float32 (LabFeatureMatrix's dtype by
    # construction) — without this, .astype() silently duplicates the whole
    # matrix in memory just to convert a dtype that's already correct.
    np.save(out / "X.npy",      X.astype(np.float32, copy=False))
    np.save(out / "X_mask.npy", (~np.isnan(X)).astype(np.uint8))
    np.save(out / "y.npy", fm.y)

    has_survival = fm.tte is not None
    if has_survival:
        np.save(out / "tte.npy",   fm.tte)
        np.save(out / "event.npy", fm.event)

    has_admission = getattr(fm, "admission_anchored", False)

    # ── load_survival.py ─────────────────────────────────────────────────
    surv_lines = f'''\
"""
load_survival.py  —  auto-generated by app_feature_extraction.py
Load {fm.task} EHR features for survival analysis.
All studies are in a single flat array — apply your own train/test split downstream.
{"Admission-anchored: each study's window is [admission_date, anchor_time] "
 "(see metadata.csv columns admission_date / days_since_admission), capped by "
 "the configured windows as an outer ceiling." if has_admission else ""}
"""
import numpy as np
import pandas as pd

BASE = "{out}"

X     = np.load(f"{{BASE}}/X.npy")        # float32, NaN = not measured
# X_mask = np.load(f"{{BASE}}/X_mask.npy") # 1 = observed, 0 = missing
y     = np.load(f"{{BASE}}/y.npy")
meta  = pd.read_csv(f"{{BASE}}/metadata.csv")
feat  = pd.read_csv(f"{{BASE}}/feature_names.csv")
{"tte   = np.load(f'{out}/tte.npy')" if has_survival else "tte   = None  # not available for this task"}
{"event = np.load(f'{out}/event.npy')" if has_survival else "event = None"}

print(f"studies: {{len(y):,}}  features: {{len(feat):,}}  prevalence: {{y.mean():.3f}}")

# ── apply your own split ──────────────────────────────────────────────────
# from sklearn.model_selection import train_test_split
# idx_tr, idx_te = train_test_split(range(len(y)), test_size=0.2, random_state=42,
#                                   stratify=y)
# X_tr, X_te = X[idx_tr], X[idx_te]
# y_tr, y_te = y[idx_tr], y[idx_te]

# ── imputation (NaN = not measured for this patient) ──────────────────────
# from sklearn.impute import SimpleImputer
# imp = SimpleImputer(strategy="median").fit(X_tr)
# X_tr = imp.transform(X_tr)

# ── scikit-survival ───────────────────────────────────────────────────────
# from sksurv.util import Surv
# from sksurv.ensemble import RandomSurvivalForest
# y_surv_tr = Surv.from_arrays(event=event[idx_tr].astype(bool), time=tte[idx_tr])
# rsf = RandomSurvivalForest(n_estimators=100).fit(X_tr, y_surv_tr)
'''
    (out / "load_survival.py").write_text(surv_lines)

    # ── load_multimodal.py ───────────────────────────────────────────────
    mm_lines = f'''\
"""
load_multimodal.py  —  auto-generated by app_feature_extraction.py
Join {fm.task} EHR features with imaging / NLP outputs
using impression_id.
"""
import numpy as np
import pandas as pd

BASE = "{out}"

X_ehr = np.load(f"{{BASE}}/X.npy")   # float32, NaN = not measured
meta  = pd.read_csv(f"{{BASE}}/metadata.csv")
feat  = pd.read_csv(f"{{BASE}}/feature_names.csv")

print(meta[["impression_id", "patient_id", "y"]].head())

# 1. Join with imaging embeddings
# imaging_df = pd.read_csv("path/to/image_embeddings.csv")
# merged = meta.merge(imaging_df, on="impression_id", how="inner")
# row_idx = meta.index.get_indexer(merged.index)
# X_ehr_aligned = X_ehr[row_idx]

# 2. Concatenate EHR + imaging
# X_image = np.load("path/to/image_embeddings.npy")[row_idx]
# X_joint = np.hstack([np.nan_to_num(X_ehr_aligned), X_image])
'''
    (out / "load_multimodal.py").write_text(mm_lines)

    return (
        f"✅ Exported {len(raw_cols):,} features "
        f"({n_dropped:,} all-missing dropped) · "
        f"{len(fm.y):,} studies · "
        f"survival: {'yes' if has_survival else 'no'}"
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def path_picker(kind: str, st) -> str:
    label, is_dir, default, help_text = DATA_SOURCES[kind]
    cur_key, browse_key = f"path_{kind}", f"browsing_{kind}"
    st.session_state.setdefault(cur_key, str(default))

    st.markdown(f"**{label}**")
    st.caption(help_text)

    c1, c2 = st.columns([6, 1])
    typed = c1.text_input("path", value=st.session_state[cur_key],
                          key=f"txt_{kind}", label_visibility="collapsed")
    if typed != st.session_state[cur_key]:
        st.session_state[cur_key] = typed
    if c2.button("Browse", key=f"btn_{kind}"):
        st.session_state[browse_key] = not st.session_state.get(browse_key, False)

    if st.session_state.get(browse_key):
        start = Path(st.session_state[cur_key]).expanduser()
        here  = start if start.is_dir() else start.parent
        if not here.exists():
            here = Path.home()
        dirs, files = list_dir_entries(here, dirs_only=is_dir)
        options = [".. (up)"] + [f"📁 {d}" for d in dirs] + [f"📄 {f}" for f in files]
        st.caption(f"in `{here}`")
        choice = st.selectbox("browse", options, key=f"sel_{kind}",
                              label_visibility="collapsed")
        b1, b2 = st.columns(2)
        if b1.button("Open", key=f"open_{kind}"):
            st.session_state[cur_key] = str(
                here.parent if choice.startswith("..") else here / choice[2:])
            st.rerun()
        if b2.button("Use this", key=f"use_{kind}", type="primary"):
            st.session_state[browse_key] = False
            st.rerun()

    ok, msgs = validate_source(kind, st.session_state[cur_key])
    (st.success if ok else st.error)(("✅ " if ok else "❌ ") + " · ".join(msgs))
    return st.session_state[cur_key]


def main() -> None:  # pragma: no cover
    import pandas as pd
    import streamlit as st

    from Custom.appd_route_b_labs import DX_TASKS, PX_TASKS

    st.set_page_config(page_title="INSPECT Feature Extraction", layout="wide")
    st.title("🧪 INSPECT EHR Feature Extraction")
    st.caption(
        "Extract structured EHR features from raw OMOP CSVs via DuckDB, "
        "explore per-impression event histories, and export flat arrays for modelling. "
        "See EHR_FEATURE_EXTRACTION_GUIDE.md for column definitions.")

    tab_load, tab_src, tab_b, tab_desc, tab_exp = st.tabs([
        "0 · Load",
        "1 · Data sources",
        "2 · Extract",
        "3 · Describe",
        "4 · Export",
    ])

    # ── 0 · Load ────────────────────────────────────────────────────────────
    with tab_load:
        st.subheader("Saved extractions")
        st.caption(
            "Load a previously saved extraction instantly — no configuration needed. "
            "Go to **Extract** to run a new one.")

        cached_list = _list_cached()
        if not cached_list:
            st.info("No saved extractions yet. Run one in the Extract tab.")
        else:
            for chash, cpath in cached_list:
                spec_file = CACHE_DIR / f"{chash}_spec.json"
                label = chash
                if spec_file.exists():
                    try:
                        s = json.loads(spec_file.read_text())
                        loinc_tag = (f"{len(s.get('loinc_codes', []))} LOINC codes"
                                     if s.get("loinc_codes") else "all LOINC")
                        adm_tag = " · admission-anchored" if s.get("admission_anchored") else ""
                        label = (f"**{s.get('task')}** · "
                                 f"windows={s.get('windows_days')} · {loinc_tag}{adm_tag}")
                    except Exception:
                        pass
                size_mb = cpath.stat().st_size / 1e6
                mtime   = datetime.datetime.fromtimestamp(
                    cpath.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                c1, c2, c3 = st.columns([5, 1, 1])
                c1.markdown(
                    f"{label}  \n"
                    f"<small>`{chash}` · {size_mb:.0f} MB · {mtime}</small>",
                    unsafe_allow_html=True)
                if c2.button("Load", key=f"load_{chash}", type="primary"):
                    with st.spinner("Loading …"):
                        with open(cpath, "rb") as _f:
                            st.session_state["fm"] = pickle.load(_f)
                        st.session_state["fm_hash"] = chash
                    st.rerun()
                if c3.button("🗑️", key=f"del_{chash}", help="Delete cache entry"):
                    for suffix in (".pkl", ".log", "_spec.json"):
                        (CACHE_DIR / f"{chash}{suffix}").unlink(missing_ok=True)
                    st.rerun()

        fm = st.session_state.get("fm")
        if fm is not None:
            st.divider()
            st.subheader(f"Loaded · {fm.task}")
            c = st.columns(4)
            c[0].metric("Studies",    f"{len(fm.y):,}")
            c[1].metric("Features",   f"{len(fm.columns):,}")
            c[2].metric("Event rate", f"{float(fm.y.mean()):.4f}",
                        help="Fraction of studies with a positive task label (y=1).")
            c[3].metric("Anchor",     fm.anchor_kind,
                        help="'dx' = features computed as of StudyTime − 1 day "
                             "(diagnostic tasks); 'px' = as of StudyTime "
                             "(prognostic tasks).")
            if getattr(fm, "admission_anchored", False):
                st.caption(
                    "🛏️ Admission-anchored: window per study is "
                    "[admission_date, anchor_time], capped by the configured "
                    "windows as an outer ceiling. See `admission_date` / "
                    "`days_since_admission` in the exported metadata.csv.")

    # ── 1 · Data sources ────────────────────────────────────────────────────
    with tab_src:
        st.subheader("Where the data lives")
        st.caption("Defaults assume the standard layout. Nothing large is read here.")
        left, right = st.columns(2)
        with left:
            paths = {k: path_picker(k, st) for k in ("cohort", "omop")}
        with right:
            paths.update({k: path_picker(k, st) for k in ("labels", "person")})

        st.divider()
        if all(validate_source(k, paths[k])[0] for k in ("cohort", "omop")):
            st.success("Required sources valid — ready to extract.")
        else:
            st.warning(
                "Cohort and OMOP directory must both validate. "
                "Labels adds survival outcomes; person.csv adds demographics.")

    # ── 2 · Extract ─────────────────────────────────────────────────────────
    with tab_b:
        st.subheader("EHR Feature Extraction")
        st.caption(
            "Scans raw OMOP CSVs out-of-core via DuckDB. "
            "Choose which feature types to extract and set individual lookback windows "
            "for each. Saved extractions appear in **tab 0 · Load**. "
            "`pip install duckdb` required in the venv.")

        # paths are inherited from tab_src; if tab_src hasn't been visited yet,
        # fall back to defaults
        omop_path   = st.session_state.get("path_omop",
                        str(DATA_SOURCES["omop"][2]))
        cohort_path = st.session_state.get("path_cohort",
                        str(DATA_SOURCES["cohort"][2]))
        labels_path = st.session_state.get("path_labels",
                        str(DATA_SOURCES["labels"][2]))
        person_path = st.session_state.get("path_person",
                        str(DATA_SOURCES["person"][2]))

        b_measurement = Path(omop_path) / "measurement.csv"
        b_concept     = Path(omop_path) / "concept.csv"

        all_tasks = sorted(DX_TASKS | PX_TASKS)
        b_task = st.selectbox("Task", all_tasks,
                              index=all_tasks.index("12_month_PH")
                              if "12_month_PH" in all_tasks else 0,
                              key="b_task")
        b_anchor = st.radio("Anchor", ["auto", "dx", "px"], horizontal=True,
                            key="b_anchor",
                            help="auto infers from task. dx = StudyTime−1d; px = StudyTime")

        # ── Feature types ──────────────────────────────────────────────────
        st.markdown("**Feature types to extract**")
        ft_r1c1, ft_r1c2, ft_r1c3 = st.columns(3)
        ft_r2c1, ft_r2c2, ft_r2c3 = st.columns(3)

        b_ft_labs  = ft_r1c1.checkbox(
            "Labs (measurement)", value=True, key="b_ft_labs",
            help="Numerical lab values from measurement.csv — mean, min, max, last, n, "
                 "days_since per LOINC code × window.")
        b_ft_diag  = ft_r1c2.checkbox(
            "Diagnoses (condition)", value=False, key="b_ft_diag",
            help="ICD-10-CM / SNOMED diagnosis codes from condition_occurrence.csv — "
                 "count of distinct event days per code.")
        b_ft_drugs = ft_r1c3.checkbox(
            "Drugs (drug_exposure)", value=False, key="b_ft_drugs",
            help="RxNorm ingredient codes from drug_exposure.csv — "
                 "count of distinct event days per drug.")
        b_ft_proc  = ft_r2c1.checkbox(
            "Procedures (procedure)", value=False, key="b_ft_proc",
            help="CPT4 / ICD-10-PCS codes from procedure_occurrence.csv — "
                 "count of distinct event days per code.")
        b_ft_obs   = ft_r2c2.checkbox(
            "Observations (observation)", value=False, key="b_ft_obs",
            help="OMOP observation domain from observation.csv — smoking status, "
                 "functional status, qualitative clinical flags, social history. "
                 "Count of distinct event days per concept.")
        b_ft_visit = ft_r2c3.checkbox(
            "Visits (visit_occurrence)", value=False, key="b_ft_visit",
            help="Inpatient admissions, outpatient encounters, and ED visits from "
                 "visit_occurrence.csv — count of distinct visit start dates per visit type.")

        b_feature_types = []
        if b_ft_labs:  b_feature_types.append("labs")
        if b_ft_diag:  b_feature_types.append("diagnoses")
        if b_ft_drugs: b_feature_types.append("drugs")
        if b_ft_proc:  b_feature_types.append("procedures")
        if b_ft_obs:   b_feature_types.append("observations")
        if b_ft_visit: b_feature_types.append("visits")

        if not b_feature_types:
            st.warning("Select at least one feature type.")

        # ── Lab-specific settings (only shown when Labs is checked) ─────────
        if b_ft_labs:
            col1, col2 = st.columns(2)
            b_windows_raw = col1.text_input(
                "Lab windows (days, comma-separated)", value="2, 7, 30, 365",
                key="b_windows",
                help="Cumulative lookback windows. '2,30,365' gives 3 window columns per lab.")
            b_loinc_raw = col2.text_area(
                "LOINC codes (leave blank = all LOINC)",
                value="", height=80, key="b_loinc",
                help="One code per line or comma-separated. E.g. 2160-0, 718-7")
            try:
                b_windows = sorted({int(x.strip()) for x in b_windows_raw.split(",")
                                    if x.strip()})
            except ValueError:
                st.error("Windows must be comma-separated integers.")
                b_windows = [2, 7, 30, 365]
            b_loinc_codes = [
                c.strip().split()[0]
                for c in b_loinc_raw.replace(",", "\n").splitlines()
                if c.strip()
            ] or None
        else:
            b_windows = [365]
            b_loinc_codes = None

        # ── Per-type lookback windows for count features ────────────────────
        # Config: (feature_type_key, label, default_days, help_text)
        _COUNT_FT_CONFIG = [
            ("diagnoses",    "Diagnoses",    365,
             "ICD-10-CM / SNOMED codes from condition_occurrence.csv."),
            ("drugs",        "Drugs",         60,
             "RxNorm ingredient codes from drug_exposure.csv."),
            ("procedures",   "Procedures",   365,
             "CPT4 / ICD-10-PCS codes from procedure_occurrence.csv."),
            ("observations", "Observations", 365,
             "OMOP observation domain (observation.csv) — smoking, functional status, etc."),
            ("visits",       "Visits",       365,
             "Inpatient / outpatient / ED visit types from visit_occurrence.csv."),
        ]
        active_count = [(ft, lbl, dflt, hlp)
                        for ft, lbl, dflt, hlp in _COUNT_FT_CONFIG
                        if ft in b_feature_types]

        b_count_windows: dict = {}
        if active_count:
            st.markdown("**Lookback window per event type** (days before anchor)")
            cw_cols = st.columns(len(active_count))
            for idx, (ft, lbl, dflt, hlp) in enumerate(active_count):
                b_count_windows[ft] = int(cw_cols[idx].number_input(
                    lbl, min_value=1, max_value=None, value=dflt, step=30,
                    key=f"b_cw_{ft}",
                    help=hlp + " No upper limit."))

        # ── Shared settings ─────────────────────────────────────────────────
        b_min_studies = st.slider(
            "Min studies per feature", 10, 500, 50, key="b_min_studies",
            help="Features (labs or codes) present in fewer studies than this are dropped.")

        b_one_per_patient_label = st.selectbox(
            "Studies per patient",
            ["All studies", "First study only", "Last study only"],
            key="b_one_per_patient",
            help="Some patients have multiple studies in the cohort. Restricting to "
                 "one per patient shrinks the cohort — and every downstream cost "
                 "(extraction time, matrix rows, long-format/timeline export size) "
                 "roughly proportionally — and avoids one patient contributing "
                 "multiple correlated rows if that matters for how you're using "
                 "the data. 'First'/'Last' is by anchor time, not upload order.",
        )
        b_one_per_patient = {
            "All studies":       None,
            "First study only":  "first",
            "Last study only":   "last",
        }[b_one_per_patient_label]

        # ── Admission-anchored extraction ────────────────────────────────────
        b_admission_anchored = st.checkbox(
            "Anchor extraction window to encompassing admission",
            value=False, key="b_admission_anchored",
            help="Instead of a fixed lookback from anchor_time for every study, "
                 "find the inpatient admission the anchor falls inside "
                 "(visit_concept_id = Inpatient or ER+Inpatient, "
                 "visit_start_date ≤ anchor ≤ visit_end_date or still open) and "
                 "extract features from [admission_start_date, anchor] instead. "
                 "The window settings above still apply as an outer ceiling per "
                 "study — e.g. a 365d lab window becomes "
                 "min(365, days since admission). "
                 "⚠️ Studies with no encompassing admission are dropped from the "
                 "cohort entirely (shrinks n and can shift prevalence). "
                 "admission_date and days_since_admission are added to "
                 "metadata.csv for the export.",
        )

        b_pre_admission_days = None
        if b_admission_anchored and b_ft_labs:
            b_pre_admission_use = st.checkbox(
                "Also extract a pre-admission lab baseline + delta",
                value=False, key="b_pre_admission_use",
                help="Adds a second lab window ending at admission_date (not "
                     "anchor_time) — 'the N days right before this admission "
                     "started' — tagged with a _preadm suffix, plus delta "
                     "columns (labs:LOINC/{code}_delta_{agg} for "
                     "last/min/max/mean) = the admission-to-anchor value "
                     "(your largest lab window above, already capped to the "
                     "admission span) minus this baseline. NaN where either "
                     "side has no measurement.",
            )
            if b_pre_admission_use:
                b_pre_admission_days = int(st.number_input(
                    "Pre-admission baseline window (days before admission)",
                    min_value=1, max_value=None, value=7, step=1,
                    key="b_pre_admission_days",
                ))
        elif b_admission_anchored and not b_ft_labs:
            st.caption(
                "ℹ️ Pre-admission baseline + delta is only available for Labs — "
                "check \"Labs (measurement)\" above to enable it.")

        # ── Concept-ancestor rollup ──────────────────────────────────────────
        with st.expander("🧬 Concept ancestor rollup (optional)", expanded=False):
            st.caption(
                "Roll up events to their ancestor concepts in the OMOP hierarchy using "
                "**concept_ancestor.csv** (~1.7 GB). This creates additional columns "
                "prefixed with `diag_anc:`, `drug_anc:`, etc. — useful for grouping "
                "rare specific codes under broader clinical categories. "
                "Requires at least one non-lab feature type to be selected. "
                "⚠️ Significantly increases extraction time."
            )
            _has_count_fts = any(
                ft != "labs" for ft in b_feature_types
            )
            b_use_anc = st.checkbox(
                "Enable concept ancestor rollup",
                value=False,
                key="b_use_ancestor",
                disabled=not _has_count_fts,
                help="Disabled when only Labs is selected — ancestor rollup applies to "
                     "Diagnoses, Drugs, Procedures, Observations, and Visits.",
            )
            b_ancestor_levels = None
            b_min_studies_anc = 100
            b_max_anc_features = 2000
            if b_use_anc and _has_count_fts:
                _use_level_limit = st.checkbox(
                    "Limit ancestor levels", value=True, key="b_limit_anc_levels",
                    help="Restrict how many levels to climb in the OMOP hierarchy. "
                         "Unchecked = all ancestors (broadest rollup, slowest)."
                )
                if _use_level_limit:
                    b_ancestor_levels = int(st.number_input(
                        "Max ancestor levels", min_value=1, max_value=10, value=2,
                        step=1, key="b_ancestor_levels",
                        help="2 = include parent and grandparent concepts only."
                    ))

                anc_col1, anc_col2 = st.columns(2)
                b_min_studies_anc = int(anc_col1.number_input(
                    "Min studies (ancestor)",
                    min_value=10, max_value=None, value=100, step=10,
                    key="b_min_studies_anc",
                    help="Ancestor codes present in fewer studies than this are dropped. "
                         "Set higher than the general threshold — ancestor codes are far "
                         "more numerous and most cover only rare sub-populations.",
                ))
                b_max_anc_features = int(anc_col2.number_input(
                    "Max ancestor features",
                    min_value=100, max_value=None, value=2000, step=100,
                    key="b_max_anc_features",
                    help="Hard cap per event table. Keeps the highest-coverage ancestor "
                         "codes. Lower = faster extraction and smaller matrix.",
                ))
            else:
                b_use_anc = False  # ensure False if no count tables

        b_spec = {
            "route":                "B",
            "task":                 b_task,
            "anchor":               b_anchor,
            "feature_types":        b_feature_types,
            "windows_days":         b_windows,
            "loinc_codes":          b_loinc_codes,
            "count_window_days":    b_count_windows,   # dict: {feature_type: days}
            "min_studies_per_lab":  b_min_studies,
            "use_concept_ancestor":  b_use_anc,
            "ancestor_levels":       b_ancestor_levels,
            "min_studies_ancestor":  b_min_studies_anc,
            "max_ancestor_features": b_max_anc_features,
            "one_study_per_patient": b_one_per_patient,
            "admission_anchored":    b_admission_anchored,
            "pre_admission_days":    b_pre_admission_days,
            "paths": {
                "cohort":      cohort_path,
                "measurement": str(b_measurement),
                "concept":     str(b_concept),
                "labels":      labels_path,
                "person":      person_path,
            },
        }
        b_hash    = _spec_hash(b_spec)
        b_pkl     = _cache_pkl(b_spec)
        b_log     = _cache_log(b_spec)
        b_proc_key = f"proc_b_{b_hash}"

        proc_info = st.session_state.get(b_proc_key)

        if proc_info is not None:
            pid, out_str = proc_info
            if Path(out_str).exists():
                st.success("✅ Extraction finished!")
                b_spec_path = _cache_spec(b_spec)
                b_spec_path.parent.mkdir(parents=True, exist_ok=True)
                b_spec_path.write_text(json.dumps(b_spec, indent=2))
                with open(out_str, "rb") as _f:
                    st.session_state["fm"] = pickle.load(_f)
                st.session_state["fm_hash"] = b_hash
                st.session_state.pop(b_proc_key)
                st.rerun()

            try:
                os.kill(pid, 0)
                still_running = True
            except OSError:
                still_running = False

            log_text = b_log.read_text() if b_log.exists() else "(waiting…)"
            st.code(log_text, language=None)

            if still_running:
                st.info(f"⏳ Running (PID {pid}) — refreshes every 5 s …")
                # NOTE: this blocks the Streamlit server thread handling this
                # session for the full 5s on every poll. Fine for the single-user
                # local usage this app is built for; if this is ever run as a
                # shared/multi-user deployment, one user's running extraction
                # will stall every other session and this should be replaced
                # with a non-blocking poll (e.g. st.fragment + a short timer, or
                # an autorefresh component) instead of time.sleep().
                time.sleep(5)
                st.rerun()
            else:
                st.error("❌ Extraction failed — check log above.")
                st.session_state.pop(b_proc_key)

        elif b_pkl.exists():
            size_mb = b_pkl.stat().st_size / 1e6
            st.success(f"✅ Cached result on disk ({size_mb:.0f} MB)")
            c1, c2 = st.columns(2)
            if c1.button("Load cached", type="primary", key="load_b_cached"):
                with st.spinner("Loading …"):
                    with open(b_pkl, "rb") as _f:
                        st.session_state["fm"] = pickle.load(_f)
                    st.session_state["fm_hash"] = b_hash
                st.rerun()
            if c2.button("Re-run", key="rerun_b"):
                b_pkl.unlink(missing_ok=True)
                st.rerun()

        else:
            if not b_measurement.exists():
                st.error(f"measurement.csv not found at {b_measurement}")
            elif not b_feature_types:
                st.error("Select at least one feature type before running.")
            elif st.button("▶ Run extraction", type="primary", key="run_b"):
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                b_spec_path = _cache_spec(b_spec)
                b_spec_path.write_text(json.dumps(b_spec, indent=2))
                proc = subprocess.Popen(
                    [sys.executable, str(ROUTE_B_WORKER_SCRIPT),
                     str(b_spec_path), str(b_pkl)],
                    stdout=open(b_log, "w"),
                    stderr=subprocess.STDOUT,
                )
                st.session_state[b_proc_key] = (proc.pid, str(b_pkl))
                st.rerun()

    # ── 3 · Describe ────────────────────────────────────────────────────────
    with tab_desc:
        fm = st.session_state.get("fm")
        if fm is None:
            st.info("Load an extraction first (tab 0 · Load, or run one in the Extract tab).")
        else:
            import pandas as pd
            from Custom.appd_route_b_labs import (
                query_events_stack, DX_TASKS as _DX, PX_TASKS as _PX,
            )
            from Custom.appd_context_descriptors import GENDER, AGE_BANDS, _age_band

            omop_dir    = st.session_state.get("path_omop",
                            str(DATA_SOURCES["omop"][2]))
            person_path = st.session_state.get("path_person",
                            str(DATA_SOURCES["person"][2]))
            concept_csv = str(Path(omop_dir) / "concept.csv")

            # ── build base metadata table ─────────────────────────────────
            @st.cache_data(show_spinner="Building case table …")
            def _build_case_table(fm_task, fm_anchor):
                _fm = st.session_state["fm"]
                base = _fm.to_frame()   # impression_id, patient_id, anchor_time, y

                if Path(person_path).is_file():
                    import csv as _csv
                    wanted = set(int(p) for p in _fm.patient_ids)
                    demo_rows = {}
                    with open(person_path) as _f:
                        for row in _csv.DictReader(_f):
                            try:
                                pid = int(row["person_id"])
                            except (KeyError, ValueError):
                                continue
                            if pid not in wanted:
                                continue
                            birth_raw = row.get("birth_DATETIME") or row.get("year_of_birth", "")
                            try:
                                bd = (datetime.datetime.fromisoformat(str(birth_raw))
                                      if "-" in str(birth_raw)
                                      else datetime.datetime(int(birth_raw), 7, 1))
                            except Exception:
                                bd = None
                            demo_rows[pid] = {
                                "sex":   GENDER.get(str(row.get("gender_concept_id", "")).strip(), "other/unknown"),
                                "birth": bd,
                            }
                    base["sex"] = [demo_rows.get(int(p), {}).get("sex", "unknown")
                                   for p in _fm.patient_ids]
                    ages = []
                    for p, t in zip(_fm.patient_ids, _fm.anchor_times):
                        bd = demo_rows.get(int(p), {}).get("birth")
                        try:
                            tt = (t if isinstance(t, datetime.datetime)
                                  else datetime.datetime.fromisoformat(str(t)))
                            ages.append(round((tt - bd).days / 365.25, 1))
                        except Exception:
                            ages.append(float("nan"))
                    base["age_years"] = ages
                    base["age_band"]  = [
                        _age_band(a) if not __import__("math").isnan(a) else "unknown"
                        for a in ages
                    ]
                return base

            base = _build_case_table(fm.task, fm.anchor_kind)

            # ── filters ───────────────────────────────────────────────────
            st.subheader("Filters")
            fc = st.columns(3)

            sel_label = fc[0].multiselect(
                "Label", ["Positive (y=1)", "Negative (y=0)"],
                default=["Positive (y=1)", "Negative (y=0)"])

            if "tte_days" in base.columns:
                sel_event = fc[1].multiselect(
                    "Survival event", ["Event observed", "Censored", "Unknown"],
                    default=["Event observed", "Censored", "Unknown"])
            else:
                sel_event = []

            if "sex" in base.columns:
                all_sex = sorted(base["sex"].unique().tolist())
                sel_sex = fc[2].multiselect("Sex", all_sex, default=all_sex)
            else:
                sel_sex = []

            if "age_band" in base.columns:
                all_bands = [f"{lo}-{hi}" if hi < 200 else f"{lo}+"
                             for lo, hi in AGE_BANDS]
                all_bands = [b for b in all_bands
                             if b in base["age_band"].unique().tolist()]
                sel_bands = st.multiselect("Age band", all_bands, default=all_bands)
            else:
                sel_bands = []

            fmask = pd.Series(True, index=base.index)
            if "Positive (y=1)" not in sel_label:
                fmask &= base["y"] != 1
            if "Negative (y=0)" not in sel_label:
                fmask &= base["y"] != 0
            if sel_sex:
                fmask &= base["sex"].isin(sel_sex)
            if sel_bands:
                fmask &= base["age_band"].isin(sel_bands)
            if "tte_days" in base.columns and sel_event:
                ev_mask = pd.Series(False, index=base.index)
                if "Event observed" in sel_event:
                    ev_mask |= base["event"] == 1
                if "Censored"       in sel_event:
                    ev_mask |= base["event"] == 0
                if "Unknown"        in sel_event:
                    ev_mask |= base["event"].isna()
                fmask &= ev_mask

            row_idx  = base[fmask].index
            filtered = base.loc[row_idx].reset_index(drop=True)

            # ── cohort summary ────────────────────────────────────────────
            st.divider()
            n_st  = len(filtered)
            n_pt  = filtered["patient_id"].nunique() if n_st else 0
            ev_rt = filtered["y"].mean() if n_st else float("nan")
            mc = st.columns(3)
            mc[0].metric("Studies",    f"{n_st:,}",
                         delta=f"{n_st - len(base):,}" if n_st != len(base) else None)
            mc[1].metric("Patients",   f"{n_pt:,}")
            mc[2].metric("Event rate", f"{ev_rt:.3f}" if n_st else "—")

            # ── cohort CSV export ─────────────────────────────────────────
            include_feats = st.checkbox(
                "Include feature columns in cohort CSV",
                value=False,
                help=f"Appends all {len(fm.columns):,} feature columns. "
                     "May be slow for large matrices.",
            )
            if include_feats:
                col_names = (
                    _load_concept_map_cached(concept_csv, st)
                    if Path(concept_csv).exists() else {}
                )
                col_labels = (fm.human_columns(col_names) if col_names
                              else list(fm.columns))
                import numpy as np
                feat_df    = pd.DataFrame(fm.X[row_idx], columns=col_labels)
                export_df  = pd.concat(
                    [filtered.reset_index(drop=True), feat_df], axis=1)
            else:
                export_df = filtered

            buf = io.StringIO()
            export_df.to_csv(buf, index=False)
            st.download_button(
                "⬇️ Export cohort CSV",
                buf.getvalue(),
                file_name=f"cohort_{fm.task}.csv",
                mime="text/csv",
            )

            # ── case table preview ────────────────────────────────────────
            st.divider()
            st.subheader(f"Cases ({n_st:,} studies)")
            _render_glossary(st, key="glossary_desc")
            st.dataframe(filtered, width="stretch", hide_index=True,
                         column_config=_glossary_column_config(filtered))

            # ── event history viewer ──────────────────────────────────────
            st.divider()
            st.subheader("Event history viewer")
            st.caption(
                "Select one or more studies from the filtered cohort to browse "
                "their raw EHR event history for a given lookback window. "
                "Data is queried live from the OMOP CSVs via DuckDB.")

            if n_st == 0:
                st.info("No studies match the current filters.")
            else:
                # impression selector — show id + label for context
                imp_options = [
                    f"{row['impression_id']}  (y={int(row['y'])})"
                    for _, row in filtered.iterrows()
                ]
                imp_labels  = list(filtered["impression_id"].astype(str))

                sel_display = st.multiselect(
                    "Select studies (impression_id)",
                    options=imp_options,
                    default=imp_options[:1],
                    max_selections=20,
                    help="Up to 20 studies. DuckDB scans all selected person timelines.",
                )
                selected_imps = [
                    imp_labels[imp_options.index(d)]
                    for d in sel_display
                    if d in imp_options
                ]

                col_win, col_tbl = st.columns(2)
                window_days = col_win.number_input(
                    "Days before anchor", min_value=1, max_value=None,
                    value=365, step=30,
                    help="Events between anchor−N days and anchor are shown. "
                         "No upper limit — EHR records can span many years.")

                table_choices = col_tbl.multiselect(
                    "Event tables",
                    ["measurement", "condition_occurrence", "drug_exposure",
                     "procedure_occurrence", "observation", "visit_occurrence"],
                    default=["measurement", "condition_occurrence", "drug_exposure",
                             "procedure_occurrence", "observation", "visit_occurrence"],
                )

                if st.button("🔍 Query events", type="primary",
                             disabled=not selected_imps or not table_choices):
                    # build impression → (person_id, anchor_time) mapping
                    imp_to_info = {
                        str(imp): (int(pid), anchor)
                        for imp, pid, anchor in zip(
                            fm.impression_ids, fm.patient_ids, fm.anchor_times)
                    }

                    concept_id_map = (
                        _load_concept_id_map_cached(concept_csv, st)
                        if Path(concept_csv).exists() else {}
                    )

                    with st.spinner(
                        f"Querying {', '.join(table_choices)} for "
                        f"{len(selected_imps)} study/studies …"):
                        try:
                            from Custom.appd_route_b_labs import query_events_stack
                            ev_df = query_events_stack(
                                omop_dir       = omop_dir,
                                selected_impressions = selected_imps,
                                imp_to_info    = imp_to_info,
                                window_days    = window_days,
                                tables         = table_choices,
                                concept_id_map = concept_id_map,
                            )
                        except Exception as e:
                            st.error(f"DuckDB query failed: {e}")
                            ev_df = None

                    if ev_df is not None:
                        if ev_df.empty:
                            st.info("No events found for the selected studies and window.")
                        else:
                            st.success(f"{len(ev_df):,} events across "
                                       f"{ev_df['source_table'].nunique()} table(s)")
                            st.dataframe(ev_df, use_container_width=True, hide_index=True,
                                         column_config=_glossary_column_config(ev_df))
                            ev_buf = io.StringIO()
                            ev_df.to_csv(ev_buf, index=False)
                            st.download_button(
                                "⬇️ Download event table",
                                ev_buf.getvalue(),
                                file_name=f"events_{'_'.join(selected_imps[:3])}.csv",
                                mime="text/csv",
                            )

    # ── 4 · Export ──────────────────────────────────────────────────────────
    with tab_exp:
        fm = st.session_state.get("fm")
        if fm is None:
            st.info("Load an extraction first (tab 0 · Load, or run one in the Extract tab).")
        else:
            omop_dir    = st.session_state.get("path_omop",
                            str(DATA_SOURCES["omop"][2]))
            concept_csv = str(Path(omop_dir) / "concept.csv")

            st.subheader(f"Export · {fm.task}")
            st.caption(
                "All studies exported together — no train/test split baked in. "
                "Apply your own split downstream using the generated `load_survival.py`.")

            fm_hash = st.session_state.get("fm_hash")
            _default_subdir = _export_subdir(fm, fm_hash)
            export_dir = st.text_input(
                "Export directory",
                value=str(DATA_ROOT / "DATA_PROCESSED" / "exports" / fm.task / _default_subdir),
                help="Defaults to a subfolder tagged with this extraction's feature "
                     "types and a short hash of its full config — so a different "
                     "extraction for the same task (e.g. drugs instead of labs) lands "
                     "in its own folder instead of overwriting X.npy / y.npy / "
                     "metadata.csv here. `extraction_spec.json` in the export "
                     "directory always records exactly what produced these files, "
                     "regardless of the folder name.",
            )

            col_opt1, col_opt2 = st.columns(2)
            drop_zero = col_opt1.checkbox(
                "Drop all-missing columns",
                value=True,
                help="Removes features that are NaN (labs) or zero (counts) for every "
                     "study in the cohort. Strongly recommended — reduces file size "
                     "substantially without losing information.",
            )
            use_human = col_opt2.checkbox(
                "Resolve human-readable column names",
                value=Path(concept_csv).exists(),
                help="Appends a second name column to feature_names.csv with the OMOP "
                     "concept name resolved via concept.csv. Requires concept.csv to be "
                     "configured in Data sources.",
            )

            _render_glossary(st, key="glossary_exp")

            st.markdown("**Files written to the export directory:**")
            st.markdown(
                "**`extraction_spec.json`** — the exact parameters that produced every "
                "other file here (task, feature types, windows, LOINC filter, thresholds, "
                "…). The source of truth for what this export *is*, independent of "
                "however the folder ends up named, copied, or moved later.  \n\n"
                "**`metadata.csv`** — one row per study: `impression_id`, `patient_id`, "
                "`anchor_time` (date the feature window ends), `y` (binary task label), "
                "`tte_days` (**t**ime-**t**o-**e**vent, in days) and `event` "
                "(1 = event observed, 0 = censored) for survival modelling, if available.  \n\n"
                "**`X.npy`** — feature matrix, float32, shape `(n_studies, n_features)`. "
                "Lab values use `NaN` for *not measured*; count features (diagnoses / drugs / "
                "procedures) use `0` for *not observed*. Rows align 1-to-1 with `metadata.csv`.  \n\n"
                "**`X_mask.npy`** — uint8 observation mask, same shape as `X.npy`. "
                "`1` = value was measured/observed; `0` = missing. Useful for "
                "imputation-aware models.  \n\n"
                "**`y.npy`** — binary label array (0/1), length `n_studies`.  \n\n"
                "**`tte.npy` / `event.npy`** — time-to-event (days) and event indicator "
                "for survival modelling; written only when survival columns are present "
                "in the cohort or labels file.  \n\n"
                "**`feature_names.csv`** — two columns: `raw` (the coded column name, "
                "e.g. `drug:RxNorm/11289_60d`) and `human` (the resolved concept name, "
                "e.g. `drug:apixaban [RxNorm/11289]_60d`). Index aligns with columns of `X.npy`.  \n\n"
                "**`load_survival.py`** — ready-to-run Python script that loads all arrays "
                "and shows how to apply a train/test split and fit a survival model.  \n\n"
                "**`load_multimodal.py`** — script showing how to join these EHR features "
                "with imaging or NLP outputs using `impression_id` as the join key."
            )

            if st.button("Export arrays", type="primary"):
                with st.spinner("Exporting …"):
                    concept_map = (
                        _load_concept_map_cached(concept_csv, st)
                        if use_human and Path(concept_csv).exists() else {}
                    )
                    msg = _do_export(fm, export_dir, drop_zero, concept_map,
                                      _fm_hash=st.session_state.get("fm_hash"))
                st.success(msg)
                st.code(export_dir)

            # ── Long-format CSV ──────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Long-format CSV")
            st.caption(
                "A human-readable alternative to the numpy arrays above. "
                "Each row is one study × one observed feature, with all study "
                "metadata repeated. Useful for inspection in Excel / R, sharing "
                "with collaborators, or feeding directly into tidy-data workflows. "
                "Demographics (`sex`, `age_years`) are treated as study metadata "
                "rather than feature rows.")

            lf_obs_only = st.checkbox(
                "Observed features only (non-zero / non-NaN)",
                value=True,
                key="lf_obs_only",
                help="Keeps only rows with a measured or non-zero value. "
                     "Uncheck to get a complete (N_studies × N_features) long table.",
            )

            if st.button("Build long-format table", key="lf_build"):
                with st.spinner("Melting feature matrix …"):
                    concept_map_lf = (
                        _load_concept_map_cached(concept_csv, st)
                        if Path(concept_csv).exists() else {}
                    )
                    try:
                        ldf = _build_long_df(fm, concept_map_lf, observed_only=lf_obs_only)
                    except MemoryError as e:
                        st.error(f"⚠️ {e}")
                        ldf = None
                if ldf is not None:
                    st.session_state["long_df"]      = ldf
                    st.session_state["long_df_task"] = fm.task

            ldf = st.session_state.get("long_df")
            # Invalidate cached table when a different extraction is loaded
            if ldf is not None and st.session_state.get("long_df_task") != fm.task:
                ldf = None
                st.session_state.pop("long_df", None)

            if ldf is not None:
                st.success(
                    f"{len(ldf):,} rows · "
                    f"{ldf['impression_id'].nunique():,} studies · "
                    f"{ldf['feature'].nunique():,} unique features")
                st.dataframe(ldf.head(200), use_container_width=True, hide_index=True,
                             column_config=_glossary_column_config(ldf))
                st.download_button(
                    # ldf.to_csv() with no path returns the CSV text directly —
                    # avoids the extra StringIO buffer + .getvalue() copy,
                    # which for a large ldf meant up to 3 simultaneous
                    # full-size copies of the same data in memory.
                    "⬇️ Download long-format CSV",
                    ldf.to_csv(index=False),
                    file_name=f"{fm.task}_long.csv",
                    mime="text/csv",
                    key="lf_download",
                )

            st.markdown("**Too large for the table above?**")
            st.caption(
                "Writes the same long-format rows straight to one CSV file "
                "on disk — a bounded-memory chunk of studies at a time, "
                "appended to the same file as it goes, so there's no "
                "separate 'fuse the pieces together' step and no size cap. "
                "Use this when 'Build long-format table' raises a memory "
                "error above (e.g. the full cohort, labs only, a wide "
                "window, no LOINC filter). Rows are grouped by "
                "study-processing order rather than globally sorted by "
                "impression_id, since sorting the whole file would need the "
                "whole file in memory."
            )
            lf_stream_path = st.text_input(
                "Output CSV path",
                value=str(Path(export_dir) / f"{fm.task}_long.csv"),
                key="lf_stream_path",
            )
            n_feat_est = max(len(fm.columns) - 9, 1)   # minus demo: columns, roughly
            default_chunk_cells = _STREAM_CHUNK_CELLS
            lf_chunk_cells = st.number_input(
                "Target cells per chunk (studies × features held in memory at once)",
                min_value=100_000, max_value=50_000_000,
                value=default_chunk_cells, step=100_000,
                key="lf_chunk_cells",
                help="Roughly 4 bytes of RAM per cell for the raw matrix slice, "
                     "plus more for the melted rows depending on how much of it "
                     "is actually observed. Lower this on a memory-constrained "
                     "machine; raise it (fewer, bigger chunks) for a faster run "
                     "on a machine with plenty of RAM.",
            )
            lf_chunk_studies = max(1, int(lf_chunk_cells) // n_feat_est)
            st.caption(
                f"≈ {lf_chunk_studies:,} studies per chunk at this extraction's "
                f"~{n_feat_est:,} feature columns "
                f"(~{lf_chunk_cells * 4 / 1e6:.0f} MB of raw matrix data per chunk)."
            )
            if st.button("💾 Stream full long-format CSV to disk", key="lf_stream_build"):
                concept_map_lf = (
                    _load_concept_map_cached(concept_csv, st)
                    if Path(concept_csv).exists() else {}
                )
                progress = st.progress(0.0, text="Starting …")

                def _on_progress(done, total):
                    progress.progress(
                        done / total if total else 1.0,
                        text=f"{done:,} / {total:,} studies …")

                with st.spinner("Streaming to disk …"):
                    n_rows = _build_long_df_streamed(
                        fm, concept_map_lf, lf_stream_path,
                        observed_only=lf_obs_only, chunk_studies=lf_chunk_studies,
                        on_progress=_on_progress,
                    )
                progress.empty()
                st.success(f"✅ Wrote {n_rows:,} rows to:")
                st.code(lf_stream_path)

            # ── Event timeline CSV ───────────────────────────────────────────
            st.markdown("---")
            st.subheader("Event timeline CSV")
            _ft = getattr(fm, "feature_types", None) or []
            _cwd = getattr(fm, "count_window_days", None) or {}
            st.caption(
                "Raw individual event occurrences — one row per event, not per "
                "aggregated feature. Includes the exact `event_date` and "
                "`days_before_ctpa` for each lab measurement, diagnosis, drug "
                "exposure, procedure, observation, or visit. For lab events the "
                "`value` column contains the measured numeric value; for coded "
                "events it is `NaN`.  \n\n"
                "**Use this for longitudinal / sequence modelling** where you need "
                "the full temporal structure, not just window aggregates.  \n\n"
                "Re-queries the raw OMOP CSVs at export time — "
                "expect a few minutes for large cohorts."
            )
            if not _ft:
                st.warning(
                    "Feature types not stored in this extraction (older pkl). "
                    "Re-run the extraction to enable timeline export."
                )
            else:
                _window_summary = []
                if "labs" in _ft:
                    _window_summary.append(f"labs ≤ {max(fm.windows_days)} d")
                for _ft_key, _days in _cwd.items():
                    _window_summary.append(f"{_ft_key} ≤ {_days} d")
                st.info(
                    f"Feature types: **{', '.join(_ft)}**  \n"
                    f"Windows: {' · '.join(_window_summary)}"
                )

                if st.button("Build event timeline", key="tl_build"):
                    from Custom.appd_route_b_labs import build_event_timeline
                    _omop_dir        = Path(omop_dir)
                    _measurement_csv = _omop_dir / "measurement.csv"
                    _concept_csv     = Path(concept_csv)
                    with st.spinner(
                        "Querying OMOP CSVs … this may take several minutes "
                        "depending on which feature types are selected."
                    ):
                        tl_df = build_event_timeline(
                            fm=fm,
                            omop_dir=_omop_dir,
                            measurement_path=_measurement_csv,
                            concept_path=_concept_csv,
                            verbose=True,
                        )
                    st.session_state["timeline_df"]      = tl_df
                    st.session_state["timeline_df_task"] = fm.task

                tl_df = st.session_state.get("timeline_df")
                if tl_df is not None and st.session_state.get("timeline_df_task") != fm.task:
                    tl_df = None
                    st.session_state.pop("timeline_df", None)

                if tl_df is not None:
                    st.success(
                        f"{len(tl_df):,} events · "
                        f"{tl_df['impression_id'].nunique():,} studies · "
                        f"{tl_df['event_type'].value_counts().to_dict()}"
                    )
                    st.dataframe(tl_df.head(200), use_container_width=True, hide_index=True,
                                 column_config=_glossary_column_config(tl_df))
                    st.download_button(
                        "⬇️ Download event timeline CSV",
                        tl_df.to_csv(index=False),
                        file_name=f"{fm.task}_event_timeline.csv",
                        mime="text/csv",
                        key="tl_download",
                    )

                st.markdown("**Too large for the table above?**")
                st.caption(
                    "This queries *raw individual events* — one row per lab "
                    "measurement / diagnosis / drug / procedure / "
                    "observation / visit, not aggregated — so for a wide "
                    "window over a large cohort it's often even bigger than "
                    "the long-format export above (tens of millions of rows "
                    "is normal, not a bug). This writes straight to a CSV "
                    "file using DuckDB's own out-of-core query engine — "
                    "Python never holds the full result. **Rows are not "
                    "globally sorted** (grouped by feature type, in "
                    "whatever order DuckDB's scan produces them) — a full "
                    "sort would need the entire result considered before "
                    "writing anything, which is what pushed this close to "
                    "the edge even with disk-spilling on. Sort after "
                    "loading if you need chronological order; every row "
                    "still carries its own `event_date`/`days_before_ctpa`."
                )
                tl_stream_path = st.text_input(
                    "Output CSV path",
                    value=str(DATA_ROOT / "DATA_PROCESSED" / "exports" / fm.task
                              / _export_subdir(fm, st.session_state.get("fm_hash"))
                              / f"{fm.task}_event_timeline.csv"),
                    key="tl_stream_path",
                )
                tl_memory_limit_gb = st.number_input(
                    "DuckDB memory limit (GB)",
                    min_value=1.0, max_value=64.0, value=4.0, step=1.0,
                    key="tl_memory_limit_gb",
                    help="How much RAM DuckDB is allowed to use before it "
                         "spills its own working set (joins, aggregation) to "
                         "the temp file, regardless of how much the machine "
                         "actually has. Deliberately conservative by default "
                         "— raise it for a faster run if you know the "
                         "machine has headroom to spare; lower it further if "
                         "this still runs the system close to its limit.",
                )
                if st.button("💾 Stream full event timeline CSV to disk", key="tl_stream_build"):
                    from Custom.appd_route_b_labs import build_event_timeline_streamed
                    _omop_dir        = Path(omop_dir)
                    _measurement_csv = _omop_dir / "measurement.csv"
                    _concept_csv     = Path(concept_csv)
                    with st.spinner(
                        "Querying and writing via DuckDB … this can take "
                        "several minutes for a wide window over a large "
                        "cohort — progress isn't shown mid-query since "
                        "DuckDB runs this as a single streaming statement."
                    ):
                        n_events = build_event_timeline_streamed(
                            fm=fm,
                            omop_dir=_omop_dir,
                            measurement_path=_measurement_csv,
                            concept_path=_concept_csv,
                            out_path=tl_stream_path,
                            memory_limit_gb=float(tl_memory_limit_gb),
                            verbose=True,
                        )
                    st.success(f"✅ Wrote {n_events:,} rows to:")
                    st.code(tl_stream_path)


if __name__ == "__main__":  # pragma: no cover
    main()
