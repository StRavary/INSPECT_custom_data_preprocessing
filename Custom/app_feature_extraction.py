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

Run:
    ../.venv_legacy/bin/python -m streamlit run ./app_feature_extraction.py
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

CACHE_DIR             = DATA_ROOT / "DATA_PROCESSED" / "femr_cache"
ROUTE_B_WORKER_SCRIPT = SCRIPT_DIR / "_route_b_worker.py"

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


def _default_concept_csv(omop_path: str) -> Path:
    return Path(omop_path) / "concept.csv"


# ---------------------------------------------------------------------------
# Concept-map cache
# ---------------------------------------------------------------------------

def _load_concept_map_cached(concept_csv: str, st):
    """Streamlit-cached wrapper around route_b_labs.load_concept_map."""
    from Custom.route_b_labs import load_concept_map

    @st.cache_resource(show_spinner="Loading OMOP concept names …")
    def _load(path: str):
        return load_concept_map(path)

    return _load(concept_csv)


def _load_concept_id_map_cached(concept_csv: str, st):
    """Streamlit-cached wrapper around route_b_labs.load_concept_id_map."""
    from Custom.route_b_labs import load_concept_id_map

    @st.cache_resource(show_spinner="Loading concept ID map …")
    def _load(path: str):
        return load_concept_id_map(path)

    return _load(concept_csv)


# ---------------------------------------------------------------------------
# Long-format (melted) export helper
# ---------------------------------------------------------------------------

def _build_long_df(fm, concept_map: dict, observed_only: bool = True) -> "pd.DataFrame":
    """Melt a LabFeatureMatrix to one row per (impression × feature).

    Metadata columns (impression_id, patient_id, anchor_time, y, tte_days, event,
    sex, age_years) are repeated on every row.  Demo features (demo:*) are
    extracted into those metadata columns and excluded from the feature rows.

    Parameters
    ----------
    fm            : LabFeatureMatrix
    concept_map   : {VOCAB/code: human_name} from load_concept_map()
    observed_only : if True, drop rows where value is NaN or 0
                    (keeps file small; uncheck to get a complete matrix)

    Returns
    -------
    pd.DataFrame with columns:
        impression_id | patient_id | anchor_time | y | tte_days | event |
        sex | age_years | feature | feature_human_readable | value
    """
    import pandas as pd
    import numpy as np
    from Custom.route_b_labs import humanize_column

    import re

    col_list = list(fm.columns)
    col_idx  = {c: i for i, c in enumerate(col_list)}

    # ── extract demographics as metadata ────────────────────────────────────
    def _get_demo(name):
        if name in col_idx:
            return fm.X[:, col_idx[name]].copy()
        return np.full(len(fm.impression_ids), np.nan, dtype=np.float32)

    is_f = _get_demo("demo:is_female")
    sex  = np.where(np.isnan(is_f), "Unknown",
           np.where(is_f == 1.0,    "Female",  "Male"))

    # ── metadata DataFrame ───────────────────────────────────────────────────
    meta = fm.to_frame().reset_index(drop=True)   # impression_id, patient_id, …
    meta["sex"]       = sex
    meta["age_years"] = _get_demo("demo:age_years")
    if "tte_days" not in meta.columns:
        meta["tte_days"] = np.nan
        meta["event"]    = np.nan

    id_vars = ["impression_id", "patient_id", "anchor_time", "y",
               "tte_days", "event", "sex", "age_years"]

    # ── feature sub-matrix (exclude demo: columns) ───────────────────────────
    feat_idx  = [i for i, c in enumerate(col_list) if not c.startswith("demo:")]
    feat_cols = [col_list[i] for i in feat_idx]

    # ── record-date lookup for each feature column ───────────────────────────
    # Labs: record_date = anchor_time - days_since  (already in the matrix)
    # Count features: record_date from fm.count_dates (most-recent event date)
    _LAB_COL_PAT     = re.compile(r'^(labs:[^_]+(?:_[^_]+)*)_([^_]+)_(\d+d)$')
    _DAYS_SINCE_PAT  = re.compile(r'^(labs:.+)_days_since_(\d+d)$')

    # Build {(base, window) → col_idx} for days_since columns
    days_since_map: dict = {}
    for i, c in enumerate(col_list):
        m = _DAYS_SINCE_PAT.match(c)
        if m:
            days_since_map[(m.group(1), m.group(2))] = i

    anchor_dt = pd.to_datetime([str(a) for a in fm.anchor_times])
    count_dates = getattr(fm, "count_dates", None)   # None for old pickles

    # Build record_date matrix (n_impressions × n_feat_cols), values = ISO date str or ''
    n_imp = len(fm.impression_ids)
    record_date_arr = np.full((n_imp, len(feat_cols)), "", dtype=object)

    for j, col in enumerate(feat_cols):
        if col.startswith("labs:"):
            m = _LAB_COL_PAT.match(col)
            if m:
                base, _agg, window = m.group(1), m.group(2), m.group(3)
                ds_idx = days_since_map.get((base, window))
                if ds_idx is not None:
                    ds_vals = fm.X[:, ds_idx].astype(float)
                    valid   = ~np.isnan(ds_vals)
                    dates   = anchor_dt - pd.to_timedelta(
                        np.where(valid, ds_vals, 0), unit="d")
                    record_date_arr[valid, j] = (
                        dates[valid].strftime("%Y-%m-%d").values)
        elif count_dates is not None and col in count_dates.columns:
            # Covers diag:, drug:, proc:, obs:, visit:, and _anc: variants
            record_date_arr[:, j] = (
                count_dates[col].fillna("").astype(str).values)

    # ── build wide feature DataFrame and melt ────────────────────────────────
    wide = pd.DataFrame(fm.X[:, feat_idx], columns=feat_cols)
    for col in id_vars:
        wide[col] = meta[col].values

    long = wide.melt(
        id_vars   = id_vars,
        var_name  = "feature",
        value_name = "value",
    )

    # Melt the record_date array in the same column order → same row order as long
    rd_wide = pd.DataFrame(record_date_arr, columns=feat_cols)
    rd_long = rd_wide.melt(var_name="feature", value_name="record_date")
    long["record_date"] = rd_long["record_date"].values

    if observed_only:
        long = long[long["value"].notna() & (long["value"] != 0.0)].copy()

    # ── human-readable feature names ─────────────────────────────────────────
    long["feature_human_readable"] = long["feature"].map(
        lambda c: humanize_column(c, concept_map)
    )

    out_cols = ["impression_id", "patient_id", "anchor_time", "y",
                "tte_days", "event", "sex", "age_years",
                "feature", "record_date", "feature_human_readable", "value"]
    return (
        long[out_cols]
        .sort_values(["impression_id", "feature"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def _do_export(fm, export_dir: str, drop_zero: bool, concept_map: dict) -> str:
    """Write flat (unsplit) files to export_dir. Returns a summary string."""
    import scipy.sparse
    import pandas as pd
    import numpy as np

    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

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

    np.save(out / "X.npy",      X.astype(np.float32))
    np.save(out / "X_mask.npy", (~np.isnan(X)).astype(np.uint8))
    np.save(out / "y.npy", fm.y)

    has_survival = fm.tte is not None
    if has_survival:
        np.save(out / "tte.npy",   fm.tte)
        np.save(out / "event.npy", fm.event)

    # ── load_survival.py ─────────────────────────────────────────────────
    surv_lines = f'''\
"""
load_survival.py  —  auto-generated by app_feature_extraction.py
Load {fm.task} EHR features for survival analysis.
All studies are in a single flat array — apply your own train/test split downstream.
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

    from Custom.route_b_labs import DX_TASKS, PX_TASKS

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
                        label = (f"**{s.get('task')}** · "
                                 f"windows={s.get('windows_days')} · {loinc_tag}")
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
            c[2].metric("Event rate", f"{float(fm.y.mean()):.4f}")
            c[3].metric("Anchor",     fm.anchor_kind)

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
            if b_use_anc and _has_count_fts:
                _use_level_limit = st.checkbox(
                    "Limit ancestor levels", value=False, key="b_limit_anc_levels",
                    help="Restrict how many levels to climb in the OMOP hierarchy. "
                         "Unchecked = all ancestors (broadest rollup)."
                )
                if _use_level_limit:
                    b_ancestor_levels = int(st.number_input(
                        "Max ancestor levels", min_value=1, max_value=10, value=2,
                        step=1, key="b_ancestor_levels",
                        help="2 = include parent and grandparent concepts only."
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
            "use_concept_ancestor": b_use_anc,
            "ancestor_levels":      b_ancestor_levels,
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
            from Custom.route_b_labs import (
                query_events_stack, DX_TASKS as _DX, PX_TASKS as _PX,
            )
            from Custom.context_descriptors import GENDER, AGE_BANDS, _age_band

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
            st.dataframe(filtered, width="stretch", hide_index=True)

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
                            from Custom.route_b_labs import query_events_stack
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
                            st.dataframe(ev_df, use_container_width=True, hide_index=True)
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

            export_dir = st.text_input(
                "Export directory",
                value=str(DATA_ROOT / "DATA_PROCESSED" / "exports" / fm.task),
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

            st.markdown("**Files written to the export directory:**")
            st.markdown(
                "**`metadata.csv`** — one row per study: `impression_id`, `patient_id`, "
                "`anchor_time` (date the feature window ends), `y` (binary task label), "
                "`tte_days` and `event` (survival outcome, if available).  \n\n"
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
                    msg = _do_export(fm, export_dir, drop_zero, concept_map)
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
                    ldf = _build_long_df(fm, concept_map_lf, observed_only=lf_obs_only)
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
                st.dataframe(ldf.head(200), use_container_width=True, hide_index=True)
                lf_buf = io.StringIO()
                ldf.to_csv(lf_buf, index=False)
                st.download_button(
                    "⬇️ Download long-format CSV",
                    lf_buf.getvalue(),
                    file_name=f"{fm.task}_long.csv",
                    mime="text/csv",
                    key="lf_download",
                )


if __name__ == "__main__":  # pragma: no cover
    main()
