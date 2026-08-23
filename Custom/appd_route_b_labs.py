"""
route_b_labs.py
---------------
Route B: extract numerical lab features from raw OMOP measurement.csv using DuckDB.

For each (study, lab-LOINC-code, time-window) triple, computes:
  last      – value of the most recent measurement before the anchor in that window
  min       – minimum value in window
  max       – maximum value in window
  mean      – mean value
  n         – number of distinct measurement days
  days_since – calendar days between the most recent measurement and the anchor

Output: LabFeatureMatrix with the same external interface as FeatureMatrix
(same attributes and methods) so all app tabs — Cohort Builder, Export — work
without changes.

Why Route B?
  Route A (FEMR) delta-encodes events, losing the actual measurement values and
  collapsing time series into counts of distinct days. measurement.csv has raw
  numeric values needed for lab-value features (last creatinine, lowest
  haemoglobin, troponin trend, etc.) which are among the strongest clinical
  predictors and are completely absent from Route A.

DuckDB is used for the 22.7 GB measurement.csv: it scans CSV out-of-core and
pushes filters down into the scan, so the full file need not fit in RAM.

Usage:

    from Custom.route_b_labs import LabExtractor

    fm = LabExtractor().build(
        task="12_month_PH",
        windows_days=[2, 7, 30, 365],
    )
    fm.X          # numpy float32 array (n_studies, n_features), NaN = not measured
    fm.columns    # feature names aligned to X
    fm.y, fm.tte, fm.event   # labels and survival
    fm.to_frame() # metadata DataFrame
    print(fm.describe())
"""

from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path defaults (same layout as temporal_features.py)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT  = SCRIPT_DIR.parent.parent

DEFAULT_MEASUREMENT       = DATA_ROOT / "DATA_RAW"  / "EHR_CSV" / "measurement.csv"
DEFAULT_CONCEPT           = DATA_ROOT / "DATA_RAW"  / "EHR_CSV" / "concept.csv"
DEFAULT_CONCEPT_ANCESTOR  = DATA_ROOT / "DATA_RAW"  / "EHR_CSV" / "concept_ancestor.csv"
DEFAULT_PERSON            = DATA_ROOT / "DATA_RAW"  / "EHR_CSV" / "person.csv"
DEFAULT_COHORT       = DATA_ROOT / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv"
DEFAULT_LABELS       = DATA_ROOT / "DATA_RAW"       / "LABELS"  / "labels_20250611.tsv"

# ---------------------------------------------------------------------------
# OMOP standard concept IDs for demographics
# ---------------------------------------------------------------------------
_GENDER_FEMALE       = 8532
_GENDER_MALE         = 8507
_ETHNICITY_HISPANIC  = 38003563
_ETHNICITY_NO_HISP   = 38003564
_RACE_WHITE          = 8527
_RACE_BLACK          = 8516
_RACE_ASIAN          = 8515

# Canonical {gender_concept_id (str) -> label} map. This is the single source
# of truth for OMOP gender decoding — appd_context_descriptors.py and the
# Streamlit app both import it rather than redefining their own copy, so the
# concept-id -> label mapping and its casing ("female"/"male") can't diverge.
GENDER = {
    str(_GENDER_MALE):   "male",
    str(_GENDER_FEMALE): "female",
}

PATIENT_ID_CANDIDATES = ("patient_id", "PatientID", "person_id")
TIME_COLUMN           = "StudyTime"
IMPRESSION_COLUMN     = "impression_id"

DEFAULT_WINDOWS = [2, 7, 30, 365]

# aggregates computed per (study, lab, window) — also used for column naming
AGGS = ["last", "min", "max", "mean", "n", "days_since"]

# ---------------------------------------------------------------------------
# Task / anchor constants (previously in temporal_features.py)
# ---------------------------------------------------------------------------

DX_TASKS = {
    "pe_positive_nlp", "pe_positive", "pe_acute", "pe_subsegmentalonly",
}
PX_TASKS = {
    "1_month_mortality", "6_month_mortality", "12_month_mortality",
    "1_month_readmission", "6_month_readmission", "12_month_readmission",
    "12_month_PH",
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural_Effusion",
}

# task -> (tte column, censoring column) in labels_20250611.tsv
SURVIVAL_COLUMNS = {
    "1_month_mortality":  ("tte_mortality",      "is_censored_mortality"),
    "6_month_mortality":  ("tte_mortality",      "is_censored_mortality"),
    "12_month_mortality": ("tte_mortality",      "is_censored_mortality"),
    "1_month_readmission":  ("tte_readmission",  "is_censored_readmission"),
    "6_month_readmission":  ("tte_readmission",  "is_censored_readmission"),
    "12_month_readmission": ("tte_readmission",  "is_censored_readmission"),
    "12_month_PH":        ("tte_PH",             "is_censored_PH"),
    "Atelectasis":        ("tte_Atelectasis",    "is_censored_Atelectasis"),
    "Cardiomegaly":       ("tte_Cardiomegaly",   "is_censored_Cardiomegaly"),
    "Consolidation":      ("tte_Consolidation",  "is_censored_Consolidation"),
    "Edema":              ("tte_Edema",          "is_censored_Edema"),
    "Pleural_Effusion":   ("tte_Pleural_Effusion","is_censored_Pleural_Effusion"),
}

TTE_MINUTES_PER_DAY = 1440.0
TRUTHY             = {"TRUE", "1", "1.0", "YES", "T"}
SKIP_LABEL_VALUES  = {"CENSORED", "CENSOR", "NAN", "NA", "NONE", ""}
CATCH_ALL_DAYS     = 36500  # 100 years

# visit_concept_ids treated as a qualifying inpatient stay for
# admission-anchored extraction: Inpatient Visit, and Emergency Room +
# Inpatient Visit (an ER visit that rolled straight into an admission).
ADMISSION_VISIT_CONCEPT_IDS = (9201, 262)

# ---------------------------------------------------------------------------
# OMOP concept-name helpers (previously in temporal_features.py)
# ---------------------------------------------------------------------------

import re as _re

# Route A window suffix: "_30 days" / "_30 days, 0:00:00"
_WINDOW_RE = _re.compile(r'(_-?\d+ days(?:, \d+:\d+:\d+)?)\s*$')
# Route B window suffix: "_365d", "_1d", etc. (always at the very end)
_WINDOW_B_RE = _re.compile(r'_(\d+d)$')
# Route B lab agg tags placed between code and window: last/min/max/mean/n/days_since
_LAB_AGGS = frozenset(("last", "min", "max", "mean", "n", "days_since"))


def load_concept_map(concept_csv) -> dict:
    """Build {VOCAB/concept_code: concept_name} from OMOP concept.csv."""
    p = Path(concept_csv)
    try:
        import pandas as pd
        df = pd.read_csv(p, usecols=["vocabulary_id", "concept_code", "concept_name"],
                         dtype=str, low_memory=False)
        df["key"] = df["vocabulary_id"] + "/" + df["concept_code"]
        return dict(zip(df["key"], df["concept_name"]))
    except ImportError:
        lookup: dict = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                lookup[f"{row['vocabulary_id']}/{row['concept_code']}"] = row["concept_name"]
        return lookup


def load_concept_id_map(concept_csv) -> dict:
    """Build {concept_id (str): concept_name} from OMOP concept.csv.

    Used by the event-stack viewer to resolve raw integer concept IDs.
    """
    p = Path(concept_csv)
    try:
        import pandas as pd
        df = pd.read_csv(p, usecols=["concept_id", "concept_name"],
                         dtype=str, low_memory=False)
        return dict(zip(df["concept_id"], df["concept_name"]))
    except ImportError:
        lookup: dict = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                lookup[row["concept_id"]] = row["concept_name"]
        return lookup


def load_concept_vocab_code_map(concept_csv) -> dict:
    """Build {concept_id (str): (vocabulary_id, concept_code)} from concept.csv.

    Separate from load_concept_id_map (which resolves concept_id -> a
    display name) because the timeline viewer's clinical-panel assignment
    (appd_clinical_panels.assign_panel) needs the (vocabulary, code) pair,
    not the name — e.g. ('LOINC', '2160-0'), not 'Creatinine'.
    """
    p = Path(concept_csv)
    try:
        import pandas as pd
        df = pd.read_csv(p, usecols=["concept_id", "vocabulary_id", "concept_code"],
                         dtype=str, low_memory=False)
        return {
            cid: (vocab, code)
            for cid, vocab, code in zip(df["concept_id"], df["vocabulary_id"], df["concept_code"])
        }
    except ImportError:
        lookup: dict = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                lookup[row["concept_id"]] = (row["vocabulary_id"], row["concept_code"])
        return lookup


def _readable_window(window_str: str) -> str:
    m = _re.match(r"_(\d+) days", window_str)
    if not m:
        return window_str
    days = int(m.group(1))
    if days >= CATCH_ALL_DAYS:
        return "_whole_history"
    if days >= 365 and days % 365 == 0:
        return f"_{days // 365}y"
    return f"_{days}d"


def humanize_column(col: str, concept_map: dict) -> str:
    """Replace raw OMOP codes in a column name with human-readable concept names.

    Handles two column formats:

    Route B  labs  : ``labs:LOINC/{code}_{agg}_{N}d``
                     → ``labs:{name} [{LOINC/code}]_{agg}_{N}d``
    Route B  counts: ``diag|drug|proc:{VOCAB}/{code}_{N}d``
                     → ``diag|drug|proc:{name} [{VOCAB/code}]_{N}d``
    Route A (legacy): ``group:{VOCAB/code} _30 days``
                     → ``group:{name} [{VOCAB/code}]_30d``
    """
    if ":" not in col:
        return col
    group, rest = col.split(":", 1)

    # ── Route B format ───────────────────────────────────────────────────────
    m_b = _WINDOW_B_RE.search(rest)
    if m_b:
        window_suffix = m_b.group(0)           # e.g. "_365d"
        before_win    = rest[: m_b.start()]    # e.g. "LOINC/2160-0_last" or "SNOMED/1755008"

        # Lab columns have an extra agg tag before the window (_last, _mean, …)
        if group == "labs":
            # Split off the agg tag: "LOINC/2160-0_last" → ("LOINC/2160-0", "last")
            parts = before_win.rsplit("_", 1)
            if len(parts) == 2 and parts[1] in _LAB_AGGS:
                code_key, agg_tag = parts
                name = concept_map.get(code_key)
                if name:
                    # Truncate very long concept names for readability
                    short = name if len(name) <= 60 else name[:57] + "…"
                    return f"{group}:{short} [{code_key}]_{agg_tag}{window_suffix}"
                return col   # no match → leave as-is
        else:
            # Count columns: before_win is "VOCAB/code" directly
            name = concept_map.get(before_win)
            if name:
                short = name if len(name) <= 60 else name[:57] + "…"
                return f"{group}:{short} [{before_win}]{window_suffix}"
        return col   # no concept match → leave as-is

    # ── Route A legacy format ─────────────────────────────────────────────────
    m_a = _WINDOW_RE.search(rest)
    if m_a:
        code_part   = rest[: m_a.start()]
        window_part = _readable_window(m_a.group(1))
    else:
        code_part   = rest
        window_part = ""
    base_code, *tail = code_part.split(" ", 1)
    suffix = f" {tail[0]}" if tail else ""
    name = concept_map.get(base_code)
    if name:
        return f"{group}:{name}{suffix} [{base_code}]{window_part}"
    return f"{group}:{code_part}{window_part}"


# ---------------------------------------------------------------------------
# Event-stack query (Describe tab)
# ---------------------------------------------------------------------------

_EVENT_TABLES = {
    "measurement": {
        "date_col":     "measurement_date",
        "datetime_col": "measurement_datetime",
        "concept_col":  "measurement_concept_id",
        "value_col":    "value_as_number",
        "label":        "measurement",
    },
    "condition_occurrence": {
        "date_col":     "condition_start_date",
        "datetime_col": "condition_start_datetime",
        "concept_col":  "condition_concept_id",
        "value_col":    None,
        "label":        "condition",
    },
    "drug_exposure": {
        "date_col":     "drug_exposure_start_date",
        "datetime_col": "drug_exposure_start_datetime",
        "concept_col":  "drug_concept_id",
        "value_col":    None,
        "label":        "drug",
    },
    "procedure_occurrence": {
        "date_col":     "procedure_date",
        "datetime_col": "procedure_datetime",
        "concept_col":  "procedure_concept_id",
        "value_col":    None,
        "label":        "procedure",
    },
    "observation": {
        "date_col":     "observation_date",
        "datetime_col": "observation_datetime",
        "concept_col":  "observation_concept_id",
        "value_col":    "value_as_number",
        "label":        "observation",
    },
    "visit_occurrence": {
        "date_col":     "visit_start_date",
        "datetime_col": "visit_start_datetime",
        "concept_col":  "visit_concept_id",
        "value_col":    None,
        "label":        "visit",
    },
}

# Configuration for count-based feature extraction (diagnoses, drugs, procedures).
# vocab_filter is appended verbatim to the WHERE clause when loading concept.csv.
_COUNT_TABLE_CONFIG = {
    "condition_occurrence": {
        "date_col":    "condition_start_date",
        "concept_col": "condition_concept_id",
        "col_prefix":  "diag",
        "vocab_filter": "AND vocabulary_id IN ('ICD10CM', 'ICD9CM', 'SNOMED')",
    },
    "drug_exposure": {
        "date_col":    "drug_exposure_start_date",
        "concept_col": "drug_concept_id",
        "col_prefix":  "drug",
        "vocab_filter": "AND vocabulary_id IN ('RxNorm', 'RxNorm Extension')",
    },
    "procedure_occurrence": {
        "date_col":    "procedure_date",
        "concept_col": "procedure_concept_id",
        "col_prefix":  "proc",
        "vocab_filter": "AND vocabulary_id IN ('CPT4', 'ICD10PCS', 'HCPCS', 'ICD9Proc')",
    },
    # Observation: smoking status, functional status, qualitative clinical flags,
    # social history — anything recorded in OMOP observation domain.
    "observation": {
        "date_col":    "observation_date",
        "concept_col": "observation_concept_id",
        "col_prefix":  "obs",
        "vocab_filter": "AND domain_id = 'Observation'",
    },
    # Visit occurrence: inpatient admissions, outpatient encounters, ED visits.
    # Count = distinct visit start dates per visit type within the lookback window.
    # end_date_col triggers additional LOS (length-of-stay) aggregate columns.
    "visit_occurrence": {
        "date_col":     "visit_start_date",
        "concept_col":  "visit_concept_id",
        "col_prefix":   "visit",
        "vocab_filter": "AND domain_id = 'Visit'",
        "end_date_col": "visit_end_date",   # enables _los_total_ and _los_max_ columns
    },
}


def query_events_stack(
    omop_dir,
    selected_impressions: list,
    imp_to_info: dict,
    window_days: int,
    tables: list,
    concept_id_map: dict,
) -> "pd.DataFrame":
    """Return a long-format DataFrame of EHR events for the selected impressions.

    Parameters
    ----------
    omop_dir : str or Path
        Directory containing the OMOP CSV files.
    selected_impressions : list[str]
        impression_ids to query.
    imp_to_info : dict
        {impression_id: (person_id, anchor_datetime)} — from the loaded fm.
    window_days : int
        How many days before the anchor to include (1 = anchor day only).
    tables : list[str]
        Subset of _EVENT_TABLES keys to query.
    concept_id_map : dict
        {concept_id_str: concept_name} — from load_concept_id_map().

    Returns
    -------
    pd.DataFrame with columns:
        impression_id, event_date, days_before_anchor, source_table,
        concept_id, concept_name, value
    Sorted by impression_id, event_date DESC.
    """
    import duckdb
    import pandas as pd

    omop_dir = Path(omop_dir)

    # Build an in-memory anchor table
    anchor_rows = []
    for imp in selected_impressions:
        info = imp_to_info.get(imp)
        if info is None:
            continue
        pid, anchor = info
        anchor_date = (anchor.date() if hasattr(anchor, "date") else
                       datetime.datetime.fromisoformat(str(anchor)).date())
        anchor_rows.append((str(imp), int(pid), str(anchor_date)))

    out_cols = ["impression_id", "event_date", "event_datetime", "days_before_anchor",
                "source_table", "concept_id", "concept_name", "value"]

    if not anchor_rows:
        return pd.DataFrame(columns=out_cols)

    con = duckdb.connect()
    con.execute("""
        CREATE TEMP TABLE _anchors (
            impression_id VARCHAR,
            person_id     BIGINT,
            anchor_date   DATE
        )
    """)
    con.executemany("INSERT INTO _anchors VALUES (?, ?, ?)", anchor_rows)

    parts = []
    for tbl in tables:
        meta = _EVENT_TABLES.get(tbl)
        if meta is None:
            continue
        csv_path = omop_dir / f"{tbl}.csv"
        if not csv_path.exists():
            continue
        date_col     = meta["date_col"]
        datetime_col = meta["datetime_col"]
        concept_col  = meta["concept_col"]
        value_expr   = (f"CAST(t.{meta['value_col']} AS VARCHAR)"
                        if meta["value_col"] else "NULL")
        label        = meta["label"]
        parts.append(f"""
            SELECT
                a.impression_id,
                CAST(t.{date_col} AS VARCHAR)                      AS event_date,
                CAST(
                    COALESCE(
                        TRY_CAST(t.{datetime_col} AS TIMESTAMP),
                        TRY_CAST(t.{date_col}     AS TIMESTAMP)
                    ) AS VARCHAR
                )                                                  AS event_datetime,
                DATEDIFF('day', CAST(t.{date_col} AS DATE),
                         a.anchor_date)                            AS days_before_anchor,
                '{label}'                                          AS source_table,
                CAST(t.{concept_col} AS VARCHAR)                   AS concept_id,
                {value_expr}                                       AS value
            FROM read_csv_auto('{csv_path}', ignore_errors=true) t
            JOIN _anchors a ON CAST(t.person_id AS BIGINT) = a.person_id
            WHERE t.{date_col} IS NOT NULL
              AND DATEDIFF('day', CAST(t.{date_col} AS DATE), a.anchor_date)
                  BETWEEN 1 AND {window_days}
        """)

    if not parts:
        con.close()
        return pd.DataFrame(columns=out_cols)

    union_sql = " UNION ALL ".join(parts)
    df = con.execute(
        f"SELECT * FROM ({union_sql}) ORDER BY impression_id, event_date DESC, source_table"
    ).df()
    con.close()

    df["concept_name"] = df["concept_id"].map(concept_id_map).fillna("")
    return df[out_cols]


# ---------------------------------------------------------------------------
# Demographic feature helpers (used by both Route A and Route B)
# ---------------------------------------------------------------------------

def load_demographic_features(
    person_csv,
    patient_ids: np.ndarray,
    anchor_times: np.ndarray,
    verbose: bool = True,
) -> tuple:
    """Load demographics from OMOP person.csv and encode as a float32 array.

    Returns (X_demo, columns):
      X_demo   : float32 (n_studies, 9)
      columns  : list of 9 feature names

    Columns:
      demo:age_years     – true age at anchor in years (NaN if birth date missing)
      demo:is_female     – 1=female, 0=male, NaN=other/unknown gender
      demo:sex_unknown   – 1 if gender not recorded (OMOP concept_id=0 or missing)
      demo:is_hispanic   – 1=Hispanic, 0=Not Hispanic, NaN=not recorded
      demo:race_white    – binary (1/0), OMOP 8527
      demo:race_black    – binary (1/0), OMOP 8516
      demo:race_asian    – binary (1/0), OMOP 8515
      demo:race_other    – 1 if race is recorded but not white/black/asian
      demo:race_unknown  – 1 if race not recorded (concept_id=0 or person not found)

    NaN means "value truly unknown/missing". For sparse matrices (Route A)
    the caller should fill NaN before converting to sparse — see
    append_demographics() which handles this automatically.
    """
    import csv as _csv

    demo: dict = {}
    with open(person_csv) as _f:
        for row in _csv.DictReader(_f):
            try:
                pid = int(row["person_id"])
            except (KeyError, ValueError):
                continue

            birth_raw = row.get("birth_DATETIME") or row.get("year_of_birth", "")
            try:
                birth = (datetime.datetime.fromisoformat(str(birth_raw))
                         if "-" in str(birth_raw)
                         else datetime.datetime(int(birth_raw), 7, 1))
            except Exception:
                birth = None

            def _int(key):
                try:
                    return int(row.get(key, 0) or 0)
                except (ValueError, TypeError):
                    return 0

            demo[pid] = {
                "birth":     birth,
                "gender":    _int("gender_concept_id"),
                "ethnicity": _int("ethnicity_concept_id"),
                "race":      _int("race_concept_id"),
            }

    n          = len(patient_ids)
    age        = np.full(n, np.nan, dtype=np.float32)
    is_female  = np.full(n, np.nan, dtype=np.float32)
    sex_unk    = np.zeros(n, dtype=np.float32)
    is_hisp    = np.full(n, np.nan, dtype=np.float32)
    race_white = np.zeros(n, dtype=np.float32)
    race_black = np.zeros(n, dtype=np.float32)
    race_asian = np.zeros(n, dtype=np.float32)
    race_other = np.zeros(n, dtype=np.float32)
    race_unk   = np.zeros(n, dtype=np.float32)

    for i, (pid, t) in enumerate(zip(patient_ids, anchor_times)):
        d = demo.get(int(pid))
        if d is None:
            # person not in person.csv at all
            sex_unk[i]  = 1.0
            race_unk[i] = 1.0
            continue

        # age at anchor
        if d["birth"] is not None:
            try:
                tt = (t if isinstance(t, datetime.datetime)
                      else datetime.datetime.fromisoformat(str(t)))
                age[i] = float((tt - d["birth"]).days / 365.25)
            except Exception:
                pass

        # sex
        g = d["gender"]
        if g == _GENDER_FEMALE:
            is_female[i] = 1.0
        elif g == _GENDER_MALE:
            is_female[i] = 0.0
        else:
            sex_unk[i] = 1.0   # concept_id=0 or non-standard

        # ethnicity
        e = d["ethnicity"]
        if e == _ETHNICITY_HISPANIC:
            is_hisp[i] = 1.0
        elif e == _ETHNICITY_NO_HISP:
            is_hisp[i] = 0.0
        # else stays NaN — OMOP often has ethnicity_concept_id=0 for "unknown"

        # race (one-hot with explicit unknown)
        r = d["race"]
        if r == _RACE_WHITE:
            race_white[i] = 1.0
        elif r == _RACE_BLACK:
            race_black[i] = 1.0
        elif r == _RACE_ASIAN:
            race_asian[i] = 1.0
        elif r != 0:
            race_other[i] = 1.0
        else:
            race_unk[i] = 1.0   # concept_id=0

    X_demo = np.column_stack([
        age, is_female, sex_unk, is_hisp,
        race_white, race_black, race_asian, race_other, race_unk,
    ]).astype(np.float32)

    columns = [
        "demo:age_years",
        "demo:is_female",
        "demo:sex_unknown",
        "demo:is_hispanic",
        "demo:race_white",
        "demo:race_black",
        "demo:race_asian",
        "demo:race_other",
        "demo:race_unknown",
    ]

    n_matched = sum(1 for p in patient_ids if int(p) in demo)
    if verbose:
        print(f"[demographics] {n_matched:,}/{n:,} studies matched in person.csv "
              f"({n_matched / n:.1%})", flush=True)

    return X_demo, columns


def append_demographics(fm, person_csv, verbose: bool = True):
    """Append demographic columns to a FeatureMatrix or LabFeatureMatrix.

    Handles the sparse/dense difference automatically:
      - Route B (dense numpy X): NaN preserved — models should impute.
      - Route A (scipy sparse X): age NaN filled with cohort median; other
        NaN (is_female, is_hispanic) filled with 0. Use demo:sex_unknown and
        demo:race_unknown columns to recover the missing-data signal.

    Returns a NEW fm object (original is not mutated).
    """
    import dataclasses

    X_demo, demo_cols = load_demographic_features(
        person_csv, fm.patient_ids, fm.anchor_times, verbose)

    is_route_b = getattr(fm, "route", "A") == "B"

    if is_route_b:
        # Dense: hstack, preserving NaN
        X_new = np.hstack([fm.X, X_demo])
    else:
        import scipy.sparse
        # Sparse: fill NaN before converting to sparse
        X_d = X_demo.copy()
        # Age: fill NaN with cohort-median age so the feature is informative
        age_col = X_d[:, 0]
        med_age = float(np.nanmedian(age_col))
        age_col[np.isnan(age_col)] = med_age
        # Binary NaN (is_female, is_hispanic): fill with 0 — sex_unknown /
        # race_unknown columns capture that these are missing
        X_d[np.isnan(X_d)] = 0.0
        X_new = scipy.sparse.hstack(
            [fm.X, scipy.sparse.csr_matrix(X_d)], format="csr")

    return dataclasses.replace(fm, X=X_new, columns=list(fm.columns) + demo_cols)


# ---------------------------------------------------------------------------
# LabFeatureMatrix — Route B result container
# ---------------------------------------------------------------------------

@dataclass
class LabFeatureMatrix:
    """Route B feature matrix — dense numeric lab features.

    X is a float32 numpy array; NaN indicates the lab was not measured in that
    window (not the same as a count of zero, which Route A would produce).

    All attributes match FeatureMatrix so the Streamlit app tabs work unchanged.
    """
    X:             np.ndarray          # (n_studies, n_features), float32, NaN = not measured
    columns:       list                # feature names aligned to X
    y:             np.ndarray          # binary label
    tte:           Optional[np.ndarray]
    event:         Optional[np.ndarray]
    patient_ids:   np.ndarray
    impression_ids: np.ndarray
    split:         np.ndarray
    anchor_times:  np.ndarray
    task:          str
    anchor_kind:   str
    windows_days:  list
    route:         str = "B"
    # Most-recent event date per (impression, count-feature column).
    # Wide DataFrame indexed by impression position (0-based), columns = count feature
    # col names, values = ISO date strings or ''.  None for old pickled extractions.
    count_dates:   Optional[object] = None   # Optional[pd.DataFrame]
    # Feature types extracted (e.g. ["labs", "diagnoses", "drugs"]) and their
    # per-type lookback windows in days.  None for old pickled extractions.
    feature_types:     Optional[object] = None   # Optional[list]
    count_window_days: Optional[object] = None   # Optional[dict] {feature_type: days}
    # Admission-anchored extraction (build(admission_anchored=True)): the
    # window per study became [admission_date, anchor_time] instead of a
    # fixed lookback, capped by windows_days/count_window_days as an outer
    # ceiling. None/False for ordinary extractions.
    admission_anchored:   bool = False
    admission_dates:      Optional[np.ndarray] = None   # ISO date strings, aligned to rows
    days_since_admission: Optional[np.ndarray] = None   # float days, admission_date -> anchor_time

    def __len__(self) -> int:
        return len(self.y)

    def mask(self, split: str) -> np.ndarray:
        return self.split == split

    def to_frame(self):
        import pandas as pd
        d = {
            "impression_id": self.impression_ids,
            "patient_id":    self.patient_ids,
            "anchor_time":   self.anchor_times,
            "y":             self.y,
        }
        if self.tte is not None:
            d["tte_days"] = self.tte
            d["event"]    = self.event
        if self.admission_dates is not None:
            d["admission_date"]        = self.admission_dates
            d["days_since_admission"]  = self.days_since_admission
        return pd.DataFrame(d)

    def human_columns(self, concept_map: dict) -> list:
        return [humanize_column(c, concept_map) for c in self.columns]

    def describe(self) -> str:
        measured = float((~np.isnan(self.X)).mean())
        # Only "labs:" columns follow the "..._<agg>_<window>d" naming scheme that
        # rsplit assumes here; count-feature (diag:/drug:/proc:/...) and demo:
        # columns don't, so they must be excluded or this overcounts distinct labs.
        n_labs = len({c.rsplit("_", 2)[0] for c in self.columns if c.startswith("labs:")})
        lines = [
            f"route=B  task={self.task}  anchor={self.anchor_kind}  "
            f"rows={len(self):,}  cols={len(self.columns):,}  labs={n_labs:,}",
            f"  windows         : {self.windows_days} days"
            + ("  (admission-anchored: capped at admission→anchor gap)"
               if self.admission_anchored else ""),
            f"  label prevalence: {float(np.nanmean(self.y)):.4f}",
            f"  measured frac   : {measured:.3f}  "
            "(fraction of study×feature pairs with an observed value)",
        ]
        if self.tte is not None:
            n_obs = int(np.nansum(self.event))
            lines.append(
                f"  events observed : {n_obs:,} "
                f"({float(np.nanmean(self.event)):.1%});  "
                f"median tte {float(np.nanmedian(self.tte)):.1f} d")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Event timeline builder — raw individual events (not aggregated)
# ---------------------------------------------------------------------------

# Maps feature_type key →
#   (OMOP table name, date_col, datetime_col, concept_col, event_type_label, value_expr)
# datetime_col is preferred over date_col when non-null (sub-day resolution).
_TIMELINE_TABLE_CONFIG: dict = {
    "diagnoses":    ("condition_occurrence", "condition_start_date",     "condition_start_datetime",     "condition_concept_id",   "diagnosis",   "NULL"),
    "drugs":        ("drug_exposure",        "drug_exposure_start_date", "drug_exposure_start_datetime", "drug_concept_id",        "drug",        "NULL"),
    "procedures":   ("procedure_occurrence", "procedure_date",           "procedure_datetime",           "procedure_concept_id",   "procedure",   "NULL"),
    "observations": ("observation",          "observation_date",         "observation_datetime",         "observation_concept_id", "observation", "CAST(t.value_as_number AS DOUBLE)"),
    "visits":       ("visit_occurrence",     "visit_start_date",        "visit_start_datetime",         "visit_concept_id",       "visit",       "NULL"),
}


def build_event_timeline(
    fm: "LabFeatureMatrix",
    omop_dir: "Path",
    measurement_path: "Path",
    concept_path: "Path",
    loinc_codes: Optional[list] = None,
    verbose: bool = True,
) -> "pd.DataFrame":
    """Return a raw event timeline — one row per individual event occurrence.

    This re-queries the raw OMOP CSVs using DuckDB. The lookback window for each
    feature type is taken from ``fm.feature_types`` and ``fm.count_window_days``
    (set automatically during extraction). Labs use ``max(fm.windows_days)``.

    Parameters
    ----------
    fm              : loaded LabFeatureMatrix (must have feature_types set)
    omop_dir        : directory containing OMOP CSV files
    measurement_path: path to measurement.csv
    concept_path    : path to concept.csv
    loinc_codes     : optional list of LOINC codes to restrict lab events
                      (None = all LOINC-coded measurements)
    verbose         : print progress

    Returns
    -------
    pd.DataFrame with columns:
        impression_id, patient_id, anchor_time, y,
        event_type, vocabulary, concept_code, concept_name,
        event_date, days_before_ctpa, value
    """
    import pandas as pd

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    try:
        import duckdb
    except ImportError:
        raise ImportError("pip install duckdb")
    import tempfile

    feature_types     = list(getattr(fm, "feature_types", None) or [])
    count_window_days = dict(getattr(fm, "count_window_days", None) or {})
    windows_days      = list(fm.windows_days) if fm.windows_days else [365]

    # Build anchors table
    anchors_df = pd.DataFrame({
        "person_id":     fm.patient_ids.astype("int64"),
        "impression_id": fm.impression_ids,
        "anchor_time":   [str(a) for a in fm.anchor_times],
        "y":             fm.y.astype(float),
    })

    # On-disk temp file, not the default in-memory database — gives DuckDB
    # somewhere to spill large intermediate state (the join against
    # measurement.csv can be sizeable for a wide window / large cohort).
    # Same pattern as _run_duckdb_and_pivot / _run_duckdb_ancestor_rollup.
    tmp_db = Path(tempfile.mktemp(suffix="_event_timeline_preview.duckdb"))
    con = duckdb.connect(str(tmp_db))
    con.execute("PRAGMA threads=4")
    con.register("anchors_tbl", anchors_df)

    # Load concept.csv once — vocabulary_id, concept_code, concept_name
    _log("[timeline] loading concept.csv …")
    con.execute(f"""
        CREATE TEMP TABLE _concept AS
        SELECT
            CAST(concept_id AS BIGINT) AS concept_id,
            vocabulary_id,
            concept_code,
            concept_name
        FROM read_csv_auto('{concept_path}', ignore_errors=true)
        WHERE concept_id IS NOT NULL
    """)

    parts = []

    # ── Labs ────────────────────────────────────────────────────────────────
    if "labs" in feature_types:
        max_window = max(windows_days)
        # Bind LOINC codes (free-typed by the user in the app) as query
        # parameters rather than splicing them into the SQL string.
        loinc_filter = ""
        loinc_params: list = []
        if loinc_codes:
            loinc_filter = f"AND c.concept_code IN ({', '.join('?' * len(loinc_codes))})"
            loinc_params = list(loinc_codes)
        _log(f"[timeline] labs: scanning measurement.csv (window={max_window} d) …")
        lab_df = con.execute(f"""
            SELECT
                a.impression_id,
                CAST(a.y AS DOUBLE)                                                      AS y,
                'lab'                                                                     AS event_type,
                COALESCE(c.vocabulary_id, 'Unknown')                                      AS vocabulary,
                COALESCE(c.concept_code,  '')                                             AS concept_code,
                COALESCE(c.concept_name,  '')                                             AS concept_name,
                COALESCE(
                    TRY_CAST(t.measurement_datetime AS TIMESTAMP),
                    CAST(t.measurement_date AS TIMESTAMP)
                )                                                                         AS event_datetime,
                CAST(t.measurement_date AS DATE)                                          AS event_date,
                -DATEDIFF('day',
                    CAST(t.measurement_date AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE))                      AS days_before_ctpa,
                CAST(t.value_as_number AS DOUBLE)                                         AS value
            FROM read_csv_auto('{measurement_path}', ignore_errors=true) t
            INNER JOIN anchors_tbl a
                    ON CAST(t.person_id AS BIGINT) = a.person_id
            LEFT  JOIN _concept c
                    ON CAST(t.measurement_concept_id AS BIGINT) = c.concept_id
            WHERE t.measurement_date IS NOT NULL
              AND CAST(t.measurement_concept_id AS BIGINT) != 0
              {loinc_filter}
              AND DATEDIFF('day',
                    CAST(t.measurement_date AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                  ) BETWEEN 1 AND {max_window}
        """, loinc_params).df()
        _log(f"  {len(lab_df):,} lab event rows")
        parts.append(lab_df)

    # ── Count feature types ──────────────────────────────────────────────────
    for ft, (tbl, date_col, datetime_col, concept_col, label, value_expr) in _TIMELINE_TABLE_CONFIG.items():
        if ft not in feature_types:
            continue
        csv_path = Path(omop_dir) / f"{tbl}.csv"
        if not csv_path.exists():
            _log(f"  WARNING: {tbl}.csv not found — skipping {ft}")
            continue
        window_days = count_window_days.get(ft, 365)
        _log(f"[timeline] {ft}: scanning {tbl}.csv (window={window_days} d) …")
        part_df = con.execute(f"""
            SELECT
                a.impression_id,
                CAST(a.y AS DOUBLE)                                                      AS y,
                '{label}'                                                                 AS event_type,
                COALESCE(c.vocabulary_id, 'Unknown')                                      AS vocabulary,
                COALESCE(c.concept_code,  '')                                             AS concept_code,
                COALESCE(c.concept_name,  '')                                             AS concept_name,
                COALESCE(
                    TRY_CAST(t.{datetime_col} AS TIMESTAMP),
                    CAST(t.{date_col} AS TIMESTAMP)
                )                                                                         AS event_datetime,
                CAST(t.{date_col} AS DATE)                                                AS event_date,
                -DATEDIFF('day',
                    CAST(t.{date_col} AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE))                      AS days_before_ctpa,
                {value_expr}                                                               AS value
            FROM read_csv_auto('{csv_path}', ignore_errors=true) t
            INNER JOIN anchors_tbl a
                    ON CAST(t.person_id AS BIGINT) = a.person_id
            LEFT  JOIN _concept c
                    ON CAST(t.{concept_col} AS BIGINT) = c.concept_id
            WHERE t.{date_col} IS NOT NULL
              AND CAST(t.{concept_col} AS BIGINT) != 0
              AND DATEDIFF('day',
                    CAST(t.{date_col} AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                  ) BETWEEN 1 AND {window_days}
        """).df()
        _log(f"  {len(part_df):,} {ft} event rows")
        parts.append(part_df)

    con.close()
    if tmp_db.exists():
        tmp_db.unlink(missing_ok=True)

    if not parts:
        return pd.DataFrame(columns=[
            "impression_id", "patient_id", "anchor_time", "y",
            "event_type", "vocabulary", "concept_code", "concept_name",
            "event_datetime", "event_date", "days_before_ctpa", "value",
        ])

    timeline = pd.concat(parts, ignore_index=True)

    # Join patient_id and anchor_time back from fm
    meta = pd.DataFrame({
        "impression_id": fm.impression_ids,
        "patient_id":    fm.patient_ids,
        "anchor_time":   fm.anchor_times,
    })
    timeline = timeline.merge(meta, on="impression_id", how="left")

    out_cols = [
        "impression_id", "patient_id", "anchor_time", "y",
        "event_type", "vocabulary", "concept_code", "concept_name",
        "event_datetime", "event_date", "days_before_ctpa", "value",
    ]
    return (
        timeline[out_cols]
        .sort_values(["impression_id", "event_datetime", "event_type"])
        .reset_index(drop=True)
    )


def build_event_timeline_streamed(
    fm: "LabFeatureMatrix",
    omop_dir: "Path",
    measurement_path: "Path",
    concept_path: "Path",
    out_path,
    loinc_codes: Optional[list] = None,
    memory_limit_gb: float = 4.0,
    verbose: bool = True,
) -> int:
    """Same output as build_event_timeline (except unsorted — see below),
    written directly to a CSV file via DuckDB's own out-of-core
    ``COPY (...) TO`` instead of being pulled into a pandas DataFrame first.

    build_event_timeline queries raw individual events — one row per lab
    measurement / diagnosis / drug / procedure / observation / visit
    occurrence, *not* aggregated — so for a wide window over a large cohort
    this can be tens of millions of rows, more than the aggregated feature
    matrix has columns. The original implementation pulls every
    per-feature-type query into memory via ``.df()``, concatenates them,
    merges in patient_id / anchor_time, then sorts — each step a full-size
    copy stacked on the last. This version instead has DuckDB do the
    filtering, joining, and CSV writing entirely inside its own engine, so
    Python never holds more than a small, fixed amount of data at any point.

    Two things had to both be true for this to actually be memory-safe in
    practice, not just in theory — the first alone (an earlier version of
    this function) still hit ~29 GB RSS + full swap on a real ~59M-row run:

    1. **On-disk temp DuckDB file**, not ``duckdb.connect()``'s default
       in-memory database — an in-memory connection has nowhere to spill
       large intermediate state. Matches `_run_duckdb_and_pivot` /
       `_run_duckdb_ancestor_rollup` elsewhere in this file.
    2. **No ``ORDER BY``.** A global sort needs the *entire* result
       considered before a single row can be written — for ~59M rows, that
       sort was almost certainly the actual peak-memory driver, disk-spill
       or not. Without it, execution is a genuinely constant-memory
       pipeline (scan → filter → join → write), bounded by one row at a
       time plus the small `anchors_tbl` hash-join build side (O(n_studies),
       not O(n_events)). The tradeoff: output rows are in whatever order
       DuckDB's parallel scan produces them — grouped by feature type, not
       globally sorted by study/time. Sort after loading if you need that
       (e.g. `pd.read_csv(...).sort_values([...])`, or let a groupby handle
       it) — cheap once it's already reduced to your own memory budget.

    `memory_limit_gb` additionally caps how much DuckDB tries to hold in
    RAM before spilling more of its own working set to the temp file,
    regardless of the system's actual available memory (DuckDB's default is
    80% of detected system RAM, which is far too permissive to rely on
    here). 4 GB is deliberately conservative; raise it for a faster run if
    you know the machine has room.

    One consequence of building `patient_id`/`anchor_time` from the SQL
    query directly instead of a pandas merge: `anchor_time` in the output
    is DuckDB's string cast of the anchor timestamp rather than however
    pandas would format the original datetime object — functionally the
    same value, cosmetically it may differ in a trailing `:00` or similar.

    Parameters
    ----------
    fm, omop_dir, measurement_path, concept_path, loinc_codes, verbose :
        see `build_event_timeline`
    out_path : destination CSV path (parent directory created if needed)

    Returns
    -------
    int : total number of rows written, counted from the output file
          itself (not re-queried from DuckDB, so the source CSVs — up to
          22+ GB for measurement.csv — are only scanned once).
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    try:
        import duckdb
    except ImportError:
        raise ImportError("pip install duckdb")
    import pandas as pd
    import tempfile

    feature_types     = list(getattr(fm, "feature_types", None) or [])
    count_window_days = dict(getattr(fm, "count_window_days", None) or {})
    windows_days      = list(fm.windows_days) if fm.windows_days else [365]

    anchors_df = pd.DataFrame({
        "person_id":     fm.patient_ids.astype("int64"),
        "impression_id": fm.impression_ids,
        "anchor_time":   [str(a) for a in fm.anchor_times],
        "y":             fm.y.astype(float),
    })

    # On-disk temp file, not the default in-memory database — gives DuckDB
    # somewhere to spill large intermediate state. See the docstring for why
    # this alone (without also dropping ORDER BY, below) was not enough.
    tmp_db = Path(tempfile.mktemp(suffix="_event_timeline.duckdb"))
    con = duckdb.connect(str(tmp_db))
    con.execute("PRAGMA threads=4")                       # cap parallelism
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")  # spill early, on purpose
    con.register("anchors_tbl", anchors_df)

    _log("[timeline] loading concept.csv …")
    con.execute(f"""
        CREATE TEMP TABLE _concept AS
        SELECT
            CAST(concept_id AS BIGINT) AS concept_id,
            vocabulary_id,
            concept_code,
            concept_name
        FROM read_csv_auto('{concept_path}', ignore_errors=true)
        WHERE concept_id IS NOT NULL
    """)

    parts_sql: list = []
    params: list = []

    # ── Labs ────────────────────────────────────────────────────────────────
    if "labs" in feature_types:
        max_window = max(windows_days)
        loinc_filter = ""
        if loinc_codes:
            loinc_filter = f"AND c.concept_code IN ({', '.join('?' * len(loinc_codes))})"
            params.extend(loinc_codes)
        _log(f"[timeline] labs: scanning measurement.csv (window={max_window} d) …")
        parts_sql.append(f"""
            SELECT
                a.impression_id,
                a.person_id                                                              AS patient_id,
                a.anchor_time,
                CAST(a.y AS DOUBLE)                                                      AS y,
                'lab'                                                                     AS event_type,
                COALESCE(c.vocabulary_id, 'Unknown')                                      AS vocabulary,
                COALESCE(c.concept_code,  '')                                             AS concept_code,
                COALESCE(c.concept_name,  '')                                             AS concept_name,
                COALESCE(
                    TRY_CAST(t.measurement_datetime AS TIMESTAMP),
                    CAST(t.measurement_date AS TIMESTAMP)
                )                                                                         AS event_datetime,
                CAST(t.measurement_date AS DATE)                                          AS event_date,
                -DATEDIFF('day',
                    CAST(t.measurement_date AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE))                      AS days_before_ctpa,
                CAST(t.value_as_number AS DOUBLE)                                         AS value
            FROM read_csv_auto('{measurement_path}', ignore_errors=true) t
            INNER JOIN anchors_tbl a
                    ON CAST(t.person_id AS BIGINT) = a.person_id
            LEFT  JOIN _concept c
                    ON CAST(t.measurement_concept_id AS BIGINT) = c.concept_id
            WHERE t.measurement_date IS NOT NULL
              AND CAST(t.measurement_concept_id AS BIGINT) != 0
              {loinc_filter}
              AND DATEDIFF('day',
                    CAST(t.measurement_date AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                  ) BETWEEN 1 AND {max_window}
        """)

    # ── Count feature types ──────────────────────────────────────────────────
    for ft, (tbl, date_col, datetime_col, concept_col, label, value_expr) in _TIMELINE_TABLE_CONFIG.items():
        if ft not in feature_types:
            continue
        csv_path = Path(omop_dir) / f"{tbl}.csv"
        if not csv_path.exists():
            _log(f"  WARNING: {tbl}.csv not found — skipping {ft}")
            continue
        window_days = count_window_days.get(ft, 365)
        _log(f"[timeline] {ft}: scanning {tbl}.csv (window={window_days} d) …")
        parts_sql.append(f"""
            SELECT
                a.impression_id,
                a.person_id                                                              AS patient_id,
                a.anchor_time,
                CAST(a.y AS DOUBLE)                                                      AS y,
                '{label}'                                                                 AS event_type,
                COALESCE(c.vocabulary_id, 'Unknown')                                      AS vocabulary,
                COALESCE(c.concept_code,  '')                                             AS concept_code,
                COALESCE(c.concept_name,  '')                                             AS concept_name,
                COALESCE(
                    TRY_CAST(t.{datetime_col} AS TIMESTAMP),
                    CAST(t.{date_col} AS TIMESTAMP)
                )                                                                         AS event_datetime,
                CAST(t.{date_col} AS DATE)                                                AS event_date,
                -DATEDIFF('day',
                    CAST(t.{date_col} AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE))                      AS days_before_ctpa,
                {value_expr}                                                               AS value
            FROM read_csv_auto('{csv_path}', ignore_errors=true) t
            INNER JOIN anchors_tbl a
                    ON CAST(t.person_id AS BIGINT) = a.person_id
            LEFT  JOIN _concept c
                    ON CAST(t.{concept_col} AS BIGINT) = c.concept_id
            WHERE t.{date_col} IS NOT NULL
              AND CAST(t.{concept_col} AS BIGINT) != 0
              AND DATEDIFF('day',
                    CAST(t.{date_col} AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                  ) BETWEEN 1 AND {window_days}
        """)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not parts_sql:
            out_path.write_text(
                "impression_id,patient_id,anchor_time,y,event_type,vocabulary,"
                "concept_code,concept_name,event_datetime,event_date,"
                "days_before_ctpa,value\n"
            )
            return 0

        union_sql = "\nUNION ALL\n".join(parts_sql)
        _log(f"[timeline] writing directly to {out_path} …")
        # POSIX path with single quotes escaped, matching how paths are quoted
        # elsewhere in this file's f-string SQL.
        escaped_out = str(out_path).replace("'", "''")
        # No ORDER BY, deliberately — see the function docstring. Rows come
        # out grouped by feature type in whatever order DuckDB's parallel
        # scan produces them, not globally sorted by study/time.
        con.execute(f"""
            COPY (
                {union_sql}
            ) TO '{escaped_out}' (HEADER, DELIMITER ',')
        """, params)
    finally:
        try:
            con.close()
        except Exception:
            pass
        if tmp_db.exists():
            tmp_db.unlink(missing_ok=True)

    # Row count from the file itself rather than re-querying DuckDB, so the
    # (potentially 22+ GB) source CSVs are only scanned once.
    with open(out_path, "rb") as f:
        n_rows = sum(1 for _ in f) - 1   # minus header
    n_rows = max(n_rows, 0)
    _log(f"[timeline] {n_rows:,} rows written")
    return n_rows


def build_timeline_skeleton_streamed(
    fm: "LabFeatureMatrix",
    omop_dir: "Path",
    measurement_path: "Path",
    concept_path: "Path",
    out_path,
    loinc_codes: Optional[list] = None,
    memory_limit_gb: float = 4.0,
    verbose: bool = True,
) -> int:
    """Cohort-wide, de-identified timeline export for the EHR timeline viewer
    (Tab 5) — identifiers, clinical panel/feature, and timing only. No
    measured values, no task label.

    Same query shape as build_event_timeline_streamed, with two differences:

    1. Every event additionally gets a `panel` column — a clinical grouping
       (e.g. "Renal function", "Anticoagulants") assigned via a LEFT JOIN
       against appd_clinical_panels.panel_lookup_rows(), not a Python pass
       over the rows (which would reintroduce exactly the memory problem
       this module spent a lot of this session getting rid of). Anything
       not in that curated taxonomy still appears, tagged "Other <type>" —
       nothing is silently dropped for lacking a panel.
    2. `value` and `y` are never selected — this artifact is deliberately
       data-minimized to (impression_id, patient_id, anchor_time, panel,
       feature identity, timing) only, suitable for a de-identified
       multi-lane timeline view, not for modelling (use
       build_event_timeline_streamed for that).

    Same memory-safety measures as build_event_timeline_streamed apply here
    for the same reasons: on-disk temp DuckDB file, capped `memory_limit_gb`,
    no ``ORDER BY`` (rows come out grouped by feature type, not globally
    sorted — sort after loading if you need chronological order).

    Parameters
    ----------
    fm, omop_dir, measurement_path, concept_path, loinc_codes,
    memory_limit_gb, verbose : see build_event_timeline_streamed
    out_path : destination CSV path (parent directory created if needed)

    Returns
    -------
    int : total number of rows written.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    try:
        import duckdb
    except ImportError:
        raise ImportError("pip install duckdb")
    import pandas as pd
    import tempfile
    from Custom.appd_clinical_panels import panel_lookup_rows

    feature_types     = list(getattr(fm, "feature_types", None) or [])
    count_window_days = dict(getattr(fm, "count_window_days", None) or {})
    windows_days      = list(fm.windows_days) if fm.windows_days else [365]

    anchors_df = pd.DataFrame({
        "person_id":     fm.patient_ids.astype("int64"),
        "impression_id": fm.impression_ids,
        "anchor_time":   [str(a) for a in fm.anchor_times],
    })

    tmp_db = Path(tempfile.mktemp(suffix="_timeline_skeleton.duckdb"))
    con = duckdb.connect(str(tmp_db))
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.register("anchors_tbl", anchors_df)

    _log("[skeleton] loading concept.csv …")
    con.execute(f"""
        CREATE TEMP TABLE _concept AS
        SELECT
            CAST(concept_id AS BIGINT) AS concept_id,
            vocabulary_id,
            concept_code,
            concept_name
        FROM read_csv_auto('{concept_path}', ignore_errors=true)
        WHERE concept_id IS NOT NULL
    """)

    _log("[skeleton] loading clinical panel taxonomy …")
    panel_df = pd.DataFrame(
        panel_lookup_rows(), columns=["vocabulary_id", "concept_code", "panel"])
    con.register("_panels", panel_df)

    parts_sql: list = []
    params: list = []

    # ── Labs ────────────────────────────────────────────────────────────────
    if "labs" in feature_types:
        max_window = max(windows_days)
        loinc_filter = ""
        if loinc_codes:
            loinc_filter = f"AND c.concept_code IN ({', '.join('?' * len(loinc_codes))})"
            params.extend(loinc_codes)
        _log(f"[skeleton] labs: scanning measurement.csv (window={max_window} d) …")
        parts_sql.append(f"""
            SELECT
                a.impression_id,
                a.person_id                                                              AS patient_id,
                a.anchor_time,
                COALESCE(p.panel, 'Other lab')                                            AS panel,
                'lab'                                                                     AS event_type,
                COALESCE(c.vocabulary_id, 'Unknown')                                      AS vocabulary,
                COALESCE(c.concept_code,  '')                                             AS concept_code,
                COALESCE(c.concept_name,  '')                                             AS concept_name,
                COALESCE(
                    TRY_CAST(t.measurement_datetime AS TIMESTAMP),
                    CAST(t.measurement_date AS TIMESTAMP)
                )                                                                         AS event_datetime,
                CAST(t.measurement_date AS DATE)                                          AS event_date,
                -DATEDIFF('day',
                    CAST(t.measurement_date AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE))                      AS days_before_ctpa
            FROM read_csv_auto('{measurement_path}', ignore_errors=true) t
            INNER JOIN anchors_tbl a
                    ON CAST(t.person_id AS BIGINT) = a.person_id
            LEFT  JOIN _concept c
                    ON CAST(t.measurement_concept_id AS BIGINT) = c.concept_id
            LEFT  JOIN _panels p
                    ON p.vocabulary_id = c.vocabulary_id AND p.concept_code = c.concept_code
            WHERE t.measurement_date IS NOT NULL
              AND CAST(t.measurement_concept_id AS BIGINT) != 0
              {loinc_filter}
              AND DATEDIFF('day',
                    CAST(t.measurement_date AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                  ) BETWEEN 1 AND {max_window}
        """)

    # ── Count feature types ──────────────────────────────────────────────────
    for ft, (tbl, date_col, datetime_col, concept_col, label, _value_expr) in _TIMELINE_TABLE_CONFIG.items():
        if ft not in feature_types:
            continue
        csv_path = Path(omop_dir) / f"{tbl}.csv"
        if not csv_path.exists():
            _log(f"  WARNING: {tbl}.csv not found — skipping {ft}")
            continue
        window_days = count_window_days.get(ft, 365)
        _log(f"[skeleton] {ft}: scanning {tbl}.csv (window={window_days} d) …")
        parts_sql.append(f"""
            SELECT
                a.impression_id,
                a.person_id                                                              AS patient_id,
                a.anchor_time,
                COALESCE(p.panel, 'Other {label}')                                        AS panel,
                '{label}'                                                                 AS event_type,
                COALESCE(c.vocabulary_id, 'Unknown')                                      AS vocabulary,
                COALESCE(c.concept_code,  '')                                             AS concept_code,
                COALESCE(c.concept_name,  '')                                             AS concept_name,
                COALESCE(
                    TRY_CAST(t.{datetime_col} AS TIMESTAMP),
                    CAST(t.{date_col} AS TIMESTAMP)
                )                                                                         AS event_datetime,
                CAST(t.{date_col} AS DATE)                                                AS event_date,
                -DATEDIFF('day',
                    CAST(t.{date_col} AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE))                      AS days_before_ctpa
            FROM read_csv_auto('{csv_path}', ignore_errors=true) t
            INNER JOIN anchors_tbl a
                    ON CAST(t.person_id AS BIGINT) = a.person_id
            LEFT  JOIN _concept c
                    ON CAST(t.{concept_col} AS BIGINT) = c.concept_id
            LEFT  JOIN _panels p
                    ON p.vocabulary_id = c.vocabulary_id AND p.concept_code = c.concept_code
            WHERE t.{date_col} IS NOT NULL
              AND CAST(t.{concept_col} AS BIGINT) != 0
              AND DATEDIFF('day',
                    CAST(t.{date_col} AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                  ) BETWEEN 1 AND {window_days}
        """)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not parts_sql:
            out_path.write_text(
                "impression_id,patient_id,anchor_time,panel,event_type,vocabulary,"
                "concept_code,concept_name,event_datetime,event_date,"
                "days_before_ctpa\n"
            )
            return 0

        union_sql = "\nUNION ALL\n".join(parts_sql)
        _log(f"[skeleton] writing directly to {out_path} …")
        escaped_out = str(out_path).replace("'", "''")
        # No ORDER BY, deliberately — same reasoning as build_event_timeline_streamed.
        con.execute(f"""
            COPY (
                {union_sql}
            ) TO '{escaped_out}' (HEADER, DELIMITER ',')
        """, params)
    finally:
        try:
            con.close()
        except Exception:
            pass
        if tmp_db.exists():
            tmp_db.unlink(missing_ok=True)

    with open(out_path, "rb") as f:
        n_rows = sum(1 for _ in f) - 1
    n_rows = max(n_rows, 0)
    _log(f"[skeleton] {n_rows:,} rows written")
    return n_rows


# ---------------------------------------------------------------------------
# LabExtractor — the extraction pipeline
# ---------------------------------------------------------------------------

class LabExtractor:
    """Extract EHR features from OMOP CSVs via DuckDB.

    Args:
        cohort                : path to cohort master CSV
        measurement_csv       : path to measurement.csv
        concept_csv           : path to concept.csv
        concept_ancestor_csv  : path to concept_ancestor.csv (optional — needed only
                                when use_concept_ancestor=True in build())
        labels                : path to labels TSV (survival data)
        person                : path to person.csv (optional — demographics)
        verbose               : print progress messages
    """

    def __init__(
        self,
        cohort:                 Optional[str] = None,
        measurement_csv:        Optional[str] = None,
        concept_csv:            Optional[str] = None,
        concept_ancestor_csv:   Optional[str] = None,
        labels:                 Optional[str] = None,
        person:                 Optional[str] = None,
        verbose:                bool = True,
    ):
        self.cohort_path      = Path(cohort          or DEFAULT_COHORT     ).expanduser()
        self.measurement_path = Path(measurement_csv or DEFAULT_MEASUREMENT ).expanduser()
        self.concept_path     = Path(concept_csv     or DEFAULT_CONCEPT    ).expanduser()
        self.labels_path      = Path(labels          or DEFAULT_LABELS     ).expanduser()
        # Infer OMOP directory from measurement.csv path (used for count-feature CSVs)
        self.omop_dir         = self.measurement_path.parent
        # concept_ancestor.csv — optional; defaults to same OMOP directory
        _anc = concept_ancestor_csv or str(DEFAULT_CONCEPT_ANCESTOR)
        self.concept_ancestor_path = Path(_anc).expanduser() if _anc else None
        # person.csv is optional — demographics are appended only if it exists
        _person = person or str(DEFAULT_PERSON)
        self.person_path = Path(_person).expanduser() if _person else None
        self.verbose     = verbose

        for p, what in [
            (self.cohort_path,      "cohort CSV"),
            (self.measurement_path, "measurement.csv"),
            (self.concept_path,     "concept.csv"),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"{what} not found: {p}")

    # -- logging ----------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- anchor -----------------------------------------------------------

    @staticmethod
    def anchor_kind_for(task: str, override: Optional[str] = None) -> str:
        if override:
            if override not in ("dx", "px"):
                raise ValueError("anchor must be 'dx' or 'px'")
            return override
        if task in DX_TASKS:
            return "dx"
        if task in PX_TASKS:
            return "px"
        raise ValueError(
            f"Unknown task '{task}'. Pass anchor='dx' or 'px' explicitly.")

    # -- cohort -----------------------------------------------------------

    def _read_cohort(
        self, task: str, anchor_kind: str,
        one_per_patient: Optional[str] = None,
    ) -> list:
        """Returns list of (person_id, anchor_dt, impression_id, split, y).

        one_per_patient : None/"all" (default) keeps every labelled study.
                           "first" keeps only the earliest-anchor study per
                           patient; "last" keeps only the latest. Patients
                           with one study are unaffected either way. Reduces
                           every downstream cost proportionally (extraction
                           time, matrix rows, long-format/timeline export
                           size) — the most direct lever if a cohort's
                           volume, not its width, is what's causing memory
                           pressure. It also removes the correlated-studies
                           issue of one patient contributing multiple rows
                           if that matters for how you're using the data.
        """
        offset = datetime.timedelta(days=1) if anchor_kind == "dx" else datetime.timedelta(0)
        rows, skipped = [], 0

        with open(self.cohort_path) as f:
            delim  = "\t" if self.cohort_path.suffix == ".tsv" else ","
            reader = csv.DictReader(f, delimiter=delim)
            pid_col = next(
                (c for c in PATIENT_ID_CANDIDATES if c in (reader.fieldnames or [])),
                None,
            )
            if pid_col is None:
                raise ValueError(
                    f"Could not find a patient-ID column in {self.cohort_path.name}. "
                    f"Expected one of {PATIENT_ID_CANDIDATES}.")

            for row in reader:
                raw = str(row.get(task, "")).strip().upper()
                if raw in SKIP_LABEL_VALUES:
                    skipped += 1
                    continue
                anchor_dt = datetime.datetime.fromisoformat(
                    row[TIME_COLUMN]) - offset
                rows.append((
                    int(row[pid_col]),
                    anchor_dt,
                    str(row[IMPRESSION_COLUMN]),
                    row.get("split", ""),
                    1.0 if raw in TRUTHY else 0.0,
                ))

        self._log(
            f"cohort: {len(rows):,} labelled rows  "
            f"({skipped:,} censored/missing skipped)")

        if one_per_patient in ("first", "last"):
            n_before = len(rows)
            # Sort by anchor time first so a plain "keep the first (or last)
            # occurrence per key" dict pass gives the earliest/latest row —
            # dicts preserve insertion order, so this needs no extra bookkeeping.
            rows.sort(key=lambda r: r[1], reverse=(one_per_patient == "last"))
            by_patient: dict = {}
            for r in rows:
                by_patient.setdefault(r[0], r)
            rows = list(by_patient.values())
            self._log(
                f"  {one_per_patient}-study-per-patient: {n_before:,} -> "
                f"{len(rows):,} rows ({n_before - len(rows):,} dropped, "
                f"{len(rows):,} unique patients)")

        return rows

    # -- survival ---------------------------------------------------------

    def _read_survival(
        self, task: str, imp_ids: list
    ) -> tuple:
        """Returns (tte_array, event_array) aligned to imp_ids, or (None, None)."""
        if task not in SURVIVAL_COLUMNS:
            self._log(f"no survival columns registered for '{task}' — tte will be None")
            return None, None

        tte_col, cen_col = SURVIVAL_COLUMNS[task]
        lookup: dict = {}

        for path, delim in [
            (self.cohort_path, "\t" if self.cohort_path.suffix == ".tsv" else ","),
            (self.labels_path, "\t"),
        ]:
            if not path.is_file():
                continue
            with open(path) as f:
                reader = csv.DictReader(f, delimiter=delim)
                fnames = reader.fieldnames or []
                if not {tte_col, cen_col} <= set(fnames):
                    continue
                for row in reader:
                    raw = str(row[tte_col]).strip()
                    if raw in ("", "NA", "nan", "NaN"):
                        continue
                    censored = str(row[cen_col]).strip().upper() in TRUTHY
                    lookup[str(row[IMPRESSION_COLUMN])] = (
                        float(raw) / TTE_MINUTES_PER_DAY,
                        0.0 if censored else 1.0,
                    )
            self._log(f"survival: {len(lookup):,} rows from {tte_col} ({path.name})")
            break

        if not lookup:
            self._log("WARNING: survival columns declared but no data found — tte=None")
            return None, None

        tte   = np.array([lookup.get(i, (np.nan, np.nan))[0] for i in imp_ids],
                         dtype=np.float32)
        event = np.array([lookup.get(i, (np.nan, np.nan))[1] for i in imp_ids],
                         dtype=np.float32)
        missing = int(np.isnan(tte).sum())
        if missing:
            self._log(f"  WARNING: {missing:,}/{len(imp_ids):,} rows "
                      "have no matching survival entry")
        return tte, event

    # -- admission anchoring ------------------------------------------------

    def _find_encompassing_admissions(self, cohort_rows: list) -> "pd.DataFrame":
        """For each cohort row, find the inpatient admission the anchor falls
        inside — i.e. visit_start_date <= anchor_date <= visit_end_date (or
        the visit is still open, visit_end_date IS NULL).

        Used by admission-anchored extraction (``build(admission_anchored=True)``):
        the feature window becomes [admission_start_date, anchor_time] per
        patient instead of a fixed number of days before anchor for everyone.
        Studies with no qualifying admission have no row in the result and are
        dropped from the cohort by the caller.

        A single DuckDB join against the whole cohort at once — no per-patient
        Python loop.

        Returns
        -------
        pd.DataFrame with columns:
            impression_id, admission_date (ISO date string),
            visit_occurrence_id, days_since_admission (int, admission -> anchor)
        One row per impression_id with a qualifying admission. Patients with
        multiple overlapping qualifying visits (a data-quality edge case) keep
        only the one with the latest visit_start_date.
        """
        import duckdb
        import pandas as pd

        visit_csv = self.omop_dir / "visit_occurrence.csv"
        if not visit_csv.exists():
            raise FileNotFoundError(
                f"visit_occurrence.csv not found at {visit_csv} — required "
                "for admission-anchored extraction.")

        anchors_df = pd.DataFrame(
            [(pid, anchor.isoformat(), imp) for pid, anchor, imp, _, _ in cohort_rows],
            columns=["person_id", "anchor_time", "impression_id"],
        )
        anchors_df["person_id"] = anchors_df["person_id"].astype("int64")

        concept_ids = ", ".join(str(c) for c in ADMISSION_VISIT_CONCEPT_IDS)
        con = duckdb.connect()
        con.register("anchors_tbl", anchors_df)
        df = con.execute(f"""
            SELECT impression_id, admission_date, visit_occurrence_id, days_since_admission
            FROM (
                SELECT
                    a.impression_id,
                    CAST(t.visit_start_date AS VARCHAR)                        AS admission_date,
                    t.visit_occurrence_id                                      AS visit_occurrence_id,
                    DATEDIFF('day', CAST(t.visit_start_date AS DATE),
                             CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE))   AS days_since_admission,
                    ROW_NUMBER() OVER (
                        PARTITION BY a.impression_id
                        ORDER BY t.visit_start_date DESC
                    )                                                         AS rn
                FROM read_csv_auto('{visit_csv}', ignore_errors=true) t
                INNER JOIN anchors_tbl a
                        ON CAST(t.person_id AS BIGINT) = a.person_id
                WHERE t.visit_concept_id IN ({concept_ids})
                  AND t.visit_start_date IS NOT NULL
                  AND CAST(t.visit_start_date AS DATE)
                        <= CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                  AND (
                        t.visit_end_date IS NULL
                        OR CAST(t.visit_end_date AS DATE)
                             >= CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                      )
            )
            WHERE rn = 1
        """).df()
        con.close()

        self._log(
            f"  encompassing admission: {df['impression_id'].nunique():,}/"
            f"{len(cohort_rows):,} studies matched "
            f"(visit_concept_id IN {ADMISSION_VISIT_CONCEPT_IDS})")
        return df

    # -- DuckDB-native scan + pivot ----------------------------------------

    def _run_duckdb_and_pivot(
        self,
        cohort_rows: list,
        windows_days: list,
        loinc_codes: Optional[list],
        min_studies: int,
        all_impression_ids: list,
        admission_dates: Optional[dict] = None,
        col_suffix: str = "",
    ) -> tuple:
        """Scan measurement.csv, aggregate per window, filter, and pivot lab
        features — all inside DuckDB with an on-disk connection.

        Replaces the ``_run_duckdb()`` + ``_pivot()`` pair used in ``build()``.
        Intermediate tables (raw scan, per-window aggregates, long-format
        feature table) live in a temporary DuckDB file and can spill to disk,
        so peak Python RAM equals the final float32 array only rather than the
        3–4× overhead that arises from long_df → melt → pivot_table in pandas.

        Requires DuckDB ≥ 0.8.0 (PIVOT statement).

        admission_dates : optional {impression_id: ISO date string}. When
            given (admission-anchored mode), each study's effective upper
            bound becomes ``LEAST(w, admission -> anchor gap)`` instead of the
            fixed ``w`` for every study — a per-row SQL expression, still one
            vectorized scan, not a per-patient loop. Every impression_id in
            ``cohort_rows`` must have an entry (the caller is expected to have
            already dropped studies with no qualifying admission).
        col_suffix : appended to every generated column name, right after the
            ``_{w}d`` window tag (e.g. ``"_preadm"`` → ``..._last_7d_preadm``).
            Used by ``build()`` to run this same function a second time with
            ``cohort_rows`` anchored at admission_date instead of anchor_time
            (a pre-admission baseline window) without colliding column names.

        Returns
        -------
        X       : float32 ndarray, shape (n_studies, n_features)
                  NaN = not measured in that window
        columns : list[str] feature names aligned to X columns
        """
        import duckdb
        import pandas as pd
        import tempfile

        windows_days = sorted(windows_days)
        max_window   = max(windows_days)

        anchors_df = pd.DataFrame(
            [(pid, anchor.isoformat(), imp)
             for pid, anchor, imp, _, _ in cohort_rows],
            columns=["person_id", "anchor_time", "impression_id"],
        )
        anchors_df["person_id"] = anchors_df["person_id"].astype("int64")
        if admission_dates is not None:
            anchors_df["admission_date"] = anchors_df["impression_id"].map(admission_dates)

        if admission_dates is not None:
            max_window_expr = (
                f"LEAST({max_window}, DATEDIFF('day', "
                "CAST(a.admission_date AS DATE), "
                "CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)))"
            )
        else:
            max_window_expr = str(max_window)

        tmp_db = Path(tempfile.mktemp(suffix="_lab_pivot.duckdb"))
        try:
            con = duckdb.connect(str(tmp_db))
            con.register("anchors_tbl", anchors_df)

            # ── 1. Load LOINC concept IDs ────────────────────────────────────
            # Bind LOINC codes (free-typed by the user in the app) as query
            # parameters rather than splicing them into the SQL string.
            loinc_params: list = []
            if loinc_codes:
                loinc_filter = f"AND concept_code IN ({', '.join('?' * len(loinc_codes))})"
                loinc_params = list(loinc_codes)
            else:
                loinc_filter = ""
            self._log("  loading LOINC concepts …")
            con.execute(f"""
                CREATE TEMP TABLE _loinc AS
                SELECT CAST(concept_id AS BIGINT) AS concept_id, concept_code
                FROM read_csv_auto('{self.concept_path}', ignore_errors=true)
                WHERE vocabulary_id = 'LOINC'
                  {loinc_filter}
            """, loinc_params)
            n_loinc = con.execute("SELECT COUNT(*) FROM _loinc").fetchone()[0]
            self._log(f"  {n_loinc:,} LOINC concepts loaded")
            if n_loinc == 0:
                raise RuntimeError(
                    "No LOINC concepts found in concept.csv. "
                    "Check vocabulary_id='LOINC' rows exist.")

            # ── 2. Scan measurement.csv → on-disk temp table ─────────────────
            #  No .df() here — result stays inside DuckDB and can spill to disk
            self._log(
                f"  scanning measurement.csv (max window="
                f"{'admission-anchored, capped at ' + str(max_window) + 'd' if admission_dates is not None else str(max_window) + ' d'}) …")
            con.execute(f"""
                CREATE TEMP TABLE _raw AS
                SELECT
                    a.impression_id,
                    lc.concept_code                   AS loinc_code,
                    CAST(m.value_as_number AS DOUBLE)  AS value,
                    CAST(DATEDIFF(
                        'day',
                        CAST(COALESCE(
                            TRY_CAST(m.measurement_datetime AS TIMESTAMP),
                            TRY_CAST(m.measurement_date     AS TIMESTAMP)
                        ) AS DATE),
                        CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                    ) AS INTEGER)                     AS days_before
                FROM read_csv_auto('{self.measurement_path}', ignore_errors=true) m
                INNER JOIN anchors_tbl a
                        ON CAST(m.person_id AS BIGINT) = a.person_id
                INNER JOIN _loinc lc
                        ON CAST(m.measurement_concept_id AS BIGINT) = lc.concept_id
                WHERE m.value_as_number IS NOT NULL
                  AND DATEDIFF('day',
                        CAST(COALESCE(
                            TRY_CAST(m.measurement_datetime AS TIMESTAMP),
                            TRY_CAST(m.measurement_date     AS TIMESTAMP)
                        ) AS DATE),
                        CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                      ) BETWEEN 1 AND {max_window_expr}
            """)
            n_raw, n_codes, n_stud = con.execute(
                "SELECT COUNT(*), COUNT(DISTINCT loinc_code), "
                "COUNT(DISTINCT impression_id) FROM _raw"
            ).fetchone()
            self._log(
                f"  {n_raw:,} measurements  "
                f"({n_codes:,} LOINC codes, {n_stud:,} studies)")

            # ── 3. Coverage filter inside DuckDB ─────────────────────────────
            con.execute(f"""
                CREATE TEMP TABLE _kept_loinc AS
                SELECT loinc_code
                FROM _raw
                GROUP BY loinc_code
                HAVING COUNT(DISTINCT impression_id) >= {min_studies}
            """)
            n_kept = con.execute("SELECT COUNT(*) FROM _kept_loinc").fetchone()[0]
            self._log(
                f"  {n_kept:,} labs kept (≥{min_studies} studies); "
                f"{n_codes - n_kept:,} dropped")
            if n_kept == 0:
                raise RuntimeError(
                    "No labs passed the min_studies filter — lower "
                    "min_studies_per_lab or broaden the LOINC filter.")

            # ── 4. Per-window aggregate tables (all on-disk) ──────────────────
            #  FIRST(value ORDER BY days_before ASC) = value of the most recent
            #  measurement (days_before=1 is closest to anchor)
            agg_defs = [
                ("last_val",       "FIRST(value ORDER BY days_before ASC)"),
                ("min_val",        "MIN(value)"),
                ("max_val",        "MAX(value)"),
                ("mean_val",       "AVG(value)"),
                ("n_val",          "COUNT(*)"),
                ("days_since_val", "MIN(days_before)"),
            ]
            agg_select_sql = ",\n                    ".join(
                f"{expr} AS {alias}" for alias, expr in agg_defs
            )
            for w in windows_days:
                con.execute(f"""
                    CREATE TEMP TABLE _agg_{w}d AS
                    SELECT
                        impression_id,
                        loinc_code,
                        {agg_select_sql}
                    FROM _raw
                    INNER JOIN _kept_loinc USING (loinc_code)
                    WHERE days_before <= {w}
                    GROUP BY impression_id, loinc_code
                """)

            # ── 5. Long-format col_name table: UNION ALL across windows/aggs ──
            agg_tags = [
                ("last_val",       "last"),
                ("min_val",        "min"),
                ("max_val",        "max"),
                ("mean_val",       "mean"),
                ("n_val",          "n"),
                ("days_since_val", "days_since"),
            ]
            union_parts = [
                f"SELECT impression_id, "
                f"'labs:LOINC/' || loinc_code || '_{tag}_{w}d{col_suffix}' AS col_name, "
                f"CAST({col} AS DOUBLE) AS value FROM _agg_{w}d"
                for w in windows_days
                for col, tag in agg_tags
            ]
            union_sql = "\nUNION ALL\n".join(union_parts)
            self._log("  building long-format feature table …")
            con.execute(f"CREATE TEMP TABLE _long AS\n{union_sql}")

            # ── 6. PIVOT inside DuckDB ─────────────────────────────────────────
            #  Single .df() call on the already-pivoted result; no pandas
            #  pivot_table, no intermediate melt DataFrame.
            self._log("  pivoting inside DuckDB …")
            wide_df = con.execute(
                "PIVOT _long ON col_name USING FIRST(value) GROUP BY impression_id"
            ).df()

        finally:
            try:
                con.close()
            except Exception:
                pass
            if tmp_db.exists():
                tmp_db.unlink(missing_ok=True)

        # ── 7. Align to cohort order, sort columns, extract numpy ─────────────
        #  Sorting columns gives deterministic alphabetical order matching the
        #  old pandas pivot_table output.
        wide_df = wide_df.set_index("impression_id").reindex(all_impression_ids)
        columns = sorted(wide_df.columns)
        X = wide_df[columns].to_numpy(dtype=np.float32)
        del wide_df   # release DataFrame immediately after numpy extraction

        self._log(
            f"  lab X shape: {X.shape}  "
            f"({(~np.isnan(X)).mean():.1%} of cells observed)")
        return X, columns

    # -- pre-admission baseline + delta (admission-anchored labs only) ----

    def _compute_pre_admission_baseline_and_delta(
        self,
        cohort_rows: list,
        admission_dates: dict,
        windows_days: list,
        loinc_codes: Optional[list],
        min_studies_per_lab: int,
        imp_ids: list,
        pre_admission_days: int,
        X_admission: np.ndarray,
        columns_admission: list,
    ) -> tuple:
        """Pre-admission lab baseline window + delta vs. the admission value.

        Runs ``_run_duckdb_and_pivot`` a second time with each study's anchor
        replaced by its admission_date and a single fixed lookback window
        (``pre_admission_days``) ending there — an ordinary (non
        admission-anchored) call, so it's simply "the N days right before
        admission", uncapped by anything else. Columns are tagged
        ``_preadm`` so they never collide with the admission-to-anchor
        columns already in ``columns_admission``.

        Delta columns (``labs:LOINC/{code}_delta_{agg}`` for agg in
        last/min/max/mean) = the admission-to-anchor value — taken from
        ``X_admission``'s largest ``windows_days`` entry, which is already
        capped to the admission span by the caller's admission-anchored
        call — minus the pre-admission baseline. NaN if either side is
        missing for a given study/lab.

        A soft failure (e.g. too few labs pass min_studies_per_lab in the
        much shorter baseline window) logs a warning and returns empty
        baseline/delta columns rather than aborting the whole extraction.

        Returns
        -------
        X_extra, columns_extra : baseline columns followed by delta columns,
            ready to ``np.hstack`` onto the combined feature matrix.
        """
        baseline_rows = [
            (pid, datetime.date.fromisoformat(admission_dates[imp]), imp, split, y)
            for pid, _anchor, imp, split, y in cohort_rows
        ]
        try:
            X_pre, columns_pre = self._run_duckdb_and_pivot(
                baseline_rows, [pre_admission_days], loinc_codes,
                min_studies_per_lab, imp_ids, admission_dates=None,
                col_suffix="_preadm")
        except RuntimeError as e:
            self._log(
                f"  WARNING: pre-admission baseline extraction failed "
                f"({e}) — skipping baseline/delta")
            return np.zeros((len(imp_ids), 0), dtype=np.float32), []

        max_w    = max(windows_days)
        pre_idx  = {c: i for i, c in enumerate(columns_pre)}

        delta_cols, delta_parts = [], []
        for agg in ("last", "min", "max", "mean"):
            suffix_adm = f"_{agg}_{max_w}d"
            for j, col in enumerate(columns_admission):
                if not col.endswith(suffix_adm):
                    continue
                code_part = col[:-len(suffix_adm)]   # e.g. "labs:LOINC/2160-0"
                pre_col   = f"{code_part}_{agg}_{pre_admission_days}d_preadm"
                if pre_col not in pre_idx:
                    continue
                delta_cols.append(f"{code_part}_delta_{agg}")
                delta_parts.append(X_admission[:, j] - X_pre[:, pre_idx[pre_col]])

        X_delta = (np.column_stack(delta_parts).astype(np.float32)
                   if delta_parts else np.zeros((len(imp_ids), 0), dtype=np.float32))

        self._log(
            f"  pre-admission baseline: {len(columns_pre):,} columns "
            f"({pre_admission_days}d before admission_date)  "
            f"delta: {len(delta_cols):,} columns "
            f"(vs. admission-to-anchor {max_w}d)")

        return np.hstack([X_pre, X_delta]), columns_pre + delta_cols

    # -- count feature extraction (diagnoses / drugs / procedures) --------

    def _run_duckdb_counts(
        self,
        cohort_rows: list,
        tables: list,
        window_days_per_table: dict,
        admission_dates: Optional[dict] = None,
    ) -> "pd.DataFrame":
        """Scan OMOP event tables and count distinct event-days per (study, code).

        Parameters
        ----------
        cohort_rows : list
            Output of ``_read_cohort()`` — list of (person_id, anchor_dt, impression_id,
            split, y).
        tables : list[str]
            Subset of ``_COUNT_TABLE_CONFIG`` keys to include, e.g.
            ``["condition_occurrence", "drug_exposure"]``.
        window_days_per_table : dict
            Mapping from OMOP table name → lookback window in days, e.g.
            ``{"condition_occurrence": 365, "drug_exposure": 60}``.
        admission_dates : optional {impression_id: ISO date string}. When
            given, each table's per-row upper bound becomes
            ``LEAST(window_days, admission -> anchor gap)`` instead of the
            fixed ``window_days`` for every study — window_days_per_table
            still acts as an outer ceiling. Every impression_id in
            ``cohort_rows`` must have an entry.

        Returns
        -------
        pd.DataFrame with columns: impression_id, col_name, day_count
            Where ``col_name`` looks like ``diag:ICD10CM/I26.9_365d``.
        """
        try:
            import duckdb
            import pandas as pd
        except ImportError:
            raise ImportError("pip install duckdb pandas  (Route B requires these packages)")

        anchors_df = pd.DataFrame(
            [
                (pid, anchor.isoformat(), imp)
                for pid, anchor, imp, _split, _y in cohort_rows
            ],
            columns=["person_id", "anchor_time", "impression_id"],
        )
        anchors_df["person_id"] = anchors_df["person_id"].astype("int64")
        if admission_dates is not None:
            anchors_df["admission_date"] = anchors_df["impression_id"].map(admission_dates)

        con = duckdb.connect()
        con.register("anchors_tbl", anchors_df)

        parts = []
        for tbl in tables:
            cfg = _COUNT_TABLE_CONFIG.get(tbl)
            if cfg is None:
                self._log(f"  WARNING: unknown table '{tbl}' — skipping")
                continue
            csv_path = self.omop_dir / f"{tbl}.csv"
            if not csv_path.exists():
                self._log(f"  WARNING: {tbl}.csv not found at {csv_path} — skipping")
                continue

            date_col     = cfg["date_col"]
            concept_col  = cfg["concept_col"]
            col_prefix   = cfg["col_prefix"]
            vocab_filter = cfg["vocab_filter"]

            self._log(f"  loading concepts for {tbl} …")
            concepts_df = con.execute(f"""
                SELECT
                    CAST(concept_id AS BIGINT) AS concept_id,
                    vocabulary_id,
                    concept_code
                FROM read_csv_auto('{self.concept_path}', ignore_errors=true)
                WHERE 1=1
                  {vocab_filter}
            """).df()
            self._log(f"    {len(concepts_df):,} concepts loaded")

            if concepts_df.empty:
                self._log(f"    WARNING: no matching concepts in concept.csv for {tbl} — skipping")
                continue

            tbl_alias  = f"_concepts_{tbl.replace('_', '')}"
            con.register(tbl_alias, concepts_df)
            window_days = window_days_per_table.get(tbl, 365)
            if admission_dates is not None:
                window_upper_expr = (
                    f"LEAST({window_days}, DATEDIFF('day', "
                    "CAST(a.admission_date AS DATE), "
                    "CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)))"
                )
            else:
                window_upper_expr = str(window_days)

            sql = f"""
                SELECT
                    a.impression_id,
                    '{col_prefix}:' || c.vocabulary_id || '/' || c.concept_code
                        || '_{window_days}d'                               AS col_name,
                    COUNT(DISTINCT CAST(t.{date_col} AS DATE))             AS day_count,
                    CAST(MAX(CAST(t.{date_col} AS DATE)) AS VARCHAR)       AS most_recent_date
                FROM read_csv_auto('{csv_path}', ignore_errors=true) t
                INNER JOIN anchors_tbl a
                        ON CAST(t.person_id AS BIGINT) = a.person_id
                INNER JOIN {tbl_alias} c
                        ON CAST(t.{concept_col} AS BIGINT) = c.concept_id
                WHERE t.{date_col} IS NOT NULL
                  AND DATEDIFF(
                        'day',
                        CAST(t.{date_col} AS DATE),
                        CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                      ) BETWEEN 1 AND {window_upper_expr}
                GROUP BY a.impression_id, c.vocabulary_id, c.concept_code
            """
            part_df = con.execute(sql).df()
            self._log(
                f"    {len(part_df):,} (study, code) rows  "
                f"({part_df['col_name'].nunique():,} distinct codes, "
                f"{part_df['impression_id'].nunique():,} studies)")
            parts.append(part_df)

            # --- LOS columns (visit_occurrence only) ----------------------------
            end_date_col = cfg.get("end_date_col")
            if end_date_col:
                self._log(f"    computing LOS features from {end_date_col} …")
                # Admission-anchored mode: the qualifying admission's
                # visit_end_date is often *after* the anchor (patient still
                # admitted at anchor time — that's the qualifying condition
                # itself), so an untruncated LOS would leak the eventual
                # total stay length. Cap at anchor_time so LOS reflects only
                # "days admitted so far", not future stay length.
                if admission_dates is not None:
                    end_date_expr = (
                        f"LEAST(CAST(t.{end_date_col} AS DATE), "
                        "CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE))"
                    )
                else:
                    end_date_expr = f"CAST(t.{end_date_col} AS DATE)"
                los_sql = f"""
                    SELECT
                        a.impression_id,
                        SUM(GREATEST(0, DATEDIFF(
                            'day',
                            CAST(t.{date_col} AS DATE),
                            {end_date_expr}
                        )))                                                 AS los_total,
                        MAX(GREATEST(0, DATEDIFF(
                            'day',
                            CAST(t.{date_col} AS DATE),
                            {end_date_expr}
                        )))                                                 AS los_max,
                        CAST(MAX(CAST(t.{date_col} AS DATE)) AS VARCHAR)   AS most_recent_date
                    FROM read_csv_auto('{csv_path}', ignore_errors=true) t
                    INNER JOIN anchors_tbl a
                            ON CAST(t.person_id AS BIGINT) = a.person_id
                    WHERE t.{date_col}     IS NOT NULL
                      AND t.{end_date_col} IS NOT NULL
                      AND DATEDIFF(
                            'day',
                            CAST(t.{date_col} AS DATE),
                            CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                          ) BETWEEN 1 AND {window_upper_expr}
                    GROUP BY a.impression_id
                """
                los_df = con.execute(los_sql).df()
                if not los_df.empty:
                    import pandas as _pd
                    los_total_df = (
                        los_df[["impression_id", "los_total", "most_recent_date"]]
                        .rename(columns={"los_total": "day_count"})
                        .assign(col_name=f"{col_prefix}:LOS/total_{window_days}d")
                    )
                    los_max_df = (
                        los_df[["impression_id", "los_max", "most_recent_date"]]
                        .rename(columns={"los_max": "day_count"})
                        .assign(col_name=f"{col_prefix}:LOS/max_{window_days}d")
                    )
                    _los_cols = ["impression_id", "col_name", "day_count", "most_recent_date"]
                    parts.append(los_total_df[_los_cols])
                    parts.append(los_max_df[_los_cols])
                    self._log(
                        f"    LOS: {los_df['impression_id'].nunique():,} studies  "
                        f"total_los range=[{los_df['los_total'].min():.0f}, "
                        f"{los_df['los_total'].max():.0f}]  "
                        f"max_los range=[{los_df['los_max'].min():.0f}, "
                        f"{los_df['los_max'].max():.0f}]"
                    )

        con.close()

        if not parts:
            import pandas as pd
            return pd.DataFrame(columns=["impression_id", "col_name", "day_count"])

        import pandas as pd
        return pd.concat(parts, ignore_index=True)

    def _pivot_counts(
        self,
        long_df: "pd.DataFrame",
        all_impression_ids: list,
        min_studies: int,
    ) -> tuple:
        """Pivot count-feature long DataFrame to (X, columns, dates_wide).

        Values are float32 counts of distinct event days (0 = not observed).
        Unlike lab features, missing codes are filled with 0, not NaN.

        Returns
        -------
        X : np.ndarray float32, shape (n, n_features)
        columns : list of str
        dates_wide : pd.DataFrame, shape (n, n_features), values = ISO date strings
            Most-recent event date per (impression, code); '' where not observed.
        """
        import pandas as pd

        empty_dates = pd.DataFrame(index=range(len(all_impression_ids)))

        if long_df.empty:
            return np.zeros((len(all_impression_ids), 0), dtype=np.float32), [], empty_dates

        # Filter: keep codes present in at least min_studies distinct studies
        if min_studies > 0:
            coverage = long_df.groupby("col_name")["impression_id"].nunique()
            keep     = set(coverage[coverage >= min_studies].index)
            n_before = long_df["col_name"].nunique()
            long_df  = long_df[long_df["col_name"].isin(keep)]
            n_after  = long_df["col_name"].nunique()
            self._log(
                f"  {n_after:,} count features kept (≥{min_studies} studies); "
                f"{n_before - n_after:,} dropped")

        if long_df.empty:
            return np.zeros((len(all_impression_ids), 0), dtype=np.float32), [], empty_dates

        # Pivot counts
        wide = long_df.pivot_table(
            index="impression_id",
            columns="col_name",
            values="day_count",
            aggfunc="sum",
            fill_value=0,
        )
        wide.columns.name = None
        wide = wide.reindex(all_impression_ids).fillna(0.0)
        columns = list(wide.columns)
        X = wide.to_numpy(dtype=np.float32)

        # Pivot most-recent dates (kept for the long-format export)
        if "most_recent_date" in long_df.columns:
            dates_wide = long_df.pivot_table(
                index="impression_id",
                columns="col_name",
                values="most_recent_date",
                aggfunc="max",   # latest date if duplicates
            )
            dates_wide.columns.name = None
            dates_wide = dates_wide.reindex(all_impression_ids).fillna("")
            # Reindex columns to match the (filtered) count columns
            dates_wide = dates_wide.reindex(columns=columns).fillna("")
        else:
            dates_wide = pd.DataFrame(
                "", index=range(len(all_impression_ids)), columns=columns)

        return X, columns, dates_wide

    def _run_duckdb_ancestor_rollup(
        self,
        cohort_rows: list,
        tables: list,
        window_days_per_table: dict,
        ancestor_levels: Optional[int] = None,
        min_studies_ancestor: int = 100,
        max_ancestor_features: int = 2000,
        admission_dates: Optional[dict] = None,
    ) -> "pd.DataFrame":
        """Roll up OMOP events to ancestor concepts via concept_ancestor.csv.

        Uses an on-disk DuckDB connection so large intermediate tables spill to
        disk rather than exhausting RAM.  concept_ancestor.csv is loaded once
        into a DuckDB temp table (filtered by level if requested).

        To avoid the explosion of sparse ancestor codes, a two-pass approach is
        used per event table:
          Pass 1 — aggregate inside DuckDB, compute per-code coverage, keep only
                   codes present in ≥ min_studies_ancestor distinct studies (up to
                   max_ancestor_features codes ordered by descending coverage).
          Pass 2 — re-aggregate over only the kept codes and return the trimmed
                   long DataFrame to Python for pivoting.

        Column names follow the pattern  ``{prefix}_anc:{vocab}/{code}_{N}d``,
        e.g. ``diag_anc:SNOMED/59282003_365d``.

        Parameters
        ----------
        cohort_rows            : from ``_read_cohort()``
        tables                 : OMOP table names (subset of ``_COUNT_TABLE_CONFIG``)
        window_days_per_table  : mapping table → lookback days
        ancestor_levels        : maximum levels to climb in the hierarchy
                                 (None = all ancestors)
        min_studies_ancestor   : drop ancestor codes present in fewer studies
                                 (default 100; higher than the direct-code threshold
                                 because ancestor codes are far more numerous)
        max_ancestor_features  : hard cap on ancestor columns per event table,
                                 keeping the highest-coverage codes (default 2000)
        admission_dates        : optional {impression_id: ISO date string}. See
                                 ``_run_duckdb_counts`` — same per-row
                                 ``LEAST(window_days, admission -> anchor gap)`` bound.

        Returns
        -------
        pd.DataFrame with columns: impression_id, col_name, day_count, most_recent_date
        """
        import pandas as pd

        if self.concept_ancestor_path is None or not self.concept_ancestor_path.exists():
            self._log(
                "  WARNING: concept_ancestor.csv not found — ancestor rollup skipped"
            )
            return pd.DataFrame(
                columns=["impression_id", "col_name", "day_count", "most_recent_date"]
            )

        try:
            import duckdb
        except ImportError:
            raise ImportError("pip install duckdb")

        import tempfile

        anchors_df = pd.DataFrame(
            [
                (pid, anchor.isoformat(), imp)
                for pid, anchor, imp, _split, _y in cohort_rows
            ],
            columns=["person_id", "anchor_time", "impression_id"],
        )
        anchors_df["person_id"] = anchors_df["person_id"].astype("int64")
        if admission_dates is not None:
            anchors_df["admission_date"] = anchors_df["impression_id"].map(admission_dates)

        # Use an on-disk DuckDB file so large temp tables can spill to disk
        tmp_db = Path(tempfile.mktemp(suffix="_ancestor_rollup.duckdb"))
        try:
            con = duckdb.connect(str(tmp_db))
            # Allow DuckDB to spill aggressively; temp dir = system default
            con.execute("PRAGMA threads=4")

            con.register("anchors_tbl", anchors_df)

            # ── Step 1: load concept.csv once ────────────────────────────────
            self._log("  [ancestor] loading concept.csv → temp table …")
            con.execute(f"""
                CREATE TEMP TABLE _concept AS
                SELECT
                    CAST(concept_id AS BIGINT) AS concept_id,
                    vocabulary_id,
                    concept_code
                FROM read_csv_auto('{self.concept_path}', ignore_errors=true)
                WHERE concept_code IS NOT NULL
            """)

            # ── Step 2: load concept_ancestor once (filtered by level) ───────
            levels_filter = (
                f"AND CAST(min_levels_of_separation AS INTEGER) BETWEEN 1 AND {ancestor_levels}"
                if ancestor_levels is not None
                else "AND CAST(min_levels_of_separation AS INTEGER) >= 1"
            )
            self._log(
                f"  [ancestor] loading concept_ancestor.csv "
                f"({'all levels' if ancestor_levels is None else f'≤{ancestor_levels} levels'}) "
                f"→ temp table …"
            )
            con.execute(f"""
                CREATE TEMP TABLE _ca AS
                SELECT
                    CAST(descendant_concept_id AS BIGINT) AS descendant_concept_id,
                    CAST(ancestor_concept_id   AS BIGINT) AS ancestor_concept_id
                FROM read_csv_auto('{self.concept_ancestor_path}', ignore_errors=true)
                WHERE 1=1
                  {levels_filter}
            """)
            n_ca = con.execute("SELECT COUNT(*) FROM _ca").fetchone()[0]
            self._log(f"    {n_ca:,} ancestor pairs retained after level filter")

            # ── Step 3: per-event-table rollup ───────────────────────────────
            parts = []
            for tbl in tables:
                cfg = _COUNT_TABLE_CONFIG.get(tbl)
                if cfg is None:
                    continue
                csv_path = self.omop_dir / f"{tbl}.csv"
                if not csv_path.exists():
                    self._log(
                        f"  WARNING: {tbl}.csv not found — ancestor rollup skipped for {tbl}"
                    )
                    continue

                date_col    = cfg["date_col"]
                concept_col = cfg["concept_col"]
                col_prefix  = cfg["col_prefix"]
                anc_prefix  = f"{col_prefix}_anc"
                window_days = window_days_per_table.get(tbl, 365)
                if admission_dates is not None:
                    window_upper_expr = (
                        f"LEAST({window_days}, DATEDIFF('day', "
                        "CAST(a.admission_date AS DATE), "
                        "CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)))"
                    )
                else:
                    window_upper_expr = str(window_days)

                self._log(f"  [ancestor] {tbl} → {window_days} d window …")

                # 3a. Distinct descendant concept_ids seen in cohort window → temp table
                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE _desc_ids AS
                    SELECT DISTINCT CAST(t.{concept_col} AS BIGINT) AS descendant_concept_id
                    FROM read_csv_auto('{csv_path}', ignore_errors=true) t
                    INNER JOIN anchors_tbl a
                            ON CAST(t.person_id AS BIGINT) = a.person_id
                    WHERE t.{date_col} IS NOT NULL
                      AND DATEDIFF(
                            'day',
                            CAST(t.{date_col} AS DATE),
                            CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                          ) BETWEEN 1 AND {window_upper_expr}
                """)
                n_desc = con.execute("SELECT COUNT(*) FROM _desc_ids").fetchone()[0]
                self._log(f"    {n_desc:,} distinct descendant concept_ids in window")
                if n_desc == 0:
                    continue

                # 3b. Ancestor concepts for those descendants → temp table
                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE _anc_concepts AS
                    SELECT DISTINCT
                        d.descendant_concept_id,
                        c.concept_id  AS ancestor_concept_id,
                        c.vocabulary_id,
                        c.concept_code
                    FROM _desc_ids d
                    INNER JOIN _ca ca
                            ON ca.descendant_concept_id = d.descendant_concept_id
                    INNER JOIN _concept c
                            ON c.concept_id = ca.ancestor_concept_id
                """)
                n_anc = con.execute(
                    "SELECT COUNT(DISTINCT ancestor_concept_id) FROM _anc_concepts"
                ).fetchone()[0]
                self._log(f"    {n_anc:,} distinct ancestor concepts")
                if n_anc == 0:
                    continue

                # 3c. Pass 1 — coverage per ancestor code (fully inside DuckDB)
                #     Keep only codes present in ≥ min_studies_ancestor studies,
                #     capped at max_ancestor_features ordered by descending coverage.
                self._log(
                    f"    pass 1: computing per-ancestor coverage "
                    f"(min={min_studies_ancestor} studies, cap={max_ancestor_features}) …"
                )
                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE _kept_anc AS
                    SELECT
                        '{anc_prefix}:' || ac.vocabulary_id || '/' || ac.concept_code
                            || '_{window_days}d'                              AS col_name,
                        COUNT(DISTINCT a.impression_id)                       AS n_studies
                    FROM read_csv_auto('{csv_path}', ignore_errors=true) t
                    INNER JOIN anchors_tbl a
                            ON CAST(t.person_id AS BIGINT) = a.person_id
                    INNER JOIN _anc_concepts ac
                            ON CAST(t.{concept_col} AS BIGINT) = ac.descendant_concept_id
                    WHERE t.{date_col} IS NOT NULL
                      AND DATEDIFF(
                            'day',
                            CAST(t.{date_col} AS DATE),
                            CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                          ) BETWEEN 1 AND {window_upper_expr}
                    GROUP BY ac.vocabulary_id, ac.concept_code
                    HAVING COUNT(DISTINCT a.impression_id) >= {min_studies_ancestor}
                    ORDER BY n_studies DESC
                    LIMIT {max_ancestor_features}
                """)
                n_kept = con.execute("SELECT COUNT(*) FROM _kept_anc").fetchone()[0]
                self._log(f"    {n_kept:,} ancestor codes pass coverage filter")
                if n_kept == 0:
                    continue

                # 3d. Pass 2 — fetch only kept codes (small result → Python)
                self._log("    pass 2: fetching filtered ancestor rows …")
                rollup_df = con.execute(f"""
                    SELECT
                        a.impression_id,
                        '{anc_prefix}:' || ac.vocabulary_id || '/' || ac.concept_code
                            || '_{window_days}d'                              AS col_name,
                        COUNT(DISTINCT CAST(t.{date_col} AS DATE))            AS day_count,
                        CAST(MAX(CAST(t.{date_col} AS DATE)) AS VARCHAR)      AS most_recent_date
                    FROM read_csv_auto('{csv_path}', ignore_errors=true) t
                    INNER JOIN anchors_tbl a
                            ON CAST(t.person_id AS BIGINT) = a.person_id
                    INNER JOIN _anc_concepts ac
                            ON CAST(t.{concept_col} AS BIGINT) = ac.descendant_concept_id
                    INNER JOIN _kept_anc k
                            ON ('{anc_prefix}:' || ac.vocabulary_id || '/' || ac.concept_code
                                || '_{window_days}d') = k.col_name
                    WHERE t.{date_col} IS NOT NULL
                      AND DATEDIFF(
                            'day',
                            CAST(t.{date_col} AS DATE),
                            CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                          ) BETWEEN 1 AND {window_upper_expr}
                    GROUP BY a.impression_id, ac.vocabulary_id, ac.concept_code
                """).df()

                self._log(
                    f"    {len(rollup_df):,} (study, ancestor) rows  "
                    f"({rollup_df['col_name'].nunique():,} distinct ancestor codes, "
                    f"{rollup_df['impression_id'].nunique():,} studies)"
                )
                parts.append(rollup_df)

        finally:
            try:
                con.close()
            except Exception:
                pass
            if tmp_db.exists():
                tmp_db.unlink(missing_ok=True)

        if not parts:
            return pd.DataFrame(
                columns=["impression_id", "col_name", "day_count", "most_recent_date"]
            )
        return pd.concat(parts, ignore_index=True)

    # -- public build -----------------------------------------------------

    def build(
        self,
        task:                   str,
        windows_days:           list = None,
        loinc_codes:            Optional[list] = None,
        anchor:                 Optional[str] = None,
        min_studies_per_lab:    int = 50,
        feature_types:          Optional[list] = None,
        count_window_days:      Optional[dict] = None,
        use_concept_ancestor:   bool = False,
        ancestor_levels:        Optional[int] = None,
        min_studies_ancestor:   int = 100,
        max_ancestor_features:  int = 2000,
        one_study_per_patient:  Optional[str] = None,
        admission_anchored:     bool = False,
        pre_admission_days:     Optional[int] = None,
    ) -> LabFeatureMatrix:
        """Run the full extraction and return a LabFeatureMatrix.

        Args:
            task                  : INSPECT task name (e.g. "12_month_PH")
            windows_days          : list of window edges in days before anchor
                                    (default: [2, 7, 30, 365])
            loinc_codes           : list of LOINC concept_codes to include
                                    (default: None = all LOINC-coded measurements)
            anchor                : "dx", "px", or None (auto from task name)
            min_studies_per_lab   : drop labs measured in fewer studies than this
            feature_types         : which feature groups to extract; any combination of
                                    "labs", "diagnoses", "drugs", "procedures",
                                    "observations", "visits"
                                    (default: ["labs"] for backward compatibility)
            count_window_days     : dict mapping feature type → lookback days, e.g.
                                    {"diagnoses": 365, "drugs": 60, "procedures": 365}.
                                    Missing keys default to 365 days.
            use_concept_ancestor  : if True, also roll up events to their OMOP ancestor
                                    concepts using concept_ancestor.csv. Columns are
                                    prefixed with ``{type}_anc:`` (e.g. ``diag_anc:``).
            ancestor_levels       : maximum number of levels to climb in the hierarchy
                                    (None = unlimited). Only used when
                                    use_concept_ancestor=True.
            min_studies_ancestor  : ancestor codes present in fewer studies than this
                                    are dropped (default 100; intentionally higher than
                                    min_studies_per_lab because ancestor codes are far
                                    more numerous and sparser).
            max_ancestor_features : hard cap on ancestor feature columns per event table,
                                    keeping the highest-coverage codes (default 2000).
            one_study_per_patient : None/"all" (default) keeps every labelled study.
                                    "first" or "last" keeps only the earliest/latest
                                    study per patient — reduces the cohort (and every
                                    downstream cost: extraction time, matrix rows,
                                    long-format/timeline export size) proportionally
                                    to how many patients have multiple studies. See
                                    `_read_cohort` for the tie-breaking rule.
            admission_anchored    : if True, restrict the cohort to studies whose
                                    anchor falls inside an inpatient admission
                                    (visit_concept_id IN ADMISSION_VISIT_CONCEPT_IDS,
                                    visit_start_date <= anchor <= visit_end_date or
                                    still open) and extract features from
                                    [admission_start_date, anchor] instead of a
                                    fixed lookback — windows_days/count_window_days
                                    still apply as an outer ceiling per study
                                    (LEAST(window, admission -> anchor gap)).
                                    Studies with no qualifying admission are
                                    dropped from the cohort entirely. See
                                    `_find_encompassing_admissions`.
            pre_admission_days    : if set (requires admission_anchored=True and
                                    "labs" in feature_types), also extract a
                                    baseline lab window of this many days
                                    *before* admission_date (columns tagged
                                    ``_preadm``, e.g. ``..._last_7d_preadm``),
                                    and add delta columns
                                    (``labs:LOINC/{code}_delta_{agg}`` for
                                    agg in last/min/max/mean) = the
                                    admission-to-anchor value (largest
                                    windows_days entry, already capped to the
                                    admission span) minus the pre-admission
                                    baseline. NaN if either side is missing.
        """
        if windows_days is None:
            windows_days = DEFAULT_WINDOWS
        if not windows_days:
            # Empty list (as opposed to None) means the caller deliberately
            # wants no fixed window ceiling — only meaningful in
            # admission-anchored mode, where the per-patient admission span
            # is itself the bound. CATCH_ALL_DAYS (100y) as the sole window
            # makes LEAST(w, admission->anchor gap) reduce to just the
            # admission->anchor gap, uncapped by anything else, and is the
            # codebase's existing "unbounded" idiom (humanize_column already
            # renders it as "_whole_history").
            if admission_anchored:
                windows_days = [CATCH_ALL_DAYS]
                self._log(
                    "  lab windows: none configured — admission-anchored "
                    "mode, using the full admission-to-anchor span only "
                    "(no fixed ceiling)")
            else:
                raise ValueError(
                    "windows_days is empty and admission_anchored=False — "
                    "there's no per-patient admission span to fall back on, "
                    "so the lab window would be unbounded for every study. "
                    "Pass explicit windows_days (e.g. [2, 7, 30, 365]) or "
                    "enable admission_anchored.")
        if feature_types is None:
            feature_types = ["labs"]
        if count_window_days is None:
            count_window_days = {}

        # Map human names → OMOP table names for count extraction
        _FT_TO_TABLE = {
            "diagnoses":    "condition_occurrence",
            "drugs":        "drug_exposure",
            "procedures":   "procedure_occurrence",
            "observations": "observation",
            "visits":       "visit_occurrence",
        }
        count_tables = [
            _FT_TO_TABLE[ft] for ft in feature_types if ft in _FT_TO_TABLE
        ]
        # Build per-table window dict (keyed by OMOP table name, fallback 365 d)
        count_window_per_table = {
            _FT_TO_TABLE[ft]: count_window_days.get(ft, 365)
            for ft in feature_types if ft in _FT_TO_TABLE
        }
        extract_labs = "labs" in feature_types

        anchor_kind = self.anchor_kind_for(task, anchor)
        self._log(
            f"[route_b] task='{task}'  anchor='{anchor_kind}'  "
            f"feature_types={feature_types}"
        )
        if extract_labs:
            self._log(f"  lab windows: {windows_days}d")
            if loinc_codes:
                self._log(f"  LOINC filter: {len(loinc_codes)} codes")
            else:
                self._log("  extracting all LOINC-coded measurements")
        if count_tables:
            self._log(f"  count tables: {count_tables}  windows: {count_window_per_table}")

        # 1. Read cohort
        cohort_rows = self._read_cohort(task, anchor_kind, one_study_per_patient)

        # 1b. Admission-anchored mode: restrict the cohort to studies with an
        #     encompassing inpatient admission, and derive a per-study
        #     [admission_start, anchor] window that steps 2-4 below cap their
        #     lookback windows against. Single vectorized DuckDB join across
        #     the whole cohort — no per-patient loop.
        admission_meta  = None
        admission_dates = None
        if admission_anchored:
            self._log("\n[admission-anchored] finding encompassing admissions …")
            admission_meta = self._find_encompassing_admissions(cohort_rows)
            admission_dates = dict(zip(admission_meta["impression_id"],
                                        admission_meta["admission_date"]))
            n_before = len(cohort_rows)
            cohort_rows = [r for r in cohort_rows if r[2] in admission_dates]
            self._log(
                f"  {n_before:,} -> {len(cohort_rows):,} studies "
                f"({n_before - len(cohort_rows):,} dropped — no encompassing "
                f"admission in visit_occurrence.csv)")
            if not cohort_rows:
                raise RuntimeError(
                    "No studies had an encompassing inpatient admission — check "
                    "that visit_occurrence.csv covers this cohort, or disable "
                    "admission_anchored.")

        imp_ids     = [r[2] for r in cohort_rows]
        pat_ids     = np.array([r[0] for r in cohort_rows], dtype=np.int64)
        anchors     = np.array([r[1] for r in cohort_rows], dtype=object)
        splits      = np.array([r[3] for r in cohort_rows], dtype=object)
        y           = np.array([r[4] for r in cohort_rows], dtype=np.float32)

        admission_date_arr       = None
        days_since_admission_arr = None
        if admission_anchored:
            _adm_aligned = admission_meta.set_index("impression_id").reindex(imp_ids)
            admission_date_arr       = _adm_aligned["admission_date"].to_numpy(dtype=object)
            days_since_admission_arr = _adm_aligned["days_since_admission"].to_numpy(dtype=np.float32)

        # 2. Lab features (measurement.csv via DuckDB)
        #    _run_duckdb_and_pivot() keeps all intermediates inside an on-disk
        #    DuckDB file so they can spill beyond RAM; only the final float32
        #    array is materialised in Python.
        X_lab, columns_lab = np.zeros((len(imp_ids), 0), dtype=np.float32), []
        if extract_labs:
            X_lab, columns_lab = self._run_duckdb_and_pivot(
                cohort_rows, windows_days, loinc_codes,
                min_studies_per_lab, imp_ids, admission_dates=admission_dates)

            # 2b. Pre-admission baseline + delta (admission-anchored only)
            if admission_anchored and pre_admission_days:
                self._log(
                    f"\n[pre-admission baseline] {pre_admission_days}d before "
                    f"admission_date …")
                X_pre, columns_pre = self._compute_pre_admission_baseline_and_delta(
                    cohort_rows, admission_dates, windows_days, loinc_codes,
                    min_studies_per_lab, imp_ids, pre_admission_days,
                    X_lab, columns_lab)
                if columns_pre:
                    X_lab       = np.hstack([X_lab, X_pre])
                    columns_lab = columns_lab + columns_pre
            elif pre_admission_days and not admission_anchored:
                self._log(
                    "  WARNING: pre_admission_days set but admission_anchored=False "
                    "— ignoring (no admission_date to anchor the baseline to)")

        # 3. Count features (condition_occurrence / drug_exposure / procedure_occurrence)
        X_count, columns_count = np.zeros((len(imp_ids), 0), dtype=np.float32), []
        count_dates = None
        if count_tables:
            self._log("\n[count features]")
            count_long = self._run_duckdb_counts(cohort_rows, count_tables,
                                                  count_window_per_table,
                                                  admission_dates=admission_dates)
            X_count, columns_count, count_dates = self._pivot_counts(
                count_long, imp_ids, min_studies_per_lab)
            self._log(f"  count X shape: {X_count.shape}")

        # 4. Concept-ancestor rollup (optional)
        X_anc, columns_anc = np.zeros((len(imp_ids), 0), dtype=np.float32), []
        count_dates_anc = None
        if use_concept_ancestor and count_tables:
            self._log("\n[concept_ancestor rollup]")
            if ancestor_levels is not None:
                self._log(f"  max levels: {ancestor_levels}")
            anc_long = self._run_duckdb_ancestor_rollup(
                cohort_rows, count_tables, count_window_per_table,
                ancestor_levels=ancestor_levels,
                min_studies_ancestor=min_studies_ancestor,
                max_ancestor_features=max_ancestor_features,
                admission_dates=admission_dates,
            )
            if not anc_long.empty:
                X_anc, columns_anc, count_dates_anc = self._pivot_counts(
                    anc_long, imp_ids, min_studies_per_lab
                )
                self._log(f"  ancestor X shape: {X_anc.shape}")

        # 5. Combine feature matrices
        all_X       = [m for m in [X_lab, X_count, X_anc] if m.shape[1] > 0]
        all_columns = columns_lab + columns_count + columns_anc
        if all_X:
            X       = np.hstack(all_X) if len(all_X) > 1 else all_X[0]
            columns = all_columns
        else:
            X       = np.zeros((len(imp_ids), 0), dtype=np.float32)
            columns = []
        self._log(f"\n  combined X shape: {X.shape}")

        # Merge count_dates from direct counts and ancestor rollup
        if count_dates is not None and count_dates_anc is not None:
            import pandas as _pd
            count_dates = _pd.concat(
                [count_dates.reset_index(drop=True),
                 count_dates_anc.reset_index(drop=True)],
                axis=1,
            )
        elif count_dates_anc is not None:
            count_dates = count_dates_anc

        # 6. Survival
        tte, event = self._read_survival(task, imp_ids)

        fm = LabFeatureMatrix(
            X                 = X,
            columns           = columns,
            y                 = y,
            tte               = tte,
            event             = event,
            patient_ids       = pat_ids,
            impression_ids    = np.array(imp_ids, dtype=object),
            split             = splits,
            anchor_times      = anchors,
            task              = task,
            anchor_kind       = anchor_kind,
            windows_days      = sorted(windows_days),
            count_dates       = count_dates,
            feature_types     = list(feature_types),
            count_window_days = dict(count_window_days),
            admission_anchored   = admission_anchored,
            admission_dates       = admission_date_arr,
            days_since_admission  = days_since_admission_arr,
        )

        # Append demographics if person.csv is available
        if self.person_path is not None and self.person_path.exists():
            self._log(f"\n[demographics] appending from {self.person_path.name} …")
            fm = append_demographics(fm, self.person_path, verbose=self.verbose)
        else:
            self._log("\n[demographics] person.csv not found — skipping demographic features")

        self._log("\n" + fm.describe())
        return fm
