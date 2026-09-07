"""
appd_route_b_constants.py
--------------------------
Module-level constants for the Route B lab-extraction pipeline.
Split from Custom/appd_route_b_labs.py.

Contains:
  - Filesystem path defaults (SCRIPT_DIR, DATA_ROOT, DEFAULT_*)
  - OMOP standard concept IDs for demographics
  - Canonical lookup dicts and scalar constants (GENDER, DEFAULT_WINDOWS, AGGS, …)
  - Task / anchor constants (DX_TASKS, PX_TASKS, SURVIVAL_COLUMNS, …)
  - _EVENT_TABLES dict (used by query_events_stack)
  - _COUNT_TABLE_CONFIG dict (used by count-feature extraction)
  - _TIMELINE_TABLE_CONFIG dict (used by all four build_* timeline functions)

No functions, no classes.
"""

from pathlib import Path

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

# visit_concept_ids treated as a qualifying "arrival" for admission-anchored
# extraction — i.e. the anchor falling inside one of these gives a study a
# per-patient start date to window from, instead of a fixed number of days
# for everyone. Deliberately broader than strict inpatient admission:
#   9201 = Inpatient Visit
#   262  = Emergency Room and Inpatient Visit (ER that rolled into admission)
#   9203 = Emergency Room Visit (ED workup that never became a formal
#          admission — still "arrived at a care unit", which is what this is
#          meant to capture; requirements clarified this should NOT be
#          restricted to admitted-only patients).
# Was named ADMISSION_VISIT_CONCEPT_IDS; renamed because "admission" implied
# a narrower, inpatient-only reading than what this is actually used for.
ARRIVAL_VISIT_CONCEPT_IDS = (9201, 262, 9203)

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
