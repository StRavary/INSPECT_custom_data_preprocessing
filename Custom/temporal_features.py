"""
temporal_features.py — COMPATIBILITY SHIM

Route A (FEMR) has been retired. All shared constants and utilities have moved
to Custom/route_b_labs.py. This shim re-exports them so any existing scripts
that import from here continue to work.
"""
from Custom.appd_route_b_labs import (  # noqa: F401  (re-export)
    DX_TASKS,
    PX_TASKS,
    SURVIVAL_COLUMNS,
    TTE_MINUTES_PER_DAY,
    TRUTHY,
    SKIP_LABEL_VALUES,
    CATCH_ALL_DAYS,
    load_concept_map,
    load_concept_id_map,
    humanize_column,
    PATIENT_ID_CANDIDATES,
    TIME_COLUMN,
    IMPRESSION_COLUMN,
    DEFAULT_COHORT,
    DEFAULT_LABELS,
)
