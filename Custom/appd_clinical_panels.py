"""
appd_clinical_panels.py
------------------------
Clinical sub-panel groupings for the EHR timeline viewer (Tab 5) — maps
individual OMOP-vocabulary codes to a clinically meaningful "panel" (lane)
name, e.g. LOINC 2160-0 (creatinine) -> "Renal function".

This is the SAME curated taxonomy documented in
appd_EHR_FEATURE_EXTRACTION_GUIDE.md §4.6 (labs/diagnoses/procedures/drugs/visits)
— kept here as data, not re-derived, so the timeline viewer's lanes and the
guide's reference tables describe the same groupings. If you add a code to
one, add it to the other.

Coverage is intentionally partial: these are the codes already identified as
clinically relevant to PE/PH in this project, not an exhaustive ontology.
Nothing is dropped for lacking a panel — anything not listed here falls into
an "Other <event type>" lane instead (see assign_panel()), so the timeline
stays complete even where the taxonomy isn't.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Labs & vitals (LOINC) — measurement.csv — mirrors guide §4.6.1
# ---------------------------------------------------------------------------
LAB_PANELS: dict[str, list[str]] = {
    "Vital signs": [
        "8867-4", "8480-6", "8462-4", "59408-5", "9279-1",
        "8310-5", "29463-7", "8302-2", "39156-5",
    ],
    "Renal function": ["2160-0", "3094-0", "33914-3"],
    "Complete blood count": [
        "718-7", "4544-3", "6690-2", "26515-7", "769-0",
        "731-0", "787-2", "785-6", "786-4",
    ],
    "Coagulation": [
        "5902-2", "6301-6", "14804-9", "3173-2", "3255-7",
        "48065-7", "25315-4", "48066-5",
    ],
    "Cardiac biomarkers": [
        "10839-9", "6598-7", "67151-1", "89579-7",
        "33762-6", "42637-9", "2532-0",
    ],
    "Metabolic panel": [
        "2951-2", "2823-3", "2075-0", "2028-9", "17861-6", "2339-0",
        "1742-6", "1920-8", "6768-6", "1975-2", "1751-7",
    ],
    "Inflammatory markers": ["1988-5", "30341-2", "33959-8"],
    "Iron studies": ["2276-4", "2498-4", "2502-3"],
    "Diabetes / endocrine": ["4548-4", "1556-0", "3016-3", "3053-6"],
    "Lipids": ["2093-3", "2089-1", "2085-9", "2571-8"],
    "Blood gas / pulmonary": ["11558-4", "2703-7", "2019-8", "19994-3", "2708-6"],
}

# ---------------------------------------------------------------------------
# Diagnoses (ICD-10-CM) — condition_occurrence.csv — mirrors guide §4.6.2
# ---------------------------------------------------------------------------
DIAG_PANELS: dict[str, list[str]] = {
    "Pulmonary embolism": [
        "I26.01", "I26.02", "I26.09", "I26.90", "I26.92",
        "I26.93", "I26.94", "Z86.711", "Z86.718",
    ],
    "Pulmonary hypertension": [
        "I27.0", "I27.20", "I27.21", "I27.22", "I27.23", "I27.24", "I27.29",
    ],
    "Deep vein thrombosis": [
        "I82.401", "I82.411", "I82.431", "I82.4Y1", "I82.A11", "I82.B11",
    ],
    "Cardiac conditions": [
        "I50.20", "I50.22", "I50.30", "I50.9", "I21.3", "I21.4",
        "I48.0", "I48.11", "I48.19", "I48.20", "I10", "I46.9",
    ],
    "Lung and respiratory": [
        "J18.9", "J44.0", "J44.1", "J44.9", "J45.20",
        "J81.0", "J90", "J84.10", "J93.11",
    ],
    "Metabolic / chronic (PE risk factors)": [
        "E11.9", "E66.9", "N18.3", "N18.4", "N18.5", "N18.6", "C34.10", "C80.1",
    ],
}

# ---------------------------------------------------------------------------
# Procedures — procedure_occurrence.csv — mirrors guide §4.6.3.
# Split by vocabulary (CPT4 vs ICD-10-PCS use disjoint code spaces).
# ---------------------------------------------------------------------------
CPT4_PANELS: dict[str, list[str]] = {
    "Imaging & diagnostic procedures": [
        "71250", "71260", "71270", "71275", "93306", "75741", "78582", "93000",
    ],
}
ICD10PCS_PANELS: dict[str, list[str]] = {
    "PE treatment procedures": ["05H633Z", "06BQ0ZZ"],
}

# ---------------------------------------------------------------------------
# Drugs (RxNorm ingredient) — drug_exposure.csv — mirrors guide §4.6.4
# ---------------------------------------------------------------------------
DRUG_PANELS: dict[str, list[str]] = {
    "Anticoagulants": [
        "11289", "5224", "67108", "321208", "1364435",
        "1114195", "1037042", "1599538", "1736754",
    ],
    "Pulmonary vasodilators": [
        "36117", "187899", "1546954", "41140", "1745276",
        "240131", "3489", "77492", "2200691",
    ],
    "Diuretics": ["4603", "1808", "38413", "3289"],
    "Cardiovascular": [
        "41493", "20352", "83367", "301542", "36567",
        "29046", "52175", "1191", "32592",
    ],
    "Thrombolytics": ["688965", "1347111"],
}

# ---------------------------------------------------------------------------
# Visits — visit_occurrence.csv — mirrors guide §4.6.5. Keyed like the
# others (vocabulary_id="Visit", concept_code=<concept_id as string>),
# matching the visit:Visit/<concept_id>_<N>d column format.
# ---------------------------------------------------------------------------
VISIT_PANELS: dict[str, list[str]] = {
    "Inpatient": ["9201", "262"],
    "Outpatient / ED": ["9202", "9203", "8883"],
    "ICU": ["32037"],
    "Home visit": ["32044"],
}

# Observations have no curated PE/PH-specific panel in the guide yet — every
# observation event falls to "Other observation" until specific codes are
# identified. Left empty deliberately rather than guessed.
OBS_PANELS: dict[str, list[str]] = {}

# appd_route_b_labs.query_events_stack's `source_table` labels (table-name
# derived: "measurement", "condition", ...) don't fully agree with the
# event_type vocabulary used by build_event_timeline_streamed /
# build_timeline_skeleton_streamed ("lab", "diagnosis", ...) — "drug" /
# "procedure" / "observation" / "visit" already match, but "measurement"
# and "condition" don't. Normalize with this map before calling
# assign_panel() on live-query results, or the "Other <type>" fallback name
# (and any code keying off event_type) will disagree depending on which
# extraction path produced the row.
SOURCE_TABLE_TO_EVENT_TYPE: dict[str, str] = {
    "measurement": "lab",
    "condition":   "diagnosis",
}


def _invert(panels: dict[str, list[str]]) -> dict[str, str]:
    return {code: panel for panel, codes in panels.items() for code in codes}


# vocabulary_id -> {concept_code -> panel}, built once at import time.
PANELS_BY_VOCAB: dict[str, dict[str, str]] = {
    "LOINC":    _invert(LAB_PANELS),
    "ICD10CM":  _invert(DIAG_PANELS),
    "CPT4":     _invert(CPT4_PANELS),
    "ICD10PCS": _invert(ICD10PCS_PANELS),
    "RxNorm":   _invert(DRUG_PANELS),
    "Visit":    _invert(VISIT_PANELS),
}

ALL_PANEL_NAMES: list[str] = sorted({
    panel
    for code_to_panel in PANELS_BY_VOCAB.values()
    for panel in code_to_panel.values()
})


def normalize_event_type(source_table: str) -> str:
    """Map query_events_stack's `source_table` value to the event_type
    vocabulary used by the batch timeline exports. Unknown/already-correct
    values pass through unchanged."""
    return SOURCE_TABLE_TO_EVENT_TYPE.get(source_table, source_table)


def assign_panel(vocabulary: str, concept_code: str, event_type: str) -> str:
    """Map a (vocabulary, code) pair to its clinical panel/lane name.

    Falls back to "Other <event_type>" for anything not in the curated
    taxonomy above (e.g. all observations today), rather than dropping the
    event or grouping it into a meaningless catch-all shared across types.
    """
    panel = PANELS_BY_VOCAB.get(vocabulary or "", {}).get(concept_code or "")
    return panel if panel else f"Other {event_type}"


def panel_lookup_rows() -> list[tuple[str, str, str]]:
    """Flat (vocabulary_id, concept_code, panel) rows.

    Register this as a DuckDB table to assign panels via SQL JOIN (see
    appd_route_b_labs.build_timeline_skeleton_streamed) so panel assignment
    stays inside DuckDB's out-of-core engine instead of needing a Python
    pass over every row — the same reasoning behind every other
    memory-safety fix in that module this session.
    """
    return [
        (vocab, code, panel)
        for vocab, code_to_panel in PANELS_BY_VOCAB.items()
        for code, panel in code_to_panel.items()
    ]
