"""
appd_route_b_concepts.py
-------------------------
OMOP concept-name helpers for the Route B lab-extraction pipeline.
Split from Custom/appd_route_b_labs.py.

Contains:
  - _WINDOW_RE, _WINDOW_B_RE, _LAB_AGGS  (regex/frozenset constants)
  - load_concept_map(concept_csv) -> dict
  - load_concept_id_map(concept_csv) -> dict
  - load_concept_vocab_code_map(concept_csv) -> dict
  - _readable_window(window_str) -> str
  - humanize_column(col, concept_map) -> str
"""

import csv
from pathlib import Path
import re as _re

from Custom.appd_route_b_constants import CATCH_ALL_DAYS

# ---------------------------------------------------------------------------
# OMOP concept-name helpers (previously in temporal_features.py)
# ---------------------------------------------------------------------------

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
