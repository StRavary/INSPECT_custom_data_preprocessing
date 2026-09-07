"""
appd_route_b_matrix.py
-----------------------
Demographic feature helpers and the LabFeatureMatrix result container for the
Route B lab-extraction pipeline.
Split from Custom/appd_route_b_labs.py.

Contains:
  - load_demographic_features(person_csv, patient_ids, anchor_times, verbose) -> tuple
  - append_demographics(fm, person_csv, verbose) -> fm
  - LabFeatureMatrix  (dataclass — the Route B result container)
"""

from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from Custom.appd_route_b_constants import (
    _GENDER_FEMALE,
    _GENDER_MALE,
    _ETHNICITY_HISPANIC,
    _ETHNICITY_NO_HISP,
    _RACE_WHITE,
    _RACE_BLACK,
    _RACE_ASIAN,
)
from Custom.appd_route_b_concepts import humanize_column

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
