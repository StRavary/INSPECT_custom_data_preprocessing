"""
temporal_features.py
--------------------
Importable API for building time-windowed count features from the INSPECT FEMR
database, with survival outcomes joined from the AIMI labels file.

    from Custom.temporal_features import (
        TemporalFeatureExtractor, FeatureGroup, VITALS_LABS, DIAG_PROC, DRUGS,
    )

    ex = TemporalFeatureExtractor()                     # paths auto-resolve
    ds = ex.build(task="12_month_PH",
                  groups=[VITALS_LABS(bins=[2, 30, 365]),
                          DIAG_PROC(bins=[365, 1825]),
                          DRUGS(bins=[30, 365])])

Anchoring: PE is the only diagnostic task (label read off this CTPA) and
anchors at StudyTime - 1 day. Everything else is prognostic -- including
12_month_PH, which is a diagnosis but is being forecast from prior data -- and
anchors at StudyTime. See the "Anchor convention" block below.

    ds.X          # scipy.sparse CSR, one row per labelled study
    ds.columns    # column names, aligned to X
    ds.y          # binary label for `task`
    ds.tte        # time-to-event in DAYS (from the labels TSV)
    ds.event      # 1 = event observed, 0 = censored
    ds.split      # 'train' / 'valid' / 'test'
    ds.to_frame() # pandas DataFrame of the metadata columns


==============================================================================
WHAT THE COLUMNS MEAN
==============================================================================
Each column is one (code, time-window) pair, produced by FEMR's
`CountFeaturizer`. The value is the number of qualifying events for that code
inside that window, counted **backwards from the anchor**.

  name pattern                     meaning
  -------------------------------  -------------------------------------------
  age                              patient age at anchor, z-scored across all
                                   labels (NOT in years -- AgeFeaturizer
                                   normalises by default)
  <group>:<CODE>_<window>          count of CODE within that window
  <group>:<CODE> <value>_<window>  string-valued events, one column per
                                   (code, value) seen more than once
  <group>:<CODE> [lo, hi)_<window> numeric-valued events bucketed by decile

`<window>` is a `datetime.timedelta` such as `30 days, 0:00:00`, meaning the
window running from the anchor back to 30 days before it. Windows are
non-overlapping and consecutive: bins [2, 30] give [T0, T0-2d] and
[T0-2d, T0-30d].

==============================================================================
HOW LOINC / ICD10 CODES ARE USED  (ontology expansion)
==============================================================================
`is_ontology_expansion=True` is set for every group. FEMR then does:

    for subcode in ontology.get_all_parents(code):
        yield subcode

so a single event increments **its own code and every ancestor** in the OMOP
concept hierarchy. FEMR's own docstring: "if we see 2 occurrences of Code C, we
count 2 occurrences of Code B and Code A" for A -> B -> C.

Three consequences worth stating in any write-up:

  1. Both leaf and parent codes appear as columns. It is not one or the other.
  2. A parent column is the SUM over its descendants, so columns are strongly
     collinear. Do not read individual coefficients as independent effects;
     prefer permutation or grouped importance.
  3. The feature space is much wider than the number of distinct raw codes.

The materialised hierarchy lives in `<extract>/ontology/all_parents`.

==============================================================================
DATA PROVENANCE CAVEATS
==============================================================================
* The FEMR extract is de-duplicated relative to the raw OMOP CSVs. Building it
  dropped 72,508,452 events via `delta_encode` (sequential duplicates of the
  same (concept_id, date) with the same value) and 130,015 via `remove_nones`.
  Counts therefore approximate "days on which this code appeared with a changed
  value", not raw record counts. The unreduced source is DATA_RAW/EHR_CSV/.
* Survival outcomes are read from labels_20250611.tsv, not re-derived. It
  provides tte_*/is_censored_* for: mortality, readmission, PH, Atelectasis,
  Cardiomegaly, Consolidation, Edema, Pleural_Effusion.

==============================================================================
TWO UPSTREAM FEMR BEHAVIOURS THIS MODULE WORKS AROUND
==============================================================================
1. `CountFeaturizer` emits only `len(time_bins)` bins but internally tracks
   `len(time_bins) + 1`. Events older than the last bin edge are counted into a
   bucket that is never emitted, i.e. silently discarded. This module appends a
   catch-all bin (default 100 years) unless `catch_all=False`.
2. `featurize()` sorts `time_bins` internally, but `get_column_name()` indexes
   the ORIGINAL list. Unsorted bins therefore produce mislabelled columns. This
   module always sorts before constructing the featurizer.
"""

from __future__ import annotations

import csv
import datetime
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_ROOT = REPO_ROOT.parent

DEFAULT_COHORT = DATA_ROOT / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv"
DEFAULT_DATABASE = DATA_ROOT / "DATA_RAW" / "EHR_FEMR_DB" / "extract"
DEFAULT_LABELS = DATA_ROOT / "DATA_RAW" / "LABELS" / "labels_20250611.tsv"

# The cohort file has shipped with both spellings depending on which version of
# 2_merge_labels.py produced it; resolve at read time rather than assuming.
PATIENT_ID_CANDIDATES = ("patient_id", "PatientID", "person_id")
TIME_COLUMN = "StudyTime"
IMPRESSION_COLUMN = "impression_id"


def resolve_patient_column(fieldnames) -> str:
    for c in PATIENT_ID_CANDIDATES:
        if c in fieldnames:
            return c
    raise KeyError(
        f"No patient id column found. Looked for {PATIENT_ID_CANDIDATES}; "
        f"file has {list(fieldnames)[:10]}...")

CATCH_ALL_DAYS = 36500  # 100 years

# --------------------------------------------------------------------------
# Anchor convention
# --------------------------------------------------------------------------
# 'dx' (diagnostic)  -> anchor at StudyTime - 1 day
# 'px' (prognostic)  -> anchor at StudyTime
#
# The distinction is whether the label describes the CTPA itself or a future
# event, NOT whether the outcome happens to be a diagnosis.
#
#   PE is the only DIAGNOSTIC task: the label *is* the finding read off this
#   CTPA, so anything recorded on the study date can encode the answer. Backing
#   the anchor off by one day removes that leakage path.
#
#   Everything else is PROGNOSTIC -- we predict whether an event occurs in a
#   future window. 12_month_PH is a diagnosis, but we are forecasting its
#   occurrence from prior data, so there is no same-day contamination and the
#   anchor stays at StudyTime. Same for mortality, readmission and the
#   radiographic findings.
#
# This matches ehr/2_generate_labels_and_features.py, which applies the one-day
# offset only in its PE branch.
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
    "1_month_mortality": ("tte_mortality", "is_censored_mortality"),
    "6_month_mortality": ("tte_mortality", "is_censored_mortality"),
    "12_month_mortality": ("tte_mortality", "is_censored_mortality"),
    "1_month_readmission": ("tte_readmission", "is_censored_readmission"),
    "6_month_readmission": ("tte_readmission", "is_censored_readmission"),
    "12_month_readmission": ("tte_readmission", "is_censored_readmission"),
    "12_month_PH": ("tte_PH", "is_censored_PH"),
    "Atelectasis": ("tte_Atelectasis", "is_censored_Atelectasis"),
    "Cardiomegaly": ("tte_Cardiomegaly", "is_censored_Cardiomegaly"),
    "Consolidation": ("tte_Consolidation", "is_censored_Consolidation"),
    "Edema": ("tte_Edema", "is_censored_Edema"),
    "Pleural_Effusion": ("tte_Pleural_Effusion", "is_censored_Pleural_Effusion"),
}

# tte in the labels file is expressed in minutes
TTE_MINUTES_PER_DAY = 1440.0

TRUTHY = {"TRUE", "1", "1.0", "YES", "T"}
SKIP_LABEL_VALUES = {"CENSORED", "CENSOR", "NAN", "NA", "NONE", ""}


# ---------------------------------------------------------------------------
# OMOP concept-name helpers
# ---------------------------------------------------------------------------

_WINDOW_RE = re.compile(r'(_-?\d+ days(?:, \d+:\d+:\d+)?)\s*$')


def load_concept_map(concept_csv) -> dict:
    """Build {VOCAB/concept_code: concept_name} from OMOP concept.csv.

    Uses pandas for speed on the large (~5-8 M row) file; falls back to
    csv.DictReader if pandas is absent.
    """
    p = Path(concept_csv)
    try:
        import pandas as pd
        df = pd.read_csv(p, usecols=["vocabulary_id", "concept_code", "concept_name"],
                         dtype=str, low_memory=False)
        df["key"] = df["vocabulary_id"] + "/" + df["concept_code"]
        return dict(zip(df["key"], df["concept_name"]))
    except ImportError:
        lookup: dict[str, str] = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                lookup[f"{row['vocabulary_id']}/{row['concept_code']}"] = row["concept_name"]
        return lookup


def _readable_window(window_str: str) -> str:
    """'_18000 days, 0:00:00' -> '_whole_history',  '_30 days, 0:00:00' -> '_30d'."""
    m = re.match(r"_(\d+) days", window_str)
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

    Example:
        'diag:CPT4/1003343_18000 days, 0:00:00'
        -> 'diag:Acute pulmonary embolism [CPT4/1003343]_whole_history'
    Falls back gracefully for any code not in the map.
    """
    if ":" not in col:
        return col  # "age"

    group, rest = col.split(":", 1)

    m = _WINDOW_RE.search(rest)
    if m:
        code_part   = rest[: m.start()]
        window_part = _readable_window(m.group(1))
    else:
        code_part   = rest
        window_part = ""

    # code_part may carry a value or bucket suffix:
    #   "CPT4/1003343"  |  "LOINC/718-7 some_val"  |  "SNOMED/73211009 [0, 5)"
    base_code, *tail = code_part.split(" ", 1)
    suffix = f" {tail[0]}" if tail else ""

    name = concept_map.get(base_code)
    if name:
        return f"{group}:{name}{suffix} [{base_code}]{window_part}"
    # code not in map — still shorten the window label
    return f"{group}:{code_part}{window_part}"


# --------------------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------------------

@dataclass
class FeatureGroup:
    """One CountFeaturizer: a set of OMOP tables with its own backward windows.

    bins: window edges in DAYS, measured back from the anchor. [2, 30] gives
          [T0, T0-2d] and [T0-2d, T0-30d]. Order is irrelevant (sorted here).
    catch_all: append a 100-year final bin so nothing older is silently lost.
    """
    name: str
    tables: frozenset
    bins: Sequence[int]
    catch_all: bool = True

    def timedeltas(self) -> list[datetime.timedelta]:
        days = sorted({int(b) for b in self.bins})
        if self.catch_all and (not days or days[-1] < CATCH_ALL_DAYS):
            days.append(CATCH_ALL_DAYS)
        return [datetime.timedelta(days=d) for d in days]


def _group(name: str, tables: Iterable[str]) -> Callable[..., FeatureGroup]:
    def make(bins: Sequence[int], catch_all: bool = True) -> FeatureGroup:
        return FeatureGroup(name, frozenset(tables), bins, catch_all)
    return make


VITALS_LABS = _group("vitals_labs", {"measurement", "observation"})
DIAG_PROC = _group("diag_proc", {"condition_occurrence", "procedure_occurrence", "device_exposure"})
DRUGS = _group("drugs", {"drug_exposure"})
VISITS = _group("visits", {"visit_occurrence", "visit_detail"})


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------

@dataclass
class FeatureMatrix:
    X: object                       # scipy.sparse CSR, (n_rows, n_cols)
    columns: list[str]
    y: np.ndarray                   # binary task label
    tte: Optional[np.ndarray]       # days to event/censoring, None if unavailable
    event: Optional[np.ndarray]     # 1 = observed, 0 = censored
    patient_ids: np.ndarray
    impression_ids: np.ndarray
    split: np.ndarray
    anchor_times: np.ndarray
    task: str
    anchor_kind: str
    groups: list[FeatureGroup] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    def mask(self, split: str) -> np.ndarray:
        return self.split == split

    def to_frame(self):
        import pandas as pd
        d = {"impression_id": self.impression_ids, "patient_id": self.patient_ids,
             "split": self.split, "anchor_time": self.anchor_times, "y": self.y}
        if self.tte is not None:
            d["tte_days"] = self.tte
            d["event"] = self.event
        return pd.DataFrame(d)

    def human_columns(self, concept_map: dict) -> list[str]:
        """Column names with OMOP concept names resolved.

        Pass the dict returned by load_concept_map(). Falls back to the
        original name (with window label shortened) for any code not found.
        """
        return [humanize_column(c, concept_map) for c in self.columns]

    def describe(self) -> str:
        lines = [
            f"task={self.task}  anchor={self.anchor_kind}  rows={len(self):,}  cols={len(self.columns):,}",
            f"  label prevalence : {self.y.mean():.4f}",
        ]
        for s in ("train", "valid", "test"):
            n = int((self.split == s).sum())
            if n:
                lines.append(f"  {s:<6} {n:>7,}  prevalence {self.y[self.split == s].mean():.4f}")
        if self.tte is not None:
            lines.append(f"  events observed  : {int(self.event.sum()):,} "
                         f"({self.event.mean():.1%});  median tte "
                         f"{np.median(self.tte):.1f} d")
        for g in self.groups:
            lines.append(f"  group '{g.name}': {[t.days for t in g.timedeltas()]} day windows")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------

class TemporalFeatureExtractor:
    def __init__(self, cohort=None, database=None, labels=None, num_threads: int = 8,
                 verbose: bool = True):
        self.cohort_path = Path(cohort or DEFAULT_COHORT).expanduser()
        self.database_path = Path(database or DEFAULT_DATABASE).expanduser()
        self.labels_path = Path(labels or DEFAULT_LABELS).expanduser()
        self.num_threads = num_threads
        self.verbose = verbose
        for p, what in ((self.cohort_path, "cohort"), (self.database_path, "database")):
            if not p.exists():
                raise FileNotFoundError(f"{what} not found: {p}")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # -- anchors ----------------------------------------------------------
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
            f"Unknown task '{task}'. Pass anchor='dx' or anchor='px' explicitly, "
            f"or add it to DX_TASKS/PX_TASKS.")

    def _read_cohort(self, task: str, anchor_kind: str):
        """-> list of (patient_id, anchor_time, impression_id, split, value)"""
        offset = datetime.timedelta(days=1) if anchor_kind == "dx" else datetime.timedelta(0)
        rows, skipped = [], 0
        with open(self.cohort_path) as f:
            reader = csv.DictReader(
                f, delimiter="\t" if self.cohort_path.suffix == ".tsv" else ",")
            if task not in reader.fieldnames:
                raise KeyError(f"Task column '{task}' not in {self.cohort_path.name}. "
                               f"Available: {[c for c in reader.fieldnames][:20]}...")
            pid_col = resolve_patient_column(reader.fieldnames)
            for row in reader:
                raw = str(row.get(task, "")).strip().upper()
                if raw in SKIP_LABEL_VALUES:
                    skipped += 1
                    continue
                t = row[TIME_COLUMN]
                t = datetime.datetime.fromisoformat(t) if isinstance(t, str) else t
                rows.append((int(row[pid_col]), t - offset,
                             str(row[IMPRESSION_COLUMN]), row.get("split", ""),
                             1.0 if raw in TRUTHY else 0.0))
        self._log(f"cohort: {len(rows):,} labelled rows ({skipped:,} censored/missing skipped)")
        return rows

    # -- survival ---------------------------------------------------------
    def _read_survival(self, task: str):
        """-> {impression_id: (tte_days, event_indicator)} or None

        The cohort master file already carries the tte_*/is_censored_* columns,
        so it is tried first -- no join, no key-mismatch risk. The labels TSV is
        the fallback for cohorts built without them.
        """
        if task not in SURVIVAL_COLUMNS:
            self._log(f"no survival columns registered for '{task}' — tte will be None")
            return None
        tte_col, cen_col = SURVIVAL_COLUMNS[task]

        for path, delim, source in (
            (self.cohort_path, "\t" if self.cohort_path.suffix == ".tsv" else ",", "cohort"),
            (self.labels_path, "\t", "labels TSV"),
        ):
            if not path or not Path(path).is_file():
                continue
            with open(path) as f:
                reader = csv.DictReader(f, delimiter=delim)
                if not reader.fieldnames or not {tte_col, cen_col} <= set(reader.fieldnames):
                    continue
                out = {}
                for row in reader:
                    raw = str(row[tte_col]).strip()
                    if raw in ("", "NA", "nan", "NaN"):
                        continue
                    censored = str(row[cen_col]).strip().upper() in TRUTHY
                    out[str(row[IMPRESSION_COLUMN])] = (
                        float(raw) / TTE_MINUTES_PER_DAY, 0.0 if censored else 1.0)
            self._log(f"survival: {len(out):,} rows from {tte_col}/{cen_col} ({source})")
            return out

        self._log(f"no {tte_col}/{cen_col} in cohort or labels file — tte will be None")
        return None

    # -- main -------------------------------------------------------------
    def build(self, task: str, groups: Sequence[FeatureGroup],
              anchor: Optional[str] = None, include_age: bool = True) -> FeatureMatrix:
        from femr.labelers import Label, LabeledPatients
        from femr.featurizers.core import FeaturizerList
        from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer

        if not groups:
            raise ValueError("Pass at least one FeatureGroup.")

        anchor_kind = self.anchor_kind_for(task, anchor)
        self._log(f"task '{task}'  anchor '{anchor_kind}' "
                  f"({'StudyTime - 1 day' if anchor_kind == 'dx' else 'StudyTime'})")

        rows = self._read_cohort(task, anchor_kind)
        by_pid = {}
        meta = {}
        for pid, t, imp, split, val in rows:
            by_pid.setdefault(pid, []).append(Label(time=t, value=bool(val)))
            meta[(pid, t)] = (imp, split, val)
        for v in by_pid.values():
            v.sort(key=lambda a: a.time)
        labeled = LabeledPatients(by_pid, "boolean")

        featurizers = [AgeFeaturizer()] if include_age else []
        for g in groups:
            tds = g.timedeltas()
            self._log(f"  group '{g.name}': tables={sorted(g.tables)} "
                      f"windows={[t.days for t in tds]}d")
            featurizers.append(
                CountFeaturizer(
                    is_ontology_expansion=True,
                    time_bins=tds,                      # already sorted ascending
                    excluded_event_filter=_TableFilter(g.tables),
                )
            )
        flist = FeaturizerList(featurizers)

        self._log("preprocessing featurizers ...")
        flist.preprocess_featurizers(str(self.database_path), labeled, self.num_threads)
        self._log("featurizing ...")
        X, y_raw, pids, times = flist.featurize(str(self.database_path), labeled, self.num_threads)

        pids = np.asarray(pids).reshape(-1)
        times = np.asarray(times).reshape(-1)
        # FEMR may return Label objects rather than scalar values; extract .value explicitly.
        try:
            y = np.asarray(
                [float(l.value) if hasattr(l, "value") else float(l) for l in y_raw],
                dtype=np.float32,
            )
        except (AttributeError, TypeError):
            y = np.asarray(y_raw, dtype=np.float32).reshape(-1)

        imp, spl = [], []
        for p, t in zip(pids, times):
            tt = t.item() if hasattr(t, "item") else t
            i_s = meta.get((int(p), tt), ("", ""))[:2]
            imp.append(i_s[0]); spl.append(i_s[1])
        imp = np.array(imp, dtype=object)
        spl = np.array(spl, dtype=object)

        surv = self._read_survival(task)
        tte = ev = None
        if surv:
            tte = np.array([surv.get(i, (np.nan, np.nan))[0] for i in imp], dtype=np.float32)
            ev = np.array([surv.get(i, (np.nan, np.nan))[1] for i in imp], dtype=np.float32)
            missing = int(np.isnan(tte).sum())
            if missing:
                self._log(f"  WARNING: {missing:,} rows have no survival entry")

        columns = _column_names(featurizers, groups, include_age)
        if len(columns) != X.shape[1]:
            self._log(f"  WARNING: {len(columns)} names for {X.shape[1]} columns — "
                      "names may be misaligned")

        fm = FeatureMatrix(X=X, columns=columns, y=y, tte=tte, event=ev,
                           patient_ids=pids, impression_ids=imp, split=spl,
                           anchor_times=times, task=task, anchor_kind=anchor_kind,
                           groups=list(groups))
        self._log("\n" + fm.describe())
        return fm


class _TableFilter:
    """`excluded_event_filter`: return True for events to EXCLUDE.

    Implemented as a module-level class rather than a closure or lambda because
    FEMR's `preprocess_featurizers` / `featurize` fan out over a multiprocessing
    Pool, which pickles the featurizers. Closures and lambdas raise
    `AttributeError: Can't pickle local object`. An instance of a top-level
    class with picklable state pickles fine.

    (The same defect exists in Custom/x_generate_binned_features.py, which
    passes lambdas — it only survives with num_threads=1.)
    """

    __slots__ = ("tables",)

    def __init__(self, tables: Iterable[str]):
        self.tables = frozenset(tables)

    def __call__(self, ev) -> bool:
        return getattr(ev, "omop_table", None) not in self.tables

    def __repr__(self) -> str:
        return f"_TableFilter({sorted(self.tables)})"

    # explicit, so __slots__ doesn't trip up the default pickling path
    def __getstate__(self):
        return {"tables": self.tables}

    def __setstate__(self, state):
        self.tables = state["tables"]


def _column_names(featurizers, groups, include_age: bool) -> list[str]:
    names: list[str] = []
    gi = 0
    for f in featurizers:
        if type(f).__name__ == "AgeFeaturizer":
            names.append("age")
            continue
        prefix = groups[gi].name if gi < len(groups) else f"group{gi}"
        gi += 1
        try:
            n = f.get_num_columns()
            names.extend(f"{prefix}:{f.get_column_name(i)}" for i in range(n))
        except Exception as e:  # pragma: no cover
            names.extend(f"{prefix}:col{i}" for i in range(f.get_num_columns()))
            print(f"  NOTE: could not name columns for '{prefix}' ({e})")
    return names
