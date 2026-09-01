"""
context_descriptors.py
----------------------
Turn a cohort into a tidy **context descriptor table**: one row per slice of the
data, columns describing that slice's composition, data density, missingness and
label characteristics.

Motivation
----------
Patient-level features answer "what happens to this patient". Descriptors answer
"what is this slice of data *like*" -- which is the input you need when studying
why a model performs well in one context and badly in another. Each row here is
a candidate observation for that analysis; attach a performance metric per slice
and you have a (descriptors -> performance) dataset.

    from Custom.temporal_features import TemporalFeatureExtractor, DIAG_PROC
    from Custom.appd_context_descriptors import ContextDescriber

    fm = TemporalFeatureExtractor().build("12_month_PH", [DIAG_PROC(bins=[365, 1825])])

    cd = ContextDescriber(fm, demographics="auto")   # joins person.csv if found
    table = cd.describe(by=["split", "sex", "age_band"])
    table = cd.attach_performance(table, {"train": 0.91, "valid": 0.86, "test": 0.85},
                                  key="split", name="auroc")

`table` is a pandas DataFrame, one row per slice, ready for feature-importance
or inverse modelling over its columns.

What is and isn't computable
----------------------------
Descriptors are derived from the FeatureMatrix plus (optionally) person.csv.
Anything requiring the raw event stream -- true history length, sampling
intervals, per-measurement values -- is NOT available here; the count matrix has
already aggregated it away. Two proxies are provided and named as such:
`frac_with_oldest_window_events` (history depth) and `mean_distinct_codes`
(record richness). For genuine temporal descriptors, extract from
DATA_RAW/EHR_CSV/ instead -- see appd_EHR_FEATURE_EXTRACTION_GUIDE.md.
"""

from __future__ import annotations

import csv
import datetime
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

# Canonical OMOP gender-concept-id -> label map, shared with the rest of the
# pipeline (see Custom.appd_route_b_labs.GENDER) so "female"/"male" labels and
# their casing can't drift between this module and the Streamlit app.
from Custom.appd_route_b_labs import GENDER

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_PERSON = DATA_ROOT / "DATA_RAW" / "EHR_CSV" / "person.csv"

AGE_BANDS = [(0, 40), (40, 55), (55, 65), (65, 75), (75, 85), (85, 200)]

# column-name prefix -> vocabulary, e.g. "diag_proc:ICD10CM/I26.0_365 days..."
_VOCAB_RE = re.compile(r"^(?:[^:]+:)?([A-Za-z0-9_]+)/")


def _age_band(age: float) -> str:
    for lo, hi in AGE_BANDS:
        if lo <= age < hi:
            return f"{lo}-{hi}" if hi < 200 else f"{lo}+"
    return "unknown"


class ContextDescriber:
    """Slice a FeatureMatrix and describe each slice."""

    def __init__(self, fm, demographics: Union[str, Path, None] = None,
                 key_columns: Optional[Sequence[str]] = None, verbose: bool = True):
        """
        fm            : FeatureMatrix from temporal_features.build()
        demographics  : path to person.csv, or "auto" to look in the standard
                        location, or None to skip demographic descriptors
        key_columns   : column names whose presence/absence defines the
                        `missing_<name>` descriptors (e.g. key labs). If None,
                        missingness descriptors are omitted.
        """
        self.fm = fm
        self.verbose = verbose
        self.key_columns = list(key_columns or [])
        self._demo = self._load_demographics(demographics)
        self._X = self._as_dense_summaries(fm.X)
        self._vocab = self._column_vocabularies(fm.columns)

    # -- setup ------------------------------------------------------------
    def _log(self, m):
        if self.verbose:
            print(m)

    def _load_demographics(self, spec):
        if spec is None:
            return None
        path = DEFAULT_PERSON if spec == "auto" else Path(spec).expanduser()
        if not path.is_file():
            self._log(f"demographics: {path} not found — demographic descriptors skipped")
            return None
        wanted = set(int(p) for p in self.fm.patient_ids)
        out = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                try:
                    pid = int(row["person_id"])
                except (KeyError, ValueError):
                    continue
                if pid not in wanted:
                    continue
                out[pid] = {
                    "sex": GENDER.get(str(row.get("gender_concept_id", "")).strip(), "other/unknown"),
                    "race_concept_id": str(row.get("race_concept_id", "")).strip(),
                    "ethnicity_concept_id": str(row.get("ethnicity_concept_id", "")).strip(),
                    "care_site_id": str(row.get("care_site_id", "")).strip(),
                    "birth": row.get("birth_DATETIME") or row.get("year_of_birth"),
                }
        self._log(f"demographics: matched {len(out):,}/{len(wanted):,} patients from {path.name}")
        return out

    @staticmethod
    def _as_dense_summaries(X):
        """Per-row (nnz, total) without densifying a sparse matrix."""
        if hasattr(X, "getnnz"):                      # scipy sparse
            nnz = X.getnnz(axis=1).astype(float)
            total = np.asarray(X.sum(axis=1)).reshape(-1).astype(float)
        else:
            A = np.asarray(X)
            nnz = (A != 0).sum(axis=1).astype(float)
            total = A.sum(axis=1).astype(float)
        return {"nnz": nnz, "total": total}

    @staticmethod
    def _column_vocabularies(columns: Sequence[str]) -> np.ndarray:
        vocab = []
        for c in columns:
            m = _VOCAB_RE.match(c)
            vocab.append(m.group(1) if m else "other")
        return np.array(vocab, dtype=object)

    # -- slicing ----------------------------------------------------------
    def _builtin(self, name: str) -> np.ndarray:
        fm, n = self.fm, len(self.fm.y)
        if name == "all":
            return np.array(["all"] * n, dtype=object)
        if name == "split":
            return np.asarray(fm.split, dtype=object)
        if name == "density_quartile":
            q = np.quantile(self._X["nnz"], [0.25, 0.5, 0.75])
            return np.array([f"Q{int(np.searchsorted(q, v)) + 1}" for v in self._X["nnz"]],
                            dtype=object)
        if name in ("sex", "race_concept_id", "ethnicity_concept_id", "care_site_id"):
            if not self._demo:
                raise ValueError(f"'{name}' needs demographics; pass demographics='auto'")
            return np.array([self._demo.get(int(p), {}).get(name, "unknown")
                             for p in fm.patient_ids], dtype=object)
        if name == "age_band":
            return np.array([_age_band(a) for a in self.ages()], dtype=object)
        raise ValueError(
            f"Unknown slicer '{name}'. Built-ins: all, split, density_quartile, "
            "age_band, sex, race_concept_id, ethnicity_concept_id, care_site_id. "
            "Or pass a dict {name: array}.")

    def ages(self) -> np.ndarray:
        """True age in years at anchor (needs demographics); NaN otherwise."""
        if not self._demo:
            return np.full(len(self.fm.y), np.nan)
        out = []
        for p, t in zip(self.fm.patient_ids, self.fm.anchor_times):
            b = self._demo.get(int(p), {}).get("birth")
            try:
                bd = (datetime.datetime.fromisoformat(str(b)) if "-" in str(b)
                      else datetime.datetime(int(b), 7, 1))
                tt = t if isinstance(t, datetime.datetime) else datetime.datetime.fromisoformat(str(t))
                out.append((tt - bd).days / 365.25)
            except Exception:
                out.append(np.nan)
        return np.array(out, dtype=float)

    def _resolve(self, by) -> dict:
        if isinstance(by, str):
            by = [by]
        if isinstance(by, Mapping):
            return {k: np.asarray(v, dtype=object) for k, v in by.items()}
        resolved = {}
        for item in by:
            if isinstance(item, Mapping):
                resolved.update({k: np.asarray(v, dtype=object) for k, v in item.items()})
            else:
                resolved[item] = self._builtin(item)
        return resolved

    # -- descriptors ------------------------------------------------------
    def _describe_rows(self, idx: np.ndarray) -> dict:
        fm = self.fm
        y = fm.y[idx]
        nnz, total = self._X["nnz"][idx], self._X["total"][idx]
        pids = fm.patient_ids[idx]

        d = {
            "n_studies":                int(idx.size),
            "n_patients":               int(len(set(pids.tolist()))),
            "event_rate":               float(y.mean())         if idx.size else np.nan,
            "n_positive":               int(y.sum()),
            "mean_distinct_codes":      float(nnz.mean())       if idx.size else np.nan,
            "median_distinct_codes":    float(np.median(nnz))   if idx.size else np.nan,
            "mean_total_count":         float(total.mean())     if idx.size else np.nan,
        }

        if fm.tte is not None:
            tte, ev = fm.tte[idx], fm.event[idx]
            ok = ~np.isnan(tte)
            d["censoring_rate"] = float(1 - np.nanmean(ev)) if ok.any() else np.nan
            d["median_tte_days"] = float(np.nanmedian(tte)) if ok.any() else np.nan

        ages = self.ages()[idx] if self._demo else np.array([np.nan])
        if np.isfinite(ages).any():
            d["age_mean"] = float(np.nanmean(ages))
            d["age_iqr"] = float(np.nanpercentile(ages, 75) - np.nanpercentile(ages, 25))

        if self._demo:
            sexes = [self._demo.get(int(p), {}).get("sex", "unknown") for p in pids]
            d["frac_female"] = float(np.mean([s == "female" for s in sexes]))
            sites = [self._demo.get(int(p), {}).get("care_site_id", "") for p in pids]
            c = Counter(sites)
            d["n_care_sites"] = int(len([k for k in c if k]))
            d["top_site_share"] = float(c.most_common(1)[0][1] / len(sites)) if sites else np.nan

        # vocabulary mix: share of nonzero column mass per vocabulary
        d.update(self._vocab_shares(idx))

        # missingness for the nominated key columns
        for name in self.key_columns:
            try:
                j = self.fm.columns.index(name)
            except ValueError:
                continue
            col = self._column(j)[idx]
            d[f"missing_{name}"] = float((col == 0).mean())

        # history-depth proxy: any events in the widest (oldest) window
        oldest = self._oldest_window_columns()
        if oldest.size:
            sub = self._submatrix_nnz(idx, oldest)
            d["frac_with_oldest_window_events"] = float((sub > 0).mean())

        return d

    def _column(self, j: int) -> np.ndarray:
        X = self.fm.X
        if hasattr(X, "getcol"):
            return np.asarray(X.getcol(j).todense()).reshape(-1)
        return np.asarray(X)[:, j]

    def _submatrix_nnz(self, idx, cols) -> np.ndarray:
        X = self.fm.X
        if hasattr(X, "tocsr"):
            return X.tocsr()[idx][:, cols].getnnz(axis=1)
        return (np.asarray(X)[np.ix_(idx, cols)] != 0).sum(axis=1)

    def _oldest_window_columns(self) -> np.ndarray:
        """Columns belonging to the largest timedelta suffix (the oldest bin)."""
        days = []
        for c in self.fm.columns:
            m = re.search(r"_(\d+) days", c)
            days.append(int(m.group(1)) if m else -1)
        days = np.array(days)
        return np.where(days == days.max())[0] if (days > 0).any() else np.array([], dtype=int)

    def _vocab_shares(self, idx) -> dict:
        out = {}
        uniq = [v for v in sorted(set(self._vocab.tolist())) if v != "other"]
        if not uniq:
            return out
        totals = {}
        for v in uniq:
            cols = np.where(self._vocab == v)[0]
            totals[v] = float(self._submatrix_nnz(idx, cols).sum())
        grand = sum(totals.values()) or 1.0
        for v in uniq:
            out[f"vocab_share_{v}"] = totals[v] / grand
        return out

    # -- public -----------------------------------------------------------
    def describe(self, by="split", min_rows: int = 1):
        """Return a tidy DataFrame, one row per slice.

        by: a built-in name, a list of them, or {name: array-aligned-to-rows}.
            Multiple entries are described independently (not crossed); to cross,
            pass a precomputed combined array.
        """
        import pandas as pd

        rows = []
        for dim, values in self._resolve(by).items():
            for level in sorted(set(values.tolist())):
                idx = np.where(values == level)[0]
                if idx.size < min_rows:
                    continue
                rows.append({"slice_dim": dim, "slice": str(level),
                             **self._describe_rows(idx)})
        df = pd.DataFrame(rows)
        self._log(f"described {len(df)} slice(s) across "
                  f"{df['slice_dim'].nunique() if len(df) else 0} dimension(s)")
        return df

    def patient_table(self) -> "pd.DataFrame":
        """One row per patient, event status as columns.

        Columns:
            patient_id, n_studies, split,
            n_positive  (studies where y=1),
            ever_positive (1 if any study was positive),
            tte_days, event_observed  (from survival data, if available),
            sex, age_years, care_site_id  (if demographics loaded)
        """
        import pandas as pd
        from collections import defaultdict

        fm   = self.fm
        ages = self.ages()
        buckets: dict = defaultdict(list)
        for i, pid in enumerate(fm.patient_ids.tolist()):
            buckets[int(pid)].append(i)

        rows = []
        for pid, indices in sorted(buckets.items()):
            idx = np.array(indices)
            y   = fm.y[idx]

            splits = fm.split[idx].tolist()
            split  = splits[0] if len(set(splits)) == 1 else "mixed"

            row: dict = {
                "patient_id":    pid,
                "n_studies":     len(idx),
                "split":         split,
                "n_positive":    int(y.sum()),
                "ever_positive": int(y.any()),
            }

            if fm.tte is not None:
                tte = fm.tte[idx]
                ev  = fm.event[idx]
                ok  = ~np.isnan(tte)
                row["tte_days"]        = float(np.nanmin(tte)) if ok.any() else np.nan
                row["event_observed"]  = int(np.nanmax(ev[ok])) if ok.any() else np.nan

            if self._demo:
                demo = self._demo.get(pid, {})
                row["sex"]          = demo.get("sex", "unknown")
                row["age_years"]    = float(np.nanmean(ages[idx]))
                row["care_site_id"] = demo.get("care_site_id", "")

            rows.append(row)

        return pd.DataFrame(rows).reset_index(drop=True)

    @staticmethod
    def attach_performance(table, metrics: Mapping[str, float],
                           key: str = "split", name: str = "metric"):
        """Join a {slice_label: value} mapping onto the descriptor table."""
        table = table.copy()
        table[name] = [
            metrics.get(s) if d == key else None
            for d, s in zip(table["slice_dim"], table["slice"])
        ]
        return table
