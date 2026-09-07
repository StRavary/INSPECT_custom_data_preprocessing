"""
appd_export.py
--------------
Export and long-format helpers for the INSPECT EHR Feature Extraction app
(app_feature_extraction.py).

What lives here
---------------
  Concept-map caches          — _load_concept_map_cached / _id_map / _vocab_map
  Export directory naming     — _export_subdir, _feature_summary
  Long-format builders        — _build_long_df, _build_long_df_streamed
    _prep_long_df             — shared O(n_studies + n_features) setup
    _melt_batch               — melt one chunk; called by both builders
  Flat-array export           — _do_export (writes X.npy, metadata.csv, …)

Memory notes
------------
_build_long_df refuses extractions that would produce more than
_LONG_DF_MAX_CELLS rows in memory; use _build_long_df_streamed for those.
_build_long_df_streamed sizes chunks by cells (studies × features), not by
a fixed row count, so the chunk size shrinks automatically for wide
extractions instead of silently stopping being a bound.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — derived from this file's location, same directory as
# app_feature_extraction.py, so they resolve to the same values.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_ROOT  = _SCRIPT_DIR.parent.parent
_CACHE_DIR  = _DATA_ROOT / "DATA_PROCESSED" / "femr_cache"

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
# Concept-map caches
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


def _load_concept_vocab_map_cached(concept_csv: str, st):
    """Streamlit-cached wrapper around route_b_labs.load_concept_vocab_code_map —
    {concept_id: (vocabulary_id, concept_code)}, used for clinical-panel
    assignment in the Timeline viewer (Tab 5)."""
    from Custom.appd_route_b_labs import load_concept_vocab_code_map

    @st.cache_resource(show_spinner="Loading concept vocabulary map …")
    def _load(path: str):
        return load_concept_vocab_code_map(path)

    return _load(concept_csv)


# ---------------------------------------------------------------------------
# Long-format (melted) export helpers
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
        import pandas as pd
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
    import numpy as np

    prep  = _prep_long_df(fm, concept_map)
    n_imp, n_feat = prep["n_imp"], prep["n_feat"]

    if chunk_studies is None:
        chunk_studies = max(1, _STREAM_CHUNK_CELLS // max(n_feat, 1))

    out_path = Path(out_path)
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
# Flat-array export
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
    import numpy as np

    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Manifest: what exactly produced these files. Independent of however
    # the export directory ends up named/moved/renamed — always copy the
    # full extraction spec if we know which cache entry this fm came from;
    # fall back to whatever fm itself still carries (older pickles predating
    # fm_hash tracking won't have a matching spec.json, but still have this
    # much on the object).
    fm_hash  = _fm_hash
    spec_src = _CACHE_DIR / f"{fm_hash}_spec.json" if fm_hash else None
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

    import pandas as pd
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
{"Admission-anchored: studies with an admission_date (see metadata.csv "
 "columns admission_date / days_since_admission) have a window of "
 "[admission_date, anchor_time], capped by the configured windows as an "
 "outer ceiling. Studies with admission_date = NaN (no qualifying arrival "
 "record) used the plain fixed window instead, same as a normal extraction."
 if has_admission else ""}
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
