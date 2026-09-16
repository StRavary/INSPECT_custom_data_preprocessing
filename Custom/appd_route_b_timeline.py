"""
appd_route_b_timeline.py
-------------------------
Event-timeline and cohort-trajectory query functions for the Route B
lab-extraction pipeline.
Split from Custom/appd_route_b_labs.py.

Contains:
  - query_events_stack(omop_dir, selected_impressions, imp_to_info,
                       window_days, tables, concept_id_map) -> pd.DataFrame
  - build_event_timeline(fm, omop_dir, measurement_path, concept_path,
                         loinc_codes, verbose) -> pd.DataFrame
  - build_event_timeline_streamed(fm, omop_dir, measurement_path, concept_path,
                                  out_path, loinc_codes, memory_limit_gb,
                                  verbose) -> int
  - build_timeline_skeleton_streamed(fm, omop_dir, measurement_path,
                                     concept_path, out_path, loinc_codes,
                                     memory_limit_gb, verbose) -> int
  - build_cohort_trajectory(fm, omop_dir, measurement_path, concept_path,
                            loinc_codes, bin_days, lookback_days,
                            memory_limit_gb, verbose) -> pd.DataFrame
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from Custom.appd_route_b_constants import (
    _EVENT_TABLES,
    _COUNT_TABLE_CONFIG,
    _TIMELINE_TABLE_CONFIG,
)
from Custom.appd_route_b_matrix import LabFeatureMatrix  # for type hints only

# ---------------------------------------------------------------------------
# Event-stack query (Describe tab)
# ---------------------------------------------------------------------------


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
# Event timeline builder — raw individual events (not aggregated)
# ---------------------------------------------------------------------------


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


def build_cohort_trajectory(
    fm: "LabFeatureMatrix",
    omop_dir: "Path",
    measurement_path: "Path",
    concept_path: "Path",
    loinc_codes: Optional[list] = None,
    bin_days: int = 7,
    lookback_days: Optional[int] = 365,
    memory_limit_gb: float = 4.0,
    verbose: bool = True,
) -> "pd.DataFrame":
    """Population-level, binned event density relative to CTPA — one row per
    (clinical panel, time bin), for a whole cohort of potentially thousands
    of studies. The aggregate counterpart to build_timeline_skeleton_streamed:
    instead of one row per raw event (tens of millions of rows for a big
    cohort — too many to usefully plot, let alone hold in memory), events are
    binned by ``bin_days``-wide windows of "days before anchor" and counted
    *inside DuckDB*, so the only thing that ever reaches Python is a summary
    table of at most (number of panels) × (number of bins) rows — a few
    thousand at the absolute widest, regardless of cohort size. This is what
    makes "an overview of the cohort's temporal trajectory" (binned, not
    per-event) tractable at population scale, where the per-study timeline
    viewer (query_events_stack / build_event_timeline*) is not — plotting
    thousands of individual patient timelines on one axis is unreadable even
    before considering memory.

    Same on-disk-connection / capped-memory-limit safety measures as
    build_event_timeline_streamed and build_timeline_skeleton_streamed, for
    the same reasons — this still has to scan the same raw OMOP CSVs.

    Parameters
    ----------
    fm, omop_dir, measurement_path, concept_path, loinc_codes,
    memory_limit_gb, verbose : see build_event_timeline_streamed
    bin_days : width of each time bin, in days before anchor. 7 (weekly) is
        a reasonable default for a ~1 year lookback; widen it for a longer
        history or narrow it for a short, dense one — the row count this
        returns is (panels × ceil(scan_window / bin_days)), so this is the
        one knob that trades resolution for how much a human can actually
        look at in one heatmap.
    lookback_days : how far back from anchor this *overview* should scan,
        independent of fm.windows_days / fm.count_window_days. Deliberately
        decoupled from the extraction's own window settings: those are often
        set generously large (e.g. to guarantee admission-anchored data
        never gets capped — see the app's admission-anchoring help text) for
        reasons that have nothing to do with what's readable in a heatmap.
        Reusing them directly here would make the number of bins (and thus
        how crowded the x-axis is) an accidental side effect of an unrelated
        setting. Each feature type is scanned out to
        ``min(lookback_days, that type's own configured window)`` — never
        further back than the extraction actually reaches, but capped at
        this value even if the extraction's window is much larger. Pass
        None to fall back to each type's own window uncapped (the old
        behavior) — the wider the cohort's real history, the more likely
        that produces hundreds of mostly-empty bins.

    Returns
    -------
    pd.DataFrame with columns:
        panel, event_type, bin_index (0 = the bin closest to anchor),
        bin_start_days, bin_end_days (the bin's [start, end] in days before
        anchor — e.g. bin_start_days=1, bin_end_days=7 for the first weekly
        bin), n_events (total event count in that panel/bin across the whole
        cohort), n_studies (distinct studies with >=1 such event),
        pct_studies (n_studies / total studies in fm, as a percentage — the
        recommended default metric to plot, since raw counts conflate "lots
        of activity" with "lots of patients still have data this far back",
        while pct_studies normalizes for that).
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

    if bin_days < 1:
        raise ValueError(f"bin_days must be >= 1, got {bin_days}")

    feature_types     = list(getattr(fm, "feature_types", None) or [])
    count_window_days = dict(getattr(fm, "count_window_days", None) or {})
    windows_days       = list(fm.windows_days) if fm.windows_days else [365]
    n_total_studies    = len(fm.impression_ids)

    out_cols = ["panel", "event_type", "bin_index", "bin_start_days",
                "bin_end_days", "n_events", "n_studies", "pct_studies"]

    anchors_df = pd.DataFrame({
        "person_id":     fm.patient_ids.astype("int64"),
        "impression_id": fm.impression_ids,
        "anchor_time":   [str(a) for a in fm.anchor_times],
    })

    tmp_db = Path(tempfile.mktemp(suffix="_cohort_trajectory.duckdb"))
    con = duckdb.connect(str(tmp_db))
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.register("anchors_tbl", anchors_df)

    try:
        _log("[trajectory] loading concept.csv …")
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

        _log("[trajectory] loading clinical panel taxonomy …")
        panel_df = pd.DataFrame(
            panel_lookup_rows(), columns=["vocabulary_id", "concept_code", "panel"])
        con.register("_panels", panel_df)

        parts_sql: list = []
        params: list = []

        # ── Labs ────────────────────────────────────────────────────────────
        if "labs" in feature_types:
            max_window = max(windows_days)
            if lookback_days is not None:
                max_window = min(max_window, lookback_days)
            loinc_filter = ""
            if loinc_codes:
                loinc_filter = f"AND c.concept_code IN ({', '.join('?' * len(loinc_codes))})"
                params.extend(loinc_codes)
            _log(f"[trajectory] labs: scanning measurement.csv (window={max_window} d) …")
            parts_sql.append(f"""
                SELECT
                    COALESCE(p.panel, 'Other lab')                                        AS panel,
                    'lab'                                                                 AS event_type,
                    CAST(FLOOR((DATEDIFF('day',
                        CAST(t.measurement_date AS DATE),
                        CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                    ) - 1) / {bin_days}) AS INTEGER)                                      AS bin_index,
                    a.impression_id
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

        # ── Count feature types ──────────────────────────────────────────────
        for ft, (tbl, date_col, _datetime_col, concept_col, label, _value_expr) in _TIMELINE_TABLE_CONFIG.items():
            if ft not in feature_types:
                continue
            csv_path = Path(omop_dir) / f"{tbl}.csv"
            if not csv_path.exists():
                _log(f"  WARNING: {tbl}.csv not found — skipping {ft}")
                continue
            window_days = count_window_days.get(ft, 365)
            if lookback_days is not None:
                window_days = min(window_days, lookback_days)
            _log(f"[trajectory] {ft}: scanning {tbl}.csv (window={window_days} d) …")
            parts_sql.append(f"""
                SELECT
                    COALESCE(p.panel, 'Other {label}')                                    AS panel,
                    '{label}'                                                             AS event_type,
                    CAST(FLOOR((DATEDIFF('day',
                        CAST(t.{date_col} AS DATE),
                        CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                    ) - 1) / {bin_days}) AS INTEGER)                                      AS bin_index,
                    a.impression_id
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

        if not parts_sql:
            _log("[trajectory] no feature types configured — returning empty result")
            return pd.DataFrame(columns=out_cols)

        union_sql = "\nUNION ALL\n".join(parts_sql)
        _log("[trajectory] aggregating inside DuckDB …")
        # The GROUP BY (not the raw UNION ALL) is what actually reaches
        # Python — one row per (panel, event_type, bin_index), never one row
        # per event. Safe to .df() directly; this result is small by
        # construction regardless of how many raw events fed into it.
        agg_df = con.execute(f"""
            SELECT
                panel, event_type, bin_index,
                COUNT(*)                        AS n_events,
                COUNT(DISTINCT impression_id)   AS n_studies
            FROM ({union_sql})
            GROUP BY panel, event_type, bin_index
            ORDER BY panel, bin_index
        """, params).df()
    finally:
        try:
            con.close()
        except Exception:
            pass
        if tmp_db.exists():
            tmp_db.unlink(missing_ok=True)

    if agg_df.empty:
        _log("[trajectory] no events matched — empty result")
        return pd.DataFrame(columns=out_cols)

    # bin_index counts bin_days-wide steps back from anchor, 0-based:
    # bin_index=0 -> days [1, bin_days], bin_index=1 -> [bin_days+1, 2*bin_days], …
    agg_df["bin_start_days"] = agg_df["bin_index"] * bin_days + 1
    agg_df["bin_end_days"]   = (agg_df["bin_index"] + 1) * bin_days
    agg_df["pct_studies"] = (
        100.0 * agg_df["n_studies"] / n_total_studies if n_total_studies else 0.0
    )

    _log(
        f"[trajectory] {len(agg_df):,} (panel, bin) rows "
        f"({agg_df['panel'].nunique():,} panels × "
        f"{agg_df['bin_index'].nunique():,} bins) from {n_total_studies:,} studies")
    return agg_df[out_cols]


def build_admission_cohort_trajectory(
    fm: "LabFeatureMatrix",
    omop_dir: "Path",
    measurement_path: "Path",
    concept_path: "Path",
    loinc_codes: Optional[list] = None,
    bin_days: int = 1,
    memory_limit_gb: float = 4.0,
    verbose: bool = True,
) -> "pd.DataFrame":
    """Population-level, binned event density relative to *admission date*.

    Same DuckDB mechanism as build_cohort_trajectory, but the x-axis is
    **days since qualifying arrival** (0 = admission day, increasing toward
    anchor/CTPA) rather than days before CTPA.

    The qualifying arrival per study is the most recent visit in
    visit_occurrence.csv whose date range encompasses the anchor_time and
    whose visit_concept_id is one of ARRIVAL_VISIT_CONCEPT_IDS (9201 =
    Inpatient, 262 = ER+Inpatient, 9203 = ER).  Studies with no such visit
    are excluded from all counts — n_total_matched in the returned DataFrame
    reflects only studies that were successfully matched.

    Parameters
    ----------
    fm, omop_dir, measurement_path, concept_path, loinc_codes,
    memory_limit_gb, verbose : see build_cohort_trajectory
    bin_days : width of each time bin in days since admission (default 1 —
        daily bins suit admission windows which are typically 1–30 days;
        use 7 for a longer admission tail).

    Returns
    -------
    pd.DataFrame with columns:
        panel, event_type, bin_index (0 = day 0 of admission),
        bin_start_days, bin_end_days (in days since admission),
        n_events, n_studies, pct_studies,
        n_total_matched (constant — studies with a qualifying admission).
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
    from Custom.appd_route_b_constants import ARRIVAL_VISIT_CONCEPT_IDS

    if bin_days < 1:
        raise ValueError(f"bin_days must be >= 1, got {bin_days}")

    visit_csv = Path(omop_dir) / "visit_occurrence.csv"
    if not visit_csv.exists():
        raise FileNotFoundError(
            f"visit_occurrence.csv not found at {visit_csv} — "
            "admission trajectory requires visit_occurrence.csv")

    feature_types     = list(getattr(fm, "feature_types", None) or [])
    count_window_days = dict(getattr(fm, "count_window_days", None) or {})
    windows_days      = list(fm.windows_days) if fm.windows_days else [365]

    out_cols = [
        "panel", "event_type", "bin_index",
        "bin_start_days", "bin_end_days",
        "n_events", "n_studies", "pct_studies", "n_total_matched",
    ]

    anchors_df = pd.DataFrame({
        "person_id":     fm.patient_ids.astype("int64"),
        "impression_id": fm.impression_ids,
        "anchor_time":   [str(a) for a in fm.anchor_times],
    })

    tmp_db = Path(tempfile.mktemp(suffix="_adm_trajectory.duckdb"))
    con = duckdb.connect(str(tmp_db))
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA memory_limit='{memory_limit_gb}GB'")
    con.register("anchors_tbl", anchors_df)

    arrival_ids_sql = ", ".join(str(v) for v in ARRIVAL_VISIT_CONCEPT_IDS)

    try:
        _log("[adm-trajectory] loading concept.csv …")
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

        _log("[adm-trajectory] loading clinical panel taxonomy …")
        panel_df = pd.DataFrame(
            panel_lookup_rows(), columns=["vocabulary_id", "concept_code", "panel"])
        con.register("_panels", panel_df)

        # ── Find the qualifying arrival per study ──────────────────────────
        # For each study, find the most recent qualifying arrival whose
        # date range encompasses anchor_time (start ≤ anchor ≤ end, or
        # still open).  Uses QUALIFY to pick one row per impression_id.
        _log("[adm-trajectory] matching qualifying arrivals from visit_occurrence.csv …")
        con.execute(f"""
            CREATE TEMP TABLE _study_admissions AS
            SELECT
                a.impression_id,
                a.person_id,
                CAST(a.anchor_time AS TIMESTAMP)                AS anchor_time,
                CAST(v.visit_start_date AS DATE)                AS admission_date,
                DATEDIFF('day',
                    CAST(v.visit_start_date AS DATE),
                    CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                )                                               AS los_days
            FROM anchors_tbl a
            INNER JOIN read_csv_auto('{visit_csv}', ignore_errors=true) v
                    ON CAST(v.person_id AS BIGINT) = a.person_id
                   AND CAST(v.visit_concept_id AS BIGINT) IN ({arrival_ids_sql})
                   AND CAST(v.visit_start_date AS DATE)
                           <= CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                   AND (
                       v.visit_end_date IS NULL
                       OR CAST(v.visit_end_date AS DATE)
                               >= CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)
                   )
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY a.impression_id
                ORDER BY CAST(v.visit_start_date AS DATE) DESC
            ) = 1
        """)

        n_matched = con.execute(
            "SELECT COUNT(*) FROM _study_admissions").fetchone()[0]
        n_total   = len(fm.impression_ids)
        _log(f"[adm-trajectory] {n_matched:,} / {n_total:,} studies matched to "
             f"a qualifying arrival")

        if n_matched == 0:
            _log("[adm-trajectory] no studies matched — returning empty result")
            return pd.DataFrame(columns=out_cols)

        parts_sql: list = []
        params: list = []

        # ── Labs ─────────────────────────────────────────────────────────
        if "labs" in feature_types:
            max_window = max(windows_days)
            loinc_filter = ""
            if loinc_codes:
                loinc_filter = (
                    f"AND c.concept_code IN "
                    f"({', '.join('?' * len(loinc_codes))})"
                )
                params.extend(loinc_codes)
            _log(f"[adm-trajectory] labs: scanning measurement.csv …")
            parts_sql.append(f"""
                SELECT
                    COALESCE(p.panel, 'Other lab')                              AS panel,
                    'lab'                                                       AS event_type,
                    CAST(FLOOR(
                        DATEDIFF('day',
                            sa.admission_date,
                            CAST(t.measurement_date AS DATE)
                        ) / {bin_days}
                    ) AS INTEGER)                                               AS bin_index,
                    sa.impression_id
                FROM read_csv_auto('{measurement_path}', ignore_errors=true) t
                INNER JOIN _study_admissions sa
                        ON CAST(t.person_id AS BIGINT) = sa.person_id
                LEFT  JOIN _concept c
                        ON CAST(t.measurement_concept_id AS BIGINT) = c.concept_id
                LEFT  JOIN _panels p
                        ON p.vocabulary_id = c.vocabulary_id
                       AND p.concept_code = c.concept_code
                WHERE t.measurement_date IS NOT NULL
                  AND CAST(t.measurement_concept_id AS BIGINT) != 0
                  {loinc_filter}
                  AND CAST(t.measurement_date AS DATE) >= sa.admission_date
                  AND CAST(t.measurement_date AS DATE)
                          <= CAST(sa.anchor_time AS DATE)
            """)

        # ── Count feature types ──────────────────────────────────────────
        for ft, (tbl, date_col, _datetime_col, concept_col, label, _value_expr) \
                in _TIMELINE_TABLE_CONFIG.items():
            if ft not in feature_types:
                continue
            csv_path = Path(omop_dir) / f"{tbl}.csv"
            if not csv_path.exists():
                _log(f"  WARNING: {tbl}.csv not found — skipping {ft}")
                continue
            _log(f"[adm-trajectory] {ft}: scanning {tbl}.csv …")
            parts_sql.append(f"""
                SELECT
                    COALESCE(p.panel, 'Other {label}')                          AS panel,
                    '{label}'                                                   AS event_type,
                    CAST(FLOOR(
                        DATEDIFF('day',
                            sa.admission_date,
                            CAST(t.{date_col} AS DATE)
                        ) / {bin_days}
                    ) AS INTEGER)                                               AS bin_index,
                    sa.impression_id
                FROM read_csv_auto('{csv_path}', ignore_errors=true) t
                INNER JOIN _study_admissions sa
                        ON CAST(t.person_id AS BIGINT) = sa.person_id
                LEFT  JOIN _concept c
                        ON CAST(t.{concept_col} AS BIGINT) = c.concept_id
                LEFT  JOIN _panels p
                        ON p.vocabulary_id = c.vocabulary_id
                       AND p.concept_code = c.concept_code
                WHERE t.{date_col} IS NOT NULL
                  AND CAST(t.{concept_col} AS BIGINT) != 0
                  AND CAST(t.{date_col} AS DATE) >= sa.admission_date
                  AND CAST(t.{date_col} AS DATE)
                          <= CAST(sa.anchor_time AS DATE)
            """)

        if not parts_sql:
            _log("[adm-trajectory] no feature types configured — returning empty result")
            return pd.DataFrame(columns=out_cols)

        union_sql = "\nUNION ALL\n".join(parts_sql)
        _log("[adm-trajectory] aggregating inside DuckDB …")
        agg_df = con.execute(f"""
            SELECT
                panel, event_type, bin_index,
                COUNT(*)                      AS n_events,
                COUNT(DISTINCT impression_id) AS n_studies
            FROM ({union_sql})
            WHERE bin_index >= 0
            GROUP BY panel, event_type, bin_index
            ORDER BY panel, bin_index
        """, params).df()

    finally:
        try:
            con.close()
        except Exception:
            pass
        if tmp_db.exists():
            tmp_db.unlink(missing_ok=True)

    if agg_df.empty:
        _log("[adm-trajectory] no events matched — empty result")
        return pd.DataFrame(columns=out_cols)

    agg_df["bin_start_days"]   = agg_df["bin_index"] * bin_days
    agg_df["bin_end_days"]     = (agg_df["bin_index"] + 1) * bin_days - 1
    agg_df["pct_studies"]      = (
        100.0 * agg_df["n_studies"] / n_matched if n_matched else 0.0
    )
    agg_df["n_total_matched"] = n_matched

    _log(
        f"[adm-trajectory] {len(agg_df):,} (panel, bin) rows "
        f"({agg_df['panel'].nunique():,} panels × "
        f"{agg_df['bin_index'].nunique():,} bins) "
        f"from {n_matched:,} admission-matched studies")
    return agg_df[out_cols]
