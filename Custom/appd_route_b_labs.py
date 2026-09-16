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

This file now re-exports all public names from the four sub-modules below so
that existing importers (appd_export.py, appd_context_descriptors.py,
appd_route_b_worker.py, app_feature_extraction.py) continue to work without
any changes.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Re-exports — keep all public names importable from this module
# ---------------------------------------------------------------------------

from Custom.appd_route_b_constants import (
    SCRIPT_DIR, DATA_ROOT,
    DEFAULT_MEASUREMENT, DEFAULT_CONCEPT, DEFAULT_CONCEPT_ANCESTOR,
    DEFAULT_PERSON, DEFAULT_COHORT, DEFAULT_LABELS,
    GENDER, PATIENT_ID_CANDIDATES, TIME_COLUMN, IMPRESSION_COLUMN,
    DEFAULT_WINDOWS, AGGS,
    DX_TASKS, PX_TASKS, SURVIVAL_COLUMNS,
    TTE_MINUTES_PER_DAY, TRUTHY, SKIP_LABEL_VALUES, CATCH_ALL_DAYS,
    ARRIVAL_VISIT_CONCEPT_IDS,
    _EVENT_TABLES, _COUNT_TABLE_CONFIG, _TIMELINE_TABLE_CONFIG,
)
from Custom.appd_route_b_concepts import (
    load_concept_map, load_concept_id_map, load_concept_vocab_code_map,
    humanize_column,
)
from Custom.appd_route_b_matrix import (
    LabFeatureMatrix, load_demographic_features, append_demographics,
)
from Custom.appd_route_b_timeline import (
    query_events_stack,
    build_event_timeline,
    build_event_timeline_streamed,
    build_timeline_skeleton_streamed,
    build_cohort_trajectory,
    build_admission_cohort_trajectory,
)

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
        """For each cohort row, find the qualifying arrival visit the anchor
        falls inside — i.e. visit_start_date <= anchor_date <= visit_end_date
        (or the visit is still open, visit_end_date IS NULL). "Qualifying"
        means visit_concept_id IN ARRIVAL_VISIT_CONCEPT_IDS — inpatient
        admission or an ED visit (not necessarily one that became a formal
        admission); see that constant's comment for the reasoning.

        Used by admission-anchored extraction (``build(admission_anchored=True)``):
        the feature window becomes [admission_start_date, anchor_time] per
        patient instead of a fixed number of days before anchor for everyone.
        Studies with no qualifying arrival simply have no row in the result —
        the caller (``build()``) keeps them in the cohort regardless and
        falls back to the plain fixed window for them, it does not drop them.

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

        concept_ids = ", ".join(str(c) for c in ARRIVAL_VISIT_CONCEPT_IDS)
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
            f"  encompassing arrival visit: {df['impression_id'].nunique():,}/"
            f"{len(cohort_rows):,} studies matched "
            f"(visit_concept_id IN {ARRIVAL_VISIT_CONCEPT_IDS})")
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
            vectorized scan, not a per-patient loop. Coverage need not be
            complete: a study with no entry (no qualifying admission/arrival
            visit) falls back to the plain fixed window ``w``, uncapped,
            rather than being excluded — the caller no longer drops
            unmatched studies from the cohort.
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
            # COALESCE(..., max_window): a study with no admission_date (no
            # qualifying arrival visit for this patient) falls back to the
            # plain fixed window instead of the DATEDIFF/LEAST producing NULL
            # — which would silently zero out that study's results (NULL
            # never satisfies BETWEEN 1 AND NULL), not just leave it uncapped.
            max_window_expr = (
                f"LEAST({max_window}, COALESCE(DATEDIFF('day', "
                "CAST(a.admission_date AS DATE), "
                f"CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)), {max_window}))"
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
        # Only studies with an admission_date have a "before admission" to
        # measure from — unlike the window-capping elsewhere in this class,
        # there's no fixed-window fallback that makes sense here. Studies
        # without one are simply left out of baseline_rows; _run_duckdb_and_
        # pivot still reindexes onto the full imp_ids list below, so they
        # come back as NaN on these columns rather than raising a KeyError
        # or being dropped from the cohort.
        baseline_rows = [
            (pid, datetime.date.fromisoformat(admission_dates[imp]), imp, split, y)
            for pid, _anchor, imp, split, y in cohort_rows
            if imp in admission_dates
        ]
        if not baseline_rows:
            self._log(
                "  WARNING: no studies have an admission_date — skipping "
                "pre-admission baseline/delta")
            return np.zeros((len(imp_ids), 0), dtype=np.float32), []
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
                # COALESCE(..., window_days): no admission_date for this study
                # -> fall back to the plain fixed window instead of NULLing
                # out the row (see _run_duckdb_and_pivot's max_window_expr).
                window_upper_expr = (
                    f"LEAST({window_days}, COALESCE(DATEDIFF('day', "
                    "CAST(a.admission_date AS DATE), "
                    f"CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)), {window_days}))"
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
                    # COALESCE(..., window_days): see _run_duckdb_and_pivot's
                    # max_window_expr — no admission_date for this study falls
                    # back to the plain fixed window, not NULL.
                    window_upper_expr = (
                        f"LEAST({window_days}, COALESCE(DATEDIFF('day', "
                        "CAST(a.admission_date AS DATE), "
                        f"CAST(CAST(a.anchor_time AS TIMESTAMP) AS DATE)), {window_days}))"
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
            admission_anchored    : if True, for studies whose anchor falls
                                    inside a qualifying arrival visit
                                    (visit_concept_id IN ARRIVAL_VISIT_CONCEPT_IDS
                                    — inpatient admission or an ED visit,
                                    visit_start_date <= anchor <= visit_end_date
                                    or still open), extract features from
                                    [arrival_date, anchor] instead of a fixed
                                    lookback — windows_days/count_window_days
                                    still apply as an outer ceiling per study
                                    (LEAST(window, arrival -> anchor gap)).
                                    Studies with no qualifying arrival record
                                    are NOT dropped — they simply use the
                                    plain fixed window like a normal
                                    extraction. See
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

        # 1b. Admission-anchored mode: for studies whose anchor falls inside a
        #     qualifying arrival visit (see ARRIVAL_VISIT_CONCEPT_IDS), derive
        #     a per-study [arrival, anchor] window that steps 2-4 below cap
        #     their lookback windows against. Single vectorized DuckDB join
        #     across the whole cohort — no per-patient loop.
        #
        #     Studies with no qualifying arrival record are KEPT in the
        #     cohort, not dropped — they simply fall back to the plain fixed
        #     windows_days/count_window_days for every feature (the SQL-side
        #     COALESCE in _run_duckdb_and_pivot etc. does this per row). This
        #     was the original design and got corrected after a requirements
        #     mismatch: the intent was always "use a relative window when we
        #     have an arrival date, else fall back to the normal fixed one,"
        #     not "restrict the cohort to admitted patients only."
        admission_meta  = None
        admission_dates = None
        if admission_anchored:
            self._log("\n[admission-anchored] finding encompassing admissions …")
            admission_meta = self._find_encompassing_admissions(cohort_rows)
            admission_dates = dict(zip(admission_meta["impression_id"],
                                        admission_meta["admission_date"]))
            n_matched = len(admission_dates)
            n_total   = len(cohort_rows)
            self._log(
                f"  {n_matched:,}/{n_total:,} studies have a qualifying "
                f"arrival record ({n_total - n_matched:,} will use the plain "
                f"fixed window instead of a relative one)")
            if n_matched == 0:
                self._log(
                    "  WARNING: zero studies matched — check that "
                    "visit_occurrence.csv covers this cohort and "
                    "ARRIVAL_VISIT_CONCEPT_IDS is right for this data; "
                    "every study will use the plain fixed window.")

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
