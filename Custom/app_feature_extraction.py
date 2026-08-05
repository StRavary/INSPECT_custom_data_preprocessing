"""
app_feature_extraction.py
-------------------------
Streamlit interface over Custom/temporal_features.py and
Custom/context_descriptors.py.

Run:

    ../.venv_legacy/bin/python -m streamlit run ./app_feature_extraction.py

Design note
-----------
Extraction is slow (FEMR preprocess + featurize across ~22k studies) while
configuration is instant. The app keeps those apart:

  * the CONFIGURE tab validates a spec and previews exactly which windows and
    anchor will be used, with no data access at all -- this is where a new user
    should spend their time;
  * EXTRACT runs the job once and caches it;
  * DESCRIBE explores slices of the cached result.

Every run can be exported as YAML so the same spec is reproducible from the
Python API without the UI.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

# ---------------------------------------------------------------------------
# Pure logic (importable and testable without a Streamlit runtime)
# ---------------------------------------------------------------------------

CATCH_ALL_DAYS = 36500

DATA_ROOT = SCRIPT_DIR.parent.parent

# kind -> (label, is_dir, default path, help)
DATA_SOURCES = {
    "cohort": ("Cohort master file", False,
               DATA_ROOT / "DATA_PROCESSED" / "cohort_0.2.0_master_file_anon.csv",
               "One row per CTPA study. Supplies PatientID, StudyTime, "
               "impression_id, split and the task label columns."),
    "database": ("FEMR database (extract/)", True,
                 DATA_ROOT / "DATA_RAW" / "EHR_FEMR_DB" / "extract",
                 "Processed patient timelines. Route A — comparable to the "
                 "published baselines."),
    "labels": ("Labels TSV", False,
               DATA_ROOT / "DATA_RAW" / "LABELS" / "labels_20250611.tsv",
               "Survival outcomes (tte_* / is_censored_*). Not re-derived."),
    "person": ("person.csv (demographics)", False,
               DATA_ROOT / "DATA_RAW" / "EHR_CSV" / "person.csv",
               "Sex, race, ethnicity, birth date, care site — used by "
               "ContextDescriber."),
    "omop": ("Raw OMOP CSVs", True,
             DATA_ROOT / "DATA_RAW" / "EHR_CSV",
             "Route B — unreduced source. Not yet consumed by the extractor; "
             "validated here so the environment can be checked."),
}

OMOP_EXPECTED = ["measurement", "observation", "condition_occurrence",
                 "drug_exposure", "procedure_occurrence", "visit_occurrence",
                 "concept", "concept_ancestor", "person"]

FEMR_EXPECTED = ["meta", "patients", "ontology", "event_metadata"]


def list_dir_entries(path, dirs_only: bool = False):
    """(subdirectories, files) for the browse widget. Never raises."""
    p = Path(path)
    try:
        entries = sorted(p.iterdir(), key=lambda e: e.name.lower())
    except (OSError, PermissionError):
        return [], []
    dirs = [e.name for e in entries if e.is_dir()]
    files = [] if dirs_only else [e.name for e in entries if e.is_file()]
    return dirs, files


def _line_count(path, cap: int = 5_000_000) -> int:
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
            if n >= cap:
                break
    return n


def validate_source(kind: str, path) -> tuple[bool, list[str]]:
    """(ok, messages). Cheap checks only — never reads a large table."""
    label, is_dir, _, _ = DATA_SOURCES[kind]
    p = Path(path).expanduser()
    if not p.exists():
        return False, [f"does not exist: {p}"]
    if is_dir and not p.is_dir():
        return False, ["expected a directory"]
    if not is_dir and not p.is_file():
        return False, ["expected a file"]

    msgs: list[str] = []
    try:
        if kind == "cohort":
            n = _line_count(p) - 1
            with open(p) as f:
                header = f.readline().rstrip("\n").split(
                    "\t" if p.suffix == ".tsv" else ",")
            hset = set(header)
            msgs.append(f"{n:,} rows, {len(header)} columns")

            # patient id ships as patient_id or PatientID depending on which
            # version of 2_merge_labels.py produced the file
            pid = next((c for c in ("patient_id", "PatientID", "person_id")
                        if c in hset), None)
            missing = {"StudyTime", "impression_id", "split"} - hset
            if pid is None:
                missing.add("patient_id / PatientID")
            if missing:
                return False, msgs + [f"missing required columns: {sorted(missing)}"]
            msgs.append(f"required columns present (patient id: {pid})")

            n_tte = len([c for c in header if c.startswith("tte_")])
            if n_tte:
                msgs.append(f"{n_tte} survival endpoints in-file — labels TSV optional")

        elif kind == "labels":
            n = _line_count(p) - 1
            with open(p) as f:
                header = f.readline().rstrip("\n").split("\t")
            tte = [c for c in header if c.startswith("tte_")]
            msgs.append(f"{n:,} rows, {len(tte)} survival endpoints")
            if not tte:
                return False, msgs + ["no tte_* columns found"]

        elif kind == "person":
            n = _line_count(p) - 1
            with open(p) as f:
                header = f.readline().rstrip("\n").split(",")
            msgs.append(f"{n:,} rows")
            if "person_id" not in header:
                return False, msgs + ["no person_id column"]

        elif kind == "database":
            present = [n for n in FEMR_EXPECTED if (p / n).exists()]
            missing = [n for n in FEMR_EXPECTED if n not in present]
            msgs.append(f"contains {', '.join(present) or 'nothing expected'}")
            if missing:
                return False, msgs + [f"missing: {', '.join(missing)}"]

        elif kind == "omop":
            found, total = [], 0
            for t in OMOP_EXPECTED:
                f = p / f"{t}.csv"
                if f.exists():
                    found.append(t)
                    total += f.stat().st_size
            msgs.append(f"{len(found)}/{len(OMOP_EXPECTED)} expected tables, "
                        f"{total / 1024**3:.1f} GB")
            if len(found) < len(OMOP_EXPECTED):
                missing = [t for t in OMOP_EXPECTED if t not in found]
                msgs.append(f"missing: {', '.join(missing)}")
                return False, msgs
    except (OSError, UnicodeDecodeError) as e:
        return False, [f"could not read: {e}"]

    return True, msgs


GROUP_TABLES = {
    "vitals_labs": ["measurement", "observation"],
    # diag_proc keeps diagnoses + procedures together (matches published baseline)
    "diag_proc":   ["condition_occurrence", "procedure_occurrence", "device_exposure"],
    # diag / proc let you set different time horizons for each
    "diag":        ["condition_occurrence"],
    "proc":        ["procedure_occurrence", "device_exposure"],
    "drugs":       ["drug_exposure"],
    "visits":      ["visit_occurrence", "visit_detail"],
}

PRESETS = {
    "Baseline (whole history, matches published GBM)": {
        "diag_proc": [CATCH_ALL_DAYS],
    },
    "Recency-aware (recommended starting point)": {
        "vitals_labs": [2, 30, 365],
        "diag_proc": [365, 1825],
        "drugs": [30, 365],
    },
    "Acute window only (last 30 days)": {
        "vitals_labs": [2, 7, 30],
        "diag_proc": [30],
        "drugs": [30],
    },
    "Split diag / proc (different horizons)": {
        "vitals_labs": [2, 30, 365],
        "diag":        [365, 1825],   # diagnoses: 1 y + 5 y lookback
        "proc":        [30, 365],     # procedures: 1 m + 1 y lookback
        "drugs":       [30, 365],
    },
}


def parse_bins(text: str) -> list[int]:
    """'2, 30, 365' -> [2, 30, 365]. Raises ValueError on bad input."""
    if not text or not text.strip():
        return []
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v <= 0:
            raise ValueError(f"window edges must be positive, got {v}")
        out.append(v)
    return sorted(set(out))


def effective_windows(bins: list[int], catch_all: bool) -> list[int]:
    """What CountFeaturizer will actually emit, after the catch-all rule."""
    days = sorted(set(bins))
    if catch_all and (not days or days[-1] < CATCH_ALL_DAYS):
        days = days + [CATCH_ALL_DAYS]
    return days


def window_intervals(days: list[int]) -> list[tuple[int, int]]:
    """[2, 30] -> [(0, 2), (2, 30)] as days back from the anchor."""
    edges = [0] + list(days)
    return list(zip(edges[:-1], edges[1:]))


def validate_spec(spec: dict) -> list[tuple[str, str]]:
    """Return [(level, message)] where level is 'error' | 'warning' | 'info'."""
    msgs: list[tuple[str, str]] = []
    groups = spec.get("groups", {})

    if not groups:
        msgs.append(("error", "Select at least one feature group."))

    for name, g in groups.items():
        bins = g.get("bins", [])
        if not bins:
            msgs.append(("error", f"'{name}' has no window edges."))
            continue
        if not g.get("catch_all", True):
            msgs.append((
                "warning",
                f"'{name}': catch-all disabled, so every event older than "
                f"{max(bins)} days is silently DISCARDED. FEMR tracks that bucket "
                f"but never emits it."))
        if len(bins) > 6:
            msgs.append(("warning",
                         f"'{name}': {len(bins)} windows multiplies the column "
                         f"count by {len(bins)}. Sparse matrices get large fast."))

    task = spec.get("task")
    anchor = spec.get("anchor") or "auto"
    if task:
        msgs.append(("info", f"Anchor for '{task}': {describe_anchor(task, anchor)}"))
    return msgs


def describe_anchor(task: str, anchor: str = "auto") -> str:
    try:
        from Custom.temporal_features import TemporalFeatureExtractor as _T
        kind = _T.anchor_kind_for(task, None if anchor == "auto" else anchor)
    except Exception:
        return "unknown (task not registered — set anchor explicitly)"
    if kind == "dx":
        return "StudyTime − 1 day  (diagnostic: label is read off this CTPA)"
    return "StudyTime  (prognostic: label is a future event)"


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

CACHE_DIR    = DATA_ROOT / "DATA_PROCESSED" / "femr_cache"
WORKER_SCRIPT = SCRIPT_DIR / "_femr_worker.py"


def _spec_hash(spec: dict) -> str:
    """Short, stable hash of the extraction spec (task + groups + paths)."""
    return hashlib.md5(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()[:12]


def _cache_pkl(spec: dict) -> Path:
    return CACHE_DIR / f"{_spec_hash(spec)}.pkl"


def _cache_log(spec: dict) -> Path:
    return CACHE_DIR / f"{_spec_hash(spec)}.log"


def _cache_spec(spec: dict) -> Path:
    return CACHE_DIR / f"{_spec_hash(spec)}_spec.json"


def _list_cached() -> list[tuple[str, Path]]:
    """Return [(spec_hash, pkl_path)] for every finished extraction."""
    if not CACHE_DIR.exists():
        return []
    return sorted(
        [(p.stem, p) for p in CACHE_DIR.glob("*.pkl")],
        key=lambda x: x[1].stat().st_mtime,
        reverse=True,
    )


def _default_concept_csv(omop_path: str) -> Path:
    """Derive concept.csv path from the OMOP directory path."""
    return Path(omop_path) / "concept.csv"


# ---------------------------------------------------------------------------
# Concept-map cache (loaded once per Streamlit session)
# ---------------------------------------------------------------------------

def _load_concept_map_cached(concept_csv: str, st):
    """Streamlit-cached wrapper around temporal_features.load_concept_map."""
    from Custom.temporal_features import load_concept_map

    @st.cache_resource(show_spinner="Loading OMOP concept names …")
    def _load(path: str):
        return load_concept_map(path)

    return _load(concept_csv)


def spec_to_yaml(spec: dict) -> str:
    """Emit a reproducible spec plus the equivalent Python call."""
    lines = ["# INSPECT feature extraction spec",
             f"# generated {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"task: {spec.get('task')}",
             f"anchor: {spec.get('anchor', 'auto')}",
             "groups:"]
    for name, g in spec.get("groups", {}).items():
        lines.append(f"  {name}:")
        lines.append(f"    bins: {g.get('bins', [])}")
        lines.append(f"    catch_all: {bool(g.get('catch_all', True))}")
        lines.append(f"    tables: {GROUP_TABLES.get(name, [])}")
    if spec.get("paths"):
        lines.append("paths:")
        for k, v in spec["paths"].items():
            lines.append(f"  {k}: {v}")
    lines += ["", "# Equivalent Python:", "#"]
    lines.append("#   from Custom.temporal_features import TemporalFeatureExtractor, \\")
    lines.append("#       " + ", ".join(sorted(spec.get("groups", {}))).upper())
    call = ", ".join(
        f"{n.upper()}(bins={g.get('bins', [])}"
        + ("" if g.get("catch_all", True) else ", catch_all=False") + ")"
        for n, g in spec.get("groups", {}).items())
    lines.append(f"#   ds = TemporalFeatureExtractor().build(")
    lines.append(f"#       task={spec.get('task')!r}, groups=[{call}])")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def path_picker(kind: str, st) -> str:  # pragma: no cover
    """Type-or-browse path selector. No third-party component needed, and it
    works over SSH port-forwarding where a native dialog would not."""
    label, is_dir, default, help_text = DATA_SOURCES[kind]
    cur_key, browse_key = f"path_{kind}", f"browsing_{kind}"
    st.session_state.setdefault(cur_key, str(default))

    st.markdown(f"**{label}**")
    st.caption(help_text)

    c1, c2 = st.columns([6, 1])
    typed = c1.text_input("path", value=st.session_state[cur_key],
                          key=f"txt_{kind}", label_visibility="collapsed")
    if typed != st.session_state[cur_key]:
        st.session_state[cur_key] = typed
    if c2.button("Browse", key=f"btn_{kind}"):
        st.session_state[browse_key] = not st.session_state.get(browse_key, False)

    if st.session_state.get(browse_key):
        start = Path(st.session_state[cur_key]).expanduser()
        here = start if start.is_dir() else start.parent
        if not here.exists():
            here = Path.home()
        dirs, files = list_dir_entries(here, dirs_only=is_dir)
        options = [".. (up)"] + [f"📁 {d}" for d in dirs] + [f"📄 {f}" for f in files]
        st.caption(f"in `{here}`")
        choice = st.selectbox("browse", options, key=f"sel_{kind}",
                              label_visibility="collapsed")
        b1, b2 = st.columns(2)
        if b1.button("Open", key=f"open_{kind}"):
            if choice.startswith(".."):
                st.session_state[cur_key] = str(here.parent)
            elif choice.startswith("📁"):
                st.session_state[cur_key] = str(here / choice[2:])
            else:
                st.session_state[cur_key] = str(here / choice[2:])
            st.rerun()
        if b2.button("Use this", key=f"use_{kind}", type="primary"):
            st.session_state[browse_key] = False
            st.rerun()

    ok, msgs = validate_source(kind, st.session_state[cur_key])
    (st.success if ok else st.error)(
        ("✅ " if ok else "❌ ") + " · ".join(msgs))
    return st.session_state[cur_key]


def main() -> None:  # pragma: no cover - requires a Streamlit runtime
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="INSPECT Feature Extraction", layout="wide")
    st.title("🧪 INSPECT EHR Feature Extraction")
    st.caption("Pick your data sources, configure and validate a spec, run the "
               "extraction once, then explore context descriptors. "
               "See EHR_FEATURE_EXTRACTION_GUIDE.md.")

    from Custom.temporal_features import DX_TASKS, PX_TASKS

    tab_src, tab_cfg, tab_run, tab_desc, tab_exp = st.tabs(
        ["0 · Data sources", "1 · Configure", "2 · Extract", "3 · Describe", "4 · Export"])

    # ---------------- 0 · data sources ----------------------------------
    with tab_src:
        st.subheader("Where the data lives")
        st.caption("Defaults assume the standard layout. Paths are validated "
                   "as you set them — nothing large is read.")
        left, right = st.columns(2)
        with left:
            paths = {k: path_picker(k, st) for k in ("cohort", "database", "labels")}
        with right:
            paths.update({k: path_picker(k, st) for k in ("person", "omop")})

        st.divider()
        states = {k: validate_source(k, v)[0] for k, v in paths.items()}
        required = ("cohort", "database")
        if all(states[k] for k in required):
            st.success("Required sources are valid — extraction can run.")
        else:
            st.error("Cohort and FEMR database must both validate before "
                     "extraction. The others are optional: labels adds survival "
                     "outcomes, person.csv adds demographic descriptors, and the "
                     "OMOP directory is Route B (not yet consumed).")

    # ---------------- sidebar -------------------------------------------
    with st.sidebar:
        st.header("Spec")
        all_tasks = sorted(DX_TASKS | PX_TASKS)
        task = st.selectbox("Task", all_tasks,
                            index=all_tasks.index("12_month_PH")
                            if "12_month_PH" in all_tasks else 0)
        anchor = st.radio("Anchor", ["auto", "dx", "px"], horizontal=True,
                          help="auto uses the dx/px table in temporal_features.py")
        st.divider()

        preset = st.selectbox("Preset", ["(custom)"] + list(PRESETS))
        base = PRESETS.get(preset, {})

        groups: dict = {}
        for gname, tables in GROUP_TABLES.items():
            on = st.checkbox(gname, value=gname in base,
                             help="OMOP tables: " + ", ".join(tables))
            if not on:
                continue
            default = ", ".join(str(b) for b in base.get(gname, [365]))
            raw = st.text_input(f"{gname} windows (days)", value=default,
                                key=f"bins_{gname}")
            ca = st.checkbox("catch-all bin", value=True, key=f"ca_{gname}",
                             help="Append a 100-year window so older events are "
                                  "not silently dropped")
            try:
                groups[gname] = {"bins": parse_bins(raw), "catch_all": ca}
            except ValueError as e:
                st.error(f"{gname}: {e}")

    spec = {"task": task, "anchor": anchor, "groups": groups,
            "paths": {k: str(v) for k, v in paths.items()}}

    # ---------------- 1 · configure -------------------------------------
    with tab_cfg:
        st.subheader("Validation")
        msgs = validate_spec(spec)
        for level, m in msgs:
            {"error": st.error, "warning": st.warning, "info": st.info}[level](m)

        st.subheader("Windows that will be created")
        rows = []
        for gname, g in groups.items():
            eff = effective_windows(g["bins"], g["catch_all"])
            for lo, hi in window_intervals(eff):
                rows.append({
                    "group": gname,
                    "window": f"T0−{lo}d → T0−{hi}d" if hi < CATCH_ALL_DAYS
                              else f"T0−{lo}d → start of record",
                    "days_back": hi,
                    "tables": ", ".join(GROUP_TABLES[gname]),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
            st.caption(f"{len(rows)} windows total. Column count ≈ "
                       f"(distinct codes after ontology expansion) × {len(rows)}.")

        st.subheader("Reproducible spec")
        y = spec_to_yaml(spec)
        st.code(y, language="yaml")
        st.download_button("Download spec .yaml", y,
                           file_name=f"spec_{task}.yaml", mime="text/yaml")

    # ---------------- 2 · extract ---------------------------------------
    with tab_run:
        h        = _spec_hash(spec)
        pkl_path = _cache_pkl(spec)
        log_path = _cache_log(spec)
        proc_key = f"proc_{h}"      # session-state key for the running subprocess

        # ── guard: config errors or missing data ─────────────────────────
        if any(l == "error" for l, _ in msgs):
            st.error("Fix the errors in Configure first.")
        elif not all(validate_source(k, spec["paths"][k])[0]
                     for k in ("cohort", "database")):
            st.error("Fix the required data sources in tab 0 first.")

        else:
            # ── a subprocess is already running for this spec ─────────────
            proc_info = st.session_state.get(proc_key)  # (pid, out_path)
            if proc_info is not None:
                pid, out_str = proc_info

                # Output file existing is the definitive completion signal —
                # check it before the pid, which can be a zombie or reused.
                if Path(out_str).exists():
                    st.success("✅ Extraction finished!")
                    with open(out_str, "rb") as _f:
                        st.session_state["fm"] = pickle.load(_f)
                    st.session_state.pop(proc_key)
                    st.rerun()

                # Output file absent — check whether the process is still alive.
                try:
                    os.kill(pid, 0)
                    still_running = True
                except OSError:
                    still_running = False

                log_text = log_path.read_text() if log_path.exists() else "(waiting for output…)"
                st.subheader("Extraction log")
                st.code(log_text, language=None)

                if still_running:
                    st.info(f"⏳ Extraction running (PID {pid}) — page refreshes every 5 s …")
                    time.sleep(5)
                    st.rerun()
                else:
                    st.error("❌ Extraction failed — check the log above.")
                    st.session_state.pop(proc_key)

            # ── cached result already on disk ─────────────────────────────
            elif pkl_path.exists():
                size_mb = pkl_path.stat().st_size / 1e6
                st.success(f"✅ Cached result on disk ({size_mb:.0f} MB) — "
                           f"spec hash `{h}`")
                c1, c2 = st.columns(2)
                if c1.button("Load", type="primary"):
                    with st.spinner("Loading …"):
                        with open(pkl_path, "rb") as _f:
                            st.session_state["fm"] = pickle.load(_f)
                    st.rerun()
                if c2.button("Re-run (overwrite cache)"):
                    pkl_path.unlink(missing_ok=True)
                    log_path.unlink(missing_ok=True)
                    st.rerun()

            # ── nothing running, nothing cached ───────────────────────────
            else:
                st.info(
                    "Extraction runs FEMR preprocess + featurize over ~22k studies. "
                    "Expect **15 – 60 minutes** depending on group config. "
                    "The result is saved to disk so lab-mates can load it without re-running."
                )

                num_threads = st.slider(
                    "Worker threads", min_value=1, max_value=16, value=4,
                    help="Higher = faster, but uses more CPU/RAM. "
                         "4 is a safe default on shared machines.")

                if st.button("🚀 Run extraction", type="primary"):
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    _cache_spec(spec).write_text(json.dumps(spec, sort_keys=True))

                    log_fh = open(log_path, "w")
                    proc = subprocess.Popen(
                        [sys.executable, str(WORKER_SCRIPT),
                         str(_cache_spec(spec)), str(pkl_path),
                         str(num_threads)],
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        close_fds=True,
                    )
                    log_fh.close()   # child inherits the fd; parent can close its copy
                    st.session_state[proc_key] = (proc.pid, str(pkl_path))
                    st.rerun()

            # ── previously computed results ───────────────────────────────
            cached_list = _list_cached()
            if cached_list:
                st.divider()
                st.subheader("Previously computed")
                st.caption("Any result saved on this machine — share the path "
                           "with lab-mates so they can load without re-running.")
                for chash, cpath in cached_list:
                    spec_file = CACHE_DIR / f"{chash}_spec.json"
                    label = chash
                    if spec_file.exists():
                        try:
                            s = json.loads(spec_file.read_text())
                            label = (f"**{s.get('task')}** · "
                                     + ", ".join(
                                         f"{n}={g.get('bins')}"
                                         for n, g in s.get("groups", {}).items()))
                        except Exception:
                            pass
                    size_mb = cpath.stat().st_size / 1e6
                    mtime   = datetime.datetime.fromtimestamp(
                        cpath.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    col1, col2 = st.columns([5, 1])
                    col1.markdown(
                        f"{label}  \n"
                        f"<small>`{chash}` · {size_mb:.0f} MB · {mtime}</small>",
                        unsafe_allow_html=True)
                    if col2.button("Load", key=f"load_{chash}"):
                        with st.spinner("Loading …"):
                            with open(cpath, "rb") as _f:
                                st.session_state["fm"] = pickle.load(_f)
                        st.rerun()

        # ── result panel (shared by all paths above) ──────────────────────
        fm = st.session_state.get("fm")
        if fm is not None:
            st.divider()
            st.subheader("Result")
            c = st.columns(4)
            c[0].metric("Studies", f"{len(fm.y):,}")
            c[1].metric("Columns", f"{len(fm.columns):,}")
            c[2].metric("Event rate", f"{fm.y.mean():.4f}")
            c[3].metric("Anchor", fm.anchor_kind)
            st.text(fm.describe())
            st.dataframe(fm.to_frame().head(50), width='stretch')

            # Show human-readable column preview if concept.csv is available
            concept_csv = _default_concept_csv(spec["paths"].get("omop", ""))
            if concept_csv.exists():
                concept_map = _load_concept_map_cached(str(concept_csv), st)
                st.caption("Sample feature columns (OMOP names resolved):")
                st.dataframe(
                    pd.DataFrame({
                        "raw": fm.columns[:50],
                        "human": fm.human_columns(concept_map)[:50],
                    }),
                    width='stretch', hide_index=True,
                )

    # ---------------- 3 · describe --------------------------------------
    with tab_desc:
        fm = st.session_state.get("fm")
        if fm is None:
            st.info("Run an extraction first.")
        else:
            from Custom.context_descriptors import ContextDescriber

            # Build concept map if available (cached after first load)
            concept_csv = _default_concept_csv(spec["paths"].get("omop", ""))
            if concept_csv.exists():
                concept_map = _load_concept_map_cached(str(concept_csv), st)
                col_labels  = fm.human_columns(concept_map)
            else:
                concept_map = {}
                col_labels  = fm.columns
            # Map human label -> raw column name for ContextDescriber
            label_to_raw = dict(zip(col_labels, fm.columns))

            dims = st.multiselect(
                "Slice by",
                ["split", "density_quartile", "age_band", "sex",
                 "race_concept_id", "ethnicity_concept_id", "care_site_id"],
                default=["split"])
            selected_labels = st.multiselect(
                "Missingness for columns",
                col_labels[:2000],
                default=[],
                help="Adds a missing_<column> row per slice. "
                     "Columns shown with OMOP concept names where available.")
            keycols = [label_to_raw[l] for l in selected_labels]

            use_demo = st.checkbox("Join person.csv demographics", value=True)
            if st.button("Describe"):
                with st.spinner("Describing …"):
                    demo = (st.session_state.get("path_person")
                            if use_demo else None)
                    cd = ContextDescriber(fm, demographics=demo,
                                          key_columns=keycols, verbose=False)
                    table = cd.describe(by=dims or ["split"])
                st.session_state["desc"] = table

            table = st.session_state.get("desc")
            if table is not None:
                st.dataframe(table, width='stretch')
                buf = io.StringIO(); table.to_csv(buf, index=False)
                st.download_button("Download descriptors .csv", buf.getvalue(),
                                   file_name=f"descriptors_{task}.csv",
                                   mime="text/csv")
                st.caption("Attach per-slice performance with "
                           "ContextDescriber.attach_performance() to build the "
                           "(descriptors → performance) dataset.")

    # ---------------- 4 · export ----------------------------------------
    with tab_exp:
        fm = st.session_state.get("fm")
        if fm is None:
            st.info("Load an extraction first (Extract tab).")
        else:
            st.subheader("Export extracted features")
            st.caption(
                "Exports per-split sparse feature matrices, metadata, and "
                "ready-to-use load scripts for survival analysis and "
                "multimodal pipelines. Join key across modalities: "
                "**impression_id**.")

            default_out = str(
                DATA_ROOT / "DATA_PROCESSED" / "exports" /
                f"{fm.task}_{datetime.datetime.now().strftime('%Y%m%d')}"
            )
            export_dir = st.text_input("Output directory", value=default_out)

            concept_csv = _default_concept_csv(spec["paths"].get("omop", ""))
            use_human = st.checkbox(
                "Human-readable column names",
                value=concept_csv.exists(),
                disabled=not concept_csv.exists(),
                help="Requires concept.csv in OMOP path (Data sources tab).",
            )

            drop_zero = st.checkbox(
                "Drop zero-variance columns",
                value=True,
                help="Removes columns that are 0 for every patient — reduces "
                     "matrix size significantly.",
            )

            st.markdown("**Outputs generated:**")
            st.markdown(
                "- `metadata.csv` — impression_id · patient_id · split · y · tte_days · event  \n"
                "- `feature_names.csv` — index · raw_name · human_name  \n"
                "- `{split}/X_sparse.npz` — scipy CSR matrix (one per split)  \n"
                "- `{split}/y.npy`, `tte.npy`, `event.npy` — aligned arrays  \n"
                "- `{split}/metadata.csv` — split-level metadata  \n"
                "- `load_survival.py` — ready-to-run script for lifelines / scikit-survival  \n"
                "- `load_multimodal.py` — shows how to join EHR features with imaging / NLP by impression_id"
            )

            if st.button("Export", type="primary"):
                with st.spinner("Exporting …"):
                    concept_map = (
                        _load_concept_map_cached(str(concept_csv), st)
                        if use_human and concept_csv.exists() else {}
                    )
                    msg = _do_export(fm, export_dir, drop_zero, concept_map)
                st.success(msg)
                st.code(export_dir)


def _do_export(fm, export_dir: str, drop_zero: bool, concept_map: dict) -> str:
    """Write split-ready files to export_dir. Returns a summary string."""
    import scipy.sparse
    import pandas as pd
    import numpy as np

    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── column names ──────────────────────────────────────────────────────
    raw_cols   = list(fm.columns)
    human_cols = (fm.human_columns(concept_map) if concept_map else raw_cols)

    # ── drop zero-variance columns ─────────────────────────────────────
    X = fm.X.tocsr()
    if drop_zero:
        col_sums = np.asarray(X.sum(axis=0)).reshape(-1)
        keep     = col_sums > 0
        X          = X[:, keep]
        raw_cols   = [c for c, k in zip(raw_cols,   keep) if k]
        human_cols = [c for c, k in zip(human_cols, keep) if k]

    n_dropped = len(fm.columns) - len(raw_cols)

    # ── feature names ────────────────────────────────────────────────────
    pd.DataFrame({"raw": raw_cols, "human": human_cols}).to_csv(
        out / "feature_names.csv", index_label="col_index")

    # ── global metadata ──────────────────────────────────────────────────
    meta = fm.to_frame()   # impression_id, patient_id, split, anchor_time, y[, tte_days, event]
    meta.to_csv(out / "metadata.csv", index=False)

    # ── per-split arrays ─────────────────────────────────────────────────
    written_splits = []
    for split in ("train", "valid", "test"):
        mask = fm.split == split
        if not mask.any():
            continue
        sd = out / split
        sd.mkdir(exist_ok=True)

        scipy.sparse.save_npz(str(sd / "X_sparse.npz"), X[mask].tocsr())
        np.save(sd / "y.npy",     fm.y[mask])
        if fm.tte is not None:
            np.save(sd / "tte.npy",   fm.tte[mask])
            np.save(sd / "event.npy", fm.event[mask])
        meta[mask].to_csv(sd / "metadata.csv", index=False)
        written_splits.append(split)

    has_survival = fm.tte is not None

    # ── load_survival.py ─────────────────────────────────────────────────
    surv_lines = f'''\
"""
load_survival.py  —  auto-generated by app_feature_extraction.py
Load {fm.task} EHR features for survival analysis.
"""
import numpy as np
import scipy.sparse
import pandas as pd

BASE = "{out}"

def load_split(split):
    """Return (X, y, tte, event, meta) for one split."""
    d    = f"{{BASE}}/{{split}}"
    X    = scipy.sparse.load_npz(f"{{d}}/X_sparse.npz")
    y    = np.load(f"{{d}}/y.npy")
    meta = pd.read_csv(f"{{d}}/metadata.csv")
    {"tte   = np.load(f'{d}/tte.npy')" if has_survival else "tte   = None  # not available for this task"}
    {"event = np.load(f'{d}/event.npy')" if has_survival else "event = None"}
    return X, y, tte, event, meta

X_tr, y_tr, tte_tr, ev_tr, meta_tr = load_split("train")
X_va, y_va, tte_va, ev_va, meta_va = load_split("valid")
X_te, y_te, tte_te, ev_te, meta_te = load_split("test")

feat = pd.read_csv(f"{{BASE}}/feature_names.csv")
print(f"train: {{X_tr.shape}},  valid: {{X_va.shape}},  test: {{X_te.shape}}")
print(f"features: {{len(feat):,}},  label prevalence (train): {{y_tr.mean():.3f}}")

# ── scikit-survival ───────────────────────────────────────────────────────
# from sksurv.util import Surv
# from sksurv.ensemble import RandomSurvivalForest
# y_surv_tr = Surv.from_arrays(event=ev_tr.astype(bool), time=tte_tr)
# rsf = RandomSurvivalForest().fit(X_tr, y_surv_tr)

# ── lifelines ─────────────────────────────────────────────────────────────
# from lifelines import CoxPHFitter
# df = meta_tr.copy()
# df["tte"] = tte_tr; df["event"] = ev_tr
# cph = CoxPHFitter().fit(df[["tte","event","y"]], "tte", "event")
'''
    (out / "load_survival.py").write_text(surv_lines)

    # ── load_multimodal.py ───────────────────────────────────────────────
    mm_lines = f'''\
"""
load_multimodal.py  —  auto-generated by app_feature_extraction.py
Join {fm.task} EHR features with imaging / NLP outputs using impression_id.
"""
import numpy as np
import scipy.sparse
import pandas as pd

BASE = "{out}"

# 1. Load EHR features for one split
split = "train"
X_ehr   = scipy.sparse.load_npz(f"{{BASE}}/{{split}}/X_sparse.npz")
meta    = pd.read_csv(f"{{BASE}}/{{split}}/metadata.csv")
feat    = pd.read_csv(f"{{BASE}}/feature_names.csv")

# meta["impression_id"] is the join key for imaging and NLP modalities.
print(meta[["impression_id","patient_id","split","y"]].head())

# 2. Example: join with imaging embeddings
# imaging_df = pd.read_csv("path/to/image_embeddings.csv")
# merged = meta.merge(imaging_df, on="impression_id", how="inner")
# aligned rows in X_ehr correspond to merged.index after the merge —
# re-index X_ehr to keep alignment:
# row_idx = meta.index.get_indexer(merged.index)
# X_ehr_aligned = X_ehr[row_idx]

# 3. Example: concatenate EHR + image embeddings for a joint model
# import numpy as np
# X_image = np.load("path/to/image_embeddings.npy")[row_idx]
# X_joint = scipy.sparse.hstack([X_ehr_aligned,
#                                 scipy.sparse.csr_matrix(X_image)])
'''
    (out / "load_multimodal.py").write_text(mm_lines)

    return (
        f"✅ Exported {len(raw_cols):,} features "
        f"({n_dropped:,} zero-variance dropped) · "
        f"splits: {', '.join(written_splits)} · "
        f"survival arrays: {'yes' if has_survival else 'no (task not registered)'}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
