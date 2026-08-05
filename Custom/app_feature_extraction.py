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
    "diag_proc": ["condition_occurrence", "procedure_occurrence", "device_exposure"],
    "drugs": ["drug_exposure"],
    "visits": ["visit_occurrence", "visit_detail"],
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

    tab_src, tab_cfg, tab_run, tab_desc = st.tabs(
        ["0 · Data sources", "1 · Configure", "2 · Extract", "3 · Describe"])

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
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
                    # process exited — check for output
                    if Path(out_str).exists():
                        st.success("✅ Extraction finished!")
                        with open(out_str, "rb") as _f:
                            st.session_state["fm"] = pickle.load(_f)
                        st.session_state.pop(proc_key)
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
            st.dataframe(fm.to_frame().head(50), use_container_width=True)

    # ---------------- 3 · describe --------------------------------------
    with tab_desc:
        fm = st.session_state.get("fm")
        if fm is None:
            st.info("Run an extraction first.")
        else:
            from Custom.context_descriptors import ContextDescriber
            dims = st.multiselect(
                "Slice by",
                ["split", "density_quartile", "age_band", "sex",
                 "race_concept_id", "ethnicity_concept_id", "care_site_id"],
                default=["split"])
            keycols = st.multiselect(
                "Missingness for columns", fm.columns[:2000], default=[],
                help="Adds missing_<column> per slice")
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
                st.dataframe(table, use_container_width=True)
                buf = io.StringIO(); table.to_csv(buf, index=False)
                st.download_button("Download descriptors .csv", buf.getvalue(),
                                   file_name=f"descriptors_{task}.csv",
                                   mime="text/csv")
                st.caption("Attach per-slice performance with "
                           "ContextDescriber.attach_performance() to build the "
                           "(descriptors → performance) dataset.")



if __name__ == "__main__":  # pragma: no cover
    main()
