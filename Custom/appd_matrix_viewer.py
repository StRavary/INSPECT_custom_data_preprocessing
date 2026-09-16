"""
appd_matrix_viewer.py
---------------------
Standalone Streamlit app for exploring exported feature matrices.

Loads the arrays written by Tab 4 · Export:
  X.npy            float32  (n_studies × n_features)  NaN = not measured (labs)
                                                       0   = not observed  (counts)
  X_mask.npy       uint8    same shape                 1 = observed, 0 = missing
  y.npy            float32  (n_studies,)               binary label (0/1)
  feature_names.csv  columns: col_index, raw, human
  metadata.csv       per-study identifiers

Three tabs:
  Coverage      — % of studies with an observed value per feature, sorted & filterable
  Distributions — per-feature value histograms, mask-aware, split by label
  Heatmap       — subsampled (studies × features) value map, raw or mask mode

Run:
    streamlit run Custom/appd_matrix_viewer.py
    # or the legacy venv:
    ../../.venv_legacy/bin/python -m streamlit run Custom/appd_matrix_viewer.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT  = SCRIPT_DIR.parent.parent

# Export root: try the canonical sibling-repo location first, then the
# in-repo fallback so the default path is always valid.
_CANONICAL_EXPORT_ROOT = DATA_ROOT.parent / "DATA_PROCESSED" / "exports"
DEFAULT_EXPORT_ROOT = (
    _CANONICAL_EXPORT_ROOT
    if _CANONICAL_EXPORT_ROOT.exists()
    else DATA_ROOT / "DATA_PROCESSED" / "exports"
)

FEATURE_PREFIXES = ["labs:", "diag:", "drug:", "proc:", "obs:", "visit:", "demo:", "other"]

MAX_HEATMAP_STUDIES  = 400
MAX_HEATMAP_FEATURES = 200
MAX_DIST_FEATURES    = 8

# Sequential single-hue palette for coverage bars (blue ramp)
_COV_COLOR  = "#2563EB"   # high coverage
_MISS_COLOR = "#DBEAFE"   # low coverage — same hue, lighter

# Categorical pair for label split in distributions (blue / orange)
_COLOR_Y0   = "#2563EB"   # y=0  negative
_COLOR_Y1   = "#EA580C"   # y=1  positive

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prefix(col: str) -> str:
    for p in FEATURE_PREFIXES[:-1]:
        if col.startswith(p):
            return p
    return "other"


def _coverage(X: np.ndarray, mask: np.ndarray, raw_names: list[str]) -> np.ndarray:
    """Per-feature fraction of studies with an observed value, as %.

    X_mask.npy is NaN-based: mask=1 wherever X is not NaN. For lab features
    that is correct. For count features (diag:, drug:, proc:, obs:, visit:)
    X stores 0 for 'not observed' and the mask is therefore 1 for every study
    — which would report 100% coverage for rare diagnoses. We override those
    columns with (X > 0) so coverage means 'at least one event recorded'.
    """
    _COUNT_PREFIXES = {"diag:", "drug:", "proc:", "obs:", "visit:"}
    obs = mask.astype(float)  # start from the NaN-based mask
    for j, raw in enumerate(raw_names):
        if _prefix(raw) in _COUNT_PREFIXES:
            obs[:, j] = (X[:, j] > 0).astype(float)
    return obs.mean(axis=0) * 100.0


# ---------------------------------------------------------------------------
# Data loading — cached so reloads don't re-read large files
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading arrays …")
def load_export(export_dir: str):
    """Return (X, mask, y, raw_names, human_names, meta_df, spec).

    All arrays are in memory; caller is responsible for keeping the export
    directory small enough to fit.  For a 50k-study × 2k-feature matrix that's
    roughly 400 MB of float32 + 100 MB for the mask.
    """
    import pandas as pd

    d = Path(export_dir)

    X    = np.load(d / "X.npy")
    mask = (np.load(d / "X_mask.npy")
            if (d / "X_mask.npy").exists()
            else (~np.isnan(X)).astype(np.uint8))
    y    = np.load(d / "y.npy") if (d / "y.npy").exists() else None
    tte  = np.load(d / "tte.npy")   if (d / "tte.npy").exists()   else None
    ev   = np.load(d / "event.npy") if (d / "event.npy").exists() else None

    if (d / "feature_names.csv").exists():
        fn = pd.read_csv(d / "feature_names.csv", index_col="col_index")
        raw_names   = fn["raw"].tolist()
        human_names = fn["human"].tolist() if "human" in fn.columns else raw_names
    else:
        raw_names   = [f"f{i}" for i in range(X.shape[1])]
        human_names = raw_names

    meta = pd.read_csv(d / "metadata.csv") if (d / "metadata.csv").exists() else None

    spec: dict = {}
    if (d / "extraction_spec.json").exists():
        import json
        spec = json.loads((d / "extraction_spec.json").read_text())

    return X, mask, y, tte, ev, raw_names, human_names, meta, spec


# ---------------------------------------------------------------------------
# Sidebar — export directory picker
# ---------------------------------------------------------------------------

def _find_export_dirs(root: Path, max_depth: int = 4) -> list[Path]:
    """Return all directories under `root` that contain X.npy, up to max_depth."""
    results = []
    if not root.exists():
        return results

    def _walk(p: Path, depth: int):
        if depth > max_depth:
            return
        if (p / "X.npy").exists():
            results.append(p)
        try:
            for child in sorted(p.iterdir()):
                if child.is_dir():
                    _walk(child, depth + 1)
        except OSError:
            pass

    _walk(root, 0)
    return results


def _dir_picker(st) -> str | None:
    st.sidebar.header("Export directory")

    # ── Auto-detect: scan the known export root ───────────────────────────
    found = _find_export_dirs(DEFAULT_EXPORT_ROOT)

    if found:
        # Label each found dir relative to DEFAULT_EXPORT_ROOT for readability
        def _label(p: Path) -> str:
            try:
                rel = p.relative_to(DEFAULT_EXPORT_ROOT)
                return str(rel)
            except ValueError:
                return str(p)

        options = {_label(p): str(p) for p in found}
        choice = st.sidebar.selectbox(
            "Detected exports",
            list(options.keys()),
            key="mv_auto_sel",
            help="All directories containing X.npy found under the export root.",
        )
        auto_dir = options[choice]
    else:
        auto_dir = None
        st.sidebar.caption("No exports detected automatically.")

    # ── Manual override ───────────────────────────────────────────────────
    st.sidebar.markdown("**Or enter path manually**")
    manual = st.sidebar.text_input(
        "Path",
        value=auto_dir or str(DEFAULT_EXPORT_ROOT),
        key="mv_manual_path",
        label_visibility="collapsed",
    )

    # Manual path wins if user has typed something different from the auto selection
    chosen = manual.strip() if manual.strip() else auto_dir

    d = Path(chosen) if chosen else None
    if d and (d / "X.npy").exists():
        sz_mb = (d / "X.npy").stat().st_size / 1e6
        st.sidebar.success(f"✅ X.npy found ({sz_mb:.0f} MB)")
        return str(d)
    elif d:
        st.sidebar.warning(f"X.npy not found in:\n`{d}`")
        return None
    else:
        st.sidebar.info("Select or enter an export directory.")
        return None


# ---------------------------------------------------------------------------
# Tab 1 — Coverage
# ---------------------------------------------------------------------------

def _tab_coverage(X, mask, raw_names, human_names, y, use_human: bool):
    import plotly.graph_objects as go
    import pandas as pd

    st.subheader("Feature coverage")
    st.caption(
        "Percentage of studies in which a value was **observed** (mask = 1) for "
        "each feature. For lab features NaN counts as unobserved; for count "
        "features (diagnoses, drugs, …) a zero value also counts as unobserved. "
        "Sort and filter to find sparse features, or set a coverage floor before "
        "downstream analysis.")

    cov = _coverage(X, mask, raw_names)   # shape (n_features,)
    display_names = human_names if use_human else raw_names
    prefixes = [_prefix(r) for r in raw_names]

    cov_df = pd.DataFrame({
        "feature":  display_names,
        "raw":      raw_names,
        "prefix":   prefixes,
        "coverage": cov,
    })

    # ── filters ───────────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([3, 2, 2])
    sel_prefix = fc1.multiselect(
        "Feature type", sorted(set(prefixes)), default=sorted(set(prefixes)),
        key="cov_prefix",
        help="Filter to specific feature types by their column prefix.")
    cov_min = fc2.slider("Min coverage (%)", 0, 100, 0, key="cov_min",
                         help="Hide features below this observed-fraction threshold.")
    top_n = fc3.number_input("Show top N (by coverage)", min_value=10,
                              max_value=len(raw_names), value=min(100, len(raw_names)),
                              step=10, key="cov_top_n")

    filt = cov_df[
        cov_df["prefix"].isin(sel_prefix) & (cov_df["coverage"] >= cov_min)
    ].nlargest(int(top_n), "coverage")

    if filt.empty:
        st.info("No features match the current filters.")
        return

    # ── summary metrics ───────────────────────────────────────────────────
    mc = st.columns(4)
    mc[0].metric("Features shown",    f"{len(filt):,}")
    mc[1].metric("Median coverage",   f"{filt['coverage'].median():.1f}%")
    mc[2].metric("≥ 50% covered",
                 f"{(filt['coverage'] >= 50).sum():,}")
    mc[3].metric("< 10% covered",
                 f"{(filt['coverage'] < 10).sum():,}")

    # ── horizontal bar chart ──────────────────────────────────────────────
    # Sort ascending so the highest-coverage feature is at the top when
    # Plotly renders a horizontal bar chart (y-axis goes bottom to top).
    plot_df = filt.sort_values("coverage", ascending=True)

    # Color: single sequential hue, darker = higher coverage
    # Map coverage 0–100 linearly to opacity 0.2–1.0 on the base hue
    colors = [
        f"rgba(37,99,235,{max(0.18, v / 100):.2f})"
        for v in plot_df["coverage"]
    ]

    fig = go.Figure(go.Bar(
        x=plot_df["coverage"],
        y=plot_df["feature"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        customdata=plot_df[["raw", "prefix"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Coverage: %{x:.1f}%<br>"
            "Raw: %{customdata[0]}<br>"
            "Type: %{customdata[1]}<extra></extra>"
        ),
        text=[f"{v:.0f}%" for v in plot_df["coverage"]],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        xaxis=dict(title="% studies observed", range=[0, 108]),
        yaxis=dict(title="", tickfont=dict(size=11)),
        height=max(360, 18 * len(plot_df)),
        margin=dict(l=10, r=10, t=30, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch", key="cov_chart")

    # ── downloadable summary ──────────────────────────────────────────────
    st.download_button(
        "⬇️ Download coverage CSV",
        cov_df.sort_values("coverage", ascending=False).to_csv(index=False),
        file_name="feature_coverage.csv",
        mime="text/csv",
        key="cov_dl",
    )


# ---------------------------------------------------------------------------
# Tab 2 — Distributions
# ---------------------------------------------------------------------------

def _tab_distributions(X, mask, raw_names, human_names, y, use_human: bool):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.subheader("Feature value distributions")
    st.caption(
        "Histogram of observed values for selected features. "
        "**Apply mask** replaces unobserved values (NaN or 0) with NaN before "
        "plotting — the histogram then shows only truly measured values. "
        "**Split by label** overlays y=0 and y=1 distributions to highlight "
        "class-conditional differences.")

    display_names = human_names if use_human else raw_names
    name_to_idx = {n: i for i, n in enumerate(display_names)}

    # ── controls ──────────────────────────────────────────────────────────
    dc1, dc2, dc3 = st.columns([5, 2, 2])

    sel_feats = dc1.multiselect(
        f"Features (max {MAX_DIST_FEATURES})",
        options=display_names,
        default=display_names[:min(4, len(display_names))],
        max_selections=MAX_DIST_FEATURES,
        key="dist_feats",
    )
    apply_mask = dc2.checkbox(
        "Apply mask",
        value=True,
        key="dist_mask",
        help="When checked, values where mask=0 are excluded from the histogram. "
             "For lab features this removes NaN (unobserved) rows; "
             "for count features it removes zeros that represent 'not observed'.",
    )
    split_label = dc3.checkbox(
        "Split by label",
        value=bool(y is not None),
        key="dist_split",
        help="Overlay y=0 (blue) and y=1 (orange) distributions on the same axis. "
             "Only available when y.npy was loaded.",
        disabled=(y is None),
    )

    n_bins = st.slider("Histogram bins", 10, 200, 50, key="dist_bins")

    if not sel_feats:
        st.info("Select at least one feature above.")
        return

    n = len(sel_feats)
    cols_per_row = min(n, 2)
    rows = (n + cols_per_row - 1) // cols_per_row

    fig = make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=sel_feats,
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )

    for idx, fname in enumerate(sel_feats):
        row = idx // cols_per_row + 1
        col = idx %  cols_per_row + 1
        fi  = name_to_idx[fname]

        vals = X[:, fi].astype(float)
        m    = mask[:, fi].astype(bool)

        def _get_vals(label_mask=None):
            v = vals.copy()
            if apply_mask:
                v[~m] = np.nan
            if label_mask is not None:
                v = v[label_mask]
            return v[np.isfinite(v)]

        if split_label and y is not None:
            v0 = _get_vals(y == 0)
            v1 = _get_vals(y == 1)
            for vals_sub, color, name_sub in [
                (v0, _COLOR_Y0, "y=0"),
                (v1, _COLOR_Y1, "y=1"),
            ]:
                if len(vals_sub) == 0:
                    continue
                fig.add_trace(go.Histogram(
                    x=vals_sub, nbinsx=n_bins,
                    name=name_sub,
                    marker_color=color,
                    opacity=0.70,
                    showlegend=(idx == 0),
                    legendgroup=name_sub,
                    hovertemplate=f"{name_sub}<br>value: %{{x}}<br>count: %{{y}}<extra></extra>",
                ), row=row, col=col)
        else:
            v = _get_vals()
            if len(v) == 0:
                continue
            fig.add_trace(go.Histogram(
                x=v, nbinsx=n_bins,
                marker_color=_COV_COLOR,
                opacity=0.85,
                showlegend=False,
                hovertemplate="value: %{x}<br>count: %{y}<extra></extra>",
            ), row=row, col=col)

        # annotate observed count
        n_obs = int(m.sum())
        n_tot = len(m)
        fig.add_annotation(
            text=f"n={n_obs:,}/{n_tot:,} ({100*n_obs/n_tot:.0f}%)",
            xref=f"x{'' if (row==1 and col==1) else idx+1} domain",
            yref=f"y{'' if (row==1 and col==1) else idx+1} domain",
            x=0.98, y=0.97, xanchor="right", yanchor="top",
            showarrow=False,
            font=dict(size=10, color="gray"),
            row=row, col=col,
        )

    fig.update_layout(
        barmode="overlay",
        height=280 * rows,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    st.plotly_chart(fig, width="stretch", key="dist_chart")


# ---------------------------------------------------------------------------
# Tab 3 — Heatmap
# ---------------------------------------------------------------------------

def _tab_heatmap(X, mask, raw_names, human_names, y, use_human: bool):
    import plotly.graph_objects as go

    st.subheader("Matrix heatmap")
    st.caption(
        "A subsampled view of the feature matrix — at most "
        f"{MAX_HEATMAP_STUDIES:,} studies × {MAX_HEATMAP_FEATURES:,} features "
        "are shown (sampled uniformly at random). Rows = studies, columns = features. "
        "**Values** mode shows measured lab values (NaN cells are grey); "
        "**Mask** mode shows 1 (observed) / 0 (missing) — useful for visualising "
        "missingness patterns across the cohort.")

    display_names = human_names if use_human else raw_names

    # ── controls ──────────────────────────────────────────────────────────
    hc1, hc2, hc3 = st.columns(3)
    mode = hc1.radio("Display mode", ["Values", "Mask"], horizontal=True,
                     key="hm_mode")
    sel_prefix = hc2.multiselect(
        "Filter by type", sorted({_prefix(r) for r in raw_names}),
        default=sorted({_prefix(r) for r in raw_names}),
        key="hm_prefix",
    )
    sort_studies = hc3.selectbox(
        "Sort studies by",
        ["Random sample", "Label (y)", "Coverage (ascending)", "Coverage (descending)"],
        key="hm_sort",
    )

    # ── feature filter ────────────────────────────────────────────────────
    feat_mask_arr = np.array([_prefix(r) in sel_prefix for r in raw_names])
    feat_idx = np.where(feat_mask_arr)[0]

    if len(feat_idx) == 0:
        st.info("No features match the selected types.")
        return

    # subsample features
    if len(feat_idx) > MAX_HEATMAP_FEATURES:
        rng = np.random.default_rng(42)
        feat_idx = rng.choice(feat_idx, size=MAX_HEATMAP_FEATURES, replace=False)
        feat_idx = np.sort(feat_idx)
        st.caption(
            f"⚠️ Showing a random sample of {MAX_HEATMAP_FEATURES:,} features "
            f"out of {feat_mask_arr.sum():,} that match the filter.")

    # ── study ordering ────────────────────────────────────────────────────
    n_studies = X.shape[0]
    rng = np.random.default_rng(0)

    if sort_studies == "Random sample":
        study_idx = rng.choice(n_studies,
                               size=min(MAX_HEATMAP_STUDIES, n_studies),
                               replace=False)
    elif sort_studies == "Label (y)" and y is not None:
        ordered = np.argsort(y)
        step = max(1, len(ordered) // MAX_HEATMAP_STUDIES)
        study_idx = ordered[::step][:MAX_HEATMAP_STUDIES]
    else:
        # sort by per-study coverage (fraction of selected features observed)
        per_study_cov = mask[:, feat_idx].mean(axis=1)
        ordered = np.argsort(per_study_cov)
        if "descending" in sort_studies:
            ordered = ordered[::-1]
        step = max(1, len(ordered) // MAX_HEATMAP_STUDIES)
        study_idx = ordered[::step][:MAX_HEATMAP_STUDIES]

    study_idx = np.sort(study_idx)

    # ── build the plot matrix ─────────────────────────────────────────────
    if mode == "Values":
        Z = X[np.ix_(study_idx, feat_idx)].astype(float)
        # Count features (diag:, drug:, proc:, obs:, visit:) store 0 for
        # "not observed" rather than NaN, so every study would show a slight
        # negative z-score instead of a grey/absent cell. Replace those zeros
        # with NaN before z-scoring so they render as missing, matching labs.
        _COUNT_PREFIXES = {"diag:", "drug:", "proc:", "obs:", "visit:"}
        for _j, _raw in enumerate([raw_names[i] for i in feat_idx]):
            if _prefix(_raw) in _COUNT_PREFIXES:
                Z[Z[:, _j] == 0, _j] = np.nan
        # Standardise each column (z-score across observed values) so that
        # different units don't swamp each other on a shared colour scale.
        with np.errstate(invalid="ignore", divide="ignore", all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                col_mean = np.nanmean(Z, axis=0)
                col_std  = np.nanstd(Z,  axis=0)
            col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
            col_std  = np.where((col_std == 0) | np.isnan(col_std), 1.0, col_std)
            Z = (Z - col_mean) / col_std
        # Drop columns where every cell is NaN in this sample — they add no
        # information and fill the heatmap with blank space.
        _col_any = ~np.all(np.isnan(Z), axis=0)
        if not np.all(_col_any):
            _keep = np.where(_col_any)[0]
            n_dropped = Z.shape[1] - len(_keep)
            Z = Z[:, _keep]
            feat_idx = feat_idx[_keep]
            if n_dropped:
                st.caption(
                    f"ℹ️ {n_dropped:,} features hidden in Values mode "
                    "(no observations in this sample — use Mask mode to see coverage).")
        # Black = NaN / zero (matches plot_bgcolor).
        # Negative z: dark blue → light blue → black.
        # Positive z: black → red → yellow.
        colorscale = [
            [0.00, "#003080"],  # dark blue      (z = -3)
            [0.30, "#2196F3"],  # medium blue    (z = -1.2)
            [0.48, "#B3D9FF"],  # pale blue      (z = -0.12)
            [0.50, "#000000"],  # black          (z =  0)
            [0.52, "#FF6030"],  # dark orange-red(z = +0.12)
            [0.70, "#FF3300"],  # vivid red      (z = +1.2)
            [1.00, "#FFE000"],  # yellow         (z = +3)
        ]
        colorbar_title = "z-score"
        zmid = 0
        zmin, zmax = -3, 3
    else:
        Z = mask[np.ix_(study_idx, feat_idx)].astype(float)
        colorscale = [[0, "#F1F5F9"], [1, _COV_COLOR]]
        colorbar_title = "observed"
        zmid = None
        zmin, zmax = 0, 1

    # Use raw names as x-axis identifiers (guaranteed unique per column) so
    # Plotly never collapses two distinct features that share a human label
    # (e.g. multiple OMOP codes that map to "Hodgkin's disease"). Human names
    # are surfaced only in the hover tooltip via customdata.
    x_raw   = [raw_names[i]   for i in feat_idx]
    x_human = [human_names[i] for i in feat_idx]
    y_labels = [str(i) for i in study_idx]

    # Heatmap customdata must match Z's shape (n_studies × n_features); broadcast
    # the per-column human name list across every row so %{customdata} works.
    custom_2d = np.array(x_human, dtype=object)[np.newaxis, :].repeat(len(study_idx), axis=0)

    fig = go.Figure(go.Heatmap(
        z=Z,
        x=x_raw,
        y=y_labels,
        colorscale=colorscale,
        zmid=zmid,
        zmin=zmin, zmax=zmax,
        colorbar=dict(title=colorbar_title, thickness=14),
        customdata=custom_2d,
        hovertemplate="study %{y}<br>%{customdata}<br>value: %{z:.3g}<extra></extra>",
    ))
    # Show human-readable tick labels on the axis, but keep unique raw values
    # as the underlying x — this prevents Plotly from merging identically-
    # labelled columns while still giving the user a readable axis.
    tick_font_size = max(7, min(11, 900 // max(len(feat_idx), 1)))
    fig.update_layout(
        xaxis=dict(
            title="Features",
            tickangle=-45,
            tickfont=dict(size=tick_font_size),
            tickmode="array",
            tickvals=x_raw,
            ticktext=x_human,
            showgrid=False,
        ),
        yaxis=dict(title="Study index", showgrid=False, autorange="reversed"),
        height=max(400, min(900, 2 * len(study_idx))),
        margin=dict(l=10, r=10, t=30, b=80),
        plot_bgcolor="black",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch", key="hm_chart")

    mc = st.columns(3)
    mc[0].metric("Studies shown",  f"{len(study_idx):,} / {n_studies:,}")
    mc[1].metric("Features shown", f"{len(feat_idx):,}")
    mc[2].metric("Overall coverage",
                 f"{mask[np.ix_(study_idx, feat_idx)].mean()*100:.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Feature Matrix Viewer", layout="wide")
    st.title("🔬 Feature Matrix Viewer")
    st.caption(
        "Explore an exported feature matrix (X.npy / X_mask.npy) without "
        "loading a full extraction. Point it at any directory written by "
        "Tab 4 · Export in app_feature_extraction.py.")

    export_dir = _dir_picker(st)
    if export_dir is None:
        st.info("Select an export directory in the sidebar to begin.")
        return

    # ── load ─────────────────────────────────────────────────────────────
    try:
        X, mask, y, tte, ev, raw_names, human_names, meta, spec = load_export(export_dir)
    except Exception as e:
        st.error(f"Failed to load export: {e}")
        return

    # ── sidebar metadata summary ─────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Loaded")
    st.sidebar.metric("Studies",  f"{X.shape[0]:,}")
    st.sidebar.metric("Features", f"{X.shape[1]:,}")
    if y is not None:
        st.sidebar.metric("Event rate", f"{float(y.mean()):.4f}")
    if spec:
        st.sidebar.caption(
            f"Task: **{spec.get('task', '?')}**  \n"
            f"Anchor: {spec.get('anchor', '?')}  \n"
            f"Windows: {spec.get('windows_days', '?')}")

    use_human = st.sidebar.checkbox(
        "Human-readable names",
        value=True,
        help="Show resolved concept names (from concept.csv) instead of raw "
             "OMOP codes. Falls back to raw names if not available.",
    )

    # ── tabs ─────────────────────────────────────────────────────────────
    tab_cov, tab_dist, tab_hm = st.tabs([
        "📊 Coverage",
        "📈 Distributions",
        "🟦 Heatmap",
    ])

    with tab_cov:
        _tab_coverage(X, mask, raw_names, human_names, y, use_human)

    with tab_dist:
        _tab_distributions(X, mask, raw_names, human_names, y, use_human)

    with tab_hm:
        _tab_heatmap(X, mask, raw_names, human_names, y, use_human)


if __name__ == "__main__":
    main()
