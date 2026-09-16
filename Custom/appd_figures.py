"""
appd_figures.py
---------------
Plotly figure builders for the INSPECT EHR Feature Extraction app
(app_feature_extraction.py).

Three visualisations live here:

  _build_timeline_figure              — per-study multi-lane swimlane
  _build_cohort_trajectory_heatmap    — population-level (panel × bin) heatmap
  _build_cohort_bubble_timeline       — same data, bubble-area encoding

All functions are pure: they take a DataFrame and a handful of scalar
parameters, return a plotly Figure, and import plotly / pandas / numpy
lazily so this module can be imported without those packages installed.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Metric label lookup — shared by the heatmap and bubble builders and by the
# Streamlit UI that formats the colour-bar title and selectbox labels.
# ---------------------------------------------------------------------------

_TRAJECTORY_METRIC_LABELS = {
    "pct_studies": "% of studies with an event",
    "n_studies":   "# studies with an event",
    "n_events":    "total event count",
}


def _build_timeline_figure(events_df: "pd.DataFrame", anchor_time, title: str = ""):
    """Multi-lane (one row per clinical panel) Plotly timeline for a single
    study's events.

    events_df needs columns: event_datetime (datetime64), panel (str),
    optionally event_type / concept_name for the hover tooltip. Marks T0
    (anchor_time) with a vertical line and gives the x-axis native
    hour/day/week/month/year zoom via Plotly's range selector — "week" isn't
    a native Plotly step type, so it's approximated as 7 days. Because the
    data only spans [anchor − window, anchor] (see query_events_stack), T0
    sits at the data's right edge, which is exactly where Plotly's
    stepmode="backward" buttons zoom from — no extra alignment needed.
    """
    import plotly.graph_objects as go
    import pandas as pd

    fig = go.Figure()
    panels = sorted(events_df["panel"].unique()) if len(events_df) else []
    concept_name = events_df.get("concept_name", pd.Series([""] * len(events_df)))
    event_type   = events_df.get("event_type",   pd.Series([""] * len(events_df)))

    for panel in panels:
        mask = events_df["panel"] == panel
        sub  = events_df[mask]
        fig.add_trace(go.Scatter(
            x=sub["event_datetime"],
            y=[panel] * len(sub),
            mode="markers",
            name=panel,
            marker=dict(size=9),
            text=concept_name[mask] if hasattr(concept_name, "__getitem__") else None,
            customdata=event_type[mask] if hasattr(event_type, "__getitem__") else None,
            hovertemplate="<b>%{text}</b><br>%{x}<br>type: %{customdata}<extra></extra>",
        ))

    if anchor_time is not None:
        fig.add_vline(x=pd.Timestamp(anchor_time), line_dash="dash",
                      line_color="crimson",
                      annotation_text="T0 (CTPA)", annotation_position="top")

    fig.update_layout(
        title=title,
        xaxis=dict(
            type="date",
            rangeselector=dict(buttons=[
                dict(count=1, label="1h", step="hour",  stepmode="backward"),
                dict(count=1, label="1d", step="day",   stepmode="backward"),
                dict(count=7, label="1w", step="day",   stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year",  stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(categoryorder="array", categoryarray=panels[::-1]),
        height=max(320, 45 * max(len(panels), 1)),
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _build_cohort_trajectory_heatmap(traj_df: "pd.DataFrame", metric: str = "pct_studies",
                                      bin_days: int = 7, title: str = "",
                                      x_axis_label: str = "Time before CTPA (T0, right edge)",
                                      reverse_x: bool = True):
    """Population-level heatmap: one row per clinical panel, one column per
    time bin relative to CTPA, color = the chosen metric. traj_df is
    appd_route_b_labs.build_cohort_trajectory()'s output — already reduced
    to (panels × bins) rows inside DuckDB, so this never touches per-event
    data regardless of cohort size.

    Panels are ordered by total activity (sum of the chosen metric across
    all bins), most active at the top — for an "at a glance" overview,
    that's more useful than alphabetical. T0 (the bin closest to anchor) is
    placed on the right edge, matching _build_timeline_figure's convention
    for the individual viewer above.
    """
    import plotly.graph_objects as go

    if traj_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title or "No events matched this cohort/window.")
        return fig

    pivot = traj_df.pivot_table(
        index="panel", columns="bin_index", values=metric, fill_value=0, aggfunc="sum")
    all_bins = range(int(traj_df["bin_index"].min()), int(traj_df["bin_index"].max()) + 1)
    pivot = pivot.reindex(columns=all_bins, fill_value=0)
    # most-active-first — pivot_table already sorted the index alphabetically;
    # re-sort by row totals instead.
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    bin_start = pivot.columns.to_numpy() * bin_days      # 0-indexed day start
    bin_end   = (pivot.columns.to_numpy() + 1) * bin_days - 1  # inclusive end
    if reverse_x:
        # CTPA-anchored: bins count backwards from T0, so show as "-Xd"
        x_labels = [f"-{e+1}d..-{s+1}d" for s, e in zip(bin_start, bin_end)]
    else:
        # Admission-anchored: bins count forward from day 0
        if bin_days == 1:
            x_labels = [f"Day {s}" for s in bin_start]
        else:
            x_labels = [f"Day {s}–{e}" for s, e in zip(bin_start, bin_end)]

    metric_label = _TRAJECTORY_METRIC_LABELS.get(metric, metric)

    # Plotly renders every category on a category axis by default — with a
    # wide lookback and a narrow bin width that's hundreds of overlapping,
    # unreadable tick labels (this is what a garbled/smeared x-axis is:
    # too many bins for the pixel width, not a rendering bug). Thin ticks
    # to a legible count; the underlying data/columns are unaffected, only
    # which labels get text drawn under them.
    _MAX_TICKS = 30
    step = max(1, -(-len(x_labels) // _MAX_TICKS))  # ceil division
    tick_labels = x_labels[::step]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.to_numpy(),
        x=x_labels,
        y=pivot.index.tolist(),
        colorscale=[
            [0.00, "#000000"],  # black        (0 — no events)
            [0.20, "#7F0000"],  # deep red
            [0.40, "#CC0000"],  # red
            [0.60, "#FF6600"],  # orange
            [0.80, "#FFD700"],  # gold
            [1.00, "#FFFF99"],  # pale yellow  (max)
        ],
        zmin=0,               # anchor black to true zero, not data minimum
        colorbar=dict(title=metric_label),
        hovertemplate="%{y}<br>%{x} before CTPA<br>" + metric_label + ": %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(
            title=x_axis_label,
            categoryorder="array", categoryarray=x_labels,
            autorange="reversed" if reverse_x else True,
            tickmode="array", tickvals=tick_labels, ticktext=tick_labels,
            tickangle=-45,
        ),
        yaxis=dict(title="Clinical panel", categoryorder="array",
                   categoryarray=pivot.index.tolist()[::-1]),
        height=max(400, 32 * max(len(pivot.index), 1)),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig


def _build_cohort_bubble_timeline(traj_df: "pd.DataFrame", metric: str = "pct_studies",
                                   title: str = ""):
    """Same (panel × bin) summary as _build_cohort_trajectory_heatmap, drawn
    as bubbles instead of heatmap cells — one lane per panel (visually close
    to the individual live viewer's swimlanes), each populated bin rendered
    as a circle whose *area* scales with the chosen metric, empty bins drawn
    as nothing at all rather than a pale cell. This is the "many small
    events agglomerate into one bigger marker" reading the heatmap's color
    scale doesn't give you as directly.

    Unlike the heatmap, x is a real numeric axis (negative days before
    anchor, 0 = CTPA) rather than a categorical bin-label axis — dragging to
    zoom is native Plotly, client-side, and free: it narrows the visible
    time range immediately with no rerun. What it does *not* do on its own
    is shrink the bin width for the newly-visible range — Plotly has no
    built-in way to call back into Python/DuckDB on a zoom gesture, and nothing
    in this stack listens for one, so genuine "zoom in and bins automatically
    subdivide into finer bubbles" isn't something this renders — that would
    need a JS-bridge component (e.g. streamlit-plotly-events) capturing the
    post-zoom x-range and re-invoking build_cohort_trajectory with a smaller
    bin_days. What's here instead: zoom narrows the view, and the bin-width
    control next to this chart lets you manually shrink bins for what's now
    visible — two deliberate actions standing in for one automatic one.
    """
    import plotly.graph_objects as go
    import numpy as np

    if traj_df.empty:
        fig = go.Figure()
        fig.update_layout(title=title or "No events matched this cohort/window.")
        return fig

    df = traj_df[traj_df[metric] > 0].copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=title or f"No non-zero {metric} values to plot.")
        return fig

    # most-active-first, same convention as the heatmap — top lane = busiest panel
    panel_order = (df.groupby("panel")[metric].sum()
                     .sort_values(ascending=False).index.tolist())
    df["x"] = -((df["bin_start_days"] + df["bin_end_days"]) / 2.0)  # negative = before CTPA

    metric_label = _TRAJECTORY_METRIC_LABELS.get(metric, metric)
    values = df[metric].to_numpy(dtype=float)
    max_val = values.max() if len(values) else 1.0
    sizeref = 2.0 * max_val / (44.0 ** 2)  # area-proportional bubbles, ~44px max diameter

    fig = go.Figure(data=go.Scatter(
        x=df["x"], y=df["panel"],
        mode="markers",
        marker=dict(
            size=values, sizemode="area", sizeref=sizeref if sizeref > 0 else 1, sizemin=4,
            color=values, colorscale="YlOrRd",
            colorbar=dict(title=metric_label),
            line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
        ),
        customdata=np.stack([df["bin_start_days"], df["bin_end_days"],
                              df["n_events"], df["n_studies"]], axis=-1),
        hovertemplate=(
            "%{y}<br>%{customdata[0]:.0f}–%{customdata[1]:.0f}d before CTPA"
            "<br>" + metric_label + ": %{marker.color:.3g}"
            "<br>n_events=%{customdata[2]:.0f} · n_studies=%{customdata[3]:.0f}"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray",
                   annotation_text="CTPA (T0)", annotation_position="top")
    fig.update_layout(
        title=title,
        xaxis=dict(title="Days before CTPA (T0 = 0, dashed line)"),
        yaxis=dict(title="Clinical panel", categoryorder="array",
                   categoryarray=panel_order[::-1]),
        height=max(400, 40 * max(len(panel_order), 1)),
        margin=dict(l=10, r=10, t=60, b=40),
    )
    return fig
