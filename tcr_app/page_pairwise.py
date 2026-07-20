"""Pairwise comparison page — compare clonotype abundances between two flexibly-defined samples.

Each sample is defined by selecting individuals, organs, cell types, and chain independently.
Abundances are summed across the selected groups for each clonotype, then plotted on a scatterplot.
"""

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tcr_app.core import render_plot_download_buttons

try:
    from scipy.stats import pearsonr, spearmanr

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def run_pairwise_comparison_page(df: pd.DataFrame) -> None:
    """Render the pairwise comparison view with scatterplot."""
    st.title("TCR Abundance Explorer")
    st.subheader("Pairwise Comparison")

    st.markdown(
        """
    Define two samples independently — pick individuals, organs, cell types, and chain for each.
    Clonotype abundances are summed within each sample, then plotted on a scatterplot.
    """
    )

    with st.sidebar:
        _render_sample_sidebar(df, "A")
        st.divider()
        _render_sample_sidebar(df, "B")
        st.divider()

        st.header("Plot settings")
        log_scale = st.checkbox("Log10 scale", value=False)
        top_n = st.number_input(
            "Top N clonotypes (0 = all)",
            min_value=0,
            max_value=10000,
            value=0,
            step=1,
        )
        min_abundance = st.number_input(
            "Min abundance (hide if both below)",
            min_value=0.0,
            value=0.0,
            step=0.1,
        )
        show_diagonal = st.checkbox("Show y=x line", value=True)

    sample_a = _get_sample_filters("A")
    sample_b = _get_sample_filters("B")

    if not sample_a["mice"] or not sample_b["mice"]:
        st.warning("Select at least one individual for both Sample A and Sample B.")
        st.stop()

    with st.spinner("Aggregating sample data…"):
        grouped_a = _aggregate_sample(df, sample_a)
        grouped_b = _aggregate_sample(df, sample_b)

    if grouped_a.empty and grouped_b.empty:
        st.warning("No data found for the selected filters.")
        st.stop()

    combined = pd.DataFrame({"abundance_A": grouped_a, "abundance_B": grouped_b}).fillna(0)
    combined.index.name = "clonotype"
    combined = combined.reset_index()

    if top_n > 0:
        combined["_max_ab"] = combined[["abundance_A", "abundance_B"]].max(axis=1)
        combined = combined.nlargest(top_n, "_max_ab").drop(columns="_max_ab")

    if min_abundance > 0:
        combined = combined[
            (combined["abundance_A"] >= min_abundance)
            | (combined["abundance_B"] >= min_abundance)
        ]

    n_total = len(combined)
    shared_mask = (combined["abundance_A"] > 0) & (combined["abundance_B"] > 0)
    only_a_mask = (combined["abundance_A"] > 0) & (combined["abundance_B"] == 0)
    only_b_mask = (combined["abundance_B"] > 0) & (combined["abundance_A"] == 0)
    n_shared = shared_mask.sum()
    n_only_a = only_a_mask.sum()
    n_only_b = only_b_mask.sum()

    desc_a = _describe_sample(sample_a)
    desc_b = _describe_sample(sample_b)

    _render_metrics(
        n_total, n_shared, n_only_a, n_only_b,
        combined[shared_mask] if n_shared > 0 else None,
    )
    st.caption(f"**Sample A:** {desc_a}  |  **Sample B:** {desc_b}")

    combined = _categorize_clonotypes(combined)
    fig = _build_scatter(combined, desc_a, desc_b, log_scale, show_diagonal)
    st.plotly_chart(fig, width='stretch')

    render_plot_download_buttons(
        fig,
        base_filename="pairwise_comparison",
        key_prefix="pairwise",
        data=combined,
        data_filename="pairwise_comparison.csv",
    )

    with st.expander("Per-clonotype abundances"):
        display = combined.set_index("clonotype")
        st.dataframe(display, width='stretch')


def _render_sample_sidebar(df: pd.DataFrame, label: str) -> None:
    """Render one sample's filter block in the sidebar."""
    llabel = label.lower()
    chain = st.selectbox(
        f"Chain {label}",
        sorted(df["chain"].unique()),
        key=f"pairwise_chain_{llabel}",
    )

    mouse_options = sorted(df["mouse"].unique())
    mice = st.multiselect(
        f"Individuals {label}",
        mouse_options,
        default=mouse_options[:1] if mouse_options else [],
        key=f"pairwise_mice_{llabel}",
    )

    if mice:
        organ_options = sorted(
            df[(df["mouse"].isin(mice)) & (df["chain"] == chain)]["organ"].unique()
        )
    else:
        organ_options = sorted(df["organ"].unique())
    organs = st.multiselect(
        f"Organs {label}",
        organ_options,
        default=organ_options[:1] if organ_options else [],
        key=f"pairwise_organs_{llabel}",
    )

    if mice and organs:
        cell_options = sorted(
            df[
                (df["mouse"].isin(mice))
                & (df["organ"].isin(organs))
                & (df["chain"] == chain)
            ]["cell_type"].unique()
        )
    else:
        cell_options = sorted(df["cell_type"].unique())
    cells = st.multiselect(
        f"Cell types {label}",
        cell_options,
        default=cell_options[:1] if cell_options else [],
        key=f"pairwise_cells_{llabel}",
    )


def _get_sample_filters(label: str) -> dict[str, Any]:
    """Read the current filter values from session state for the given sample."""
    llabel = label.lower()
    return {
        "chain": st.session_state.get(f"pairwise_chain_{llabel}"),
        "mice": st.session_state.get(f"pairwise_mice_{llabel}", []),
        "organs": st.session_state.get(f"pairwise_organs_{llabel}", []),
        "cells": st.session_state.get(f"pairwise_cells_{llabel}", []),
    }


def _aggregate_sample(df: pd.DataFrame, sample: dict) -> pd.Series:
    """Sum abundance per clonotype across the sample's filter scope."""
    if not sample["mice"]:
        return pd.Series(dtype=float)
    subset = df[df["mouse"].isin(sample["mice"]) & (df["chain"] == sample["chain"])]
    if sample["organs"]:
        subset = subset[subset["organ"].isin(sample["organs"])]
    if sample["cells"]:
        subset = subset[subset["cell_type"].isin(sample["cells"])]
    return subset.groupby("clonotype", as_index=True)["abundance"].sum()


def _describe_sample(sample: dict, max_items: int = 3) -> str:
    """Build a human-readable label listing actual selected values."""
    parts = [sample["chain"]]
    if sample["mice"]:
        parts.append(_join_values(sample["mice"], max_items))
    if sample["organs"]:
        parts.append(_join_values(sample["organs"], max_items))
    if sample["cells"]:
        parts.append(_join_values(sample["cells"], max_items))
    return " | ".join(parts)


def _join_values(values: list[str], max_items: int) -> str:
    if len(values) <= max_items:
        return ", ".join(values)
    return ", ".join(values[:max_items]) + f", +{len(values) - max_items} more"


def _render_metrics(
    n_total: int,
    n_shared: int,
    n_only_a: int,
    n_only_b: int,
    shared_df: pd.DataFrame | None,
) -> None:
    """Display summary metric boxes."""
    cols = st.columns(5)
    cols[0].metric("Clonotypes", f"{n_total:,}")
    cols[1].metric("Shared", f"{n_shared:,}")
    cols[2].metric("Only in A", f"{n_only_a:,}")
    cols[3].metric("Only in B", f"{n_only_b:,}")

    corr = ""
    if shared_df is not None and len(shared_df) >= 3 and SCIPY_AVAILABLE:
        try:
            x = shared_df["abundance_A"].values
            y = shared_df["abundance_B"].values
            r_p, _ = pearsonr(x, y)
            r_s, _ = spearmanr(x, y)
            corr = f"r={r_p:.3f}  ρ={r_s:.3f}"
        except Exception:
            pass
    cols[4].metric("Correlation", corr or "—")


def _categorize_clonotypes(combined: pd.DataFrame) -> pd.DataFrame:
    """Add a `_category` column for coloring the scatterplot.

    Categories (priority order):
    - "Top 100 A & B" — in top 100 abundance of both samples
    - "Top 100 A"     — in top 100 abundance of Sample A only
    - "Top 100 B"     — in top 100 abundance of Sample B only
    - "Zero"          — abundance is 0 in both samples
    - "Other"         — everything else
    """
    result = combined.copy()

    if result.empty:
        result["_category"] = pd.Series(dtype=str)
        return result

    top_a = set(
        result.nlargest(100, "abundance_A", keep="all").index
    )
    top_b = set(
        result.nlargest(100, "abundance_B", keep="all").index
    )

    def _assign_category(idx: int, row: pd.Series) -> str:
        in_a = idx in top_a
        in_b = idx in top_b
        if in_a and in_b:
            return "Top 100 A & B"
        if in_a:
            return "Top 100 A"
        if in_b:
            return "Top 100 B"
        if row["abundance_A"] == 0 and row["abundance_B"] == 0:
            return "Zero"
        return "Other"

    result["_category"] = pd.Series(
        {idx: _assign_category(idx, result.loc[idx]) for idx in result.index},
        dtype=str,
    )
    return result


def _compute_pseudo_zero(combined: pd.DataFrame) -> float:
    non_zero = combined[["abundance_A", "abundance_B"]].replace(0, pd.NA).stack()
    smallest = non_zero.min(skipna=True)
    if pd.isna(smallest):
        return 1e-6
    return smallest / 10


def _build_tick_values(pseudo_zero: float) -> tuple[list[float], list[str]]:
    vals: list[float] = [pseudo_zero]
    texts: list[str] = ["0"]
    v = pseudo_zero * 10
    max_v = 1e4
    while v <= max_v:
        vals.append(v)
        texts.append(f"1e{int(round(math.log10(v)))}")
        v *= 10
    return vals, texts


def _build_scatter(
    combined: pd.DataFrame,
    desc_a: str,
    desc_b: str,
    log_scale: bool,
    show_diagonal: bool,
) -> go.Figure:
    """Build a publication-quality pairwise scatterplot."""
    cat_colors: dict[str, str] = {
        "Top 100 A & B": "#3b2e5c",
        "Top 100 A": "#c44536",
        "Top 100 B": "#2a6f97",
        "Zero": "#d3d3d3",
        "Other": "#bcc3cd",
    }
    cat_order = ["Top 100 A & B", "Top 100 A", "Top 100 B", "Other", "Zero"]

    plot_data = combined.copy()

    pseudo_zero = _compute_pseudo_zero(combined)
    if log_scale:
        plot_data["abundance_A"] = plot_data["abundance_A"].replace(0, pseudo_zero)
        plot_data["abundance_B"] = plot_data["abundance_B"].replace(0, pseudo_zero)

    has_aggregate = plot_data["_category"].isin(cat_order).any()

    fig = go.Figure()

    if has_aggregate:
        for cat in cat_order:
            sub = plot_data[plot_data["_category"] == cat]
            if sub.empty:
                continue
            color = cat_colors[cat]
            is_zero_cat = cat == "Zero"

            nonzero = sub[
                (combined.loc[sub.index, "abundance_A"] > 0) & (combined.loc[sub.index, "abundance_B"] > 0)
            ]
            zeroed = sub[
                (combined.loc[sub.index, "abundance_A"] == 0) | (combined.loc[sub.index, "abundance_B"] == 0)
            ]

            if not nonzero.empty:
                fig.add_trace(go.Scatter(
                    x=nonzero["abundance_A"],
                    y=nonzero["abundance_B"],
                    mode="markers",
                    marker=dict(
                        size=3 if is_zero_cat else 6,
                        color="rgba(0,0,0,0)" if is_zero_cat else color,
                        line=dict(width=0.5, color="#b0b0b0" if is_zero_cat else "rgba(0,0,0,0.4)"),
                        opacity=0.4 if is_zero_cat else 0.85,
                    ),
                    name=cat,
                    legendgroup=cat,
                    showlegend=True,
                    hovertemplate=(
                        f"<b>%{{customdata[0]}}</b><br>"
                        f"Sample A: %{{x:.4g}}<br>"
                        f"Sample B: %{{y:.4g}}<extra></extra>"
                    ),
                    customdata=np.stack([nonzero["clonotype"]], axis=1),
                ))

            if not zeroed.empty:
                fig.add_trace(go.Scatter(
                    x=zeroed["abundance_A"],
                    y=zeroed["abundance_B"],
                    mode="markers",
                    marker=dict(
                        size=3 if is_zero_cat else 4,
                        color="rgba(0,0,0,0)",
                        line=dict(width=0.5 if is_zero_cat else 0.8, color="#b0b0b0" if is_zero_cat else color),
                        opacity=0.4 if is_zero_cat else 0.6,
                    ),
                    name=cat,
                    legendgroup=cat,
                    showlegend=nonzero.empty,
                    hovertemplate=(
                        f"<b>%{{customdata[0]}}</b><br>"
                        f"Sample A: %{{customdata[1]:.4g}}<br>"
                        f"Sample B: %{{customdata[2]:.4g}}<extra></extra>"
                    ),
                    customdata=np.stack([
                        zeroed["clonotype"],
                        combined.loc[zeroed.index, "abundance_A"],
                        combined.loc[zeroed.index, "abundance_B"],
                    ], axis=1),
                ))
    else:
        fig.add_trace(go.Scatter(
            x=plot_data.get("abundance_A", []),
            y=plot_data.get("abundance_B", []),
            mode="markers",
        ))

    fig.update_layout(
        template="plotly_white",
        height=600,
        font_family="Arial, Helvetica, sans-serif",
        font_size=12,
        title_font_size=16,
        legend_title_text="",
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bordercolor="rgba(0,0,0,0)",
            font_size=11,
        ),
        margin=dict(l=60, r=120, t=40, b=60),
    )
    fig.update_xaxes(
        title_text=f"Sample A — {desc_a}",
        title_font_size=14,
        showline=True,
        linewidth=1,
        linecolor="black",
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#e0e0e0",
        ticks="outside",
        ticklen=4,
        tickfont_size=12,
    )
    fig.update_yaxes(
        title_text=f"Sample B — {desc_b}",
        title_font_size=14,
        showline=True,
        linewidth=1,
        linecolor="black",
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#e0e0e0",
        ticks="outside",
        ticklen=4,
        tickfont_size=12,
    )

    if log_scale:
        fig.update_xaxes(type="log")
        tickvals, ticktext = _build_tick_values(pseudo_zero)
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext)
        fig.update_yaxes(tickvals=tickvals, ticktext=ticktext)

    if show_diagonal:
        max_val = max(plot_data["abundance_A"].max(), plot_data["abundance_B"].max())
        fig.add_shape(
            type="line",
            x0=pseudo_zero if log_scale else 0,
            y0=pseudo_zero if log_scale else 0,
            x1=max_val,
            y1=max_val,
            line={"color": "#333333", "width": 1.5, "dash": "dot"},
        )

    shared = combined[(combined["abundance_A"] > 0) & (combined["abundance_B"] > 0)]
    if len(shared) >= 3 and SCIPY_AVAILABLE:
        try:
            r_p, _ = pearsonr(shared["abundance_A"], shared["abundance_B"])
            r_s, _ = spearmanr(shared["abundance_A"], shared["abundance_B"])
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.05,
                y=0.95,
                text=f"r = {r_p:.3f} ρ = {r_s:.3f}",
                showarrow=False,
                font=dict(size=12, color="#333333"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#cccccc",
                borderwidth=0.5,
                borderpad=4,
            )
        except Exception:
            pass

    return fig
