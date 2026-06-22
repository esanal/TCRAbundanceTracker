"""Cohort summary page — aggregate analysis across all subjects.

Features:
- Individual metrics table
- CD4/CD8 line plots across organ/cell per subject
- Cosine similarity and Pearson correlation matrices across subjects
- Per-subject clonotype presence grids (Kiki plots)
- Across-subject median count bar charts
- VDJdb enrichment pipeline
"""

import math
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tcr_app.core import (
    _build_summary_lineage_abundance_figure,
    build_aggregate_row_count_figure,
    build_clonotype_presence_count_histogram,
    build_clonotype_presence_grid_dataframe,
    build_clonotype_presence_grid_figure,
    build_highlighted_tick_labels,
    calculate_mouse_correlation,
    calculate_mouse_cosine_similarity,
    classify_cd4_cd8,
    enrich_clonotypes_with_vdjdb,
    get_organ_cell_order,
    nucleotide_to_amino_acid_cdr3,
    render_clonotype_presence_grid_legend,
    render_plot_download_buttons,
    summarize_clonotype_presence_grid_counts,
)


def run_summary_all_page(df: pd.DataFrame) -> None:
    """Render the cohort-level aggregate analysis view."""
    st.title("TCR Abundance Explorer")
    st.subheader("Cohort summary")
    st.markdown(
        """
    Aggregate of all individuals (or mice) to show largest n (selected below) clonotypes across the organ|subset groups.
    """
    )
    
    with st.sidebar:
        st.header("Filters")
        chain_selected = st.selectbox(
            "Chain", sorted(df["chain"].unique()))

        organ_selected = st.multiselect(
            "Organ", sorted(df["organ"].unique()), default=sorted(df["organ"].unique())
        )
        cell_selected = st.multiselect(
            "Cell type",
            sorted(df["cell_type"].unique()),
            default=sorted(df["cell_type"].unique()),
        )

    filtered = df[
        (df["organ"].isin(organ_selected))
        & (df["cell_type"].isin(cell_selected))
        & (df["chain"] == chain_selected)
    ].copy()

    if filtered.empty:
        st.warning("No data match the selected filters.")
        st.stop()

    total_mice = filtered["mouse"].nunique()

    metrics_cols = st.columns(1)
    metrics_cols[0].metric("Individuals", total_mice)

    if total_mice == 0:
        st.warning("No individual records available to summarize.")
        return

    chain_counts = filtered["chain"].value_counts()
    if not chain_counts.empty:
        sorted_chains = sorted(chain_counts.items(), key=lambda item: -item[1])
        display_items = sorted_chains[:3]  # Display max 3
        extra = len(sorted_chains) - len(display_items)
        summary_text = ", ".join(f"{chain} ({count})" for chain, count in display_items)
        if extra > 0:
            summary_text = f"{summary_text}, +{extra} more"
        st.caption(f"Chain counts: {summary_text}")
    
    mouse_summary = (
        filtered.groupby("mouse", as_index=False)
        .agg(
            total_clonotypes=("clonotype", "nunique"),
            organs=("organ", "nunique"),
            cell_types=("cell_type", "nunique"),
        )
    )

    st.markdown("### Individual metrics")
    st.dataframe(
        mouse_summary[
            [
                "mouse",
                "total_clonotypes",
                "organs",
                "cell_types",
            ]
        ],
        width="content",
    )


    st.subheader("Abundance by Organ/Cell (Top Clonotypes)")
    
    select_cols = st.columns(2)
    organ_cell_options = sorted(filtered["organ_cell"].unique())
    with select_cols[0]:
        subset_selected = st.selectbox(
            "Rank top clonotypes by organ/cell combination",
            organ_cell_options,
            index=0,
            help="Select a subset whose top clones across all individuals will be shown.",
        )
    max_clonotypes = (
            filtered[filtered["organ_cell"] == subset_selected]["clonotype"].nunique()
    )
    max_clonotypes = max(1, max_clonotypes)
    
    with select_cols[1]:
        top_n = st.number_input(
            "Select number of largest clonotypes to display",
            min_value=1,
            max_value=max_clonotypes,
            value=min(10, max_clonotypes),
            step=1,
            help="Choose how many of the most abundant clonotypes should appear on the CD4/CD8 line plots.",
        )

    clono_totals = (
        filtered[filtered["organ_cell"] == subset_selected]
        .sort_values(
            ["mouse", "abundance"],
            ascending=[True, False],
            kind="mergesort",
        )
    )
    scale_cols = st.columns(2)
    with scale_cols[0]:
        log_axis_summary = st.checkbox(
            "Log10 scale",
            value=True,
            help=(
                "Display values on a log10 axis. Missing clone values are shown with a "
                "dynamic pseudo-0 per mouse and organ|cell"
            ),
        )
    with scale_cols[1]:
        normalize_topn_summary = st.checkbox(
            f"Normalize by mouse+organ/cell top-{int(top_n)} denominator",
            value=False,
            help=(
                f"For each mouse and organ/cell group, divide abundance by that group's sum "
                f"of the top-{int(top_n)} abundances, then multiply by 100."
            ),
        )
    if log_axis_summary:
        st.caption(
            "Open-circle markers indicate imputed 0 values per organ|cell (pseudo-0 used for Log10 display)."
        )
    sort_organ_cell = st.selectbox("Sort organ/cell axis by", ["organ", "trm"])

    filtered["cd_group"] = filtered["cell_type"].apply(classify_cd4_cd8)
    filtered["norm_topN"] = (
        filtered.groupby(["mouse", "organ_cell"])["abundance"]
        .transform(lambda x: x.nlargest(int(top_n)).sum())
    )
    topClones = _render_summary_subset_top_clonotype_section(
        filtered=filtered,
        subset_selected=subset_selected,
        top_n=int(top_n),
        sort_organ_cell=sort_organ_cell,
        log_axis_summary=log_axis_summary,
        normalize_topn_summary=normalize_topn_summary,
    )

    cosine_by_lineage: Dict[str, pd.DataFrame] = {}
    correlation_by_lineage: Dict[str, pd.DataFrame] = {}
    for lineage in ["CD4", "CD8"]:
        lineage_df = topClones[topClones["cd_group"] == lineage].copy()
        if lineage_df.empty:
            continue
        lineage_cosine = calculate_mouse_cosine_similarity(
            lineage_df, value_col="abundance_for_metric"
        )
        lineage_correlation = calculate_mouse_correlation(
            lineage_df, value_col="abundance_for_metric"
        )
        if not lineage_cosine.empty:
            cosine_by_lineage[lineage] = lineage_cosine
        if not lineage_correlation.empty:
            correlation_by_lineage[lineage] = lineage_correlation

    summary_metric_view = st.selectbox(
        "Similarity analysis to show",
        ("Cosine", "Correlation"),
        index=0,
        help=(
            "Choose which similarity metric section to display. "
            "Both analyses are still computed and available."
        ),
    )

    if summary_metric_view == "Cosine":
        st.subheader("Cosine Similarity Across Individuals (Top Clones)")
        st.caption(
            "Cosine similarity is rank-based: each individual's vector uses "
            "(clonotype rank, organ/cell) abundance features from the CD4/CD8 plots."
        )
        if not cosine_by_lineage:
            st.info("At least two individuals with non-empty vectors are required to compute cosine similarity.")
        else:
            for lineage, similarity_df in cosine_by_lineage.items():
                st.markdown(f"**{lineage} cosine matrix**")
                heatmap_fig = px.imshow(
                    similarity_df,
                    zmin=0,
                    zmax=1,
                    color_continuous_scale="Blues",
                    text_auto=".2f",
                    labels={"x": "Individual", "y": "Individual", "color": "Cosine similarity"},
                )
                heatmap_fig.update_layout(height=420)
                st.plotly_chart(heatmap_fig, width="stretch")
                render_plot_download_buttons(
                    heatmap_fig,
                    base_filename=f"cosine_heatmap_{lineage}".replace(" ", "_"),
                    key_prefix=f"cosine_heatmap_{lineage}",
                )
                st.dataframe(similarity_df.round(4), width="stretch")
    else:
        st.subheader("Correlation Across Individuals (Top Clones)")
        st.caption(
            "Pearson correlation is rank-based: each individual's vector uses "
            "(clonotype rank, organ/cell) abundance features from the CD4/CD8 plots."
        )
        if not correlation_by_lineage:
            st.info("At least two individuals with non-empty vectors are required to compute correlation.")
        else:
            for lineage, similarity_df in correlation_by_lineage.items():
                st.markdown(f"**{lineage} correlation matrix**")
                heatmap_fig = px.imshow(
                    similarity_df,
                    zmin=-1,
                    zmax=1,
                    color_continuous_scale="Blues",
                    text_auto=".2f",
                    labels={"x": "Individual", "y": "Individual", "color": "Correlation"},
                )
                heatmap_fig.update_layout(height=420)
                st.plotly_chart(heatmap_fig, width="stretch")
                render_plot_download_buttons(
                    heatmap_fig,
                    base_filename=f"correlation_heatmap_{lineage}".replace(" ", "_"),
                    key_prefix=f"correlation_heatmap_{lineage}",
                )
                st.dataframe(similarity_df.round(4), width="stretch")

    all_selection_cosine_records: List[Dict[str, object]] = []
    all_selection_correlation_records: List[Dict[str, object]] = []
    for subset_option in organ_cell_options:
        subset_totals = (
            filtered[filtered["organ_cell"] == subset_option]
            .groupby(["mouse", "clonotype"], as_index=False)["abundance"]
            .sum()
            .sort_values(
                ["mouse", "abundance", "clonotype"],
                ascending=[True, False, True],
                kind="mergesort",
            )
        )
        if subset_totals.empty:
            continue

        subset_max_clonotypes = max(
            1,
            int(
                filtered[filtered["organ_cell"] == subset_option]["clonotype"].nunique()
            ),
        )
        subset_top_n = min(int(top_n), subset_max_clonotypes)

        subset_ranked = subset_totals.groupby("mouse").head(subset_top_n).copy()
        subset_ranked["clonotype_rank"] = (
            subset_ranked.groupby("mouse").cumcount() + 1
        )
        subset_top_clones = pd.merge(
            subset_ranked[["mouse", "clonotype", "clonotype_rank"]],
            filtered,
            how="left",
            on=["mouse", "clonotype"],
        )
        subset_top_clones["cd_group"] = subset_top_clones["cell_type"].apply(
            classify_cd4_cd8
        )
        subset_top_clones["abundance_for_metric"] = subset_top_clones[
            "abundance"
        ].astype(float)
        if normalize_topn_summary:
            subset_mouse_totals = subset_top_clones.groupby(
                ["mouse", "organ_cell"]
            )["abundance_for_metric"].transform("sum")
            subset_top_clones["abundance_for_metric"] = np.where(
                subset_mouse_totals > 0,
                (subset_top_clones["abundance_for_metric"] / subset_mouse_totals)
                * 100.0,
                0.0,
            )
        for lineage in ["CD4", "CD8"]:
            subset_lineage_df = subset_top_clones[
                subset_top_clones["cd_group"] == lineage
            ].copy()
            subset_cosine = calculate_mouse_cosine_similarity(
                subset_lineage_df, value_col="abundance_for_metric"
            )
            if not subset_cosine.empty:
                cos_values = subset_cosine.to_numpy(dtype=float)
                if cos_values.shape[0] >= 2:
                    cos_pairwise = cos_values[np.triu_indices(cos_values.shape[0], k=1)]
                    cos_pairwise = cos_pairwise[~np.isnan(cos_pairwise)]
                    if cos_pairwise.size > 0:
                        mean_cosine = float(np.mean(cos_pairwise))
                        if cos_pairwise.size > 1:
                            std_cosine = float(np.std(cos_pairwise, ddof=1))
                            cos_se = std_cosine / math.sqrt(cos_pairwise.size)
                            cos_ci_radius = 1.96 * cos_se
                        else:
                            cos_ci_radius = 0.0
                        all_selection_cosine_records.append(
                            {
                                "organ_cell": subset_option,
                                "lineage": lineage,
                                "mean_cosine": mean_cosine,
                                "ci_low": max(0.0, mean_cosine - cos_ci_radius),
                                "ci_high": min(1.0, mean_cosine + cos_ci_radius),
                                "n_pairs": int(cos_pairwise.size),
                            }
                        )

            subset_correlation = calculate_mouse_correlation(
                subset_lineage_df, value_col="abundance_for_metric"
            )
            if not subset_correlation.empty:
                corr_values = subset_correlation.to_numpy(dtype=float)
                if corr_values.shape[0] >= 2:
                    corr_pairwise = corr_values[np.triu_indices(corr_values.shape[0], k=1)]
                    corr_pairwise = corr_pairwise[~np.isnan(corr_pairwise)]
                    if corr_pairwise.size > 0:
                        mean_correlation = float(np.mean(corr_pairwise))
                        if corr_pairwise.size > 1:
                            std_correlation = float(np.std(corr_pairwise, ddof=1))
                            corr_se = std_correlation / math.sqrt(corr_pairwise.size)
                            corr_ci_radius = 1.96 * corr_se
                        else:
                            corr_ci_radius = 0.0
                        all_selection_correlation_records.append(
                            {
                                "organ_cell": subset_option,
                                "lineage": lineage,
                                "mean_correlation": mean_correlation,
                                "ci_low": max(-1.0, mean_correlation - corr_ci_radius),
                                "ci_high": min(1.0, mean_correlation + corr_ci_radius),
                                "n_pairs": int(corr_pairwise.size),
                            }
                        )

    filtered_with_lineage = filtered.copy()
    if True:
        filtered_with_lineage["cd_group"] = filtered_with_lineage["cell_type"].apply(
            classify_cd4_cd8
        )
        lineage_colors = {"CD4": "#1f77b4", "CD8": "#ff7f0e"}

        if summary_metric_view == "Cosine":
            st.subheader("Cosine Similarity Across Individuals (All Organ/Cell Selections)")
            st.caption(
                "Single summary plot across all organ/cell selections. "
                "Lines show mean pairwise mouse cosine similarity; shaded bands show 95% confidence ranges."
            )
            if not all_selection_cosine_records:
                st.info(
                    "No all-selection cosine matrices could be computed with the current filters."
                )
            else:
                cosine_summary_df = pd.DataFrame(all_selection_cosine_records)
                for lineage in ["CD4", "CD8"]:
                    lineage_axis_options = get_organ_cell_order(
                        filtered_with_lineage[
                            filtered_with_lineage["cd_group"] == lineage
                        ],
                        sort_organ_cell,
                    )
                    if not lineage_axis_options:
                        continue
                    lineage_stats = cosine_summary_df[
                        cosine_summary_df["lineage"] == lineage
                    ].copy()
                    if lineage_stats.empty:
                        continue
                    summary_plot = go.Figure()
                    lineage_stats = lineage_stats[
                        lineage_stats["organ_cell"].isin(lineage_axis_options)
                    ].copy()
                    if lineage_stats.empty:
                        continue
                    lineage_stats["organ_cell"] = pd.Categorical(
                        lineage_stats["organ_cell"],
                        categories=lineage_axis_options,
                        ordered=True,
                    )
                    lineage_stats = lineage_stats.sort_values("organ_cell")
                    x_values = lineage_stats["organ_cell"].astype(str).tolist()
                    lower = lineage_stats["ci_low"].tolist()
                    upper = lineage_stats["ci_high"].tolist()
                    mean = lineage_stats["mean_cosine"].tolist()
                    color = lineage_colors.get(lineage, "#444444")
                    summary_plot.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=upper,
                            mode="lines",
                            line={"width": 0},
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    summary_plot.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=lower,
                            mode="lines",
                            line={"width": 0},
                            fill="tonexty",
                            fillcolor=color.replace(")", ", 0.18)").replace("rgb", "rgba")
                            if color.startswith("rgb(")
                            else "rgba(31,119,180,0.18)"
                            if lineage == "CD4"
                            else "rgba(255,127,14,0.18)",
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    summary_plot.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=mean,
                            mode="lines+markers",
                            line={"color": color, "width": 2},
                            marker={"size": 7},
                            customdata=lineage_stats[
                                ["ci_low", "ci_high", "n_pairs"]
                            ].to_numpy(),
                            hovertemplate=(
                                "Selection: %{x}<br>"
                                "Mean cosine: %{y:.3f}<br>"
                                "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>"
                                "Pairs: %{customdata[2]}<extra>"
                                + lineage
                                + "</extra>"
                            ),
                        )
                    )
                    summary_plot.update_layout(
                        height=420,
                        title=f"{lineage} mean cosine similarity across selections",
                        yaxis_title="Cosine similarity",
                        xaxis_title="Organ/Cell selection",
                        yaxis={"range": [0, 1]},
                        showlegend=False,
                    )
                    summary_plot.update_xaxes(
                        categoryorder="array",
                        categoryarray=lineage_axis_options,
                    )
                    st.plotly_chart(summary_plot, width="stretch")
                    render_plot_download_buttons(
                        summary_plot,
                        base_filename=f"cosine_summary_{lineage}".replace(" ", "_"),
                        key_prefix=f"cosine_summary_{lineage}",
                    )
        else:
            st.subheader("Correlation Across Individuals (All Organ/Cell Selections)")
            st.caption(
                "Single summary plot across all organ/cell selections. "
                "Lines show mean pairwise mouse correlation; shaded bands show 95% confidence ranges."
            )
            if not all_selection_correlation_records:
                st.info(
                    "No all-selection correlation matrices could be computed with the current filters."
                )
            else:
                correlation_summary_df = pd.DataFrame(all_selection_correlation_records)
                for lineage in ["CD4", "CD8"]:
                    lineage_axis_options = get_organ_cell_order(
                        filtered_with_lineage[
                            filtered_with_lineage["cd_group"] == lineage
                        ],
                        sort_organ_cell,
                    )
                    if not lineage_axis_options:
                        continue
                    lineage_stats = correlation_summary_df[
                        correlation_summary_df["lineage"] == lineage
                    ].copy()
                    if lineage_stats.empty:
                        continue
                    summary_plot = go.Figure()
                    lineage_stats = lineage_stats[
                        lineage_stats["organ_cell"].isin(lineage_axis_options)
                    ].copy()
                    if lineage_stats.empty:
                        continue
                    lineage_stats["organ_cell"] = pd.Categorical(
                        lineage_stats["organ_cell"],
                        categories=lineage_axis_options,
                        ordered=True,
                    )
                    lineage_stats = lineage_stats.sort_values("organ_cell")
                    x_values = lineage_stats["organ_cell"].astype(str).tolist()
                    lower = lineage_stats["ci_low"].tolist()
                    upper = lineage_stats["ci_high"].tolist()
                    mean = lineage_stats["mean_correlation"].tolist()
                    color = lineage_colors.get(lineage, "#444444")
                    summary_plot.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=upper,
                            mode="lines",
                            line={"width": 0},
                            hoverinfo="skip",
                            showlegend=False,
                            name=f"{lineage} upper",
                        )
                    )
                    summary_plot.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=lower,
                            mode="lines",
                            line={"width": 0},
                            fill="tonexty",
                            fillcolor=color.replace(")", ", 0.18)").replace("rgb", "rgba")
                            if color.startswith("rgb(")
                            else "rgba(31,119,180,0.18)"
                            if lineage == "CD4"
                            else "rgba(255,127,14,0.18)",
                            hoverinfo="skip",
                            showlegend=False,
                            name=f"{lineage} CI",
                        )
                    )
                    summary_plot.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=mean,
                            mode="lines+markers",
                            name=f"{lineage} mean",
                            line={"color": color, "width": 2},
                            marker={"size": 7},
                            customdata=lineage_stats[
                                ["ci_low", "ci_high", "n_pairs"]
                            ].to_numpy(),
                            hovertemplate=(
                                "Selection: %{x}<br>"
                                "Mean correlation: %{y:.3f}<br>"
                                "95% CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>"
                                "Pairs: %{customdata[2]}<extra>"
                                + lineage
                                + "</extra>"
                            ),
                        )
                    )
                    summary_plot.update_layout(
                        height=420,
                        title=f"{lineage} mean correlation across selections",
                        yaxis_title="Correlation",
                        xaxis_title="Organ/Cell selection",
                        yaxis={"range": [-1, 1]},
                        showlegend=False,
                    )
                    summary_plot.update_xaxes(
                        categoryorder="array",
                        categoryarray=lineage_axis_options,
                    )
                    st.plotly_chart(summary_plot, width="stretch")
                    render_plot_download_buttons(
                        summary_plot,
                        base_filename=f"correlation_summary_{lineage}".replace(" ", "_"),
                        key_prefix=f"correlation_summary_{lineage}",
                    )

    st.subheader("Clonotype Presence Grid Across Organ/Cell")
    st.caption(
        "Separate grids are shown for CD4 and CD8. One grid per individual (no pooling across mice). "
        f"Columns are each mouse's lineage-specific top {int(top_n)} clonotypes from {subset_selected}. "
        f"Red: in that row organ|cell top-{int(top_n)}, "
        f"Blue: present but not in that row top-{int(top_n)}, "
        "White: not present."
    )
    
    # input columns for Kiki plot
    grid_selection_cols_sum = st.columns(2)
    with grid_selection_cols_sum[0]:
        query_top_n_sum = st.text_input(
                f"Search {subset_selected} clonotypes within the top N clonotypes in other organ|cell groups:",
                #min_value=1,
                #max_value=1000,
                value=500
        )
    with grid_selection_cols_sum[1]:
        show_clonotype_sequences_summary = st.checkbox(
            "Show clonotype sequences on x-axis",
            value=False,
            key="summary_grid_show_sequences",
            help="Turn off to use compact rank labels (C1, C2, ...).",
        )


    plotted_any_grid = False
    pooled_lineage_row_summaries: Dict[str, List[pd.DataFrame]] = {}
    for lineage in ["CD4", "CD8"]:
        st.markdown(f"**{lineage}**")
        lineage_filtered_summary = filtered_with_lineage[
            filtered_with_lineage["cd_group"] == lineage
        ].copy()
        shared_summary_grid_rows = get_organ_cell_order(
            lineage_filtered_summary, sort_organ_cell
        )
        if lineage_filtered_summary.empty:
            st.info(f"No {lineage} data available for the current filters.")
            continue
        mouse_ids = sorted(lineage_filtered_summary["mouse"].unique())
        lineage_plotted = False
        lineage_row_summaries: List[pd.DataFrame] = []
        grid_columns = 2
        for row_start in range(0, len(mouse_ids), grid_columns):
            row_mouse_ids = mouse_ids[row_start : row_start + grid_columns]
            row_cols = st.columns(grid_columns)
            for col_idx, mouse_id in enumerate(row_mouse_ids):
                with row_cols[col_idx]:
                    mouse_df = lineage_filtered_summary[
                        lineage_filtered_summary["mouse"] == mouse_id
                    ].copy()
                    mouse_reference_totals = (
                        mouse_df[mouse_df["organ_cell"] == subset_selected]
                        .groupby("clonotype", as_index=False)["abundance"]
                        .sum()
                        .sort_values(
                            ["abundance", "clonotype"],
                            ascending=[False, True],
                            kind="mergesort",
                        )
                    )
                    mouse_grid_clonotypes = (
                        mouse_reference_totals.head(int(top_n))["clonotype"]
                        .astype(str)
                        .tolist()
                    )
                    st.markdown(
                        f"<div style='text-align:center;font-weight:700'>{mouse_id}</div>",
                        unsafe_allow_html=True,
                    )
                    if not mouse_grid_clonotypes:
                        st.info(
                            f"No {lineage} top clonotypes available for {mouse_id} in {subset_selected}."
                        )
                        continue
                    summary_grid_fig = build_clonotype_presence_grid_figure(
                        df=mouse_df,
                        selected_clonotypes=mouse_grid_clonotypes,
                        top_n=int(top_n),
                        query_top_n = query_top_n_sum,
                        selected_organ_cell=subset_selected,
                        show_clonotype_sequences=show_clonotype_sequences_summary,
                        row_categories=shared_summary_grid_rows,
                    )
                    if summary_grid_fig is None:
                        st.info(f"No {lineage} clonotype presence grid available for {mouse_id}.")
                        continue
                    plotted_any_grid = True
                    lineage_plotted = True
                    st.plotly_chart(summary_grid_fig, width="stretch")
                    render_plot_download_buttons(
                        summary_grid_fig,
                        base_filename=(
                            f"clonotype_presence_grid_{lineage}_{mouse_id}_{subset_selected}"
                        ).replace(" ", "_"),
                        key_prefix=f"summary_grid_{lineage}_{mouse_id}",
                    )
                    summary_grid_counts = build_clonotype_presence_grid_dataframe(
                        df=mouse_df,
                        selected_clonotypes=mouse_grid_clonotypes,
                        top_n=int(top_n),
                        query_top_n=query_top_n_sum,
                        row_categories=shared_summary_grid_rows,
                    )
                    row_count_summary, col_count_summary = summarize_clonotype_presence_grid_counts(
                        grid_df=summary_grid_counts,
                        top_in_row_label=f"Top-{int(top_n)} in row",
                        present_not_top_label=f"Present (not Top-{int(top_n)} in row)",
                    )
                    lineage_row_summaries.append(row_count_summary)
                    st.markdown(
                        f"**{mouse_id} {lineage} row and clonotype counts**"
                    )
                    histogram_cols = st.columns(2)
                    row_hist_fig = build_clonotype_presence_count_histogram(
                        summary_df=row_count_summary,
                        axis_label="Row",
                        top_n=int(top_n),
                        selected_label=subset_selected,
                    )
                    col_hist_fig = build_clonotype_presence_count_histogram(
                        summary_df=col_count_summary,
                        axis_label="Column",
                        top_n=int(top_n),
                        show_clonotype_sequences=show_clonotype_sequences_summary,
                    )
                    if row_hist_fig is not None:
                        with histogram_cols[0]:
                            st.plotly_chart(row_hist_fig, width="stretch")
                            render_plot_download_buttons(
                                row_hist_fig,
                                base_filename=f"row_histogram_{lineage}_{mouse_id}_{subset_selected}".replace(" ", "_"),
                                key_prefix=f"summary_row_hist_{lineage}_{mouse_id}",
                            )
                    else:
                        with histogram_cols[0]:
                            st.info(f"No {mouse_id} {lineage} row-count plot available.")
                    if col_hist_fig is not None:
                        with histogram_cols[1]:
                            st.plotly_chart(col_hist_fig, width="stretch")
                            render_plot_download_buttons(
                                col_hist_fig,
                                base_filename=f"col_histogram_{lineage}_{mouse_id}_{subset_selected}".replace(" ", "_"),
                                key_prefix=f"summary_col_hist_{lineage}_{mouse_id}",
                            )
                    else:
                        with histogram_cols[1]:
                            st.info(f"No {mouse_id} {lineage} clonotype-count plot available.")
        if lineage_row_summaries:
            pooled_lineage_row_summaries[lineage] = lineage_row_summaries
        if not lineage_plotted:
            st.info(f"No per-individual {lineage} clonotype presence grids available.")
        if lineage == "CD4" and lineage_plotted:
            render_clonotype_presence_grid_legend(int(top_n))

    if not plotted_any_grid:
        st.info("No per-individual clonotype presence grids available for the current filters.")
    else:
        render_clonotype_presence_grid_legend(int(top_n))

    st.subheader("Top clonotypes across individuals")
    st.caption("Sorted by total abundance for the selected subset across all mice.")

    table_df = topClones.copy()
    vdj_controls = st.columns(3)
    with vdj_controls[0]:
        enable_vdjdb = st.checkbox(
            "Enrich with VDJdb",
            value=False,
            help="Translate nucleotide clonotypes to amino-acid CDR3, query VDJdb, then append match metadata columns.",
        )
    with vdj_controls[1]:
        max_vdjdb_queries = st.number_input(
            "Max VDJdb sequence queries",
            min_value=1,
            max_value=1000,
            value=min(200, max(1, table_df["clonotype"].nunique())),
            step=1,
            disabled=not enable_vdjdb,
        )
    with vdj_controls[2]:
        allow_fuzzy_vdjdb = st.checkbox(
            "Fuzzy fallback search",
            value=False,
            help="If no exact match is found, use VDJdb sequence fuzzy search (1 substitution/insertion/deletion).",
            disabled=not enable_vdjdb,
        )

    if enable_vdjdb:
        table_df["clonotype_lookup"] = table_df["clonotype"].astype(str).str.strip().str.upper()
        with st.spinner("Querying VDJdb for clonotype annotations..."):
            vdjdb_df = enrich_clonotypes_with_vdjdb(
                clonotypes=table_df["clonotype_lookup"].tolist(),
                chain_value=chain_selected,
                max_queries=int(max_vdjdb_queries),
                allow_fuzzy_fallback=allow_fuzzy_vdjdb,
            )
        if vdjdb_df.empty:
            st.info("No clonotypes were sent to VDJdb.")
        else:
            vdjdb_df = vdjdb_df.rename(columns={"clonotype": "clonotype_lookup"})
            table_df = table_df.merge(vdjdb_df, on="clonotype_lookup", how="left")
            queried_n = int(vdjdb_df["clonotype_lookup"].nunique())
            matched_n = int((vdjdb_df["vdjdb_match_count"] > 0).sum())
            error_n = int(vdjdb_df["vdjdb_error"].astype(str).str.len().gt(0).sum())
            st.caption(
                f"VDJdb queried {queried_n} unique sequences; {matched_n} had at least one match."
            )
            if error_n > 0:
                st.warning(
                    f"VDJdb queries reported {error_n} errors. Table is shown with available results."
                )
        table_df = table_df.drop(columns=["clonotype_lookup"], errors="ignore")

    st.dataframe(table_df, width="stretch")


def _render_summary_subset_top_clonotype_section(
    filtered: pd.DataFrame,
    subset_selected: str,
    top_n: int,
    sort_organ_cell: str,
    log_axis_summary: bool,
    normalize_topn_summary: bool,
    pooled_only: bool = False,
) -> pd.DataFrame:
    """Render CD4/CD8 line plots and pooled count bar charts for a single organ|cells selection.

    When pooled_only=True only the pooled bar charts are rendered (no individual line plots).
    Returns a DataFrame with the selected top clonotypes and their metadata for downstream
    similarity analysis.
    """
    subset_max_clonotypes = max(
        1,
        int(filtered[filtered["organ_cell"] == subset_selected]["clonotype"].nunique()),
    )
    subset_top_n = min(int(top_n), subset_max_clonotypes)

    subset_totals = (
        filtered[filtered["organ_cell"] == subset_selected]
        .sort_values(
            ["mouse", "abundance", "clonotype"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    selected_clonotypes = subset_totals.groupby("mouse").head(subset_top_n).copy()
    selected_clonotypes["clonotype_rank"] = (
        selected_clonotypes.groupby("mouse").cumcount() + 1
    )
    selected_clonotype_list = list(
        dict.fromkeys(selected_clonotypes["clonotype"].astype(str).tolist())
    )
    if not selected_clonotype_list:
        st.info("No clonotypes available for this selection.")
        return pd.DataFrame()

    topClones = pd.merge(
        selected_clonotypes[["mouse", "clonotype", "clonotype_rank"]],
        filtered,
        how="left",
        on=["mouse", "clonotype"],
    )
    topClones["cd_group"] = topClones["cell_type"].apply(classify_cd4_cd8)
    topClones["abundance_for_metric"] = topClones["abundance"].astype(float)
    if normalize_topn_summary:
        topClones["abundance_for_metric"] = np.where(
            topClones["norm_topN"] > 0,
            (topClones["abundance_for_metric"] / topClones["norm_topN"]) * 100.0,
            0.0,
        )
    y_axis_title = "Normalized % Pool Size" if normalize_topn_summary else "% Pool Size"

    if not pooled_only:
        if topClones.empty:
            st.info("No clonotypes available for the lineage plots.")
        else:
            for lineage in ["CD4", "CD8"]:
                lineage_df = topClones[topClones["cd_group"] == lineage].copy()
                st.markdown(f"**{lineage} clonotype abundance across individuals**")
                if lineage_df.empty:
                    st.info(f"No {lineage} subsets found for {subset_selected}.")
                    continue
                lineage_fig = _build_summary_lineage_abundance_figure(
                    lineage_df=lineage_df,
                    selected_clonotypes=selected_clonotype_list,
                    sort_organ_cell=sort_organ_cell,
                    log_axis_summary=log_axis_summary,
                    normalize_topn_summary=normalize_topn_summary,
                    selected_label=subset_selected,
                    lineage=lineage,
                )
                if lineage_fig is None:
                    st.info(f"No {lineage} abundance plot could be built for {subset_selected}.")
                else:
                    st.plotly_chart(lineage_fig, width="stretch")
                    render_plot_download_buttons(
                        lineage_fig,
                        base_filename=f"lineage_abundance_{lineage}_{subset_selected}".replace(" ", "_"),
                        key_prefix=f"lineage_abundance_{lineage}",
                    )

    pooled_lineage_row_summaries: Dict[str, List[pd.DataFrame]] = {}
    for lineage in ["CD4", "CD8"]:
        lineage_filtered_summary = filtered[filtered["cd_group"] == lineage].copy()
        if lineage_filtered_summary.empty:
            continue
        shared_summary_grid_rows = get_organ_cell_order(
            lineage_filtered_summary, sort_organ_cell
        )
        lineage_row_summaries: List[pd.DataFrame] = []
        for mouse_id in sorted(lineage_filtered_summary["mouse"].unique()):
            mouse_df = lineage_filtered_summary[
                lineage_filtered_summary["mouse"] == mouse_id
            ].copy()
            mouse_reference_totals = (
                mouse_df[mouse_df["organ_cell"] == subset_selected]
                .groupby("clonotype", as_index=False)["abundance"]
                .sum()
                .sort_values(
                    ["abundance", "clonotype"],
                    ascending=[False, True],
                    kind="mergesort",
                )
            )
            mouse_grid_clonotypes = (
                mouse_reference_totals.head(subset_top_n)["clonotype"]
                .astype(str)
                .tolist()
            )
            if not mouse_grid_clonotypes:
                continue
            summary_grid_counts = build_clonotype_presence_grid_dataframe(
                df=mouse_df,
                selected_clonotypes=mouse_grid_clonotypes,
                top_n=subset_top_n,
                query_top_n="all",
                row_categories=shared_summary_grid_rows,
            )
            row_count_summary, _ = summarize_clonotype_presence_grid_counts(
                grid_df=summary_grid_counts,
                top_in_row_label=f"Top-{subset_top_n} in row",
                present_not_top_label=f"Present (not Top-{subset_top_n} in row)",
            )
            if not row_count_summary.empty:
                lineage_row_summaries.append(row_count_summary)
        if lineage_row_summaries:
            pooled_lineage_row_summaries[lineage] = lineage_row_summaries

    if pooled_lineage_row_summaries:
        st.subheader("Across-subject median counts")
        st.caption(
            "Bars show the per-organ|cell median across individuals. "
            "Dots show the individual mouse values for red, blue, and total counts."
        )
        for lineage in ["CD4", "CD8"]:
            lineage_summaries = pooled_lineage_row_summaries.get(lineage, [])
            if not lineage_summaries:
                continue
            st.markdown(f"**{lineage}**")
            aggregate_row_fig = build_aggregate_row_count_figure(
                row_summaries=lineage_summaries,
                top_n=subset_top_n,
                selected_label=subset_selected,
            )
            if aggregate_row_fig is not None:
                st.plotly_chart(aggregate_row_fig, width="stretch")
                render_plot_download_buttons(
                    aggregate_row_fig,
                    base_filename=f"aggregate_row_count_{lineage}_{subset_selected}".replace(" ", "_"),
                    key_prefix=f"aggregate_row_{lineage}_{subset_selected}".replace(" ", "_"),
                )
    else:
        st.info("No pooled row summaries could be computed for this selection.")

    return topClones
