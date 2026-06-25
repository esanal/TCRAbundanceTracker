"""Single-subject by organ/cell page — deep-dive into one mouse and chain.

Provides a single-mouse, single-chain deep-dive with:
- Top-N heatmap, CD4/CD8 line plots, PyVis network, chord diagram
- Organ/cell clonotype sharing matrix and network centrality metrics
- Stacked abundance flow, Kiki-style presence grid, and filtered data download
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from tcr_app.core import (
    DEFAULT_EDGE_WIDTH_SCALE,
    DEFAULT_GRAVITY,
    DEFAULT_SPRING_LENGTH,
    SELECTED_ORGAN_CELL_COLOR,
    build_clonotype_color_map,
    build_clonotype_presence_count_histogram,
    build_clonotype_presence_grid_dataframe,
    build_clonotype_presence_grid_figure,
    build_entity_chord_figure,
    build_highlighted_tick_labels,
    build_organ_cell_clonotype_edges,
    build_organ_cell_clonotype_network_html,
    build_stacked_clonotype_band_figure,
    calculate_network_metrics,
    calculate_organ_cell_sharing,
    classify_cd4_cd8,
    get_organ_cell_order,
    render_clonotype_presence_grid_legend,
    render_plot_download_buttons,
    summarize_clonotype_presence_grid_counts,
)


def run_per_individual_page(df: pd.DataFrame) -> None:
    """Render the single-mouse analysis view with all visualizations."""
    st.title("TCR Abundance Explorer")
    st.subheader("Single-subject by organ/cell")

    st.markdown(
        """
    Upload a CSV file of clonotype sequences with mouse, organ, cell type, chain, and abundance.
    The app will help you explore clonotypes per mouse, organ and cell type and visualize abundance patterns.
    """
    )


    with st.sidebar:
        st.header("Filters")
        mouse_selected = st.selectbox("Mouse", sorted(df["mouse"].unique()))
        chain_selected = st.selectbox(
            "Chain", sorted(df["chain"].unique()))

        mouse_data = df[
            (df["mouse"] == mouse_selected) & (df["chain"] == chain_selected)
        ]
        organ_opts = sorted(mouse_data["organ"].unique())
        cell_opts = sorted(mouse_data["cell_type"].unique())

        organ_selected = st.multiselect("Organ", organ_opts, default=organ_opts)
        cell_selected = st.multiselect("Cell type", cell_opts, default=cell_opts)

    filtered = df[
        (df["mouse"] == mouse_selected)
        & (df["organ"].isin(organ_selected))
        & (df["cell_type"].isin(cell_selected))
        & (df["chain"] == chain_selected)
    ].copy()

    if filtered.empty:
        st.warning("No data match the selected filters.")
        st.stop()

    summary_cols = st.columns(5)
    summary_cols[0].metric("Mouse/Individual", mouse_selected)
    summary_cols[1].metric("Chain", chain_selected)
    summary_cols[2].metric("Clonotypes", filtered["clonotype"].nunique())
    summary_cols[3].metric("Organs", filtered["organ"].nunique())
    summary_cols[4].metric("Cell types", filtered["cell_type"].nunique())

    st.subheader("Abundance by Organ/Cell (Top Clonotypes)")

    select_cols = st.columns(2)
    organ_cell_options = sorted(filtered["organ_cell"].unique())
    with select_cols[0]:
        top_n_scope = st.selectbox(
            "Rank top clonotypes by organ/cell combination",
            organ_cell_options,
        )
    max_clonotypes = (
        filtered[filtered["organ_cell"] == top_n_scope]["clonotype"].nunique()
    )
    max_clonotypes = max(1, max_clonotypes)
    with select_cols[1]:
        top_n = st.number_input(
            "Select number of largest clonotypes to display",
            min_value=1,
            max_value=max_clonotypes,
            value=min(10, max_clonotypes),
            step=1,
        )
    clono_totals = (
        filtered[filtered["organ_cell"] == top_n_scope]
        .groupby("clonotype", as_index=False)["abundance"]
        .sum()
        .sort_values(["abundance", "clonotype"], ascending=[False, True], kind="mergesort")
    )
    filtered["norm_topN"] = (
        filtered.groupby("organ_cell")["abundance"]
        .transform(lambda x: x.nlargest(int(top_n)).sum())
    )
    selected_clonotypes = clono_totals.head(top_n)["clonotype"].tolist()
    heatmap_df = (
        filtered[filtered["clonotype"].isin(selected_clonotypes)]
        .groupby(["clonotype", "organ_cell"], as_index=False)["abundance"]
        .sum()
    )
    heatmap_pivot = heatmap_df.pivot(
        index="clonotype", columns="organ_cell", values="abundance"
    ).fillna(0)
    heatmap_fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Organ/Cell", y="Clonotype", color="Abundance"),
        aspect="auto",
        color_continuous_scale="viridis",
    )
    heatmap_fig.update_layout(height=500)
    heatmap_x_categories = heatmap_pivot.columns.tolist()
    heatmap_fig.update_xaxes(
        tickmode="array",
        tickvals=heatmap_x_categories,
        ticktext=build_highlighted_tick_labels(
            heatmap_x_categories,
            top_n_scope,
        ),
    )

    st.plotly_chart(heatmap_fig, width="stretch")
    render_plot_download_buttons(
        heatmap_fig,
        base_filename=f"abundance_heatmap_{mouse_selected}_{chain_selected}".replace(" ", "_"),
        key_prefix=f"heatmap_{mouse_selected}",
        data=heatmap_pivot,
        data_filename=f"abundance_heatmap_{mouse_selected}_{chain_selected}.csv".replace(" ", "_"),
    )

    st.subheader("Clonotype Abundance Line Plot: CD4 vs CD8")
    st.caption(
        "Separate line plots for CD4 and CD8 cells so each lineage can be compared independently. "
        "Y-axis shows raw abundance by default; enable top-N normalization if needed."
    )
    scale_cols = st.columns(2)
    with scale_cols[0]:
        log_axis = st.checkbox(
            "Log10 scale",
            value=False,
            help=(
                "Display values on a log10 axis. Missing clone values are shown with a "
                "dynamic pseudo-0 per organ|cell"
            ),
        )
    with scale_cols[1]:
        normalize_topn_individual = st.checkbox(
            f"Normalize by organ/cell top-{int(top_n)} denominator",
            value=True,
            help=(
                f"For each organ/cell group, divide abundance by that group's sum of the "
                f"top-{int(top_n)} abundances (within the current mouse), then multiply by 100."
            ),
        )
    y_axis_title = (
        "Normalized % Pool Size" if normalize_topn_individual else "% Pool Size"
    )
    if log_axis:
        st.caption(
            "Open-circle markers indicate imputed 0 values per organ|cell (pseudo-0)."
        )
    sort_organ_cell = st.selectbox("Sort organ/cell axis by", ["organ", "trm"])
    lineage_filtered = filtered[filtered["clonotype"].isin(selected_clonotypes)].copy()
    lineage_filtered["cd_group"] = lineage_filtered["cell_type"].apply(classify_cd4_cd8)
    clonotype_color_map = build_clonotype_color_map(selected_clonotypes)
    for lineage in ["CD4", "CD8"]:
        lineage_df = lineage_filtered[lineage_filtered["cd_group"] == lineage].copy()
        st.markdown(f"**{lineage} clonotype abundance across organ/cell**")
        if lineage_df.empty:
            st.info(f"No {lineage} cells found for current filters.")
            continue
        lineage_df["abundance_for_metric"] = lineage_df["abundance"].astype(float)
        if normalize_topn_individual:
            lineage_df["abundance_for_metric"] = np.where(
                lineage_df["norm_topN"] > 0,
                (lineage_df["abundance_for_metric"] / lineage_df["norm_topN"]) * 100.0,
                0.0,
            )
        lineage_organ_cells = get_organ_cell_order(lineage_df, sort_organ_cell)
        # 1. Pivot to create the grid automatically (clonotypes x organ_cells)
        # This handles the "missing" combinations by filling them with 0 immediately
        lineage_pivot = lineage_df.pivot_table(
            index="clonotype", 
            columns="organ_cell", 
            values="abundance_for_metric",
            aggfunc="sum", 
            fill_value=0
        ).reindex(index=selected_clonotypes, columns=lineage_organ_cells, fill_value=0)
        # 2. Flatten back to "long" format for Plotly
        organ_cell_line = lineage_pivot.reset_index().melt(
            id_vars="clonotype", 
            value_name="abundance"
        )
        organ_cell_line["is_pseudo"] = False
        organ_cell_line["pool_pct"] = organ_cell_line["abundance"].astype(float)
        organ_cell_line["pool_pct_plot"] = organ_cell_line["pool_pct"]
        if log_axis:
            positive_lineage = organ_cell_line[organ_cell_line["pool_pct"] > 0].copy()
            pseudo_by_group = (
                positive_lineage.groupby("organ_cell", as_index=False)["pool_pct"]
                .min()
                .rename(columns={"pool_pct": "pseudo_zero"})
            )
            pseudo_by_group["pseudo_zero"] = (pseudo_by_group["pseudo_zero"]) / (pseudo_by_group["pseudo_zero"]+1)
            organ_cell_line = organ_cell_line.merge(
                pseudo_by_group,
                on="organ_cell",
                how="left",
            )
            lineage_min_positive = (
                float(positive_lineage["pool_pct"].min())
                if not positive_lineage.empty
                else np.nan
            )
            lineage_pseudo = (
                lineage_min_positive / (lineage_min_positive+1)
                if not np.isnan(lineage_min_positive)
                else float(np.finfo(float).tiny)
            )
            organ_cell_line["pseudo_zero"] = organ_cell_line["pseudo_zero"].fillna(
                lineage_pseudo
            )
            organ_cell_line["is_pseudo"] = organ_cell_line["pool_pct_plot"] <= 0
            organ_cell_line["pool_pct_plot"] = np.where(
                organ_cell_line["pool_pct_plot"] > 0,
                organ_cell_line["pool_pct_plot"],
                organ_cell_line["pseudo_zero"],
            )
        line_fig = px.line(
            organ_cell_line,
            x="organ_cell",
            y="pool_pct_plot",
            color="clonotype",
            markers=True,
            color_discrete_map=clonotype_color_map,
            labels={"organ_cell": "Organ/Cell", "pool_pct_plot": y_axis_title},
        )
        yaxis_config = {
            "title": y_axis_title,
            "type": "log" if log_axis else "linear",
        }
        line_fig.update_layout(height=420, yaxis=yaxis_config)
        line_fig.update_xaxes(
            tickmode="array",
            tickvals=lineage_organ_cells,
            ticktext=build_highlighted_tick_labels(
                lineage_organ_cells,
                top_n_scope,
            ),
        )
        pseudo_points = organ_cell_line[organ_cell_line["is_pseudo"]].copy()
        if not pseudo_points.empty:
            line_fig.add_trace(
                go.Scatter(
                    x=pseudo_points["organ_cell"],
                    y=pseudo_points["pool_pct_plot"],
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "size": 11,
                        "color": "#222222",
                        "line": {"width": 1.5, "color": "#222222"},
                    },
                    name="Pseudo-0",
                    showlegend=True,
                    customdata=pseudo_points[["clonotype"]].to_numpy(),
                    hovertemplate=(
                        "Organ/Cell: %{x}<br>"
                        "Value: %{y:.4g}<br>"
                        "Clonotype: %{customdata[0]}<br>"
                        "Imputed from pseudo-0<extra></extra>"
                    ),
                )
            )
        st.plotly_chart(line_fig, width="stretch")
        render_plot_download_buttons(
            line_fig,
            base_filename=f"line_plot_{lineage}_{mouse_selected}".replace(" ", "_"),
            key_prefix=f"line_{lineage}",
            data=organ_cell_line,
            data_filename=f"line_plot_{lineage}_{mouse_selected}.csv".replace(" ", "_"),
            data_index=False,
        )


    st.subheader("Hierarchical Organ/Cell-Clonotype Network")
    st.caption(
        "Organ/cell nodes are on the left and clonotypes on the right. "
        "Organ/cell node size reflects total clone-sharing with other organ/cell nodes."
    )
    network_cols = st.columns(3)
    with network_cols[0]:
        physics_mode = st.selectbox(
            "Network physics preset",
            [
                "Balanced (default)",
                "Weak repulsion (long links)",
                "Compact clusters (tight)",
                "Force Atlas 2 (attractive)",
                "No physics",
            ],
            index=0,
            help="Switch between physics behaviors; presets adjust repulsion/spring behavior.",
        )
    with network_cols[1]:
        st.caption(f"Network clonotypes: using current top selection ({len(selected_clonotypes)}).")
        show_clonotype_labels = st.checkbox("Show clonotype labels", value=False)
        node_font_size = st.slider(
            "Node label font size",
            min_value=12,
            max_value=30,
            value=18,
            step=1,
            help="Adjust the font size for organ/cell and clonotype node labels.",
        )
    with network_cols[2]:
        min_edge_abundance = st.number_input(
            "Minimum organ/cell-clonotype edge abundance",
            min_value=0.0,
            max_value=float(filtered["abundance"].max()),
            value=0.0,
            step=1.0,
            help="Filter out low-abundance organ/cell/clonotype edges.",
        )
    edge_width_scale = DEFAULT_EDGE_WIDTH_SCALE
    gravity = DEFAULT_GRAVITY
    spring_length = DEFAULT_SPRING_LENGTH

    edge_df = build_organ_cell_clonotype_edges(
        filtered,
        selected_clonotypes,
        min_edge_abundance=min_edge_abundance,
    )
    pair_df, organ_cell_summary, shared_matrix = calculate_organ_cell_sharing(edge_df)
    selected_metric_node = st.session_state.get("network_metrics_selected_node")

    network_html = build_organ_cell_clonotype_network_html(
        edge_df=edge_df,
        organ_cell_summary=organ_cell_summary,
        selected_organ_cell=top_n_scope,
        selected_node=selected_metric_node,
        show_clonotype_labels=show_clonotype_labels,
        gravity=gravity,
        spring_length=spring_length,
        edge_width_scale=edge_width_scale,
        physics_mode=physics_mode,
        node_font_size=node_font_size,
    )
    if network_html:
        components.html(network_html, height=580, scrolling=True)
    else:
        st.warning(
            "No network edges remain after the current filters and minimum edge threshold."
        )

    st.subheader("Entity-Level Chord View (Clone to Organ/Cell)")
    st.caption(
        "Clones can connect to more than two organ/cell groups. "
        "Use filters to focus on shared clones and keep the plot readable."
    )
    chord_cols = st.columns(2)
    with chord_cols[0]:
        chord_only_shared = st.checkbox(
            "Only clones shared by >=2 organ/cell groups",
            value=True,
        )
    with chord_cols[1]:
        available_chord_clones = int(edge_df["clonotype"].nunique()) if not edge_df.empty else 1
        chord_max_clonotypes = st.slider(
            "Max clonotypes in chord view",
            min_value=1,
            max_value=max(1, available_chord_clones),
            value=min(20, max(1, available_chord_clones)),
            step=1,
        )
    chord_fig, chord_edges = build_entity_chord_figure(
        edge_df=edge_df,
        only_shared_clones=chord_only_shared,
        max_clonotypes=chord_max_clonotypes,
    )
    if chord_fig is None or chord_edges.empty:
        st.info("No clonotypes match the chord filters.")
    else:
        st.plotly_chart(chord_fig, width="stretch")
        render_plot_download_buttons(
            chord_fig,
            base_filename=f"chord_diagram_{mouse_selected}".replace(" ", "_"),
            key_prefix="chord",
            data=chord_edges,
            data_filename=f"chord_edges_{mouse_selected}.csv".replace(" ", "_"),
            data_index=False,
        )
        st.caption(
            f"Chord edges shown: {len(chord_edges)} "
            f"across {chord_edges['clonotype'].nunique()} clonotypes and "
            f"{chord_edges['organ_cell'].nunique()} organ/cell groups."
        )

    st.markdown("**Organ/cell pairs with most shared clonotypes**")
    if pair_df.empty:
        st.info("No organ/cell pairs share clonotypes at the current threshold.")
    else:
        st.dataframe(pair_df.head(20), width="stretch")
        shared_heatmap = px.imshow(
            shared_matrix,
            labels={"x": "Organ/Cell", "y": "Organ/Cell", "color": "Shared clonotypes"},
            color_continuous_scale="Blues",
            aspect="auto",
        )
        shared_heatmap.update_layout(height=420)
        st.plotly_chart(shared_heatmap, width="stretch")
        render_plot_download_buttons(
            shared_heatmap,
            base_filename=f"shared_clonotypes_heatmap_{mouse_selected}".replace(" ", "_"),
            key_prefix="shared_heatmap",
            data=shared_matrix,
            data_filename=f"shared_clonotypes_{mouse_selected}.csv".replace(" ", "_"),
        )

    st.subheader("Network Metrics")
    metrics_df = calculate_network_metrics(
        edge_df=edge_df,
        organ_cell_summary=organ_cell_summary,
    )
    if metrics_df.empty:
        st.info("No network metrics available for the current edge threshold.")
    else:
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            top_pair_text = (
                f"{pair_df.iloc[0]['organ_cell_a']} <-> {pair_df.iloc[0]['organ_cell_b']}"
                if not pair_df.empty
                else "N/A"
            )
            top_pair_value = (
                f"{pair_df.iloc[0]['shared_clonotypes']:.0f}"
                if not pair_df.empty
                else "0"
            )
            st.metric("Top shared organ/cell pair", top_pair_text, top_pair_value)
        with metric_col2:
            top_node = (
                organ_cell_summary.iloc[0]
                if not organ_cell_summary.empty
                else pd.Series({"organ_cell": "N/A", "total_shared_clonotypes": 0.0})
            )
            st.metric(
                "Most connected organ/cell node",
                str(top_node["organ_cell"]),
                f"{float(top_node['total_shared_clonotypes']):.0f} shared clones",
            )
        metric_actions = st.columns([3, 1])
        with metric_actions[0]:
            if selected_metric_node:
                st.caption(f"Highlighted node in network: {selected_metric_node}")
        with metric_actions[1]:
            if st.button("Clear node highlight", width="stretch"):
                st.session_state["network_metrics_selected_node"] = None
                st.rerun()

        metrics_display_df = metrics_df.copy()
        metrics_display_df["weighted_degree"] = metrics_display_df["weighted_degree"].round(2)
        metrics_display_df["betweenness_centrality"] = metrics_display_df[
            "betweenness_centrality"
        ].round(4)
        metrics_display_df["total_shared_clonotypes"] = metrics_display_df[
            "total_shared_clonotypes"
        ].round(0)
        metric_selection_event = st.dataframe(
            metrics_display_df,
            width="stretch",
            key="network_metrics_table",
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_rows = metric_selection_event.selection.get("rows", [])
        if selected_rows:
            selected_row_idx = int(selected_rows[0])
            clicked_node = str(metrics_display_df.iloc[selected_row_idx]["node"])
            if clicked_node != selected_metric_node:
                st.session_state["network_metrics_selected_node"] = clicked_node
                st.rerun()

    st.subheader("Stacked Clonotype Abundance Flow")
    st.caption(
        "Stacked bars show clonotype % pool size per organ/cell. "
        "Semi-transparent colored bands connect adjacent organ/cell segments for the same clonotype."
    )
    for lineage in ["CD4", "CD8"]:
        lineage_df = lineage_filtered[lineage_filtered["cd_group"] == lineage].copy()
        st.markdown(f"**{lineage} stacked abundance flow**")
        if lineage_df.empty:
            st.info(f"No {lineage} cells found for current filters.")
            continue
        stacked_flow_fig = build_stacked_clonotype_band_figure(
            lineage_df=lineage_df,
            selected_clonotypes=selected_clonotypes,
            selected_organ_cell=top_n_scope,
            lineage_label=lineage,
            clonotype_color_map=clonotype_color_map,
        )
        if stacked_flow_fig is None:
            st.info(f"No non-zero {lineage} abundances available for stacked flow.")
            continue
        st.plotly_chart(stacked_flow_fig, width="stretch")
        render_plot_download_buttons(
            stacked_flow_fig,
            base_filename=f"stacked_flow_{lineage}_{mouse_selected}".replace(" ", "_"),
            key_prefix=f"stacked_flow_{lineage}",
            data=lineage_df,
            data_filename=f"stacked_flow_{lineage}_{mouse_selected}.csv".replace(" ", "_"),
            data_index=False,
        )

    st.subheader("Clonotype Presence Grid Across Organ/Cell")
    st.caption(
        "Separate grids are shown for CD4 and CD8. "
        f"Columns are lineage-specific top {int(top_n)} clonotypes from {top_n_scope}. "
        f"Red: in that row organ|cell top-{int(top_n)}, "
        f"Blue: present but not in that row top-{int(top_n)}, "
        "White: not present."
    )

    # input columns for Kiki plot
    grid_selection_cols = st.columns(2)
    with grid_selection_cols[0]:
        query_top_n = st.text_input(
                f"Search {top_n_scope} clonotypes within the top N clonotypes in other organ|cell groups:",
                #min_value=1,
                #max_value=1000,
                value=500
        )
    with grid_selection_cols[1]:
        show_clonotype_sequences = st.checkbox(
            "Show clonotype sequences on x-axis",
            value=False,
            help="Turn off to use compact rank labels (C1, C2, ...).",
        )

    grid_source_with_lineage = filtered.copy()
    grid_source_with_lineage["cd_group"] = grid_source_with_lineage["cell_type"].apply(
        classify_cd4_cd8
    )
    plotted_any_grid = False
    for lineage in ["CD4", "CD8"]:
        st.markdown(f"**{lineage}**")
        lineage_plotted = False
        lineage_grid_df = grid_source_with_lineage[
            grid_source_with_lineage["cd_group"] == lineage
        ].copy()
        shared_grid_rows = sorted(lineage_grid_df["organ_cell"].unique())
        if lineage_grid_df.empty:
            st.info(f"No {lineage} data available for the current filters.")
            continue
        lineage_reference_totals = (
            lineage_grid_df[lineage_grid_df["organ_cell"] == top_n_scope]
            .groupby("clonotype", as_index=False)["abundance"]
            .sum()
            .sort_values(
                ["abundance", "clonotype"],
                ascending=[False, True],
                kind="mergesort",
            )
        )
        lineage_grid_clonotypes = (
            lineage_reference_totals.head(int(top_n))["clonotype"].astype(str).tolist()
        )
        if not lineage_grid_clonotypes:
            st.info(
                f"No {lineage} top clonotypes available in reference organ|cell {top_n_scope}."
            )
            continue
        grid_fig = build_clonotype_presence_grid_figure(
            df=lineage_grid_df,
            selected_clonotypes=lineage_grid_clonotypes,
            top_n=int(top_n),
            query_top_n = query_top_n,
            selected_organ_cell=top_n_scope,
            show_clonotype_sequences=show_clonotype_sequences,
            row_categories=shared_grid_rows,
        )
        if grid_fig is None:
            st.info(f"No {lineage} clonotype presence grid available for current filters.")
            continue
        plotted_any_grid = True
        lineage_plotted = True
        st.plotly_chart(grid_fig, width="stretch")
        render_plot_download_buttons(
            grid_fig,
            base_filename=f"clonotype_presence_grid_{lineage}_{top_n_scope}".replace(" ", "_"),
            key_prefix=f"lineage_grid_{lineage}",
            data=lineage_grid_df,
            data_filename=f"presence_grid_{lineage}_{top_n_scope}.csv".replace(" ", "_"),
            data_index=False,
        )
        lineage_grid_counts = build_clonotype_presence_grid_dataframe(
            df=lineage_grid_df,
            selected_clonotypes=lineage_grid_clonotypes,
            top_n=int(top_n),
            query_top_n=query_top_n,
            row_categories=shared_grid_rows,
        )
        row_count_summary, col_count_summary = summarize_clonotype_presence_grid_counts(
            grid_df=lineage_grid_counts,
            top_in_row_label=f"Top-{int(top_n)} in row",
            present_not_top_label=f"Present (not Top-{int(top_n)} in row)",
        )
        st.markdown(f"**{lineage} row and clonotype counts**")
        histogram_cols = st.columns(2)
        row_hist_fig = build_clonotype_presence_count_histogram(
            summary_df=row_count_summary,
            axis_label="Row",
            top_n=int(top_n),
            selected_label=top_n_scope,
        )
        col_hist_fig = build_clonotype_presence_count_histogram(
            summary_df=col_count_summary,
            axis_label="Column",
            top_n=int(top_n),
            show_clonotype_sequences=show_clonotype_sequences,
        )
        if row_hist_fig is not None:
            with histogram_cols[0]:
                st.plotly_chart(row_hist_fig, width="stretch")
                render_plot_download_buttons(
                    row_hist_fig,
                    base_filename=f"row_histogram_{lineage}_{mouse_selected}_{top_n_scope}".replace(" ", "_"),
                    key_prefix=f"row_hist_{lineage}",
                    data=row_count_summary,
                    data_filename=f"row_histogram_{lineage}_{mouse_selected}_{top_n_scope}.csv".replace(" ", "_"),
                    data_index=False,
                )
        else:
            with histogram_cols[0]:
                st.info(f"No {lineage} row-count plot available.")
        if col_hist_fig is not None:
            with histogram_cols[1]:
                st.plotly_chart(col_hist_fig, width="stretch")
                render_plot_download_buttons(
                    col_hist_fig,
                    base_filename=f"col_histogram_{lineage}_{mouse_selected}_{top_n_scope}".replace(" ", "_"),
                    key_prefix=f"col_hist_{lineage}",
                    data=col_count_summary,
                    data_filename=f"col_histogram_{lineage}_{mouse_selected}_{top_n_scope}.csv".replace(" ", "_"),
                    data_index=False,
                )
        else:
            with histogram_cols[1]:
                st.info(f"No {lineage} clonotype-count plot available.")
        if lineage == "CD4" and lineage_plotted:
            render_clonotype_presence_grid_legend(int(top_n))
    if plotted_any_grid:
        render_clonotype_presence_grid_legend(int(top_n))
    else:
        st.info("No lineage-specific clonotype presence grids available for current filters.")

    st.subheader("Filtered Data")
    st.dataframe(filtered, width="stretch")

    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download filtered data", csv_buffer.getvalue(), file_name="filtered_clonotypes.csv"
    )
