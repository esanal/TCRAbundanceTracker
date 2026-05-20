from .core import *

def run_per_individual_page(df: pd.DataFrame):
    st.title("TCR Abundance Explorer")
    st.subheader("Per individual")

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

        organ_selected = st.multiselect(
            "Organ", sorted(df["organ"].unique()), default=sorted(df["organ"].unique())
        )
        cell_selected = st.multiselect(
            "Cell type",
            sorted(df["cell_type"].unique()),
            default=sorted(df["cell_type"].unique()),
        )

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
        else:
            with histogram_cols[0]:
                st.info(f"No {lineage} row-count plot available.")
        if col_hist_fig is not None:
            with histogram_cols[1]:
                st.plotly_chart(col_hist_fig, width="stretch")
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


#########################################
#### Summary page of all individuals ####
#########################################
def run_summary_all_page(df: pd.DataFrame):
    st.title("TCR Abundance Explorer")
    st.subheader("Summary all individuals")
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
        display_items = sorted_chains[:3] # Display max 3
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
    # Select organ|clonotype
    organ_cell_options = sorted(filtered["organ_cell"].unique())
    with select_cols[0]:
        subset_selected = st.selectbox(
            "Rank top clonotypes by organ/cell combination",
            organ_cell_options,
            index=0,
            help="Select a subset whose top clones across all individuals will be shown.",
        )
    # Select top n between 1 and max clonotypes
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

    # Count top n clonotypes per mouse
    clono_totals = (
        filtered[filtered["organ_cell"] == subset_selected]
        #.groupby(["mouse"], as_index=False)["abundance"]
        #.sum()
        .sort_values(
            ["mouse", "abundance"],
            ascending=[True, False],
            kind="mergesort",
        )
    )
    selected_clonotypes = clono_totals.groupby("mouse").head(top_n).copy()
    selected_clonotypes["clonotype_rank"] = (
        selected_clonotypes.groupby("mouse").cumcount() + 1
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

 
    # Calculate normalization factor per organ|cell
    filtered["norm_topN"] = (
            filtered.groupby(["mouse", "organ_cell"])["abundance"]
            .transform(lambda x: x.nlargest(top_n).sum())
            )
    # Grab top clones' occurance in other organ|cells 
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
    y_axis_title = (
        "Normalized % Pool Size" if normalize_topn_summary else "% Pool Size"
    )
    
    if topClones.empty:
        st.info("No clonotypes available for the lineage plots.")
    else:
        cosine_by_lineage: Dict[str, pd.DataFrame] = {}
        correlation_by_lineage: Dict[str, pd.DataFrame] = {}
        for lineage in ["CD4", "CD8"]:
            lineage_df = topClones[topClones["cd_group"] == lineage].copy()
            st.markdown(f'**{lineage} clonotype abundance across organ/cell**')
            if lineage_df.empty:
                st.info(f"No {lineage} subsets found for the current filter.")
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
            lineage_organ_cells = get_organ_cell_order(lineage_df, sort_organ_cell)
            # 1. Pivot (clonotypes x organ_cells) per individual!
            all_mice = lineage_df.mouse.unique()
            lineage_pivot = lineage_df.pivot_table(
                index=["clonotype", "mouse"], 
                columns=["organ_cell"], 
                values="abundance_for_metric", 
                aggfunc="sum",
                fill_value=0
            ).reset_index()
            ordered_columns = ["clonotype", "mouse", *lineage_organ_cells]
            lineage_pivot = lineage_pivot.reindex(columns=ordered_columns, fill_value=0)
            # 2. Flatten back to "long"
            organ_cell_line = lineage_pivot.melt(
                    id_vars=["clonotype", "mouse"],
                    value_vars=lineage_organ_cells,
                    var_name="organ_cell",
                    value_name="abundance",
            )
            organ_cell_line["organ_cell"] = pd.Categorical(
                organ_cell_line["organ_cell"],
                categories=lineage_organ_cells,
                ordered=True,
            )
            organ_cell_line = organ_cell_line.sort_values(
                ["mouse", "clonotype", "organ_cell"],
                kind="mergesort",
            )
            organ_cell_line["is_pseudo"] = False
            
            if log_axis_summary:
                # Per mouse+organ_cell pseudo-zero:
                # half of the lowest positive abundance observed for that lineage.
                positive_lineage = lineage_df[lineage_df["abundance_for_metric"] > 0].copy()
                pseudo_by_group = (
                    positive_lineage.groupby(["mouse", "organ_cell"], as_index=False)[
                        "abundance_for_metric"
                    ]
                    .min()
                    .rename(columns={"abundance_for_metric": "pseudo_zero"})
                )
                pseudo_by_group["pseudo_zero"] = pseudo_by_group["pseudo_zero"] / (pseudo_by_group["pseudo_zero"]+1)
                organ_cell_line = organ_cell_line.merge(
                    pseudo_by_group,
                    on=["mouse", "organ_cell"],
                    how="left",
                )

                # Fallbacks when a specific mouse+organ_cell has no observed positive top-N value.
                mouse_level_pseudo = (
                    positive_lineage.groupby("mouse", as_index=False)["abundance_for_metric"]
                    .min()
                    .rename(columns={"abundance_for_metric": "mouse_pseudo_zero"})
                )
                mouse_level_pseudo["mouse_pseudo_zero"] = (
                    mouse_level_pseudo["mouse_pseudo_zero"] / (mouse_level_pseudo["mouse_pseudo_zero"]+1)
                )
                organ_cell_line = organ_cell_line.merge(
                    mouse_level_pseudo,
                    on="mouse",
                    how="left",
                )
                lineage_min_positive = (
                    float(positive_lineage["abundance_for_metric"].min())
                    if not positive_lineage.empty
                    else np.nan
                )
                lineage_pseudo = (
                    lineage_min_positive / (lineage_min_positive+1)
                    if not np.isnan(lineage_min_positive)
                    else float(np.finfo(float).tiny)
                )
                organ_cell_line["pseudo_zero"] = organ_cell_line["pseudo_zero"].fillna(
                    organ_cell_line["mouse_pseudo_zero"]
                )
                organ_cell_line["pseudo_zero"] = organ_cell_line["pseudo_zero"].fillna(
                    lineage_pseudo
                )
                organ_cell_line["is_pseudo"] = organ_cell_line["abundance"] <= 0
                organ_cell_line["abundance"] = np.where(
                    organ_cell_line["abundance"] > 0,
                    organ_cell_line["abundance"],
                    organ_cell_line["pseudo_zero"],
                )

            cd_fig = px.line(
                organ_cell_line,
                x="organ_cell",
                y="abundance",
                color="mouse",
                #line_dash="clonotype",
                line_group="clonotype",
                markers=True,
                labels={
                    "organ_cell": "Organ/Cell",
                    "abundance": y_axis_title,
                    "mouse": "Individual",
                },
                title=f"{lineage} clonotype abundance across individuals",
                category_orders={
                    "organ_cell": lineage_organ_cells,
                    "mouse": all_mice,
                    "clonotype": selected_clonotypes,
                },
            )
            yaxis_config = {
                "title": y_axis_title,
                "type": "log" if log_axis_summary else "linear",
            }
            if log_axis_summary:
                tick_exponents = list(range(-5, 3))
                yaxis_config.update(
                    {
                        "tickvals": [10 ** exp for exp in tick_exponents],
                        "ticktext": [
                            "0" if exp == -5 else f"10{superscript(exp)}"
                            for exp in tick_exponents
                        ],
                        "range": [tick_exponents[0], tick_exponents[-1]],
                    }
                )
            cd_fig.update_layout(
                height=420,
                yaxis=yaxis_config,
                xaxis_title="Organ/Cell",
            )


            cd_fig.update_xaxes(
                tickmode="array",
                tickvals=lineage_organ_cells,
                ticktext=build_highlighted_tick_labels(
                    lineage_organ_cells,
                    subset_selected,
                ),
            )

            #cd_fig.update_xaxes(tickangle=-45)
            pseudo_points = organ_cell_line[organ_cell_line["is_pseudo"]].copy()
            if not pseudo_points.empty:
                cd_fig.add_trace(
                    go.Scatter(
                        x=pseudo_points["organ_cell"],
                        y=pseudo_points["abundance"],
                        mode="markers",
                        marker={
                            "symbol": "circle-open",
                            "size": 11,
                            "color": "#222222",
                            "line": {"width": 1.5, "color": "#222222"},
                        },
                        name="Pseudo-0",
                        showlegend=True,
                        customdata=pseudo_points[["mouse", "clonotype"]].to_numpy(),
                        hovertemplate=(
                            "Organ/Cell: %{x}<br>"
                            "Value: %{y:.4g}<br>"
                            "Mouse: %{customdata[0]}<br>"
                            "Clonotype: %{customdata[1]}<br>"
                            "Imputed from pseudo-0<extra></extra>"
                        ),
                    )
                )
            st.plotly_chart(cd_fig, width="stretch")

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
                    else:
                        with histogram_cols[0]:
                            st.info(f"No {mouse_id} {lineage} row-count plot available.")
                    if col_hist_fig is not None:
                        with histogram_cols[1]:
                            st.plotly_chart(col_hist_fig, width="stretch")
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
        if pooled_lineage_row_summaries:
            st.subheader("Pooled Organ/Cell Counts Across Individuals")
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
                    top_n=int(top_n),
                    selected_label=subset_selected,
                )
                if aggregate_row_fig is not None:
                    st.plotly_chart(aggregate_row_fig, width="stretch")

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


def run_all_clonotype_flow_page(df: pd.DataFrame):
    st.title("TCR Abundance Explorer")
    st.subheader("Top-N flow per individual")
    st.markdown(
        """
    Per selected mouse, this page shows stacked clonotype abundance flow across organ|cell groups.
    CD4 and CD8 are displayed in separate plots. For each organ|cell, top-N clonotypes are selected,
    and the plot shows how those selected clonotypes flow across organ|cell groups.
    """
    )

    with st.sidebar:
        st.header("Flow filters")
        chain_selected = st.selectbox("Chain", sorted(df["chain"].unique()))
        organ_selected = st.multiselect(
            "Organ",
            sorted(df["organ"].unique()),
            default=sorted(df["organ"].unique()),
        )
        cell_selected = st.multiselect(
            "Cell type",
            sorted(df["cell_type"].unique()),
            default=sorted(df["cell_type"].unique()),
        )
        top_n = st.number_input(
            "Top N clonotypes per organ|cell",
            min_value=1,
            max_value=500,
            value=10,
            step=1,
            help=(
                "For each mouse and lineage, select top N clonotypes within every organ|cell, "
                "then show the union of those clonotypes in the flow plot."
            ),
        )
        sort_organ_cell = st.selectbox("Sort organ/cell axis by", ["organ", "trm"])

    filtered = df[
        (df["chain"] == chain_selected)
        & (df["organ"].isin(organ_selected))
        & (df["cell_type"].isin(cell_selected))
    ].copy()

    if filtered.empty:
        st.warning("No data match the selected filters.")
        st.stop()

    available_mice = sorted(filtered["mouse"].unique())
    selected_mice = st.multiselect(
        "Individuals (mice) to display",
        options=available_mice,
        default=available_mice[: min(len(available_mice), 4)],
        help="Each selected mouse renders separate CD4 and CD8 flow plots using all clonotypes.",
    )
    if not selected_mice:
        st.info("Select at least one mouse to render flow plots.")
        st.stop()

    filtered = filtered[filtered["mouse"].isin(selected_mice)].copy()
    filtered["cd_group"] = filtered["cell_type"].apply(classify_cd4_cd8)

    st.caption(
        f"Within each mouse and lineage, stacked bars show top-flow clonotypes individually plus one aggregated 'Other clonotypes' bar segment. Only per-row top {int(top_n)} clonotypes are highlighted with color and flow bands."
    )

    for mouse_id in selected_mice:
        mouse_df = filtered[filtered["mouse"] == mouse_id].copy()
        st.markdown(f"### {mouse_id}")
        if mouse_df.empty:
            st.info(f"No data available for {mouse_id} under current filters.")
            continue

        mouse_metrics = st.columns(4)
        mouse_metrics[0].metric("Clonotypes", int(mouse_df["clonotype"].nunique()))
        mouse_metrics[1].metric("Organs", int(mouse_df["organ"].nunique()))
        mouse_metrics[2].metric("Cell types", int(mouse_df["cell_type"].nunique()))
        mouse_metrics[3].metric("Organ|cell groups", int(mouse_df["organ_cell"].nunique()))

        for lineage in ["CD4", "CD8"]:
            lineage_df = mouse_df[mouse_df["cd_group"] == lineage].copy()
            st.markdown(f"**{lineage} all-clonotype stacked abundance flow**")
            if lineage_df.empty:
                st.info(f"No {lineage} cells found for {mouse_id}.")
                continue

            per_row_ranked = (
                lineage_df.groupby(["organ_cell", "clonotype"], as_index=False)["abundance"]
                .sum()
                .sort_values(
                    ["organ_cell", "abundance", "clonotype"],
                    ascending=[True, False, True],
                    kind="mergesort",
                )
            )
            lineage_topn_df = per_row_ranked.groupby("organ_cell").head(int(top_n)).copy()
            lineage_flow_candidates = (
                lineage_topn_df
                .groupby("clonotype", as_index=False)["abundance"]
                .sum()
                .sort_values(["abundance", "clonotype"], ascending=[False, True], kind="mergesort")
            )["clonotype"].astype(str).tolist()
            lineage_clonotypes = lineage_flow_candidates[: int(top_n)]
            st.caption(
                f"{lineage}: showing top {len(lineage_clonotypes)} highlighted clonotypes (from per-organ|cell top-{int(top_n)} candidates); remaining clonotypes are aggregated as 'Other clonotypes'."
            )
            if not lineage_clonotypes:
                st.info(f"No {lineage} clonotypes available after top-{int(top_n)} row filtering.")
                continue
            clonotype_color_map = build_clonotype_color_map(lineage_clonotypes)
            lineage_organ_cells = get_organ_cell_order(lineage_df, sort_organ_cell)
            stacked_flow_fig = build_stacked_clonotype_band_figure(
                lineage_df=lineage_df,
                selected_clonotypes=lineage_clonotypes,
                selected_organ_cell="",
                lineage_label=f"{mouse_id} {lineage}",
                clonotype_color_map=clonotype_color_map,
                organ_cell_order=lineage_organ_cells,
                aggregate_non_selected=True,
                non_selected_label="Other clonotypes",
            )
            if stacked_flow_fig is None:
                st.info(f"No non-zero {lineage} abundances available for {mouse_id}.")
                continue
            stacked_flow_fig.update_layout(height=430)
            st.plotly_chart(stacked_flow_fig, width="stretch")

        st.markdown("---")
