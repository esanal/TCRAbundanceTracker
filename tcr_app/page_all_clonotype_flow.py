"""All-clonotype flow page — per-mouse stacked abundance flow across organ/cell groups.

For each selected mouse and lineage (CD4/CD8), ranks clonotypes per organ|cell row,
selects the top-N union, and renders stacked bar/band flow figures showing how the
same clonotypes distribute across anatomical sites.
"""

import pandas as pd
import streamlit as st

from tcr_app.core import (
    build_clonotype_color_map,
    build_stacked_clonotype_band_figure,
    classify_cd4_cd8,
    get_organ_cell_order,
    render_plot_download_buttons,
)


def run_all_clonotype_flow_page(df: pd.DataFrame) -> None:
    """Render per-mouse stacked flow plots for CD4 and CD8 lineages."""
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
            render_plot_download_buttons(
                stacked_flow_fig,
                base_filename=f"all_clonotype_flow_{mouse_id}_{lineage}".replace(" ", "_"),
                key_prefix=f"flow_{mouse_id}_{lineage}",
            )

        st.markdown("---")
