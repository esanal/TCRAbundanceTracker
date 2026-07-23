"""Organ distribution page — detection count distribution plots per organ/cell.

For each individual, iterates over the selected organs. For each organ, finds the
top N clonotypes by abundance and counts how many of the selected organs (origin
included) detect each clonotype. Displays either a ridge plot or an overlaid
histogram per individual.
"""

from typing import Dict, List

import pandas as pd
import streamlit as st

from tcr_app.core import (
    build_clonotype_detection_histogram_figure,
    build_clonotype_detection_ridge_figure,
    render_plot_download_buttons,
)


def run_organ_distribution_page(df: pd.DataFrame) -> None:
    st.title("TCR Abundance Explorer")
    st.subheader("Clonotype Organ Spread")
    st.caption(
        "For each individual, find the top N clonotypes per organ/cell by abundance, "
        "then count how many of the selected organs detect each clonotype. "
        "Ridge lines show the distribution for each organ's top N clonotypes; "
        "choose a histogram overlay for a direct frequency view."
    )

    with st.sidebar:
        st.header("Filters")
        chain_selected = st.selectbox(
            "Chain", sorted(df["chain"].unique()), key="org_dist_chain"
        )

        chain_df = df[df["chain"] == chain_selected]
        organ_cell_options = sorted(chain_df["organ_cell"].unique())

        per_organ_clonotypes = chain_df.groupby("organ_cell")["clonotype"].nunique()
        max_possible = int(per_organ_clonotypes.max()) if not per_organ_clonotypes.empty else 1
        top_n = st.number_input(
            "Top N clonotypes",
            min_value=1,
            max_value=max_possible,
            value=min(10, max_possible),
            key="org_dist_topn",
        )

        organ_cells_selected = st.multiselect(
            "Organ/Cell",
            organ_cell_options,
            default=organ_cell_options,
            key="org_dist_organs",
        )

        plot_style = st.radio(
            "Plot style",
            ["Ridge plot", "Histograms (per organ)"],
            key="org_dist_style",
        )

        kde_bandwidth = 0.6
        if "Ridge" in plot_style:
            kde_bandwidth = st.slider(
                "KDE smoothness (bandwidth)",
                min_value=0.05,
                max_value=2.0,
                value=0.6,
                step=0.05,
                key="org_dist_bandwidth",
            )

    filtered = df[
        (df["chain"] == chain_selected)
        & (df["organ_cell"].isin(organ_cells_selected))
    ].copy()

    if filtered.empty or not organ_cells_selected:
        st.info("Select at least one organ/cell.")
        return

    mice = sorted(filtered["mouse"].unique())
    n_mice = len(mice)
    if n_mice == 0:
        st.info("No individuals found for the selected filters.")
        return

    progress_bar = st.progress(0, text="Processing individuals...")

    for mouse_idx, mouse_id in enumerate(mice):
        progress_bar.progress(
            (mouse_idx + 1) / n_mice,
            text=f"Processing {mouse_id} ({mouse_idx + 1}/{n_mice})",
        )

        mouse_df = filtered[filtered["mouse"] == mouse_id]
        per_organ: Dict[str, List[int]] = {}

        for organ in organ_cells_selected:
            organ_df = mouse_df[mouse_df["organ_cell"] == organ]
            if organ_df.empty:
                continue

            top_clonotypes = (
                organ_df.groupby("clonotype")["abundance"]
                .sum()
                .sort_values(ascending=False)
                .head(int(top_n))
                .index.tolist()
            )
            if not top_clonotypes:
                continue

            detection_counts: List[int] = []
            for clonotype in top_clonotypes:
                count = mouse_df[mouse_df["clonotype"] == clonotype][
                    "organ_cell"
                ].nunique()
                detection_counts.append(count)

            per_organ[organ] = detection_counts

        if not per_organ:
            continue

        if "Ridge" in plot_style:
            fig = build_clonotype_detection_ridge_figure(
                per_organ_detection_counts=per_organ,
                top_n=int(top_n),
                mouse_id=mouse_id,
                bandwidth=kde_bandwidth,
            )
        else:
            fig = build_clonotype_detection_histogram_figure(
                per_organ_detection_counts=per_organ,
                top_n=int(top_n),
                mouse_id=mouse_id,
            )
        if fig is None:
            continue

        with st.expander(
            f"{mouse_id} ({len(per_organ)} organ/cell types, top {int(top_n)})",
            expanded=(mouse_idx == 0),
        ):
            st.plotly_chart(fig, width="stretch")

            csv_records: List[Dict[str, object]] = []
            for organ, counts in per_organ.items():
                for c in counts:
                    csv_records.append(
                        {
                            "mouse": mouse_id,
                            "organ_cell": organ,
                            "detection_count": c,
                        }
                    )

            csv_data = pd.DataFrame(csv_records) if csv_records else None
            style_prefix = "ridge" if "Ridge" in plot_style else "hist"
            safe_fn = (
                f"{style_prefix}_{chain_selected.lower()}_{mouse_id}_top{int(top_n)}"
                .replace(" ", "_")
                .replace("|", "_")
            )

            render_plot_download_buttons(
                fig,
                base_filename=safe_fn,
                key_prefix=f"{style_prefix}_{mouse_id}",
                data=csv_data,
                data_filename=f"{safe_fn}.csv",
                data_index=False,
            )

    progress_bar.empty()
    st.success(f"Done — {n_mice} individual(s) processed.")
