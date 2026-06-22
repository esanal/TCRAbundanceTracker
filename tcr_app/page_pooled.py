"""Top-N clonotype summary page — batch-renders per-organ/cell top-N analyses.

Iterates over all organ|cells pairs in a 2-column grid layout and reuses the shared
render function from the summary page with pooled_only=True (no CD4/CD8 line plots,
only the pooled count bar charts at the bottom).
"""

import numpy as np
import pandas as pd
import streamlit as st

from tcr_app.core import (
    classify_cd4_cd8,
    get_organ_cell_order,
)
from tcr_app.page_summary_all import _render_summary_subset_top_clonotype_section


def run_summary_all_individuals_pooled_page(df: pd.DataFrame) -> None:
    """Render per-organ|cells pooled count summaries in a 2-column grid."""
    st.title("TCR Abundance Explorer")
    st.subheader("Top-N clonotype summary")
    st.markdown(
        """
    Browse every organ|cell combination from the summary view and render the same
    rank-top clonotype plots and pooled count summaries for each selection.
    """
    )

    with st.sidebar:
        st.header("Filters")
        chain_selected = st.selectbox("Chain", sorted(df["chain"].unique()))
        organ_selected = st.multiselect(
            "Organ", sorted(df["organ"].unique()), default=sorted(df["organ"].unique())
        )
        cell_selected = st.multiselect(
            "Cell type",
            sorted(df["cell_type"].unique()),
            default=sorted(df["cell_type"].unique()),
        )
        top_n = st.number_input(
            "Top N clonotypes per mouse",
            min_value=1,
            max_value=500,
            value=10,
            step=1,
            help=(
                "For each organ|cell selection, use the same top-N logic as the summary "
                "page and cap it to the available clonotypes in that selection."
            ),
        )
        log_axis_summary = st.checkbox("Log10 scale", value=True)
        normalize_topn_summary = st.checkbox(
            "Normalize by mouse+organ/cell top-N denominator",
            value=False,
            help=(
                "Divide abundances by the per-mouse, per-organ/cell sum of the top-N "
                "clonotypes before plotting."
            ),
        )
        sort_organ_cell = st.selectbox("Sort organ/cell axis by", ["organ", "trm"])

    filtered = df[
        (df["organ"].isin(organ_selected))
        & (df["cell_type"].isin(cell_selected))
        & (df["chain"] == chain_selected)
    ].copy()

    if filtered.empty:
        st.warning("No data match the selected filters.")
        st.stop()

    filtered["cd_group"] = filtered["cell_type"].apply(classify_cd4_cd8)
    filtered["norm_topN"] = (
        filtered.groupby(["mouse", "organ_cell"])["abundance"]
        .transform(lambda x: x.nlargest(int(top_n)).sum())
    )

    organ_cell_options = sorted(filtered["organ_cell"].unique())
    if not organ_cell_options:
        st.info("No organ/cell combinations are available for the current filters.")
        return

    st.caption(
        f"Rendering {len(organ_cell_options)} organ|cell selections with chain {chain_selected}."
    )

    for i in range(0, len(organ_cell_options), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx >= len(organ_cell_options):
                break
            subset_selected = organ_cell_options[idx]
            with cols[j]:
                subset_max_clonotypes = max(
                    1,
                    int(filtered[filtered["organ_cell"] == subset_selected]["clonotype"].nunique()),
                )
                subset_top_n = min(int(top_n), subset_max_clonotypes)
                st.caption(
                    f"Using top {subset_top_n} clonotypes here (selection max: {subset_max_clonotypes})."
                )
                _render_summary_subset_top_clonotype_section(
                    filtered=filtered,
                    subset_selected=subset_selected,
                    top_n=int(top_n),
                    sort_organ_cell=sort_organ_cell,
                    log_axis_summary=log_axis_summary,
                    normalize_topn_summary=normalize_topn_summary,
                    pooled_only=True,
                )
