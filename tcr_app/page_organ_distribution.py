"""Organ distribution page — detection count distribution plots per organ/cell.

For each individual, iterates over the selected organs. For each organ, finds the
top N clonotypes by abundance and counts how many of the selected organs (origin
included) detect each clonotype. Displays either a ridge plot or an overlaid
histogram per individual. The downloadable CSV is a presence matrix: one row per
clonotype with a 0/1 column for every compared organ/cell, so the row sum of those
columns equals the detection count.
"""

from typing import Dict, List, Set, Tuple

import pandas as pd
import streamlit as st

from tcr_app.core import (
    build_clonotype_detection_histogram_figure,
    build_clonotype_detection_ridge_figure,
    render_plot_download_buttons,
)


def _compute_membership(
    mouse_df: pd.DataFrame,
    organ_cells_selected: List[str],
    top_n: int,
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, Dict[str, Set[str]]],
    Dict[str, List[int]],
]:
    """For each reference organ/cell, find its top clonotypes and their spread.

    Returns ``(per_organ_top, per_organ_membership, per_organ_detection_counts)``:
    - top clonotypes: reference organ -> clonotypes ordered by abundance (desc)
    - membership: reference organ -> clonotype -> set of organ/cells detected in
    - detection counts: reference organ -> per-clonotype detection count (len of
      the membership set)
    """
    clonotype_organs: Dict[str, Set[str]] = {}
    for clonotype, organ_cell in zip(mouse_df["clonotype"], mouse_df["organ_cell"]):
        clonotype_organs.setdefault(str(clonotype), set()).add(str(organ_cell))

    per_organ_top: Dict[str, List[str]] = {}
    per_organ_membership: Dict[str, Dict[str, Set[str]]] = {}
    per_organ_detection_counts: Dict[str, List[int]] = {}

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

        membership: Dict[str, Set[str]] = {}
        detection_counts: List[int] = []
        for clonotype in top_clonotypes:
            clonotype = str(clonotype)
            detected = clonotype_organs.get(clonotype, set())
            membership[clonotype] = detected
            detection_counts.append(len(detected))

        per_organ_top[organ] = top_clonotypes
        per_organ_membership[organ] = membership
        per_organ_detection_counts[organ] = detection_counts

    return per_organ_top, per_organ_membership, per_organ_detection_counts


def _merge_pooled_membership(
    pooled_membership: Dict[str, Dict[str, Set[str]]],
    per_organ_membership: Dict[str, Dict[str, Set[str]]],
) -> None:
    """Merge one mouse's membership into the pooled one (union of tissues across mice)."""
    for organ, membership in per_organ_membership.items():
        target = pooled_membership.setdefault(organ, {})
        for clonotype, detected in membership.items():
            target.setdefault(clonotype, set()).update(detected)


def _build_presence_csv(
    mouse_id: str,
    per_organ_top: Dict[str, List[str]],
    per_organ_membership: Dict[str, Dict[str, Set[str]]],
    organ_cells_selected: List[str],
) -> pd.DataFrame:
    """Build the presence-matrix CSV for download.

    One row per (reference organ/cell, clonotype, rank). For every compared
    organ/cell there is a 0/1 column (1 = clonotype detected there), and the row
    sum of those columns equals ``detection_count``.
    """
    records: List[Dict[str, object]] = []
    for organ, clonotypes in per_organ_top.items():
        membership = per_organ_membership.get(organ, {})
        for rank, clonotype in enumerate(clonotypes, start=1):
            clonotype = str(clonotype)
            detected = membership.get(clonotype, set())
            row: Dict[str, object] = {
                "mouse": mouse_id,
                "reference_organ_cell": organ,
                "clonotype": clonotype,
                "top_rank": rank,
            }
            for oc in organ_cells_selected:
                row[oc] = 1 if oc in detected else 0
            row["detection_count"] = len(detected)
            records.append(row)
    return pd.DataFrame(records) if records else pd.DataFrame()


def run_organ_distribution_page(df: pd.DataFrame) -> None:
    st.title("TCR Abundance Explorer")
    st.subheader("Clonotype Organ Spread")
    st.caption(
        "For each individual, find the top N clonotypes per organ/cell by abundance, "
        "then count how many of the selected organs detect each clonotype. "
        "Ridge lines show the distribution for each organ's top N clonotypes; "
        "choose a histogram overlay for a direct frequency view. The CSV download "
        "is a presence matrix: one row per clonotype with a 0/1 column per compared "
        "organ/cell, so the row sum of those columns equals the detection count."
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
    pooled_per_organ: Dict[str, List[int]] = {}
    pooled_membership: Dict[str, Dict[str, Set[str]]] = {}

    for mouse_idx, mouse_id in enumerate(mice):
        progress_bar.progress(
            (mouse_idx + 1) / n_mice,
            text=f"Processing {mouse_id} ({mouse_idx + 1}/{n_mice})",
        )

        mouse_df = filtered[filtered["mouse"] == mouse_id]
        per_organ_top, per_organ_membership, per_organ_detection_counts = (
            _compute_membership(mouse_df, organ_cells_selected, int(top_n))
        )

        if not per_organ_top:
            continue

        for organ, counts in per_organ_detection_counts.items():
            pooled_per_organ.setdefault(organ, []).extend(counts)
        _merge_pooled_membership(pooled_membership, per_organ_membership)

        if "Ridge" in plot_style:
            fig = build_clonotype_detection_ridge_figure(
                per_organ_detection_counts=per_organ_detection_counts,
                top_n=int(top_n),
                mouse_id=mouse_id,
                bandwidth=kde_bandwidth,
            )
        else:
            fig = build_clonotype_detection_histogram_figure(
                per_organ_detection_counts=per_organ_detection_counts,
                top_n=int(top_n),
                mouse_id=mouse_id,
            )
        if fig is None:
            continue

        with st.expander(
            f"{mouse_id} ({len(per_organ_top)} organ/cell types, top {int(top_n)})",
            expanded=(mouse_idx == 0),
        ):
            st.plotly_chart(fig, width="stretch")

            csv_data = _build_presence_csv(
                mouse_id, per_organ_top, per_organ_membership, organ_cells_selected
            )
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

    if pooled_per_organ:
        combined_mouse_id = "All individuals (pooled)"
        if "Ridge" in plot_style:
            fig = build_clonotype_detection_ridge_figure(
                per_organ_detection_counts=pooled_per_organ,
                top_n=int(top_n),
                mouse_id=combined_mouse_id,
                bandwidth=kde_bandwidth,
            )
        else:
            fig = build_clonotype_detection_histogram_figure(
                per_organ_detection_counts=pooled_per_organ,
                top_n=int(top_n),
                mouse_id=combined_mouse_id,
            )
        if fig is not None:
            with st.expander(
                f"{combined_mouse_id} ({len(pooled_per_organ)} organ/cell types, top {int(top_n)})",
                expanded=False,
            ):
                st.plotly_chart(fig, width="stretch")

                pooled_top: Dict[str, List[str]] = {
                    organ: sorted(
                        membership,
                        key=lambda c: (-len(membership[c]), c),
                    )
                    for organ, membership in pooled_membership.items()
                }
                csv_data = _build_presence_csv(
                    "all", pooled_top, pooled_membership, organ_cells_selected
                )
                style_prefix = "ridge" if "Ridge" in plot_style else "hist"
                safe_fn = (
                    f"{style_prefix}_all_{chain_selected.lower()}_top{int(top_n)}"
                    .replace(" ", "_")
                    .replace("|", "_")
                )

                render_plot_download_buttons(
                    fig,
                    base_filename=safe_fn,
                    key_prefix=f"{style_prefix}_all",
                    data=csv_data,
                    data_filename=f"{safe_fn}.csv",
                    data_index=False,
                )

    progress_bar.empty()
    st.success(f"Done — {n_mice} individual(s) processed.")