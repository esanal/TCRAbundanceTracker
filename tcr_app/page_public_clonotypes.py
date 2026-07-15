"""Public clonotypes page — shared clonotype network across individuals.

Features:
- Sidebar filter by chain, individuals, organs, cell types
- Hierarchy network (Individual → Organ → Subset → Clonotype)
- Clonotypes shared by >= N individuals
- Expandable summary table with shared-by counts and optional VDJdb antigen annotations
"""

from typing import Optional

import streamlit as st
import pandas as pd
from streamlit.components.v1 import html

from tcr_app.core import build_public_clonotype_network_html, enrich_clonotypes_with_vdjdb
from tcr_app.precomputed.load import load as load_precomputed


def _get_precomputed() -> Optional[dict]:
    """Load precomputed data for the current dataset key from session state or disk."""
    key = (
        st.session_state.get("precomputed_key")
        if "precomputed" in st.session_state
        else None
    )
    if key is not None:
        return st.session_state["precomputed"]
    if "dataset_name" in st.session_state:
        pre = load_precomputed(st.session_state["dataset_name"])
        if pre is not None:
            st.session_state["precomputed"] = pre
            st.session_state["precomputed_key"] = st.session_state["dataset_name"]
        return pre
    return None


def run_public_clonotypes_page(df: pd.DataFrame) -> None:
    """Render the public clonotype network view."""
    st.title("TCR Abundance Explorer")
    st.subheader("Public Clonotypes (shared across individuals)")
    st.markdown(
        """
    Visualise clonotypes that appear in multiple individuals.
    Each clonotype appears as a **single red node**; edges to different individuals/organs/subtypes show the sharing pattern.
    """
    )

    with st.sidebar:
        st.header("Filters")

        chain_selected = st.selectbox(
            "Chain", sorted(df["chain"].unique())
        )

        mouse_options = sorted(df["mouse"].unique())
        mouse_selected = st.multiselect(
            "Individuals", mouse_options, default=mouse_options
        )

        if mouse_selected:
            organ_options = sorted(
                df[(df["mouse"].isin(mouse_selected)) & (df["chain"] == chain_selected)]["organ"].unique()
            )
        else:
            organ_options = sorted(df["organ"].unique())
        organ_selected = st.multiselect(
            "Organ", organ_options, default=organ_options
        )

        if mouse_selected and organ_selected:
            cell_options = sorted(
                df[
                    (df["mouse"].isin(mouse_selected))
                    & (df["organ"].isin(organ_selected))
                    & (df["chain"] == chain_selected)
                ]["cell_type"].unique()
            )
        else:
            cell_options = sorted(df["cell_type"].unique())
        cell_selected = st.multiselect(
            "Cell type", cell_options, default=cell_options
        )

    filtered = df[
        (df["mouse"].isin(mouse_selected if mouse_selected else df["mouse"].unique()))
        & (df["organ"].isin(organ_selected if organ_selected else df["organ"].unique()))
        & (df["cell_type"].isin(cell_selected if cell_selected else df["cell_type"].unique()))
        & (df["chain"] == chain_selected)
    ].copy()

    if filtered.empty:
        st.warning("No data match the selected filters.")
        return

    with st.sidebar:
        st.header("Network settings")
        n_individuals_selected = len(mouse_selected)
        min_individuals = st.number_input(
            "Min individuals sharing",
            min_value=2,
            max_value=n_individuals_selected,
            value=min(2, n_individuals_selected),
        )
        max_clonotypes = st.number_input(
            "Max clonotypes to show", min_value=1, max_value=1000, value=100
        )
        show_clonotype_labels = st.checkbox("Show clonotype labels", value=False)
        node_font_size = st.slider("Node label font size", 10, 30, 16)

        st.divider()
        st.markdown("**Physics**")
        physics_mode = st.selectbox(
            "Physics mode",
            ["No physics", "Barnes-Hut", "Weak repulsion", "Compact clusters", "Force Atlas 2"],
            index=0,
        )
        has_physics = physics_mode != "No physics"
        gravity = st.slider(
            "Gravity",
            min_value=-2000, max_value=0, value=-800,
            step=50,
            disabled=not has_physics,
        )
        spring_length = st.slider(
            "Spring length",
            min_value=0, max_value=500, value=200,
            step=10,
            disabled=not has_physics,
        )

    # Stats
    clono_mouse_counts = filtered.groupby("clonotype")["mouse"].nunique()
    public_mask = clono_mouse_counts >= min_individuals
    n_public = public_mask.sum()
    n_total_clonos = len(filtered["clonotype"].unique())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total clonotypes", n_total_clonos)
    col2.metric(f"Public (≥{min_individuals} individuals)", n_public)
    col3.metric("Individuals selected", n_individuals_selected)

    if n_public == 0:
        st.info(
            f"No clonotypes are shared by ≥ {min_individuals} individuals with the current filters. "
            "Try lowering the minimum sharing count or selecting more individuals."
        )
        enable_vdjdb, max_vdjdb_queries, allow_fuzzy_vdjdb = _vdjdb_controls()
        vdjdb_df: Optional[pd.DataFrame] = None
        if enable_vdjdb and n_total_clonos > 0:
            vdjdb_df = _enrich_with_vdjdb(
                filtered, chain_selected, max_vdjdb_queries, allow_fuzzy_vdjdb
            )
        with st.expander("Clonotype sharing table (all)"):
            _show_public_table(filtered, clono_mouse_counts, chain_selected, vdjdb_df)
        return

    with st.spinner("Building public clonotype network…"):
        network_html = build_public_clonotype_network_html(
            filtered,
            min_individuals=min_individuals,
            max_clonotypes=max_clonotypes,
            show_clonotype_labels=show_clonotype_labels,
            node_font_size=node_font_size,
            physics_mode=physics_mode,
            gravity=gravity,
            spring_length=spring_length,
        )

    if network_html:
        html(network_html, height=720, scrolling=True)
    else:
        st.warning("Could not generate network view.")

    enable_vdjdb, max_vdjdb_queries, allow_fuzzy_vdjdb = _vdjdb_controls()
    vdjdb_df: Optional[pd.DataFrame] = None
    if enable_vdjdb and n_total_clonos > 0:
        vdjdb_df = _enrich_with_vdjdb(
            filtered, chain_selected, max_vdjdb_queries, allow_fuzzy_vdjdb
        )

    with st.expander("Clonotype sharing table", expanded=False):
        _show_public_table(filtered, clono_mouse_counts, chain_selected, vdjdb_df)


def _vdjdb_controls() -> tuple:
    """Render VDJdb toggle controls and return (enable, max_queries, allow_fuzzy)."""
    vdj_controls = st.columns([1, 1, 1])
    with vdj_controls[0]:
        enable = st.checkbox(
            "Enrich with VDJdb",
            value=False,
            help="Translate clonotypes to amino-acid CDR3, query VDJdb, then append match columns.",
        )
    with vdj_controls[1]:
        max_q = st.number_input(
            "Max VDJdb queries",
            min_value=1, max_value=1000, value=200, step=1,
            disabled=not enable,
        )
    with vdj_controls[2]:
        fuzzy = st.checkbox(
            "Fuzzy fallback",
            value=False,
            help="If no exact match, use VDJdb sequence fuzzy search.",
            disabled=not enable,
        )
    return enable, int(max_q), fuzzy


def _enrich_with_vdjdb(
    df: pd.DataFrame,
    chain_value: str,
    max_queries: int,
    allow_fuzzy: bool,
) -> pd.DataFrame:
    """Return VDJdb-enriched DataFrame for unique clonotypes in `df`, or empty if unavailable."""
    unique_clonos = list(
        dict.fromkeys(df["clonotype"].astype(str).str.strip().str.upper())
    )
    if not unique_clonos:
        return pd.DataFrame()

    pre = _get_precomputed()
    if pre is not None:
        pre_vdjdb = pre.get(f"{chain_value}_vdjdb_enriched")
        if pre_vdjdb is not None and not pre_vdjdb.empty:
            clono_set = set(unique_clonos)
            pre_vdjdb_filtered = pre_vdjdb[pre_vdjdb["clonotype"].isin(clono_set)].copy()
            if not pre_vdjdb_filtered.empty:
                matched_n = int((pre_vdjdb_filtered["vdjdb_match_count"] > 0).sum())
                st.caption(
                    f"VDJdb (precomputed): {len(pre_vdjdb_filtered)} sequences; "
                    f"{matched_n} had at least one match."
                )
                return pre_vdjdb_filtered
            st.info("No clonotypes matched precomputed VDJdb enrichment.")
            return pd.DataFrame()

    max_q = min(max_queries, len(unique_clonos))
    with st.spinner("Querying VDJdb for clonotype annotations..."):
        vdjdb_result = enrich_clonotypes_with_vdjdb(
            clonotypes=unique_clonos[:max_q],
            chain_value=chain_value,
            max_queries=max_q,
            allow_fuzzy_fallback=allow_fuzzy,
        )
    if vdjdb_result.empty:
        st.info("No clonotypes were sent to VDJdb.")
        return pd.DataFrame()

    queried_n = int(vdjdb_result["clonotype"].nunique())
    matched_n = int((vdjdb_result["vdjdb_match_count"] > 0).sum())
    error_n = int(vdjdb_result["vdjdb_error"].astype(str).str.len().gt(0).sum())
    st.caption(
        f"VDJdb queried {queried_n} unique sequences; {matched_n} had at least one match."
    )
    if error_n > 0:
        st.warning(f"VDJdb queries reported {error_n} errors.")
    return vdjdb_result


def _show_public_table(
    df: pd.DataFrame,
    clono_mouse_counts: pd.Series,
    chain_value: str,
    vdjdb_df: Optional[pd.DataFrame] = None,
) -> None:
    """Show a dataframe of public clonotypes with sharing stats and optional VDJdb annotations."""
    public_clonos = clono_mouse_counts[clono_mouse_counts >= 2].index.tolist()
    if not public_clonos:
        st.write("No public clonotypes found.")
        return

    public_df = df[df["clonotype"].isin(public_clonos)].copy()

    summary = (
        public_df.groupby("clonotype")
        .agg(
            shared_by_n=("mouse", "nunique"),
            individuals=("mouse", lambda x: ", ".join(dict.fromkeys(x))),
            organs=("organ", lambda x: ", ".join(dict.fromkeys(x))),
            cell_types=("cell_type", lambda x: ", ".join(dict.fromkeys(x))),
            total_abundance=("abundance", "sum"),
        )
        .reset_index()
        .sort_values("shared_by_n", ascending=False)
    )

    summary.insert(1, "Chain", chain_value)
    summary.columns = [
        "Clonotype",
        "Chain",
        "Individuals sharing",
        "Individuals",
        "Organs",
        "Cell types",
        "Total abundance",
    ]

    if vdjdb_df is not None and not vdjdb_df.empty:
        merge_cols = [
            "vdjdb_match_count",
            "vdjdb_top_antigen",
            "vdjdb_top_gene",
            "vdjdb_top_species",
            "vdjdb_top_mhc",
            "vdjdb_top_score",
            "vdjdb_query_cdr3_aa",
            "vdjdb_has_paired_record",
            "vdjdb_paired_cdr3_all",
            "vdjdb_paired_cdr3_tra",
            "vdjdb_paired_cdr3_trb",
            "vdjdb_error",
        ]
        vdjdb_subset = vdjdb_df[["clonotype"] + [c for c in merge_cols if c in vdjdb_df.columns]].copy()
        vdjdb_subset["clonotype"] = vdjdb_subset["clonotype"].astype(str).str.strip().str.upper()
        summary["clonotype_lookup"] = summary["Clonotype"].astype(str).str.strip().str.upper()
        summary = summary.merge(vdjdb_subset, left_on="clonotype_lookup", right_on="clonotype", how="left", suffixes=("", "_vdj"))
        summary = summary.drop(columns=["clonotype_lookup", "clonotype_vdj"], errors="ignore")
        summary = summary.drop(columns=[c for c in summary.columns if c.endswith("_vdj")], errors="ignore")

    st.dataframe(summary, use_container_width=True, hide_index=True)
