"""Core business logic and visualization engine for the TCR Abundance Explorer.

Provides all shared functions used across pages:
- Data loading, column normalization, and validation
- Clonotype sharing, network metrics, and similarity calculations
- All Plotly figure builders (heatmaps, line plots, stacked flow, presence grids, chord diagrams)
- PyVis interactive network HTML generation
- VDJdb TCR-antigen annotation integration
"""
import copy
import io

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
from urllib import error, request

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from streamlit.runtime.scriptrunner import get_script_run_ctx
import math
import numpy as np

CANONICAL_COLUMNS: Dict[str, List[str]] = {
    "mouse": ["mouse", "individual", "mouse_id", "animal", "animal_id"],
    "organ": ["organ", "tissue"],
    "cell_type": ["cell_type", "celltype", "cell type", "cell", "celltype", "cell.type", "Cell"],
    "chain": ["chain", "tcr_chain"],
    "clonotype": ["clonotype", "clonetype", "cdr3", "sequence", "tcr", "nSeqCDR3"],
    "abundance": ["abundance", "count", "frequency", "freq", "Abundance"]
}

REQUIRED_COLUMNS = ["mouse", "organ", "cell_type", "chain", "clonotype", "abundance"]
SELECTED_ORGAN_CELL_COLOR = "#d62728"
DEFAULT_EDGE_WIDTH_SCALE = 0.2
DEFAULT_GRAVITY = -2200
DEFAULT_SPRING_LENGTH = 180
PSEUDO_ZERO = 1e-4
VDJDB_API_BASE_URL = "https://vdjdb.com/api/database"
DNA_BASES = set("ACGTUN")
CODON_TABLE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
#EXAMPLE_DATASET_FILENAME = "test.abundance.1.csv"
EXAMPLE_DATASET_FILENAME = "output.csv"
FALLBACK_EXAMPLE_CSV = """mouse,organ,cell_type,chain,clonotype,abundance,sample
MouseA,Spleen,CD4,TCRB,CLN001,120,S1
MouseA,Spleen,CD8,TCRB,CLN002,80,S1
MouseA,Lung,CD4,TCRB,CLN001,40,S2
MouseA,Lung,CD8,TCRB,CLN003,65,S2
MouseB,Spleen,CD4,TCRA,CLN010,95,S3
MouseB,Lung,CD8,TCRA,CLN011,55,S4
"""


def load_example_dataframe() -> pd.DataFrame:
    """Load the bundled example CSV dataset, or fall back to a hardcoded mini example."""
    example_path_candidates = [
        Path(EXAMPLE_DATASET_FILENAME),
        Path(__file__).resolve().parent / EXAMPLE_DATASET_FILENAME,
    ]
    for path in example_path_candidates:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    return pd.read_csv(io.StringIO(FALLBACK_EXAMPLE_CSV), low_memory=False)


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Map variant column names to canonical names using CANONICAL_COLUMNS."""
    mapping: Dict[str, str] = {}
    lower_cols = {col.lower(): col for col in df.columns}
    for canonical, options in CANONICAL_COLUMNS.items():
        for option in options:
            if option.lower() in lower_cols:
                mapping[lower_cols[option.lower()]] = canonical
                break
    df = df.rename(columns=mapping)
    return df, mapping


def validate_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check that all required columns are present after normalization."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def classify_cd4_cd8(cell_type: str) -> str:
    """Classify a cell type string as 'CD4', 'CD8', or 'Other' using word-boundary regex."""
    upper = str(cell_type).upper()
    if re.search(r"\bCD8\b", upper):
        return "CD8"
    if re.search(r"\bCD4\b", upper):
        return "CD4"
    return "Other"


def get_organ_cell_order(df: pd.DataFrame, sort_mode: str) -> List[str]:
    """Return an ordered list of organ|cells groups.

    'organ' mode: alphabetical sort.
    'trm' mode: sort by cell type prefix, then organ (for TRM-focused layouts).
    """
    if df.empty:
        return []
    if "organ_cell" not in df.columns:
        return []
    if sort_mode == "trm":
        df_local = df.copy()
        df_local["organ_cell"] = df_local["organ_cell"].astype(str)
        if "organ" not in df_local.columns or "cell_type" not in df_local.columns:
            split_cols = df_local["organ_cell"].str.split(" | ", n=1, regex=False, expand=True)
            if split_cols.shape[1] >= 2:
                df_local["organ"] = split_cols[0]
                df_local["cell_type"] = split_cols[1]
            else:
                return sorted(df_local["organ_cell"].unique())
        df_local["organ"] = df_local["organ"].astype(str).str.strip()
        df_local["cell_type"] = df_local["cell_type"].astype(str).str.strip()
        df_local["cell_sort_key"] = (
            df_local["cell_type"]
            .str.split()
            .str[:2]
            .str.join(" ")
            .str.lower()
        )

        cell_order = (
            df_local
            #.drop_duplicates("cell_type")
            .sort_values(["cell_sort_key", "organ"], kind="mergesort")["organ_cell"]
            .tolist()
        )
        # Keep stable order while removing duplicates.
        cell_order = list(dict.fromkeys(cell_order))
        #existing_pairs = set(
        #    zip(df_local["cell_type"].tolist(), df_local["organ"].tolist())
        #)
        #print(df_local)
        #ordered: List[str] = []
        #for cell in cell_order:
        #    for organ in organ_order:
        #        if (cell, organ) in existing_pairs:
        #            ordered.append(f"{organ} | {cell}")
        return cell_order
    return sorted(df["organ_cell"].astype(str).unique())


def build_organ_cell_clonotype_edges(
    df: pd.DataFrame,
    top_clonotypes: List[str],
    min_edge_abundance: float,
) -> pd.DataFrame:
    """Build a DataFrame of organ|cells-to-clonotype edges with summed abundances.

    Filters to the provided top clonotypes and applies a minimum abundance threshold.
    """
    filtered = df[df["clonotype"].isin(top_clonotypes)].copy()
    edge_df = filtered.groupby(["organ_cell", "clonotype"], as_index=False)["abundance"].sum()
    edge_df["abundance"] = pd.to_numeric(edge_df["abundance"], errors="coerce").fillna(0)
    edge_df = edge_df[edge_df["abundance"] >= min_edge_abundance].copy()
    return edge_df


def calculate_organ_cell_sharing(
    edge_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute clonotype sharing between organ|cells groups.

    Returns:
        pair_df: pairwise shared clonotype counts between organ|cells groups.
        organ_cell_summary: per-group total shared + unique clonotype counts.
        shared_matrix: organ|cells x organ|cells presence co-occurrence matrix.
    """
    if edge_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    presence = (
        edge_df.assign(present=1)
        .pivot_table(
            index="organ_cell",
            columns="clonotype",
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .astype(int)
    )
    shared_matrix = presence.dot(presence.T)
    diagonal = pd.Series(
        shared_matrix.values.diagonal(),
        index=shared_matrix.index,
        dtype=float,
    )
    for node in shared_matrix.index:
        shared_matrix.loc[node, node] = 0
    organ_cell_sharing_score = shared_matrix.sum(axis=1).astype(float)

    pair_records: List[Dict[str, float]] = []
    organ_cells = list(shared_matrix.index)
    for idx, node_a in enumerate(organ_cells):
        for node_b in organ_cells[idx + 1 :]:
            shared_count = float(shared_matrix.loc[node_a, node_b])
            if shared_count > 0:
                pair_records.append(
                    {
                        "organ_cell_a": node_a,
                        "organ_cell_b": node_b,
                        "shared_clonotypes": shared_count,
                    }
                )
    pair_df = pd.DataFrame(pair_records)
    if pair_df.empty:
        pair_df = pd.DataFrame(
            columns=["organ_cell_a", "organ_cell_b", "shared_clonotypes"]
        )
    else:
        pair_df = pair_df.sort_values(
            by="shared_clonotypes",
            ascending=False,
        ).reset_index(drop=True)

    organ_cell_summary = (
        pd.DataFrame(
            {
                "organ_cell": organ_cell_sharing_score.index,
                "total_shared_clonotypes": organ_cell_sharing_score.values,
                "unique_clonotypes": diagonal.reindex(organ_cell_sharing_score.index).values,
            }
        )
        .sort_values("total_shared_clonotypes", ascending=False)
        .reset_index(drop=True)
    )
    return pair_df, organ_cell_summary, shared_matrix

def build_highlighted_tick_labels(
    categories: List[str],
    selected_category: str,
    highlight_color: str = SELECTED_ORGAN_CELL_COLOR,
) -> List[str]:
    """Wrap the selected category in an HTML span with a highlight color for Plotly tick labels."""
    tick_labels: List[str] = []
    for category in categories:
        if category == selected_category:
            tick_labels.append(
                f"<span style='color:{highlight_color}; font-weight:700'>{category}</span>"
            )
        else:
            tick_labels.append(category)
    return tick_labels


def build_organ_cell_clonotype_network_html(
    edge_df: pd.DataFrame,
    organ_cell_summary: pd.DataFrame,
    selected_organ_cell: str,
    selected_node: Optional[str],
    show_clonotype_labels: bool,
    gravity: int,
    spring_length: int,
    edge_width_scale: float,
    physics_mode: str,
    node_font_size: int,
) -> str:
    """Generate an interactive PyVis (vis.js) HTML network of organ|cells and clonotype nodes.

    Organ|cells nodes are placed on an outer ring, clonotype nodes on an inner ring.
    Edge widths are proportional to abundance. Returns an HTML string for embedding.
    """
    if edge_df.empty:
        return ""

    organ_cell_score_map = (
        organ_cell_summary.set_index("organ_cell")["total_shared_clonotypes"].to_dict()
        if not organ_cell_summary.empty
        else {}
    )
    net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="#222222")

    def make_physics_config() -> Dict[str, object]:
        base = {
            "enabled": physics_mode != "No physics",
            "solver": "barnesHut"
            if physics_mode != "Force Atlas 2"
            else "forceAtlas2Based",
            "barnesHut": {
                "gravitationalConstant": gravity,
                "centralGravity": 0.3,
                "springLength": spring_length,
                "avoidOverlap": 0.5,
            },
        }
        if "Weak repulsion" in physics_mode:
            base["barnesHut"].update(
                {
                    "springLength": spring_length * 1.4,
                    "gravitationalConstant": gravity * 0.65,
                    "centralGravity": 0.08,
                }
            )
        elif "Compact clusters" in physics_mode:
            base["barnesHut"].update(
                {
                    "springLength": spring_length * 0.7,
                    "centralGravity": 0.5,
                    "avoidOverlap": 0.8,
                }
            )
        elif "Force Atlas 2" in physics_mode:
            base["forceAtlas2Based"] = {
                "adjustSizes": False,
                "centralGravity": 0.01,
                "springLength": spring_length,
                "springConstant": 0.01,
                "damping": 0.6,
            }
        elif "No physics" in physics_mode:
            base["barnesHut"].update({"gravitationalConstant": 0, "springLength": 0})
        return base

    organ_cell_nodes = sorted(edge_df["organ_cell"].unique())
    clonotype_nodes = sorted(edge_df["clonotype"].unique())
    center_x = 0
    center_y = 0
    organ_radius = 250
    clonotype_radius = 130

    max_node_score = max(organ_cell_score_map.values(), default=0.0)
    min_node_size = 18
    size_range = 22
    clonotype_font_size = max(node_font_size - 2, 14)
    for idx, organ_cell in enumerate(organ_cell_nodes):
        score = float(organ_cell_score_map.get(organ_cell, 0.0))
        normalized = (score / max_node_score) if max_node_score > 0 else 0.0
        node_size = min_node_size + normalized * size_range
        is_selected = organ_cell == selected_organ_cell or organ_cell == selected_node
        net.add_node(
            organ_cell,
            label=organ_cell.replace(" | ", "\n"),
            color=(
                {
                    "background": SELECTED_ORGAN_CELL_COLOR,
                    "border": "#8b1d1d",
                    "highlight": {"background": SELECTED_ORGAN_CELL_COLOR, "border": "#8b1d1d"},
                }
                if is_selected
                else "#1f77b4"
            ),
            title=f"Organ/Cell: {organ_cell}<br>Total shared clonotypes: {score:.0f}",
            shape="box",
            level=0,
            value=node_size + 6 if is_selected else node_size,
            font={"size": node_font_size, "color": "#ffffff"},
            borderWidth=3 if is_selected else 1,
            x=center_x + organ_radius * math.cos(2 * math.pi * idx / len(organ_cell_nodes)),
            y=center_y + organ_radius * math.sin(2 * math.pi * idx / len(organ_cell_nodes)),
        )

    for idx, clonotype in enumerate(clonotype_nodes):
        display_label = clonotype if show_clonotype_labels else " "
        display_size = 12 if show_clonotype_labels else 0
        is_selected_clonotype = clonotype == selected_node
        net.add_node(
            clonotype,
            label=display_label,
            color=(
                {
                    "background": SELECTED_ORGAN_CELL_COLOR,
                    "border": "#8b1d1d",
                    "highlight": {"background": SELECTED_ORGAN_CELL_COLOR, "border": "#8b1d1d"},
                }
                if is_selected_clonotype
                else "#ff7f0e"
            ),
            title=f"Clonotype: {clonotype}",
            font={"size": clonotype_font_size, "color": "#ffffff"},
            level=1,
            borderWidth=3 if is_selected_clonotype else 1,
            x=center_x + clonotype_radius * math.cos(2 * math.pi * idx / len(clonotype_nodes)),
            y=center_y + clonotype_radius * math.sin(2 * math.pi * idx / len(clonotype_nodes)),
        )

    for _, row in edge_df.iterrows():
        net.add_edge(
            row["organ_cell"],
            row["clonotype"],
            value=row["abundance"],
            width=max(1, row["abundance"] * edge_width_scale),
            title=f"Abundance: {row['abundance']}",
        )

    options = {
        "physics": make_physics_config(),
        "nodes": {
            "shape": "dot",
            "size": 12,
            "font": {"size": node_font_size, "color": "#ffffff"},
            "color": {
                "highlight": {
                    "border": "FF0000",
                    "background": "FF5555",
                }
            },
        },
        "edges": {
            "color": {"inherit": True},
            "smooth": False,
        },
        "interaction": {
            "navigationButtons": True,
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
        },
        "layout": {"improvedLayout": False},
    }
    net.set_options(json.dumps(options))
    return net.generate_html()


def superscript(exponent: int) -> str:
    """Convert an integer to a superscript Unicode string (e.g., 3 → '³', -5 → '⁻⁵')."""
    sup_digits = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹",
    }
    if exponent == 0:
        return sup_digits["0"]
    parts = []
    if exponent < 0:
        parts.append("⁻")
        exponent = abs(exponent)
    for digit in str(exponent):
        parts.append(sup_digits.get(digit, digit))
    return "".join(parts)


def _hex_to_rgba(color: str, alpha: float) -> str:
    """Convert a hex color string (#RRGGBB) or rgb() to rgba() with the given alpha."""
    color = str(color).strip()
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("rgb(") and color.endswith(")"):
        return color.replace("rgb(", "rgba(").replace(")", f", {alpha})")
    return f"rgba(68,68,68,{alpha})"


def build_clonotype_color_map(clonotypes: List[str]) -> Dict[str, str]:
    """Assign a stable color from a combined Plotly/Dark24/Alphabet/Set3 palette to each clonotype."""
    palette = (
        px.colors.qualitative.Plotly
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Alphabet
        + px.colors.qualitative.Set3
    )
    ordered = [str(clonotype) for clonotype in clonotypes]
    return {
        clonotype: palette[idx % len(palette)]
        for idx, clonotype in enumerate(ordered)
    }


def build_stacked_clonotype_band_figure(
    lineage_df: pd.DataFrame,
    selected_clonotypes: List[str],
    selected_organ_cell: str,
    lineage_label: str,
    clonotype_color_map: Dict[str, str],
    organ_cell_order: Optional[List[str]] = None,
    aggregate_non_selected: bool = False,
    non_selected_label: str = "Other clonotypes",
) -> Optional[go.Figure]:
    """Build a stacked bar figure with semi-transparent connecting bands between adjacent organ|cells bars.

    Each selected clonotype gets a colored bar segment and a polygon band that connects
    its top/bottom across consecutive x-axis positions, showing abundance flow.
    """
    if lineage_df.empty:
        return None

    lineage_df = lineage_df.copy()
    lineage_df["clonotype"] = lineage_df["clonotype"].astype(str)
    available_organ_cells = sorted(lineage_df["organ_cell"].astype(str).unique())
    if organ_cell_order:
        available_set = set(available_organ_cells)
        lineage_organ_cells = [item for item in organ_cell_order if item in available_set]
        ordered_set = set(lineage_organ_cells)
        lineage_organ_cells.extend(
            [item for item in available_organ_cells if item not in ordered_set]
        )
    else:
        lineage_organ_cells = available_organ_cells
    if not lineage_organ_cells:
        return None

    all_clonotypes = (
        lineage_df.groupby("clonotype", as_index=False)["abundance"]
        .sum()
        .sort_values(["abundance", "clonotype"], ascending=[False, True], kind="mergesort")
    )["clonotype"].astype(str).tolist()
    lineage_pivot = (
        lineage_df.pivot_table(
            index="clonotype",
            columns="organ_cell",
            values="abundance",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=all_clonotypes, columns=lineage_organ_cells, fill_value=0.0)
        .astype(float)
    )
    lineage_pool_size = float(lineage_df["abundance"].sum())
    #print(lineage_pool_size)
    #if lineage_pool_size > 0:
    #lineage_pivot = (lineage_pivot / lineage_pool_size) * 100.0

    selected_set = {str(clonotype) for clonotype in selected_clonotypes}
    active_all_clonotypes = [
        clonotype
        for clonotype in all_clonotypes
        if clonotype in lineage_pivot.index and float(lineage_pivot.loc[clonotype].sum()) > 0
    ]
    if not active_all_clonotypes:
        return None
    selected_active_clonotypes = [
        clonotype for clonotype in active_all_clonotypes if clonotype in selected_set
    ]

    if aggregate_non_selected:
        non_selected_clonotypes = [
            clonotype for clonotype in active_all_clonotypes if clonotype not in selected_set
        ]
        bar_clonotypes = selected_active_clonotypes.copy()
        if non_selected_clonotypes:
            bar_clonotypes.append(non_selected_label)
    else:
        bar_clonotypes = [
            clonotype for clonotype in selected_clonotypes if clonotype in active_all_clonotypes
        ]
        if not bar_clonotypes:
            return None

    x_values = list(range(len(lineage_organ_cells)))
    cumulative = np.zeros(len(lineage_organ_cells), dtype=float)
    segment_bounds: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for clonotype in bar_clonotypes:
        if aggregate_non_selected and clonotype == non_selected_label:
            y_vals = np.zeros(len(lineage_organ_cells), dtype=float)
            for other_clonotype in non_selected_clonotypes:
                y_vals = y_vals + lineage_pivot.loc[other_clonotype].to_numpy(dtype=float)
        else:
            y_vals = lineage_pivot.loc[clonotype].to_numpy(dtype=float)
        lower = cumulative.copy()
        upper = lower + y_vals
        segment_bounds[clonotype] = (lower, upper, y_vals)
        cumulative = upper

    flow_fig = go.Figure()
    bar_width = 0.62
    bar_half_width = bar_width / 2.0

    for clonotype in selected_active_clonotypes:
        if clonotype not in segment_bounds:
            continue
        lower, upper, y_vals = segment_bounds[clonotype]
        band_color = _hex_to_rgba(clonotype_color_map.get(clonotype, "#444444"), 0.23)
        polygon_x: List[float] = []
        polygon_y: List[float] = []
        for idx in range(len(x_values) - 1):
            if y_vals[idx] <= 0 and y_vals[idx + 1] <= 0:
                continue
            left_x = x_values[idx] + bar_half_width
            right_x = x_values[idx + 1] - bar_half_width
            polygon_x.extend([left_x, right_x, right_x, left_x, left_x, None])
            polygon_y.extend(
                [
                    lower[idx],
                    lower[idx + 1],
                    upper[idx + 1],
                    upper[idx],
                    lower[idx],
                    None,
                ]
            )
        if polygon_x:
            flow_fig.add_trace(
                go.Scatter(
                    x=polygon_x,
                    y=polygon_y,
                    mode="lines",
                    line={"width": 0},
                    fill="toself",
                    fillcolor=band_color,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=clonotype,
                )
            )

    for clonotype in bar_clonotypes:
        _, _, y_vals = segment_bounds[clonotype]
        is_selected = clonotype in selected_set or (not aggregate_non_selected)
        marker_color = (
            clonotype_color_map.get(clonotype, "#444444")
            if is_selected and clonotype != non_selected_label
            else "#d9d9d9"
        )
        flow_fig.add_trace(
            go.Bar(
                x=x_values,
                y=y_vals,
                width=[bar_width] * len(x_values),
                name=clonotype,
                marker={"color": marker_color},
                opacity=1.0 if (is_selected and clonotype != non_selected_label) else 0.7,
                legendgroup=clonotype,
                showlegend=(clonotype in selected_set) or (clonotype == non_selected_label),
                hovertemplate=(
                    "Organ/Cell: %{customdata[0]}<br>"
                    "Clonotype: %{fullData.name}<br>"
                    "Abundance: %{y:.3f}%<extra></extra>"
                ),
                customdata=np.array(lineage_organ_cells, dtype=object).reshape(-1, 1),
            )
        )

    flow_fig.update_layout(
        barmode="stack",
        bargap=0.0,
        height=460,
        title=f"{lineage_label} stacked clonotype abundance flow",
        xaxis_title="Organ/Cell",
        yaxis_title="% Pool Size",
        legend_title="Clonotype",
        legend={"groupclick": "togglegroup"},
    )
    flow_fig.update_xaxes(
        tickmode="array",
        tickvals=x_values,
        ticktext=build_highlighted_tick_labels(lineage_organ_cells, selected_organ_cell),
    )

    return flow_fig


def build_clonotype_presence_grid_figure(
    df: pd.DataFrame,
    selected_clonotypes: List[str],
    top_n: int,
    query_top_n: str,
    selected_organ_cell: str,
    show_clonotype_sequences: bool = True,
    x_spacing: float = 0.62,
    row_categories: Optional[List[str]] = None,
) -> Optional[go.Figure]:
    """Build a Kiki-style presence grid: rows = organ|cells, columns = clonotypes.

    Red dot: clonotype is in the top-N of that row.
    Blue dot: clonotype is present but not in the top-N of that row.
    White/empty: clonotype is absent from that row.

    Side annotations show R(ed)/B(lue)/T(otal) counts per row and per column.
    """
    top_in_row_label = f"Top-{int(top_n)} in row"
    present_not_top_label = f"Present (not Top-{int(top_n)} in row)"
    grid_df = build_clonotype_presence_grid_dataframe(
        df=df,
        selected_clonotypes=selected_clonotypes,
        top_n=top_n,
        query_top_n=query_top_n,
        row_categories=row_categories,
        x_spacing=x_spacing,
    )
    if grid_df.empty:
        return None

    grid_rows = (
        list(row_categories)
        if row_categories is not None
        else sorted(df["organ_cell"].unique())
    )
    grid_cols = [str(clonotype) for clonotype in selected_clonotypes]
    x_positions = [idx * x_spacing for idx in range(len(grid_cols))]
    x_tick_labels = (
        grid_cols
        if show_clonotype_sequences
        else [f"C{idx + 1}" for idx in range(len(grid_cols))]
    )

    row_summary, col_summary = summarize_clonotype_presence_grid_counts(
        grid_df=grid_df,
        top_in_row_label=top_in_row_label,
        present_not_top_label=present_not_top_label,
    )

    grid_fig = go.Figure()
    status_styles = [
        (top_in_row_label, "#d62728", "#7f1d1d"),
        (present_not_top_label, "#1f77b4", "#1f77b4"),
        ("Not present", "#ffffff", "#9ca3af"),
    ]
    for status, fill_color, line_color in status_styles:
        status_df = grid_df[grid_df["status"] == status]
        if status_df.empty:
            continue
        grid_fig.add_trace(
            go.Scatter(
                x=status_df["col_idx"],
                y=status_df["row_idx"],
                mode="markers",
                name=status,
                marker={
                    "size": 12,
                    "symbol": "circle",
                    "color": fill_color,
                    "line": {"width": 1.3, "color": line_color},
                },
                customdata=status_df[["organ_cell", "clonotype", "status"]].to_numpy(),
                hovertemplate=(
                    "Organ/Cell: %{customdata[0]}<br>"
                    "Clonotype: %{customdata[1]}<br>"
                    "State: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    grid_fig.update_layout(
        height=max(300, 80 + 38 * len(grid_rows)),
        xaxis_title=f"Top {int(top_n)} clonotypes from selected organ|cell",
        yaxis_title=None,
        plot_bgcolor="#ffffff",
        showlegend=False,
        margin={"l": 80, "r": 110, "t": 80, "b": 70},
    )
    grid_fig.update_xaxes(
        tickmode="array",
        tickvals=x_positions,
        ticktext=x_tick_labels,
        tickangle=90 if show_clonotype_sequences else 0,
        range=[-0.3 * x_spacing, (len(grid_cols) - 1 + 0.3) * x_spacing],
        showgrid=False,
        zeroline=False,
    )
    grid_fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(grid_rows))),
        ticktext=build_highlighted_tick_labels(grid_rows, selected_organ_cell),
        autorange="reversed",
        range=[len(grid_rows) - 0.5, -0.5],
        showgrid=False,
        zeroline=False,
    )

    for _, row in row_summary.iterrows():
        red_count = int(row.get(top_in_row_label, 0))
        blue_count = int(row.get(present_not_top_label, 0))
        total_count = red_count + blue_count
        grid_fig.add_annotation(
            x=1.01,
            xref="paper",
            y=float(row["row_idx"]),
            yref="y",
            text=(
                f"<span style='color:#d62728'>R:{red_count}</span> "
                f"<span style='color:#1f77b4'>B:{blue_count}</span> "
                f"<span style='color:#111827'>T:{total_count}</span>"
            ),
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            align="left",
            font={"size": 11},
        )

    top_annotation_y = -0.85
    for _, col in col_summary.iterrows():
        red_count = int(col.get(top_in_row_label, 0))
        blue_count = int(col.get(present_not_top_label, 0))
        total_count = red_count + blue_count
        grid_fig.add_annotation(
            x=float(col["col_idx"]),
            xref="x",
            y=top_annotation_y,
            yref="y",
            text=(
                f"<span style='color:#d62728'>R:{red_count}</span><br>"
                f"<span style='color:#1f77b4'>B:{blue_count}</span><br>"
                f"<span style='color:#111827'>T:{total_count}</span>"
            ),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            align="center",
            font={"size": 11},
        )

    return grid_fig


def build_clonotype_presence_grid_dataframe(
    df: pd.DataFrame,
    selected_clonotypes: List[str],
    top_n: int,
    query_top_n: str,
    row_categories: Optional[List[str]] = None,
    x_spacing: float = 0.62,
) -> pd.DataFrame:
    """Build the underlying DataFrame for the Kiki presence grid.

    Each row records (row_idx, col_idx, organ_cell, clonotype, status) where
    status is one of 'Top-N in row', 'Present (not Top-N in row)', or 'Not present'.
    The query_top_n parameter controls how many clonotypes to consider "present" per row
    (use 'all' for unlimited, or a number to cap it).
    """
    if df.empty or not selected_clonotypes:
        return pd.DataFrame()

    organ_cell_totals = (
        df.groupby(["organ_cell", "clonotype"], as_index=False)["abundance"]
        .sum()
        .sort_values(
            ["organ_cell", "abundance", "clonotype"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    row_topn = organ_cell_totals.groupby("organ_cell").head(int(top_n)).copy()
    row_topn_map = row_topn.groupby("organ_cell")["clonotype"].apply(set).to_dict()
    if query_top_n == "all":
        row_present_map = organ_cell_totals[organ_cell_totals["abundance"] > 0].groupby(
            "organ_cell"
        )["clonotype"].apply(set).to_dict()
    elif query_top_n:
        row_present_map = (
            organ_cell_totals[organ_cell_totals["abundance"] > 0]
            .groupby("organ_cell")
            .apply(
                lambda x: x.nlargest(int(query_top_n), "abundance", keep="all"),
                include_groups=False,
            )
            .groupby(level=0)["clonotype"]
            .apply(set)
            .to_dict()
        )
    else:
        row_present_map = {}

    available_rows = set(df["organ_cell"].unique())
    grid_rows = (
        list(row_categories)
        if row_categories is not None
        else sorted(df["organ_cell"].unique())
    )
    grid_cols = [str(clonotype) for clonotype in selected_clonotypes]
    if not grid_rows or not grid_cols:
        return pd.DataFrame()

    x_positions = [idx * x_spacing for idx in range(len(grid_cols))]
    top_in_row_label = f"Top-{int(top_n)} in row"
    present_not_top_label = f"Present (not Top-{int(top_n)} in row)"

    grid_records: List[Dict[str, object]] = []
    for row_idx, organ_cell in enumerate(grid_rows):
        if organ_cell not in available_rows:
            continue
        topn_set = row_topn_map.get(organ_cell, set())
        present_set = row_present_map.get(organ_cell, set())
        for col_idx, clonotype in enumerate(grid_cols):
            if clonotype in topn_set:
                status = top_in_row_label
            elif clonotype in present_set:
                status = present_not_top_label
            else:
                status = "Not present"
            grid_records.append(
                {
                    "row_idx": row_idx,
                    "col_idx": x_positions[col_idx],
                    "organ_cell": organ_cell,
                    "clonotype": clonotype,
                    "status": status,
                }
            )
    return pd.DataFrame(grid_records)


def summarize_clonotype_presence_grid_counts(
    grid_df: pd.DataFrame,
    top_in_row_label: str,
    present_not_top_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate the presence grid into per-row and per-column Red/Blue/Total count summaries."""
    if grid_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    row_summary = (
        grid_df.groupby(["row_idx", "organ_cell"], as_index=False)
        .agg(
            **{
                top_in_row_label: (
                    "status",
                    lambda s: int((s == top_in_row_label).sum()),
                ),
                present_not_top_label: (
                    "status",
                    lambda s: int((s == present_not_top_label).sum()),
                ),
            }
        )
        .sort_values(["row_idx", "organ_cell"], kind="mergesort")
        .reset_index(drop=True)
    )
    col_summary = (
        grid_df.groupby(["col_idx", "clonotype"], as_index=False)
        .agg(
            **{
                top_in_row_label: (
                    "status",
                    lambda s: int((s == top_in_row_label).sum()),
                ),
                present_not_top_label: (
                    "status",
                    lambda s: int((s == present_not_top_label).sum()),
                ),
            }
        )
        .sort_values(["col_idx", "clonotype"], kind="mergesort")
        .reset_index(drop=True)
    )
    for summary_df in [row_summary, col_summary]:
        if top_in_row_label not in summary_df.columns:
            summary_df[top_in_row_label] = 0
        if present_not_top_label not in summary_df.columns:
            summary_df[present_not_top_label] = 0
        summary_df["total"] = (
            summary_df[top_in_row_label].astype(int)
            + summary_df[present_not_top_label].astype(int)
        )
    return row_summary, col_summary


def build_clonotype_presence_count_histogram(
    summary_df: pd.DataFrame,
    axis_label: str,
    top_n: int,
    selected_label: Optional[str] = None,
    show_clonotype_sequences: bool = True,
) -> Optional[go.Figure]:
    """Build a grouped bar chart of Red/Blue/Total counts per row or per column from the presence grid summary."""
    if summary_df.empty:
        return None

    top_in_row_label = f"Top-{int(top_n)} in row"
    present_not_top_label = f"Present (not Top-{int(top_n)} in row)"
    value_df = summary_df.rename(
        columns={
            top_in_row_label: "Red",
            present_not_top_label: "Blue",
            "total": "Total",
        }
    )
    label_column = "organ_cell" if axis_label == "Row" else "clonotype"
    display_label_column = label_column
    if axis_label != "Row" and not show_clonotype_sequences:
        value_df = value_df.copy()
        value_df["display_label"] = [
            f"C{idx + 1}" for idx in range(len(value_df))
        ]
        display_label_column = "display_label"
    plot_df = value_df[[label_column, "Red", "Blue", "Total"]].copy()
    if display_label_column != label_column:
        plot_df["display_label"] = value_df["display_label"]
        plot_df = plot_df.rename(columns={"display_label": "label"})
    else:
        plot_df = plot_df.rename(columns={label_column: "label"})
    plot_df["label"] = plot_df["label"].astype(str)
    plot_df = plot_df.melt(
        id_vars="label",
        value_vars=["Red", "Blue", "Total"],
        var_name="count_type",
        value_name="count",
    )
    if plot_df.empty:
        return None

    fig = px.bar(
        plot_df,
        x="label",
        y="count",
        color="count_type",
        barmode="group",
        color_discrete_map={"Red": "#d62728", "Blue": "#1f77b4", "Total": "#111827"},
        labels={
            "label": axis_label,
            "count": "Count",
            "count_type": "Series",
        },
        category_orders={"count_type": ["Red", "Blue", "Total"]},
    )
    fig.update_layout(
        title=f"{axis_label} counts",
        height=360,
        margin={"l": 20, "r": 20, "t": 60, "b": 40},
        bargap=0.15,
        yaxis={"range": [0, int(top_n)], "title": "Count"},
    )
    if axis_label == "Row":
        tickvals = value_df["organ_cell"].astype(str).tolist()
        ticktext = build_highlighted_tick_labels(tickvals, selected_label or "")
        fig.update_xaxes(
            tickangle=45,
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
        )
    else:
        fig.update_xaxes(tickangle=90)
    return fig


def build_aggregate_row_count_figure(
    row_summaries: List[pd.DataFrame],
    top_n: int,
    selected_label: Optional[str] = None,
) -> Optional[go.Figure]:
    """Aggregate row summaries across individuals into a grouped bar chart of median Red/Blue/Total counts.

    Bars show the median across individuals; overlaid scatter dots show individual values.
    """
    if not row_summaries:
        return None

    non_empty = [summary.copy() for summary in row_summaries if not summary.empty]
    if not non_empty:
        return None

    top_in_row_label = f"Top-{int(top_n)} in row"
    present_not_top_label = f"Present (not Top-{int(top_n)} in row)"
    combined = pd.concat(non_empty, ignore_index=True)
    combined = combined.rename(
        columns={
            top_in_row_label: "Red",
            present_not_top_label: "Blue",
            "total": "Total",
        }
    )

    median = (
        combined.groupby(["row_idx", "organ_cell"], as_index=False)[["Red", "Blue", "Total"]]
        .median()
        .sort_values(["row_idx", "organ_cell"], kind="mergesort")
        .reset_index(drop=True)
    )
    plot_df = median.melt(
        id_vars=["row_idx", "organ_cell"],
        value_vars=["Red", "Blue", "Total"],
        var_name="count_type",
        value_name="median_count",
    )
    individual_df = combined.melt(
        id_vars=["row_idx", "organ_cell"],
        value_vars=["Red", "Blue", "Total"],
        var_name="count_type",
        value_name="count",
    )
    if plot_df.empty:
        return None

    colors = {"Red": "#d62728", "Blue": "#1f77b4", "Total": "#111827"}
    count_types = ["Red", "Blue", "Total"]
    x_labels = median["organ_cell"].astype(str).tolist()
    x_positions = list(range(len(x_labels)))
    position_lookup = {
        (row["row_idx"], row["organ_cell"]): idx
        for idx, row in median.sort_values(["row_idx", "organ_cell"], kind="mergesort").iterrows()
    }
    offset_map = {"Red": -0.26, "Blue": 0.0, "Total": 0.26}
    bar_width = 0.22
    fig = go.Figure()
    for count_type in count_types:
        current = plot_df[plot_df["count_type"] == count_type].sort_values(
            ["row_idx", "organ_cell"], kind="mergesort"
        )
        bar_x = [pos + offset_map[count_type] for pos in x_positions]
        fig.add_trace(
            go.Bar(
                x=bar_x,
                y=current["median_count"],
                width=bar_width,
                marker_color=colors[count_type],
                name=count_type,
                legendgroup=count_type,
                customdata=current[["organ_cell"]].to_numpy(),
                hovertemplate=(
                    "Organ/Cell: %{customdata[0]}<br>"
                    f"{count_type} median: "
                    "%{y:.1f}<extra></extra>"
                ),
            )
        )
    for count_type in count_types:
        current = individual_df[individual_df["count_type"] == count_type]#.sort_values(
                #["row_idx", "organ_cell"], kind="mergesort"
        #)
        if current.empty:
            continue
        dot_x = current["row_idx"].astype(float) + offset_map[count_type]
        dot_x = [pos + offset_map[count_type] for pos in x_positions]
        dot_x = [
            position_lookup[(row["row_idx"], row["organ_cell"])] + offset_map[count_type]
            for _, row in current.iterrows()
        ]
        fig.add_trace(
            go.Scatter(
                x=dot_x,
                y=current["count"],
                mode="markers",
                marker={
                    "symbol": "circle",
                    "size": 8,
                    "color": colors[count_type],
                    "opacity": 0.65,
                    "line": {"width": 0.8, "color": "#ffffff"},
                },
                name=f"{count_type} individuals",
                legendgroup=count_type,
                hovertemplate=(
                    "Organ/Cell: %{customdata[0]}<br>"
                    f"{count_type} individual count: "
                    "%{y:.1f}<extra></extra>"
                ),
                customdata=current[["organ_cell"]].to_numpy(),
            )
        )
    fig.update_layout(
        title="Median counts across individuals",
        height=380,
        margin={"l": 20, "r": 20, "t": 60, "b": 40},
        bargap=0,
        barmode="overlay",
    )
    fig.update_yaxes(range=[0, None], title="Count")
    tickvals = x_positions
    fig.update_xaxes(
        tickangle=45,
        tickmode="array",
        tickvals=tickvals,
        ticktext=build_highlighted_tick_labels(x_labels, selected_label or ""),
        range=[-0.6, len(x_positions) - 1 + 0.6],
    )
    return fig


def render_clonotype_presence_grid_legend(top_n: int) -> None:
    """Render a 3-column HTML legend for the Kiki presence grid (top-N, present, not present)."""
    legend_cols = st.columns(3)
    legend_items = [
        (f"Top-{int(top_n)} in row", "#d62728", "#7f1d1d"),
        (f"Present (not Top-{int(top_n)} in row)", "#1f77b4", "#1f77b4"),
        ("Not present", "#ffffff", "#9ca3af"),
    ]
    for col, (label, fill_color, border_color) in zip(legend_cols, legend_items):
        with col:
            st.markdown(
                (
                    f"<div style='display:flex;align-items:center;gap:8px;'>"
                    f"<span style='display:inline-block;width:12px;height:12px;"
                    f"border-radius:50%;background:{fill_color};"
                    f"border:1.5px solid {border_color};'></span>"
                    f"<span>{label}</span></div>"
                ),
                unsafe_allow_html=True,
            )


def render_plot_download_buttons(
    fig: go.Figure,
    base_filename: str,
    key_prefix: str,
) -> None:
    """Render two Streamlit download buttons (PNG and PDF) for a given Plotly figure.

    Silently handles cases where Kaleido is not available for image export.
    """
    safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", base_filename).strip("_") or "plot"

    export_fig = copy.deepcopy(fig)
    export_fig.update_xaxes(automargin=True)
    export_fig.update_yaxes(automargin=True)
    orig_height = export_fig.layout.height or 500
    export_fig.update_layout(
        width=1400,
        height=orig_height + 300,
        margin={"l": 200, "r": 80, "t": 80, "b": 400},
    )

    png_bytes: Optional[bytes] = None
    pdf_bytes: Optional[bytes] = None
    try:
        png_bytes = export_fig.to_image(format="png", scale=2)
    except Exception:
        png_bytes = None
    try:
        pdf_bytes = export_fig.to_image(format="pdf")
    except Exception:
        pdf_bytes = None

    png_col, pdf_col, _ = st.columns([1, 1, 6], gap="small")
    with png_col:
        if png_bytes is not None:
            st.download_button(
                "PNG",
                data=png_bytes,
                file_name=f"{safe_filename}.png",
                mime="image/png",
                key=f"{key_prefix}_download_png",
                use_container_width=True,
            )
    with pdf_col:
        if pdf_bytes is not None:
            st.download_button(
                "PDF",
                data=pdf_bytes,
                file_name=f"{safe_filename}.pdf",
                mime="application/pdf",
                key=f"{key_prefix}_download_pdf",
                use_container_width=True,
            )


def calculate_network_metrics(
    edge_df: pd.DataFrame,
    organ_cell_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compute graph-theoretic metrics (degree, weighted degree, betweenness centrality)
    for the organ|cells-clonotype bipartite network using NetworkX.
    """
    if edge_df.empty:
        return pd.DataFrame()

    graph = nx.Graph()
    for _, row in edge_df.iterrows():
        graph.add_edge(row["organ_cell"], row["clonotype"], weight=row["abundance"])

    betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    organ_cell_score_map = (
        organ_cell_summary.set_index("organ_cell")["total_shared_clonotypes"].to_dict()
        if not organ_cell_summary.empty
        else {}
    )
    organ_cell_nodes = set(edge_df["organ_cell"].unique())

    records = []
    for node in graph.nodes:
        node_type = "organ/cell" if node in organ_cell_nodes else "clonotype"
        records.append(
            {
                "node": node,
                "node_type": node_type,
                "degree": degree.get(node, 0),
                "weighted_degree": weighted_degree.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "total_shared_clonotypes": (
                    float(organ_cell_score_map.get(node, 0.0))
                    if node_type == "organ/cell"
                    else 0.0
                ),
            }
        )

    metrics_df = pd.DataFrame(records).sort_values(
        by=["betweenness_centrality", "weighted_degree"], ascending=False
    )
    return metrics_df


def _arc_positions(labels: List[str], start_deg: float, end_deg: float) -> Dict[str, Tuple[float, float]]:
    """Generate (x, y) positions for labels along an arc from start_deg to end_deg on the unit circle."""
    if not labels:
        return {}
    if len(labels) == 1:
        angle = math.radians((start_deg + end_deg) / 2.0)
        return {labels[0]: (math.cos(angle), math.sin(angle))}
    span = end_deg - start_deg
    coords: Dict[str, Tuple[float, float]] = {}
    for idx, label in enumerate(labels):
        angle_deg = start_deg + (span * idx / (len(labels) - 1))
        angle = math.radians(angle_deg)
        coords[label] = (math.cos(angle), math.sin(angle))
    return coords


def build_entity_chord_figure(
    edge_df: pd.DataFrame,
    only_shared_clones: bool,
    max_clonotypes: int,
) -> Tuple[Optional[go.Figure], pd.DataFrame]:
    """Build a chord-like diagram connecting organ|cells nodes (bottom arc) to clonotype nodes (top arc).

    Bezier curves connect nodes with line widths proportional to abundance.
    Returns (figure, chord_edges_df).
    """
    if edge_df.empty:
        return None, pd.DataFrame()

    clone_summary = (
        edge_df.groupby("clonotype", as_index=False)
        .agg(
            organ_cell_count=("organ_cell", "nunique"),
            total_abundance=("abundance", "sum"),
        )
        .sort_values(["organ_cell_count", "total_abundance"], ascending=[False, False])
        .reset_index(drop=True)
    )

    if only_shared_clones:
        clone_summary = clone_summary[clone_summary["organ_cell_count"] >= 2].copy()

    if clone_summary.empty:
        return None, pd.DataFrame()

    selected_clones = clone_summary.head(max_clonotypes)["clonotype"].tolist()
    chord_edges = edge_df[edge_df["clonotype"].isin(selected_clones)].copy()
    if chord_edges.empty:
        return None, pd.DataFrame()

    organ_nodes = sorted(chord_edges["organ_cell"].unique())
    clone_nodes = (
        clone_summary.set_index("clonotype")
        .loc[selected_clones]
        .sort_values(["organ_cell_count", "total_abundance"], ascending=[False, False])
        .index.tolist()
    )
    organ_pos = _arc_positions(organ_nodes, 110, 250)
    clone_pos = _arc_positions(clone_nodes, -70, 70)
    positions = {**organ_pos, **clone_pos}

    max_edge_abundance = float(chord_edges["abundance"].max()) if not chord_edges.empty else 1.0
    max_edge_abundance = max(max_edge_abundance, 1.0)
    clone_degree_map = clone_summary.set_index("clonotype")["organ_cell_count"].to_dict()

    fig = go.Figure()
    for _, row in chord_edges.sort_values("abundance").iterrows():
        source = str(row["organ_cell"])
        target = str(row["clonotype"])
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        points = 24
        x_vals: List[float] = []
        y_vals: List[float] = []
        for step in range(points + 1):
            t = step / points
            one_minus_t = 1 - t
            x = (one_minus_t * one_minus_t * x0) + (2 * one_minus_t * t * 0.0) + (t * t * x1)
            y = (one_minus_t * one_minus_t * y0) + (2 * one_minus_t * t * 0.0) + (t * t * y1)
            x_vals.append(x)
            y_vals.append(y)
        width = 1 + (5 * float(row["abundance"]) / max_edge_abundance)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                line={"color": "rgba(31, 119, 180, 0.35)", "width": width},
                hovertemplate=(
                    f"Organ/Cell: {source}<br>"
                    f"Clonotype: {target}<br>"
                    f"Abundance: {float(row['abundance']):.2f}<br>"
                    f"Groups for clone: {int(clone_degree_map.get(target, 0))}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[organ_pos[label][0] for label in organ_nodes],
            y=[organ_pos[label][1] for label in organ_nodes],
            mode="markers+text",
            text=organ_nodes,
            textposition="middle left",
            marker={"size": 14, "color": "#1f77b4"},
            name="Organ/Cell",
            hovertemplate="%{text}<extra>Organ/Cell</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[clone_pos[label][0] for label in clone_nodes],
            y=[clone_pos[label][1] for label in clone_nodes],
            mode="markers+text",
            text=clone_nodes,
            textposition="middle right",
            marker={"size": 11, "color": "#ff7f0e"},
            name="Clonotype",
            hovertemplate="%{text}<extra>Clonotype</extra>",
        )
    )

    fig.update_layout(
        height=760,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        xaxis={"visible": False, "range": [-1.3, 1.3]},
        yaxis={"visible": False, "range": [-1.2, 1.2]},
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    return fig, chord_edges


def prepare_summary_line_data(
    df: pd.DataFrame, selected_clonotypes: List[str], lineage: str
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Prepare a long-format DataFrame for per-mouse, per-clonotype line plots within a single lineage.

    Fills missing (mouse, organ_cell, clonotype) combinations with zero. Computes pool_pct
    as percentage of that mouse's lineage total abundance.
    Returns (lineage_plot, all_mice_list, organ_cells_list).
    """
    all_mice = sorted(df["mouse"].unique())
    df_lineage = (
        df.assign(cd_group=df["cell_type"].apply(classify_cd4_cd8))
        .query("cd_group == @lineage")
        .copy()
    )
    lineage_totals = (
        df_lineage.groupby("mouse", as_index=False)["abundance"]
        .sum()
        .rename(columns={"abundance": "lineage_abundance"})
    )
    lineage_totals = (
        pd.DataFrame({"mouse": all_mice})
        .merge(lineage_totals, on="mouse", how="left")
        .fillna({"lineage_abundance": 0.0})
    )
    organ_cells = sorted(df_lineage["organ_cell"].unique())
    combos = (
        pd.MultiIndex.from_product(
            [all_mice, organ_cells, selected_clonotypes],
            names=["mouse", "organ_cell", "clonotype"],
        )
        .to_frame(index=False)
    )
    aggregated = (
        df_lineage[df_lineage["clonotype"].isin(selected_clonotypes)]
        .groupby(["mouse", "organ_cell", "clonotype"], as_index=False)["abundance"]
        .sum()
    )
    lineage_plot = combos.merge(aggregated, on=["mouse", "organ_cell", "clonotype"], how="left")
    lineage_plot["abundance"] = lineage_plot["abundance"].fillna(0.0)
    lineage_plot = lineage_plot.merge(lineage_totals, on="mouse", how="left")
    lineage_plot["lineage_abundance"] = lineage_plot["lineage_abundance"].fillna(0.0)
    lineage_plot["pool_pct"] = (
        lineage_plot["abundance"]
        / lineage_plot["lineage_abundance"].replace(0, np.nan)
        * 100.0
    ).fillna(0.0)
    return lineage_plot, all_mice, organ_cells


def calculate_mouse_cosine_similarity(
    df: pd.DataFrame, value_col: str = "abundance"
) -> pd.DataFrame:
    """Compute pairwise cosine similarity between mice using (clonotype_rank, organ_cell) feature vectors."""
    if df.empty:
        return pd.DataFrame()
    if "clonotype_rank" not in df.columns:
        return pd.DataFrame()
    if value_col not in df.columns:
        return pd.DataFrame()

    feature_matrix = df.pivot_table(
        index="mouse",
        columns=["clonotype_rank", "organ_cell"],
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    if feature_matrix.empty or feature_matrix.shape[0] < 2:
        return pd.DataFrame()

    feature_values = feature_matrix.to_numpy(dtype=float)
    norms = np.linalg.norm(feature_values, axis=1, keepdims=True)
    normalized = np.divide(
        feature_values,
        norms,
        out=np.zeros_like(feature_values, dtype=float),
        where=norms != 0,
    )
    similarity = normalized @ normalized.T
    return pd.DataFrame(
        similarity,
        index=feature_matrix.index,
        columns=feature_matrix.index,
    )


def calculate_mouse_correlation(
    df: pd.DataFrame, value_col: str = "abundance"
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation between mice using (clonotype_rank, organ_cell) feature vectors."""
    if df.empty:
        return pd.DataFrame()
    if "clonotype_rank" not in df.columns:
        return pd.DataFrame()
    if value_col not in df.columns:
        return pd.DataFrame()

    feature_matrix = df.pivot_table(
        index="mouse",
        columns=["clonotype_rank", "organ_cell"],
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    if feature_matrix.empty or feature_matrix.shape[0] < 2:
        return pd.DataFrame()

    return feature_matrix.T.corr(method="pearson")


def map_chain_to_vdjdb_gene(chain_value: str) -> Optional[str]:
    """Map a TCR chain value to the VDJdb gene parameter ('TRA' or 'TRB'), or None if unrecognized."""
    value = str(chain_value).upper()
    if "TRA" in value or value.endswith("A"):
        return "TRA"
    if "TRB" in value or value.endswith("B"):
        return "TRB"
    return None


def nucleotide_to_amino_acid_cdr3(seq: str) -> str:
    """Translate a nucleotide CDR3 sequence to amino acids using the standard codon table.

    If the input is already an amino-acid sequence (non-DNA characters), it is returned as-is.
    Terminal stop codons are removed; internal stops cause the translation to be rejected.
    Returns an empty string on failure.
    """
    if not seq:
        return ""

    # If sequence is already amino-acid-like, keep it as-is for VDJdb search.
    if not set(seq).issubset(DNA_BASES):
        return seq

    if len(seq) < 3:
        return ""
    usable_len = len(seq) - (len(seq) % 3)
    if usable_len <= 0:
        return ""
    seq = seq[:usable_len]

    aa_chars: List[str] = []
    for i in range(0, len(seq), 3):
        codon = seq[i : i + 3]
        if len(codon) < 3:
            continue
        if any(base not in {"A", "C", "G", "T"} for base in codon):
            aa_chars.append("X")
            continue
        aa_chars.append(CODON_TABLE.get(codon, "X"))

    aa = "".join(aa_chars)
    if not aa:
        return ""

    # Remove terminal stop; internal stop usually indicates invalid CDR3 frame.
    if aa.endswith("*"):
        aa = aa[:-1]
    if "*" in aa:
        return ""
    return aa


def _vdjdb_post_search(payload: Dict[str, Any], timeout_seconds: int = 15) -> Dict[str, Any]:
    """POST a search payload to the VDJdb API /search endpoint and return the parsed JSON response."""
    endpoint = f"{VDJDB_API_BASE_URL}/search"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_vdjdb_metadata() -> Dict[str, Any]:
    """Fetch VDJdb column metadata (cached for 24 hours)."""
    endpoint = f"{VDJDB_API_BASE_URL}/meta"
    req = request.Request(endpoint, method="GET")
    with request.urlopen(req, timeout=15) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    return payload.get("metadata", {})


def _build_vdjdb_column_index(metadata: Dict[str, Any]) -> Dict[str, int]:
    """Convert VDJdb metadata column list to a name-to-index mapping."""
    columns = metadata.get("columns", [])
    if not isinstance(columns, list):
        return {}
    return {
        str(col.get("name")): idx
        for idx, col in enumerate(columns)
        if isinstance(col, dict) and "name" in col
    }


def _extract_entry_by_candidates(
    entries: List[Any], index_by_name: Dict[str, int], candidates: List[str]
) -> str:
    """Extract the first non-empty entry field matching one of the candidate column names."""
    for name in candidates:
        idx = index_by_name.get(name)
        if idx is None:
            continue
        if 0 <= idx < len(entries):
            value = str(entries[idx]).strip()
            if value:
                return value
    return ""


def _mode_or_empty(values: List[str]) -> str:
    """Return the most frequent non-empty value, or empty string if none exist."""
    cleaned = [v for v in values if str(v).strip()]
    if not cleaned:
        return ""
    return pd.Series(cleaned).value_counts().index[0]


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def query_vdjdb_sequence(
    sequence: str,
    gene: str = "",
    allow_fuzzy_fallback: bool = False,
) -> Dict[str, Any]:
    """Query VDJdb for a single CDR3 amino-acid sequence.

    Returns a dict with match count, top gene/species/antigen/MHC, score,
    and paired-chain CDR3 sequences. Results are cached with a 12-hour TTL.

    If allow_fuzzy_fallback is True and no exact match is found, retries with a
    sequence-edit-distance search (1 substitution/insertion/deletion).
    """
    seq = str(sequence).strip().upper()
    if not seq:
        return {
            "clonotype": sequence,
            "vdjdb_match_count": 0,
            "vdjdb_top_gene": "",
            "vdjdb_top_species": "",
            "vdjdb_top_antigen": "",
            "vdjdb_top_mhc": "",
            "vdjdb_top_score": np.nan,
            "vdjdb_has_paired_record": False,
            "vdjdb_paired_cdr3_all": "",
            "vdjdb_paired_cdr3_tra": "",
            "vdjdb_paired_cdr3_trb": "",
            "vdjdb_error": "empty sequence",
        }

    base_filters: List[Dict[str, Any]] = [
        {"column": "cdr3", "value": seq, "filterType": "exact", "negative": False}
    ]
    gene_filter = str(gene).strip().upper()
    if gene_filter in {"TRA", "TRB"}:
        base_filters.append(
            {"column": "gene", "value": gene_filter, "filterType": "exact", "negative": False}
        )

    payload = {"filters": base_filters, "paired": True}
    try:
        search_result = _vdjdb_post_search(payload)
        if allow_fuzzy_fallback and int(search_result.get("recordsFound", 0)) == 0:
            fuzzy_filters = [
                {
                    "column": "cdr3",
                    "value": f"{seq}:1:1:1",
                    "filterType": "sequence",
                    "negative": False,
                }
            ]
            if gene_filter in {"TRA", "TRB"}:
                fuzzy_filters.append(
                    {
                        "column": "gene",
                        "value": gene_filter,
                        "filterType": "exact",
                        "negative": False,
                    }
                )
            search_result = _vdjdb_post_search({"filters": fuzzy_filters, "paired": True})
    except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return {
            "clonotype": sequence,
            "vdjdb_match_count": 0,
            "vdjdb_top_gene": "",
            "vdjdb_top_species": "",
            "vdjdb_top_antigen": "",
            "vdjdb_top_mhc": "",
            "vdjdb_top_score": np.nan,
            "vdjdb_has_paired_record": False,
            "vdjdb_paired_cdr3_all": "",
            "vdjdb_paired_cdr3_tra": "",
            "vdjdb_paired_cdr3_trb": "",
            "vdjdb_error": str(exc),
        }

    rows = search_result.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    records_found = int(search_result.get("recordsFound", 0))
    try:
        metadata = fetch_vdjdb_metadata()
    except Exception:
        metadata = {}
    index_by_name = _build_vdjdb_column_index(metadata)

    genes: List[str] = []
    species_values: List[str] = []
    antigens: List[str] = []
    mhc_values: List[str] = []
    scores: List[float] = []
    paired_all: List[str] = []
    paired_tra: List[str] = []
    paired_trb: List[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        entries = row.get("entries", [])
        if not isinstance(entries, list):
            continue
        genes.append(_extract_entry_by_candidates(entries, index_by_name, ["gene"]))
        species_values.append(
            _extract_entry_by_candidates(
                entries,
                index_by_name,
                ["species", "species.name", "speciesname"],
            )
        )
        antigens.append(
            _extract_entry_by_candidates(
                entries,
                index_by_name,
                ["antigen.epitope", "epitope", "antigen.gene", "antigen.species"],
            )
        )
        mhc_values.append(
            _extract_entry_by_candidates(
                entries,
                index_by_name,
                ["mhc.a", "mhc.b", "mhc.class", "mhc"],
            )
        )
        score_raw = _extract_entry_by_candidates(
            entries,
            index_by_name,
            ["vdjdb.score", "score"],
        )
        if score_raw:
            try:
                scores.append(float(score_raw))
            except ValueError:
                pass

        metadata_row = row.get("metadata", {})
        paired_id = str(metadata_row.get("pairedID", "0")).strip()
        row_gene = _extract_entry_by_candidates(entries, index_by_name, ["gene"]).upper()
        row_cdr3 = _extract_entry_by_candidates(entries, index_by_name, ["cdr3"]).upper()
        if paired_id and paired_id != "0" and row_cdr3 and row_cdr3 != seq:
            paired_all.append(row_cdr3)
            if row_gene == "TRA":
                paired_tra.append(row_cdr3)
            elif row_gene == "TRB":
                paired_trb.append(row_cdr3)

    return {
        "clonotype": seq,
        "vdjdb_match_count": records_found,
        "vdjdb_top_gene": _mode_or_empty(genes),
        "vdjdb_top_species": _mode_or_empty(species_values),
        "vdjdb_top_antigen": _mode_or_empty(antigens),
        "vdjdb_top_mhc": _mode_or_empty(mhc_values),
        "vdjdb_top_score": max(scores) if scores else np.nan,
        "vdjdb_has_paired_record": len(set(paired_all)) > 0,
        "vdjdb_paired_cdr3_all": ";".join(sorted(set(paired_all))),
        "vdjdb_paired_cdr3_tra": ";".join(sorted(set(paired_tra))),
        "vdjdb_paired_cdr3_trb": ";".join(sorted(set(paired_trb))),
        "vdjdb_error": "",
    }


def enrich_clonotypes_with_vdjdb(
    clonotypes: List[str],
    chain_value: str,
    max_queries: int = 100,
    allow_fuzzy_fallback: bool = False,
) -> pd.DataFrame:
    """Enrich a list of nucleotide clonotype sequences with VDJdb antigen annotations.

    Translates each nucleotide CDR3 to amino acids, queries the VDJdb API,
    and returns a DataFrame with match metadata (gene, species, antigen, MHC, score, paired chain).
    Deduplicates identical sequences before querying and caches results server-side.
    """
    unique_nt_sequences = list(
        dict.fromkeys([str(c).strip().upper() for c in clonotypes if str(c).strip()])
    )
    if max_queries > 0:
        unique_nt_sequences = unique_nt_sequences[: int(max_queries)]

    if not unique_nt_sequences:
        return pd.DataFrame(
            columns=[
                "clonotype",
                "vdjdb_query_cdr3_aa",
                "vdjdb_match_count",
                "vdjdb_top_gene",
                "vdjdb_top_species",
                "vdjdb_top_antigen",
                "vdjdb_top_mhc",
                "vdjdb_top_score",
                "vdjdb_has_paired_record",
                "vdjdb_paired_cdr3_all",
                "vdjdb_paired_cdr3_tra",
                "vdjdb_paired_cdr3_trb",
                "vdjdb_error",
            ]
        )

    gene = map_chain_to_vdjdb_gene(chain_value) or ""
    records: List[Dict[str, Any]] = []
    aa_result_cache: Dict[str, Dict[str, Any]] = {}

    for nt_seq in unique_nt_sequences:
        aa_seq = nucleotide_to_amino_acid_cdr3(nt_seq)
        if not aa_seq:
            records.append(
                {
                    "clonotype": nt_seq,
                    "vdjdb_query_cdr3_aa": "",
                    "vdjdb_match_count": 0,
                    "vdjdb_top_gene": "",
                    "vdjdb_top_species": "",
                    "vdjdb_top_antigen": "",
                    "vdjdb_top_mhc": "",
                    "vdjdb_top_score": np.nan,
                    "vdjdb_has_paired_record": False,
                    "vdjdb_paired_cdr3_all": "",
                    "vdjdb_paired_cdr3_tra": "",
                    "vdjdb_paired_cdr3_trb": "",
                    "vdjdb_error": "unable to translate nucleotide CDR3 to amino acid",
                }
            )
            continue

        if aa_seq not in aa_result_cache:
            aa_result_cache[aa_seq] = query_vdjdb_sequence(
                sequence=aa_seq,
                gene=gene,
                allow_fuzzy_fallback=allow_fuzzy_fallback,
            )
        aa_result = aa_result_cache[aa_seq].copy()
        aa_result["clonotype"] = nt_seq
        aa_result["vdjdb_query_cdr3_aa"] = aa_seq
        records.append(aa_result)

    return pd.DataFrame(records)


def _build_summary_lineage_abundance_figure(
    lineage_df: pd.DataFrame,
    selected_clonotypes: List[str],
    sort_organ_cell: str,
    log_axis_summary: bool,
    normalize_topn_summary: bool,
    selected_label: str,
    lineage: str,
) -> Optional[go.Figure]:
    """Build a multi-mouse line plot for a single lineage (CD4 or CD8).

    Each mouse gets a separate line per clonotype. Supports log10 y-axis with
    dynamic per-(mouse, organ_cell) pseudo-zero imputation for missing values.
    """
    if lineage_df.empty or not selected_clonotypes:
        return None

    lineage_df = lineage_df.copy()
    lineage_df["abundance_for_metric"] = lineage_df["abundance"].astype(float)
    if normalize_topn_summary:
        lineage_df["abundance_for_metric"] = np.where(
            lineage_df["norm_topN"] > 0,
            (lineage_df["abundance_for_metric"] / lineage_df["norm_topN"]) * 100.0,
            0.0,
        )

    lineage_organ_cells = get_organ_cell_order(lineage_df, sort_organ_cell)
    if not lineage_organ_cells:
        return None

    all_mice = sorted(lineage_df["mouse"].astype(str).unique())
    lineage_pivot = lineage_df.pivot_table(
        index=["clonotype", "mouse"],
        columns=["organ_cell"],
        values="abundance_for_metric",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    ordered_columns = ["clonotype", "mouse", *lineage_organ_cells]
    lineage_pivot = lineage_pivot.reindex(columns=ordered_columns, fill_value=0)
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
        positive_lineage = lineage_df[lineage_df["abundance_for_metric"] > 0].copy()
        pseudo_by_group = (
            positive_lineage.groupby(["mouse", "organ_cell"], as_index=False)[
                "abundance_for_metric"
            ]
            .min()
            .rename(columns={"abundance_for_metric": "pseudo_zero"})
        )
        pseudo_by_group["pseudo_zero"] = (
            pseudo_by_group["pseudo_zero"] / (pseudo_by_group["pseudo_zero"] + 1)
        )
        organ_cell_line = organ_cell_line.merge(
            pseudo_by_group,
            on=["mouse", "organ_cell"],
            how="left",
        )

        mouse_level_pseudo = (
            positive_lineage.groupby("mouse", as_index=False)["abundance_for_metric"]
            .min()
            .rename(columns={"abundance_for_metric": "mouse_pseudo_zero"})
        )
        mouse_level_pseudo["mouse_pseudo_zero"] = (
            mouse_level_pseudo["mouse_pseudo_zero"]
            / (mouse_level_pseudo["mouse_pseudo_zero"] + 1)
        )
        lineage_min_positive = (
            float(positive_lineage["abundance_for_metric"].min())
            if not positive_lineage.empty
            else np.nan
        )
        lineage_pseudo = (
            lineage_min_positive / (lineage_min_positive + 1)
            if not np.isnan(lineage_min_positive)
            else float(np.finfo(float).tiny)
        )
        organ_cell_line = organ_cell_line.merge(
            mouse_level_pseudo,
            on="mouse",
            how="left",
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
        line_group="clonotype",
        markers=True,
        labels={
            "organ_cell": "Organ/Cell",
            "abundance": "Normalized % Pool Size" if normalize_topn_summary else "% Pool Size",
            "mouse": "Individual",
        },
        title=f"{lineage} clonotype abundance across individuals for {selected_label}",
        category_orders={
            "organ_cell": lineage_organ_cells,
            "mouse": all_mice,
            "clonotype": selected_clonotypes,
        },
    )
    yaxis_config = {
        "title": "Normalized % Pool Size" if normalize_topn_summary else "% Pool Size",
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
    cd_fig.update_layout(height=420, yaxis=yaxis_config, xaxis_title="Organ/Cell")
    cd_fig.update_xaxes(
        tickmode="array",
        tickvals=lineage_organ_cells,
        ticktext=build_highlighted_tick_labels(lineage_organ_cells, selected_label),
    )

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

    return cd_fig


def load_dataset_from_sidebar() -> pd.DataFrame:
    """Render the data-source controls in the sidebar and return the loaded/normalized DataFrame.

    Supports bundled example dataset or user-uploaded CSV.
    Normalizes column names, validates required columns, coerces types,
    and creates the composite 'organ_cell' column.
    """
    with st.sidebar:
        st.header("Data")
        use_example = st.checkbox("Use example dataset", value=True)
        uploaded_file = st.file_uploader("Upload clonotype dataset", type=["csv"])

    data_source = "__bundled_example__" if use_example else uploaded_file

    if use_example:
        st.info(
            f"Using the bundled example dataset ({EXAMPLE_DATASET_FILENAME}). "
            "Uncheck 'Use example dataset' to upload a new file."
        )

    if not use_example and data_source is None:
        st.info("Upload a CSV file to begin.")
        if get_script_run_ctx() is None:
            raise SystemExit("No file uploaded. Run with `streamlit run app.py`.")
        st.stop()

    try:
        df = (
            load_example_dataframe()
            if use_example
            else pd.read_csv(data_source, low_memory=False)
        )
    except Exception as exc:
        st.error(f"Unable to read file: {exc}")
        st.stop()

    df, _ = normalize_columns(df)
    valid, missing = validate_columns(df)
    if not valid:
        st.error("Missing required columns: " + ", ".join(missing))
        st.write("Detected columns:", list(df.columns))
        st.stop()

    for col in ["mouse", "organ", "cell_type", "chain", "clonotype"]:
        df[col] = df[col].astype(str)
    df["abundance"] = pd.to_numeric(df["abundance"], errors="coerce").fillna(0.0)
    df["organ_cell"] = df["organ"] + " | " + df["cell_type"]

    return df
