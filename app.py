import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

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

st.set_page_config(page_title="TCR Abundance Explorer", layout="wide")

CANONICAL_COLUMNS: Dict[str, List[str]] = {
    "mouse": ["mouse", "individual", "mouse_id", "animal", "animal_id"],
    "organ": ["organ", "tissue"],
    "cell_type": ["cell_type", "celltype", "cell type", "cell", "celltype", "cell.type"],
    "chain": ["chain", "tcr_chain"],
    "clonotype": ["clonotype", "clonetype", "cdr3", "sequence", "tcr", "nSeqCDR3"],
    "abundance": ["abundance", "count", "frequency", "freq"]
}

REQUIRED_COLUMNS = ["mouse", "organ", "cell_type", "chain", "clonotype", "abundance"]
SELECTED_ORGAN_CELL_COLOR = "#d62728"
DEFAULT_EDGE_WIDTH_SCALE = 0.2
DEFAULT_GRAVITY = -2200
DEFAULT_SPRING_LENGTH = 180
PSEUDO_ZERO = 1e-4
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
    example_path_candidates = [
        Path(EXAMPLE_DATASET_FILENAME),
        Path(__file__).resolve().parent / EXAMPLE_DATASET_FILENAME,
    ]
    for path in example_path_candidates:
        if path.exists():
            return pd.read_csv(path, low_memory=False)
    return pd.read_csv(io.StringIO(FALLBACK_EXAMPLE_CSV), low_memory=False)


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
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
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def classify_cd4_cd8(cell_type: str) -> str:
    upper = str(cell_type).upper()
    if re.search(r"\bCD8\b", upper):
        return "CD8"
    if re.search(r"\bCD4\b", upper):
        return "CD4"
    return "Other"


def build_organ_cell_clonotype_edges(
    df: pd.DataFrame,
    top_clonotypes: List[str],
    min_edge_abundance: float,
) -> pd.DataFrame:
    filtered = df[df["clonotype"].isin(top_clonotypes)].copy()
    edge_df = filtered.groupby(["organ_cell", "clonotype"], as_index=False)["abundance"].sum()
    edge_df["abundance"] = pd.to_numeric(edge_df["abundance"], errors="coerce").fillna(0)
    edge_df = edge_df[edge_df["abundance"] >= min_edge_abundance].copy()
    return edge_df


def calculate_organ_cell_sharing(
    edge_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
) -> Optional[go.Figure]:
    if lineage_df.empty:
        return None

    lineage_organ_cells = sorted(lineage_df["organ_cell"].unique())
    if not lineage_organ_cells:
        return None

    lineage_pivot = (
        lineage_df.pivot_table(
            index="clonotype",
            columns="organ_cell",
            values="abundance",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reindex(index=selected_clonotypes, columns=lineage_organ_cells, fill_value=0.0)
        .astype(float)
    )
    lineage_pool_size = float(lineage_df["abundance"].sum())
    if lineage_pool_size > 0:
        lineage_pivot = (lineage_pivot / lineage_pool_size) * 100.0

    active_clonotypes = [
        clonotype
        for clonotype in selected_clonotypes
        if clonotype in lineage_pivot.index and float(lineage_pivot.loc[clonotype].sum()) > 0
    ]
    if not active_clonotypes:
        return None

    x_values = list(range(len(lineage_organ_cells)))
    bar_half_width = 0.32
    cumulative = np.zeros(len(lineage_organ_cells), dtype=float)
    segment_bounds: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for clonotype in active_clonotypes:
        y_vals = lineage_pivot.loc[clonotype].to_numpy(dtype=float)
        lower = cumulative.copy()
        upper = lower + y_vals
        segment_bounds[clonotype] = (lower, upper, y_vals)
        cumulative = upper

    flow_fig = go.Figure()
    bar_width = 0.62
    bar_half_width = bar_width / 2.0

    for clonotype in active_clonotypes:
        lower, upper, y_vals = segment_bounds[clonotype]
        band_color = _hex_to_rgba(clonotype_color_map[clonotype], 0.23)
        for idx in range(len(x_values) - 1):
            if y_vals[idx] <= 0 and y_vals[idx + 1] <= 0:
                continue
            left_x = x_values[idx] + bar_half_width
            right_x = x_values[idx + 1] - bar_half_width
            flow_fig.add_trace(
                go.Scatter(
                    x=[left_x, right_x, right_x, left_x, left_x],
                    y=[
                        lower[idx],
                        lower[idx + 1],
                        upper[idx + 1],
                        upper[idx],
                        lower[idx],
                    ],
                    mode="lines",
                    line={"width": 0},
                    fill="toself",
                    fillcolor=band_color,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    for clonotype in active_clonotypes:
        _, _, y_vals = segment_bounds[clonotype]
        flow_fig.add_trace(
            go.Bar(
                x=x_values,
                y=y_vals,
                width=[bar_width] * len(x_values),
                name=clonotype,
                marker={"color": clonotype_color_map[clonotype]},
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
    selected_organ_cell: str,
    show_clonotype_sequences: bool = True,
    x_spacing: float = 0.62,
    row_categories: Optional[List[str]] = None,
) -> Optional[go.Figure]:
    if df.empty or not selected_clonotypes:
        return None

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
    row_present_map = organ_cell_totals[organ_cell_totals["abundance"] > 0].groupby(
        "organ_cell"
    )["clonotype"].apply(set).to_dict()
    available_rows = set(df["organ_cell"].unique())

    grid_rows = (
        list(row_categories)
        if row_categories is not None
        else sorted(df["organ_cell"].unique())
    )
    grid_cols = [str(clonotype) for clonotype in selected_clonotypes]
    if not grid_rows or not grid_cols:
        return None

    x_positions = [idx * x_spacing for idx in range(len(grid_cols))]
    x_tick_labels = (
        grid_cols
        if show_clonotype_sequences
        else [f"C{idx + 1}" for idx in range(len(grid_cols))]
    )
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
    grid_df = pd.DataFrame(grid_records)
    if grid_df.empty:
        return None

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
        margin={"l": 80, "r": 20, "t": 40, "b": 70},
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
    return grid_fig


def render_clonotype_presence_grid_legend(top_n: int) -> None:
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
    safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", base_filename).strip("_") or "plot"
    png_bytes: Optional[bytes] = None
    pdf_bytes: Optional[bytes] = None
    try:
        png_bytes = fig.to_image(format="png", scale=2)
    except Exception:
        png_bytes = None
    try:
        pdf_bytes = fig.to_image(format="pdf")
    except Exception:
        pdf_bytes = None

    button_cols = st.columns(2)
    with button_cols[0]:
        if png_bytes is not None:
            st.download_button(
                "Download PNG",
                data=png_bytes,
                file_name=f"{safe_filename}.png",
                mime="image/png",
                key=f"{key_prefix}_download_png",
            )
    with button_cols[1]:
        if pdf_bytes is not None:
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"{safe_filename}.pdf",
                mime="application/pdf",
                key=f"{key_prefix}_download_pdf",
            )


def calculate_network_metrics(
    edge_df: pd.DataFrame,
    organ_cell_summary: pd.DataFrame,
) -> pd.DataFrame:
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


def load_dataset_from_sidebar() -> pd.DataFrame:
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
        "Y-axis shows percent of the lineage pool size; toggle the log option if needed."
    )
    log_axis = st.checkbox(
        "Log10 scale",
        value=True,
        help=(
            "Display % pool size on a log10 axis. Missing clone values are shown with a "
            "dynamic pseudo-0 per organ|cell: 0.5 x the lowest observed abundance."
        ),
    )
    if log_axis:
        st.caption(
            "Open-circle markers indicate imputed 0 values per organ|cell (pseudo-0 used for Log10 display)."
        )
    lineage_filtered = filtered[filtered["clonotype"].isin(selected_clonotypes)].copy()
    lineage_filtered["cd_group"] = lineage_filtered["cell_type"].apply(classify_cd4_cd8)
    clonotype_color_map = build_clonotype_color_map(selected_clonotypes)
    for lineage in ["CD4", "CD8"]:
        lineage_df = lineage_filtered[lineage_filtered["cd_group"] == lineage].copy()
        st.markdown(f"**{lineage} clonotype abundance across organ/cell**")
        if lineage_df.empty:
            st.info(f"No {lineage} cells found for current filters.")
            continue
        lineage_organ_cells = sorted(lineage_df["organ_cell"].unique())
        # 1. Pivot to create the grid automatically (clonotypes x organ_cells)
        # This handles the "missing" combinations by filling them with 0 immediately
        lineage_pivot = lineage_df.pivot_table(
            index="clonotype", 
            columns="organ_cell", 
            values="abundance", 
            aggfunc="sum", 
            fill_value=0
        ).reindex(index=selected_clonotypes, columns=lineage_organ_cells, fill_value=0)
        # 2. Flatten back to "long" format for Plotly
        organ_cell_line = lineage_pivot.reset_index().melt(
            id_vars="clonotype", 
            value_name="abundance"
        )
        organ_cell_line["is_pseudo"] = False
        lineage_pool_size = float(lineage_df["abundance"].sum())
        if lineage_pool_size > 0:
            organ_cell_line["pool_pct"] = (
                organ_cell_line["abundance"] / lineage_pool_size * 100.0
            )
        else:
            organ_cell_line["pool_pct"] = 0.0
        organ_cell_line["pool_pct_plot"] = organ_cell_line["pool_pct"]
        if log_axis:
            positive_lineage = organ_cell_line[organ_cell_line["pool_pct"] > 0].copy()
            pseudo_by_group = (
                positive_lineage.groupby("organ_cell", as_index=False)["pool_pct"]
                .min()
                .rename(columns={"pool_pct": "pseudo_zero"})
            )
            pseudo_by_group["pseudo_zero"] = pseudo_by_group["pseudo_zero"] * 0.5
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
                lineage_min_positive * 0.5
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
            labels={"organ_cell": "Organ/Cell", "pool_pct_plot": "% Pool Size"},
        )
        yaxis_config = {
            "title": "% Pool Size",
            "type": "log" if log_axis else "linear",
        }
        if log_axis:
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
        else:
            yaxis_config.setdefault("range", [0, 100])
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
        .groupby(["mouse", "clonotype"], as_index=False)["abundance"]
        .sum()
        .sort_values(
            ["mouse", "abundance", "clonotype"],
            ascending=[True, False, True],
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
                "dynamic pseudo-0 per mouse and organ|cell: 0.5 x the lowest observed abundance."
            ),
        )
    with scale_cols[1]:
        normalize_topn_summary = st.checkbox(
            f"Normalize top-{int(top_n)} per mouse to 100%",
            value=False,
            help=(
                f"Normalize selected top-{int(top_n)} clone abundances so each mouse sums to 100%."
            ),
        )
    if log_axis_summary:
        st.caption(
            "Open-circle markers indicate imputed 0 values per organ|cell (pseudo-0 used for Log10 display)."
        )
    # Grab top clones only
    topClones = pd.merge(
        selected_clonotypes[["mouse", "clonotype", "clonotype_rank"]],
        filtered,
        how="left",
        on=["mouse", "clonotype"],
    )

    topClones["cd_group"] = topClones["cell_type"].apply(classify_cd4_cd8)
    topClones["abundance_for_metric"] = topClones["abundance"].astype(float)
    if normalize_topn_summary:
        mouse_totals = topClones.groupby(["mouse","organ_cell"])["abundance_for_metric"].transform("sum")
        topClones["abundance_for_metric"] = np.where(
            mouse_totals > 0,
            (topClones["abundance_for_metric"] / mouse_totals) * 100.0,
            0.0,
        )
    y_axis_title = (
        "Normalized abundance (%)" if normalize_topn_summary else "% Pool Size"
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
            lineage_organ_cells = sorted(lineage_df["organ_cell"].unique())
            # 1. Pivot (clonotypes x organ_cells) per individual!
            all_mice = lineage_df.mouse.unique()
            lineage_pivot = lineage_df.pivot_table(
                index=["clonotype", "mouse"], 
                columns=["organ_cell"], 
                values="abundance_for_metric", 
                aggfunc="sum",
                fill_value=0
            ).reset_index()
            # 2. Flatten back to "long"
            organ_cell_line = lineage_pivot.melt(
                    id_vars=["clonotype","mouse"],
                    value_name="abundance"
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
                pseudo_by_group["pseudo_zero"] = pseudo_by_group["pseudo_zero"] * 0.5
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
                    mouse_level_pseudo["mouse_pseudo_zero"] * 0.5
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
                    lineage_min_positive * 0.5
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
                    lineage_axis_options = sorted(
                        filtered_with_lineage[
                            filtered_with_lineage["cd_group"] == lineage
                        ]["organ_cell"].unique()
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
                    lineage_axis_options = sorted(
                        filtered_with_lineage[
                            filtered_with_lineage["cd_group"] == lineage
                        ]["organ_cell"].unique()
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
    show_clonotype_sequences_summary = st.checkbox(
        "Show clonotype sequences on x-axis",
        value=False,
        key="summary_grid_show_sequences",
        help="Turn off to use compact rank labels (C1, C2, ...).",
    )
    plotted_any_grid = False
    for lineage in ["CD4", "CD8"]:
        st.markdown(f"**{lineage}**")
        lineage_filtered_summary = filtered_with_lineage[
            filtered_with_lineage["cd_group"] == lineage
        ].copy()
        shared_summary_grid_rows = sorted(lineage_filtered_summary["organ_cell"].unique())
        if lineage_filtered_summary.empty:
            st.info(f"No {lineage} data available for the current filters.")
            continue
        mouse_ids = sorted(lineage_filtered_summary["mouse"].unique())
        lineage_plotted = False
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
        if not lineage_plotted:
            st.info(f"No per-individual {lineage} clonotype presence grids available.")
        if lineage == "CD4" and lineage_plotted:
            render_clonotype_presence_grid_legend(int(top_n))

    if not plotted_any_grid:
        st.info("No per-individual clonotype presence grids available for the current filters.")
    else:
        render_clonotype_presence_grid_legend(int(top_n))

    display_count = max(10, top_n)
    st.subheader("Top clonotypes across individuals")
    st.caption("Sorted by total abundance for the selected subset across all mice.")
    st.dataframe(topClones, width="stretch")


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ("Per individual", "Summary all individuals"),
        index=0,
        help="Switch between the detailed per-mouse view and the cohort summary.",
    )

    df = load_dataset_from_sidebar()

    if page == "Per individual":
        run_per_individual_page(df)
    else:
        run_summary_all_page(df)

if __name__ == "__main__":
    main()
