import io
import json
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from streamlit.runtime.scriptrunner import get_script_run_ctx

st.set_page_config(page_title="TCR Abundance Explorer", layout="wide")

CANONICAL_COLUMNS: Dict[str, List[str]] = {
    "mouse": ["mouse", "individual", "mouse_id", "animal", "animal_id"],
    "organ": ["organ", "tissue"],
    "cell_type": ["cell_type", "celltype", "cell type", "cell", "celltype", "cell.type"],
    "chain": ["chain", "tcr_chain"],
    "clonotype": ["clonotype", "clonetype", "cdr3", "sequence", "tcr", "nSeqCDR3"],
    "abundance": ["abundance", "count", "frequency", "freq"],
    "sample": ["sample", "sample_id", "sample id", "sample_name"],
}

REQUIRED_COLUMNS = ["mouse", "organ", "cell_type", "chain", "clonotype", "abundance"]
SELECTED_ORGAN_CELL_COLOR = "#d62728"


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
    if "CD4" in upper:
        return "CD4"
    if "CD8" in upper:
        return "CD8"
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
) -> str:
    if edge_df.empty:
        return ""

    organ_cell_score_map = (
        organ_cell_summary.set_index("organ_cell")["total_shared_clonotypes"].to_dict()
        if not organ_cell_summary.empty
        else {}
    )
    net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="#222222")

    organ_cell_nodes = sorted(edge_df["organ_cell"].unique())
    clonotype_nodes = sorted(edge_df["clonotype"].unique())

    max_node_score = max(organ_cell_score_map.values(), default=0.0)
    min_node_size = 18
    size_range = 22
    for organ_cell in organ_cell_nodes:
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
            font={"size": 12, "color": "#ffffff" if is_selected else "#222222"},
            borderWidth=3 if is_selected else 1,
        )

    for clonotype in clonotype_nodes:
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
            font={"size": display_size, "color": "#ffffff" if is_selected_clonotype else "#222222"},
            level=1,
            borderWidth=3 if is_selected_clonotype else 1,
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
        "physics": {
            "enabled": False,
            "barnesHut": {
                "gravitationalConstant": gravity,
                "springLength": spring_length,
            },
        },
        "nodes": {
            "shape": "dot",
            "size": 12,
            "font": {"size": 12},
            "color": {
                "highlight": {
                    "border": "FF0000",
                    "background": "FF5555"
                }
            }
        },
        "edges": {
            "color": {"inherit": True},
            "smooth": {
                "enabled": True,
                "type": "cubicBezier",
                "forceDirection": "vertical",
                "roundness": 0.35,
            },
        },
        "interaction": {
            "navigationButtons": True,
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
        },
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "hubsize",
                "nodeSpacing": 190,
                "levelSeparation": 300,
                "treeSpacing": 260,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
            }
        },
    }
    net.set_options(json.dumps(options))
    return net.generate_html()


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


st.title("TCR Abundance Explorer")

with st.sidebar:
    st.header("Data")
    use_example = st.checkbox("Use example dataset", value=True)
    uploaded_file = st.file_uploader("Upload clonotype dataset", type=["csv"])

st.markdown(
    """
Upload a CSV file of clonotype sequences with mouse, organ, cell type, chain, and abundance.
The app will help you explore clonotypes per mouse, organ and cell type and visualize abundance patterns.
"""
)

if use_example:
    st.info(
        "Using the bundled example dataset (test.abundance.1.csv). "
        "Uncheck 'Use example dataset' to upload a new file."
    )
    uploaded_file = "test.abundance.1.csv"

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    if get_script_run_ctx() is None:
        raise SystemExit("No file uploaded. Run with `streamlit run app.py`.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:  # pragma: no cover - UI validation
    st.error(f"Unable to read file: {exc}")
    st.stop()

df, mapping = normalize_columns(df)

valid, missing = validate_columns(df)
if not valid:
    st.error(
        "Missing required columns: " + ", ".join(missing)
    )
    st.write("Detected columns:", list(df.columns))
    st.stop()

if "sample" not in df.columns:
    st.warning("No sample column detected; sample-based charts will be hidden.")

for col in ["mouse", "organ", "cell_type", "chain", "clonotype"]:
    df[col] = df[col].astype(str)
df["abundance"] = pd.to_numeric(df["abundance"], errors="coerce").fillna(0.0)

with st.sidebar:
    st.header("Filters")
    mouse_selected = st.selectbox("Mouse", sorted(df["mouse"].unique()))
    organ_selected = st.multiselect(
        "Organ", sorted(df["organ"].unique()), default=sorted(df["organ"].unique())
    )
    cell_selected = st.multiselect(
        "Cell type",
        sorted(df["cell_type"].unique()),
        default=sorted(df["cell_type"].unique()),
    )
    chain_selected = st.multiselect(
        "Chain", sorted(df["chain"].unique()), default=sorted(df["chain"].unique())
    )
filtered = df[
    (df["mouse"] == mouse_selected)
    & (df["organ"].isin(organ_selected))
    & (df["cell_type"].isin(cell_selected))
    & (df["chain"].isin(chain_selected))
].copy()

if filtered.empty:
    st.warning("No data match the selected filters.")
    st.stop()

filtered["organ_cell"] = filtered["organ"] + " | " + filtered["cell_type"]

summary_cols = st.columns(4)
summary_cols[0].metric("Clonotypes", filtered["clonotype"].nunique())
summary_cols[1].metric("Organs", filtered["organ"].nunique())
summary_cols[2].metric("Cell types", filtered["cell_type"].nunique())
summary_cols[3].metric("Total abundance", f"{filtered['abundance'].sum():,.0f}")

st.subheader("Abundance by Organ/Cell (Top Clonotypes)")
organ_cell_options = sorted(filtered["organ_cell"].unique())
top_n_scope = st.selectbox(
    "Rank top clonotypes by organ/cell combination",
    organ_cell_options,
)
max_clonotypes = (
    filtered[filtered["organ_cell"] == top_n_scope]["clonotype"].nunique()
)
max_clonotypes = max(1, max_clonotypes)
top_n = st.slider(
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
    .sort_values("abundance", ascending=False)
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

st.plotly_chart(heatmap_fig, use_container_width=True)

if "sample" in filtered.columns:
    st.subheader("Abundance by Sample")
    sample_df = (
        filtered.groupby(["sample", "organ"], as_index=False)["abundance"].sum()
    )
    sample_fig = px.bar(
        sample_df,
        x="sample",
        y="abundance",
        color="organ",
        barmode="stack",
        labels={"abundance": "Abundance", "sample": "Sample"},
    )
    sample_fig.update_layout(height=400)
    st.plotly_chart(sample_fig, use_container_width=True)

st.subheader("Clonotype Abundance Line Plot: CD4 vs CD8")
st.caption(
    "Separate line plots for CD4 and CD8 cells so each lineage can be compared independently. "
    "Y-axis shows percent of lineage pool size."
)
lineage_filtered = filtered[filtered["clonotype"].isin(selected_clonotypes)].copy()
lineage_filtered["cd_group"] = lineage_filtered["cell_type"].apply(classify_cd4_cd8)
for lineage in ["CD4", "CD8"]:
    lineage_df = lineage_filtered[lineage_filtered["cd_group"] == lineage].copy()
    st.markdown(f"**{lineage} clonotype abundance across organ/cell**")
    if lineage_df.empty:
        st.info(f"No {lineage} cells found for current filters.")
        continue
    lineage_organ_cells = sorted(lineage_df["organ_cell"].unique())
    lineage_line_df = lineage_df.groupby(
        ["clonotype", "organ_cell"],
        as_index=False,
    )["abundance"].sum()
    organ_cell_all = pd.DataFrame({"organ_cell": lineage_organ_cells})
    clonotype_all = pd.DataFrame({"clonotype": selected_clonotypes})
    line_grid = clonotype_all.merge(organ_cell_all, how="cross")
    organ_cell_line = line_grid.merge(
        lineage_line_df,
        on=["clonotype", "organ_cell"],
        how="left",
    ).fillna({"abundance": 0})
    lineage_pool_size = float(lineage_df["abundance"].sum())
    if lineage_pool_size > 0:
        organ_cell_line["pool_pct"] = (
            organ_cell_line["abundance"] / lineage_pool_size * 100.0
        )
    else:
        organ_cell_line["pool_pct"] = 0.0
    line_fig = px.line(
        organ_cell_line,
        x="organ_cell",
        y="pool_pct",
        color="clonotype",
        markers=True,
        labels={"organ_cell": "Organ/Cell", "pool_pct": "% Pool Size"},
    )
    line_fig.update_layout(height=420)
    line_fig.update_xaxes(
        tickmode="array",
        tickvals=lineage_organ_cells,
        ticktext=build_highlighted_tick_labels(
            lineage_organ_cells,
            top_n_scope,
        ),
    )
    st.plotly_chart(line_fig, use_container_width=True)

st.subheader("Hierarchical Organ/Cell-Clonotype Network")
st.caption(
    "Organ/cell nodes are on the left and clonotypes on the right. "
    "Organ/cell node size reflects total clone-sharing with other organ/cell nodes."
)
network_control_cols = st.columns(2)
with network_control_cols[0]:
    st.caption(f"Network clonotypes: using current top selection ({len(selected_clonotypes)}).")
    show_clonotype_labels = st.checkbox("Show clonotype labels", value=False)
    min_edge_abundance = st.slider(
        "Minimum organ/cell-clonotype edge abundance",
        min_value=0.0,
        max_value=float(filtered["abundance"].max()),
        value=0.0,
        step=1.0,
        help="Filter out low-abundance organ/cell-clonotype edges.",
    )
with network_control_cols[1]:
    edge_width_scale = st.slider(
        "Edge width scale",
        min_value=0.05,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )
    gravity = st.slider(
        "Node repulsion (gravity)",
        min_value=-5000,
        max_value=-500,
        value=-2200,
        step=100,
    )
    spring_length = st.slider(
        "Spring length",
        min_value=50,
        max_value=400,
        value=180,
        step=10,
    )

edge_df = build_organ_cell_clonotype_edges(
    filtered,
    selected_clonotypes,
    min_edge_abundance=min_edge_abundance,
)
pair_df, organ_cell_summary, shared_matrix = calculate_organ_cell_sharing(edge_df)
selected_metric_node = st.session_state.get("network_metrics_selected_node")

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
    st.plotly_chart(shared_heatmap, use_container_width=True)

network_html = build_organ_cell_clonotype_network_html(
    edge_df=edge_df,
    organ_cell_summary=organ_cell_summary,
    selected_organ_cell=top_n_scope,
    selected_node=selected_metric_node,
    show_clonotype_labels=show_clonotype_labels,
    gravity=gravity,
    spring_length=spring_length,
    edge_width_scale=edge_width_scale,
)
if network_html:
    components.html(network_html, height=580, scrolling=True)
else:
    st.warning(
        "No network edges remain after the current filters and minimum edge threshold."
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
        if st.button("Clear node highlight", use_container_width=True):
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
    
st.subheader("Filtered Data")
st.dataframe(filtered, width="stretch")

csv_buffer = io.StringIO()
filtered.to_csv(csv_buffer, index=False)
st.download_button(
    "Download filtered data", csv_buffer.getvalue(), file_name="filtered_clonotypes.csv"
)
