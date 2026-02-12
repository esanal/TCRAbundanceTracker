import io
import json
from typing import Dict, List, Tuple
import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import get_script_run_ctx
from pyvis.network import Network
import numpy as np
from typing import Tuple, List

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


def build_occurrence_network_html(
    df: pd.DataFrame,
    top_clonotypes: List[str],
    show_clonotype_labels: bool,
    min_edge_abundance: float,
    gravity: int,
    spring_length: int,
    edge_width_scale: float,
    organ_box_margin: int,
) -> str:
    filtered = df[df["clonotype"].isin(top_clonotypes)].copy()
    edge_df = (
        filtered.groupby(["clonotype", "organ_cell"], as_index=False)["abundance"].sum()
    )
    # Convert to numeric and filter out zero/negative abundances
    edge_df["abundance"] = pd.to_numeric(edge_df["abundance"], errors="coerce").fillna(0)
    #edge_df = edge_df[edge_df["abundance"] > 0].copy()
    edge_df = edge_df[edge_df["abundance"] >= min_edge_abundance].copy()

    if edge_df.empty:
        return ""

    net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut()

    # Only include nodes that have edges with non-zero abundance
    clonotype_nodes = sorted(edge_df["clonotype"].unique())
    organ_nodes = sorted(edge_df["organ_cell"].unique())

    for clonotype in clonotype_nodes:
        display_label = clonotype if show_clonotype_labels else " "
        display_size = 12 if show_clonotype_labels else 0

        net.add_node(
            clonotype,
            label=display_label,
            color="#ff7f0e",
            title=f"Clonotype: {clonotype}", # Tooltip still works!
            font={"size": display_size}
        )
        
    for organ_cell in organ_nodes:
        wrapped_label = organ_cell.replace(" | ", "\n")
        net.add_node(
            organ_cell,
            label=wrapped_label,
            color="#1f77b4",
            title=f"Organ/Cell: {organ_cell}",
            shape="box",
            margin=organ_box_margin,
            font={"size": 12}
        )

    for _, row in edge_df.iterrows():
        net.add_edge(
            row["clonotype"],
            row["organ_cell"],
            value=row["abundance"],
            width=max(1, row["abundance"] * edge_width_scale),
            title=f"Abundance: {row['abundance']}",
        )

    options = {
        "physics": {
            "enabled": True,
            "barnesHut": {
              "gravitationalConstant": gravity,
                "springLength": spring_length,
            },
        },
        "nodes": {
            "shape": "dot",
            "size": 12,
            "font": {"size": 12},
        },
        "edges": {
            "color": {"inherit": True},
            "smooth": False,
        },
        "interaction": {
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
        },
    }
    net.set_options(json.dumps(options))
    
    return net.generate_html()


def calculate_network_metrics(
    df: pd.DataFrame,
    top_clonotypes: List[str],
    min_edge_abundance: float,
) -> pd.DataFrame:
    filtered = df[df["clonotype"].isin(top_clonotypes)].copy()
    edge_df = (
        filtered.groupby(["clonotype", "organ_cell"], as_index=False)["abundance"].sum()
    )
    edge_df["abundance"] = pd.to_numeric(edge_df["abundance"], errors="coerce").fillna(0)
    edge_df = edge_df[edge_df["abundance"] >= min_edge_abundance].copy()

    if edge_df.empty:
        return pd.DataFrame()

    graph = nx.Graph()
    for _, row in edge_df.iterrows():
        graph.add_edge(row["clonotype"], row["organ_cell"], weight=row["abundance"])

    betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))

    records = []
    for node in graph.nodes:
        node_type = "clonotype" if node in set(edge_df["clonotype"].unique()) else "organ/cell"
        records.append(
            {
                "node": node,
                "node_type": node_type,
                "degree": degree.get(node, 0),
                "weighted_degree": weighted_degree.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
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

st.plotly_chart(heatmap_fig, width="stretch")

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
    st.plotly_chart(sample_fig, width="stretch")

st.subheader("Clonotype Abundance Across Organ/Cell")
line_df = (
    filtered[filtered["clonotype"].isin(selected_clonotypes)]
    .groupby(["clonotype", "organ_cell"], as_index=False)["abundance"]
    .sum()
)
organ_cell_all = pd.DataFrame({"organ_cell": organ_cell_options})
clonotype_all = pd.DataFrame({"clonotype": selected_clonotypes})
line_grid = clonotype_all.merge(organ_cell_all, how="cross")
organ_cell_line = line_grid.merge(
    line_df, on=["clonotype", "organ_cell"], how="left"
).fillna({"abundance": 0})
line_fig = px.line(
    organ_cell_line,
    x="organ_cell",
    y="abundance",
    color="clonotype",
    markers=True,
    labels={"organ_cell": "Organ/Cell", "abundance": "% pool size"},
)
line_fig.update_layout(height=400)
st.plotly_chart(line_fig, width="stretch")

st.subheader("Occurrence Network (Drag to Explore)")
st.caption(
    "The network links clonotypes (orange) to organ/cell combinations (blue). "
    "Edges appear when a clonotype is detected in an organ/cell subset, and "
    "thicker connections reflect higher summed abundance."
)
# show_clonotype_labels = st.checkbox("Show clonotype labels", value=True)
network_control_cols = st.columns(2)
with network_control_cols[0]:
    show_clonotype_labels = st.checkbox("Show clonotype labels", value=False)
    min_edge_abundance = st.slider(
        "Minimum edge abundance",
        min_value=0.0,
        max_value=float(filtered["abundance"].max()),
        value=0.0,
        step=1.0,
        help="Filter out low-abundance clonotype-to-organ/cell edges.",
    )
    edge_width_scale = st.slider(
        "Edge width scale",
        min_value=0.05,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Increase to make high-abundance edges visually thicker.",
    )
with network_control_cols[1]:
    gravity = st.slider(
        "Node repulsion (gravity)",
        min_value=-5000,
        max_value=-500,
        value=-2000,
        step=100,
        help="More negative values push nodes farther apart.",
    )
    spring_length = st.slider(
        "Spring length",
        min_value=50,
        max_value=400,
        value=160,
        step=10,
        help="Higher values increase preferred distance between connected nodes.",
    )
    organ_box_margin = st.slider(
        "Organ/cell label padding",
        min_value=5,
        max_value=30,
        value=10,
        step=1,
        help="Padding inside rectangular organ/cell nodes.",
    )
    
network_html = build_occurrence_network_html(
    filtered,
    selected_clonotypes,
    show_clonotype_labels=show_clonotype_labels,
    min_edge_abundance=min_edge_abundance,
    gravity=gravity,
    spring_length=spring_length,
    edge_width_scale=edge_width_scale,
    organ_box_margin=organ_box_margin,
)

if network_html:
    components.html(network_html, height=580, scrolling=True)
else:
    st.warning(
        "No edges remain after applying the current minimum edge abundance threshold. "
        "Lower the threshold to render the network."
    )

st.subheader("Network Metrics")
st.caption(
    "Centrality metrics are computed on the displayed bipartite network. "
    "Use them to identify connector clonotypes and high-traffic organ/cell nodes."
)
# components.html(network_html, height=580, scrolling=True)
metrics_df = calculate_network_metrics(
    filtered,
    selected_clonotypes,
    min_edge_abundance=min_edge_abundance,
)

if metrics_df.empty:
    st.info("No network metrics available for the current edge threshold.")
else:
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        top_connector = metrics_df.iloc[0]
        st.metric(
            "Top node by betweenness",
            top_connector["node"],
            f"{top_connector['betweenness_centrality']:.3f}",
        )
    with metric_col2:
        st.metric(
            "Nodes in current network",
            f"{metrics_df['node'].nunique()}",
            f"{(metrics_df['node_type'] == 'clonotype').sum()} clonotypes",
        )

    st.dataframe(
        metrics_df.style.format(
            {
                "weighted_degree": "{:.2f}",
                "betweenness_centrality": "{:.4f}",
            }
        ),
        width="stretch",
    )
    
st.subheader("Filtered Data")
st.dataframe(filtered, width="stretch")

csv_buffer = io.StringIO()
filtered.to_csv(csv_buffer, index=False)
st.download_button(
    "Download filtered data", csv_buffer.getvalue(), file_name="filtered_clonotypes.csv"
)
