import io
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from streamlit.runtime.scriptrunner import get_script_run_ctx
from pyvis.network import Network

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
) -> str:
    filtered = df[df["clonotype"].isin(top_clonotypes)].copy()
    edge_df = (
        filtered.groupby(["clonotype", "organ_cell"], as_index=False)["abundance"].sum()
    )
    # Convert to numeric and filter out zero/negative abundances
    edge_df["abundance"] = pd.to_numeric(edge_df["abundance"], errors="coerce").fillna(0)
    edge_df = edge_df[edge_df["abundance"] > 0].copy()

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
            margin=10,
            font={"size": 12}
        )

    for _, row in edge_df.iterrows():
        net.add_edge(
            row["clonotype"],
            row["organ_cell"],
            value=row["abundance"],
            title=f"Abundance: {row['abundance']}",
        )

    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -2000,
              "springLength": 160
            }
          },
          "nodes": {
            "shape": "dot",
            "size": 12,
            "font": { "size": 12 }
          },
          "edges": {
            "color": { "inherit": true },
            "smooth": false
          },
          "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
          }
        }
        """
    )
    
    return net.generate_html()


st.title("TCR Abundance Explorer")

with st.sidebar:
    st.header("Data")
    use_example = st.checkbox("Use example dataset", value=True)
    uploaded_file = st.file_uploader("Upload clonotype dataset", type=["csv"])

st.markdown(
    """
Upload a CSV file of clonotype sequences with mouse, organ, cell type, chain, and abundance.
The app will help you explore clonotypes per mouse and visualize abundance patterns.
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
    "Top clonotypes",
    min_value=1,
    max_value=max_clonotypes,
    value=min(15, max_clonotypes),
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
    labels={"organ_cell": "Organ/Cell", "abundance": "Abundance"},
)
line_fig.update_layout(height=400)
st.plotly_chart(line_fig, width="stretch")

st.subheader("Occurrence Network (Drag to Explore)")
st.caption(
    "The network links clonotypes (orange) to organ/cell combinations (blue). "
    "Edges appear when a clonotype is detected in an organ/cell subset, and "
    "thicker connections reflect higher summed abundance."
)
show_clonotype_labels = st.checkbox("Show clonotype labels", value=True)
network_html = build_occurrence_network_html(
    filtered,
    selected_clonotypes,
    show_clonotype_labels=show_clonotype_labels,
)
components.html(network_html, height=580, scrolling=True)

st.subheader("Filtered Data")
st.dataframe(filtered, width="stretch")

csv_buffer = io.StringIO()
filtered.to_csv(csv_buffer, index=False)
st.download_button(
    "Download filtered data", csv_buffer.getvalue(), file_name="filtered_clonotypes.csv"
)
