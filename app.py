"""Main entry point for the TCR Abundance Explorer Streamlit application.

Launches a multi-page interactive web app for exploring T-cell receptor (TCR)
clonotype abundance patterns across biological samples (mice, organs, cell types).
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

from tcr_app.core import load_dataset_from_sidebar
from tcr_app.page_per_individual import run_per_individual_page
from tcr_app.page_summary_all import run_summary_all_page
from tcr_app.page_pooled import run_summary_all_individuals_pooled_page
from tcr_app.page_all_clonotype_flow import run_all_clonotype_flow_page
from tcr_app.page_public_clonotypes import run_public_clonotypes_page
from tcr_app.page_pairwise import run_pairwise_comparison_page


st.set_page_config(page_title="TCR Abundance Explorer", layout="wide")


def _check_kaleido() -> str | None:
    """Return None if Kaleido export works, or an error message if it doesn't."""
    try:
        import kaleido as _k  # noqa: F401 — ensure the package itself is importable
    except ImportError as exc:
        return f"kaleido pip package not installed ({exc}). Add `kaleido>=0.2.1` to pyproject.toml."

    try:
        fig = go.Figure(go.Scatter(x=[0], y=[0]))
        pio.to_image(fig, format="png")
        return None
    except Exception as exc:
        return f"Kaleido render failed: {type(exc).__name__}: {exc}"


def main() -> None:
    """Set up sidebar navigation, load data, and dispatch to the selected page."""
    kaleido_err = _check_kaleido()
    if kaleido_err is not None:
        st.sidebar.warning(
            f"Figure download (PNG/PDF) unavailable.\n\n{kaleido_err}"
            "\n\nIf deploying on Streamlit Cloud, ensure `packages.txt` "
            "in the repo root contains:\n"
            "```\nlibgl1\nchromium\n```"
        )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        (
            "Single-subject by organ/cell",
            "Cohort summary",
            "Top-N clonotype summary",
            "Clonotype flow by subject",
            "Public clonotypes",
            "Pairwise comparison",
        ),
        index=0,
        help="Switch between single-subject, cohort, top-N, clonotype flow, public clonotype, and pairwise views.",
    )

    df = load_dataset_from_sidebar()

    st.sidebar.checkbox(
        "Show download buttons (PNG/PDF/CSV)",
        value=False,
        key="show_download_buttons",
        help="When disabled, download buttons are hidden and export computations are skipped for faster performance.",
    )



    if page == "Single-subject by organ/cell":
        run_per_individual_page(df)
    elif page == "Cohort summary":
        run_summary_all_page(df)
    elif page == "Top-N clonotype summary":
        run_summary_all_individuals_pooled_page(df)
    elif page == "Public clonotypes":
        run_public_clonotypes_page(df)
    elif page == "Pairwise comparison":
        run_pairwise_comparison_page(df)
    elif page == "Clonotype flow by subject":
        run_all_clonotype_flow_page(df)


if __name__ == "__main__":
    main()
