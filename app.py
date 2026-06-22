"""Main entry point for the TCR Abundance Explorer Streamlit application.

Launches a multi-page interactive web app for exploring T-cell receptor (TCR)
clonotype abundance patterns across biological samples (mice, organs, cell types).
"""

import streamlit as st

from tcr_app.core import load_dataset_from_sidebar
from tcr_app.page_per_individual import run_per_individual_page
from tcr_app.page_summary_all import run_summary_all_page
from tcr_app.page_pooled import run_summary_all_individuals_pooled_page
from tcr_app.page_all_clonotype_flow import run_all_clonotype_flow_page


st.set_page_config(page_title="TCR Abundance Explorer", layout="wide")


def main() -> None:
    """Set up sidebar navigation, load data, and dispatch to the selected page."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        (
            "Single-subject by organ/cell",
            "Cohort summary",
            "Top-N clonotype summary",
            "Clonotype flow by subject",
        ),
        index=0,
        help="Switch between single-subject, cohort, top-N, and clonotype flow views.",
    )

    df = load_dataset_from_sidebar()

    if page == "Single-subject by organ/cell":
        run_per_individual_page(df)
    elif page == "Cohort summary":
        run_summary_all_page(df)
    elif page == "Top-N clonotype summary":
        run_summary_all_individuals_pooled_page(df)
    else:
        run_all_clonotype_flow_page(df)


if __name__ == "__main__":
    main()
