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
            "Per individual",
            "Summary all individuals",
            "Pooled organ|cell counts",
            "All-clonotype flow",
        ),
        index=0,
        help="Switch between detailed view, cohort summary, and per-mouse all-clonotype flow.",
    )

    df = load_dataset_from_sidebar()

    if page == "Per individual":
        run_per_individual_page(df)
    elif page == "Summary all individuals":
        run_summary_all_page(df)
    elif page == "Pooled organ|cell counts":
        run_summary_all_individuals_pooled_page(df)
    else:
        run_all_clonotype_flow_page(df)


if __name__ == "__main__":
    main()
