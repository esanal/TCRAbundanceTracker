import streamlit as st

from tcr_app.core import load_dataset_from_sidebar
from tcr_app.pages import (
    run_all_clonotype_flow_page,
    run_per_individual_page,
    run_summary_all_page,
)


st.set_page_config(page_title="TCR Abundance Explorer", layout="wide")


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ("Per individual", "Summary all individuals", "All-clonotype flow"),
        index=0,
        help="Switch between detailed view, cohort summary, and per-mouse all-clonotype flow.",
    )

    df = load_dataset_from_sidebar()

    if page == "Per individual":
        run_per_individual_page(df)
    elif page == "Summary all individuals":
        run_summary_all_page(df)
    else:
        run_all_clonotype_flow_page(df)


if __name__ == "__main__":
    main()
