# TCR Abundance Explorer

An interactive Streamlit app for exploring TCR clonotypes abundance across mice, individual, organs, cell types and chains.

## Run on streamlit.io
https://tcrexplorer.streamlit.app/

## Features
- Filter by mouse, organ, cell type, and chain.
- Heatmap of top clonotype abundance by organ/cell combinations.
- Draggable occurrence network linking clonotypes to organ/cell combinations.
- Download filtered datasets for downstream analysis.
- Organ|Cell group summaries. Organ|Cell based clonotype occurance of all individuals can be summarized.

## Run local: use uv
This project uses uv for fast and reliable dependency management.

## Expected columns in the data
Each row is a clonotype with the expected columns:
- `mouse` or `individual`
- `organ`
- `cell_type`
- `chain`
- `clonotype` (or `nSeqCDR3`)
- `abundance`
- Optional: `sample`

## Naming requirements of the cell_type column
In the current version, "CD4" or "CD8" strings are expected in the cell_type.
Subset names should start with either CD4 or CD8 and a white space after is needed.
After the white space further description of the subset is expected such as "CD4 Memory".

## Run locally

Install Streamlit (https://docs.streamlit.io/get-started/installation), requirements.txt and run via
```bash
streamlit run app.py
```

Upload your CSV file in the UI to begin exploring.
