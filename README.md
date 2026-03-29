# TCR Abundance Explorer

An interactive Streamlit app for exploring TCR clonotype abundance across mice, individual, organs, cell types, and chains.

## Run on streamlit.io
https://tcrexplorer.streamlit.app/

## Features
- Filter by mouse, organ, cell type, and chain.
- Heatmap of top clonotype abundance by organ/cell combinations.
- Draggable occurrence network linking clonotypes to organ/cell combinations.
- Download filtered datasets for downstream analysis.
- Organ|Cell group summaries. Organ|Cell based clonotype occurance of all individuals can be summarized.

## Expected columns in the data
Each row is a clonotype with the expected columns:
- `mouse` or `individual`
- `organ`
- `cell_type`
- `chain`
- `clonotype` (or `nSeqCDR3`)
- `abundance`
- Optional: `sample`

## Run locally

Install Streamlit (https://docs.streamlit.io/get-started/installation), requirements.txt and run via
```bash
streamlit run app.py
```

## Large uploads
To allow CSV uploads larger than 200 MB, Streamlit reads the `.streamlit/config.toml`
file in this repo (set to 1024 MB). Adjust `maxUploadSize` if you need a different limit.

Upload your CSV file in the UI to begin exploring.
