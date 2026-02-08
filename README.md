# TCR Abundance Explorer

An interactive Streamlit app for exploring TCR clonotype abundance across mice, individual, organs, cell types, and chains.

## Features
- Filter by mouse, organ, cell type, and chain.
- Heatmap of top clonotype abundance by organ/cell combinations.
- Sample-level stacked bar chart (when a `sample` column is provided).
- Draggable occurrence network linking clonotypes to organ/cell combinations.
- Download filtered datasets for downstream analysis.

## Expected columns
The app looks for these columns (case-insensitive):
- `mouse`
- `individual`
- `organ`
- `cell_type`
- `chain`
- `clonotype` (or `nSeqCDR3`)
- `abundance`
- Optional: `sample`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
## Run on streamlit.io
https://tcrexplorer.streamlit.app/

## Large uploads
To allow CSV uploads larger than 200 MB, Streamlit reads the `.streamlit/config.toml`
file in this repo (set to 1024 MB). Adjust `maxUploadSize` if you need a different limit.

## Troubleshooting
If you see `ValueError: numpy.dtype size changed`, reinstall from scratch to ensure compatible
binary wheels are installed (the requirements pin NumPy <2.0 and pandas <2.2).
```bash
pip uninstall -y numpy pandas pyarrow
pip install -r requirements.txt
```

Upload your CSV file in the UI to begin exploring.
