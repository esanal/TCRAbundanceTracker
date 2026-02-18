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
uv sync --frozen
uv run streamlit run app.py
```

## Package as a single executable (Windows/macOS)
PyInstaller can bundle this app into a one-file executable that launches Streamlit automatically.
The build script uses `uv` and pulls `pyinstaller` through `uv run --with`.

### macOS/Linux
```bash
./build_executable.sh
```

### Windows (PowerShell)
```powershell
./build_executable.ps1
```

Outputs:
- macOS: `dist/TCRAbundanceExplorer`
- Windows: `dist/TCRAbundanceExplorer.exe`

Important: cross-compiling is not supported by PyInstaller. Build on macOS for macOS binaries and on Windows for Windows binaries.

### CI build for both platforms
A GitHub Actions workflow is included at `.github/workflows/build-executables.yml`.
- Run it manually (`workflow_dispatch`) or push a tag like `v1.0.0`.
- Download artifacts named `TCRAbundanceExplorer-macos` and `TCRAbundanceExplorer-windows`.

## Run on streamlit.io
https://tcrexplorer.streamlit.app/

## Large uploads
To allow CSV uploads larger than 200 MB, Streamlit reads the `.streamlit/config.toml`
file in this repo (set to 1024 MB). Adjust `maxUploadSize` if you need a different limit.
```bash
uv sync --reinstall --frozen
```

Upload your CSV file in the UI to begin exploring.
