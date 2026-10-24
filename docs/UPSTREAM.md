# Upstream reference — Stone Soup

| Field | Value |
|-------|-------|
| Repository | https://github.com/dstl/Stone-Soup |
| Release tag | `v1.9.1` |
| Pinned commit (tag peel) | `a4336b920a799cfe0a77ecb05867c5deeb371c7a` |
| Annotated tag object | `d9e6fb16f5ae176817aeb6a6fc3a39f544694408` |
| Upstream license | MIT (Copyright Defence Science and Technology Laboratory UK) |
| Clone used for audit | Shallow clone at tag `v1.9.1` (2026-10-24) |

## Python reference environment (oracle / fixtures)

Recorded when fixtures were generated (`tools/python/requirements.txt`):

- Python ≥ 3.10
- `stonesoup` installed editable from the pinned commit
- NumPy, SciPy, Matplotlib (transitive via Stone Soup)

## Inventory (at pinned commit)

| Category | Count (Python files, excl. `tests/`) |
|----------|--------------------------------------|
| `stonesoup/` components | 198 |
| Tutorials | 22 |
| Examples | 46 |
| Demos | 5 |
| **Total matrix rows** | **271** |

Full line-by-line mapping: `docs/PORTING_MATRIX.csv`.

## Dataset and dependency licenses

- **Stone Soup source**: MIT (`LICENSE` in upstream repo).
- **Bundled demo CSVs** (e.g. OpenSky, AIS Solent): upstream demo data; treat as separate from library MIT — verify per-file before redistribution in Rust builds.
- **Python oracle dependencies**: NumPy/SciPy (BSD), Matplotlib (PSF-based), Plotly (MIT) — used only under `tools/python/`, not in the Rust runtime product.

## Rust port license

This repository is MIT (see root `LICENSE`). Derived algorithm implementations are independent code; retain upstream MIT notice where substantial text or equations are quoted from Stone Soup documentation.
