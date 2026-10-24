# WASM viewer — tutorial final states

PNG captures of the **Stone Soup Rust** egui viewer (`apps/stonesoup-viewer`) running as **WebAssembly**, each catalogue entry stepped to completion (**01–29**).

| File | Catalogue | Title (registry) |
|------|-----------|------------------|
| `tutorial_01.png` | 01 | Kalman filter introduction |
| `tutorial_02.png` | 02 | Extended Kalman filter |
| `tutorial_03.png` | 03 | Unscented Kalman filter |
| `tutorial_04.png` | 04 | Particle filter |
| `tutorial_05.png` | 05 | Data association (clutter) |
| `tutorial_06.png` | 06 | Multi-target (GNN) |
| `tutorial_07.png` | 07 | Probabilistic data association |
| `tutorial_08.png` | 08 | Joint PDA |
| `tutorial_09.png` | 09 | Initiators & deleters |
| `tutorial_10.png` | 10 | Simulation & tracking stack |
| `tutorial_11.png` | 11 | Particle resamplers |
| `tutorial_12.png` | 12 | Information filter |
| `tutorial_13.png` | 13 | Gaussian mixture PHD |
| `tutorial_14.png` | 14 | ASD filter |
| `tutorial_15.png` | 15 | KD-tree GNN |
| `tutorial_16.png` | 16 | Single sensor management |
| `tutorial_17.png` | 17 | AKKF |
| `tutorial_18.png` | 18 | Multi-sensor management |
| `tutorial_19.png` | 19 | Optimised sensor management |
| `tutorial_20.png` | 20 | Information vs network architectures |
| `tutorial_21.png` | 21 | Avoiding data incest |
| `tutorial_22.png` | 22 | Introduction to architectures |
| `tutorial_23.png` | 23 | Simple OOSM (fixed lag) |
| `tutorial_24.png` | 24 | Kalman filter OOSM |
| `tutorial_25.png` | 25 | Particle filter OOSM |
| `tutorial_26.png` | 26 | OOSM Algorithm C |
| `tutorial_27.png` | 27 | Bearings-only EKF (moving platform) |
| `tutorial_28.png` | 28 | Multi-tracker comparison (manoeuvring targets) |
| `tutorial_29.png` | 29 | Ensemble Kalman filter |

## Regenerate

From the repo root:

```bash
bash tools/scripts/capture_wasm_tutorial_screenshots.sh
```

Requires [Trunk](https://trunkrs.dev/), Python 3, and Playwright (installed automatically by the script if missing).

## Deep links (local `trunk serve` or `dist/`)

```text
http://localhost:8080/?tutorial=29&finish=1
```

When stepping completes, the page title becomes `stonesoup-viewer-ready` (for automation).

## Python reference plots

Viewer-style matplotlib PNGs for the same catalogue: [../python/README.md](../python/README.md). Regenerate with `bash tools/scripts/capture_python_tutorial_screenshots.sh`.

## Tutorial 11 vs Stone Soup (Python)

See prior section in git history for resampling cross-check commands (`plot_lot5_resampling_reference.py`, `capture_wasm_tutorial11_steps.sh`, `compare_lot5_graphics.py`).
