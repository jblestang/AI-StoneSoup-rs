# WASM viewer — tutorial final states

PNG captures of the **Stone Soup Rust** egui viewer (`apps/stonesoup-viewer`) running as **WebAssembly**, each tutorial stepped to completion (catalogue tutorials 01–15).

| File | Tutorial |
|------|----------|
| `tutorial_01.png` | 01 — Kalman filter introduction |
| `tutorial_02.png` | 02 — Extended Kalman filter |
| `tutorial_03.png` | 03 — Unscented Kalman filter |
| `tutorial_04.png` | 04 — Particle filter |
| `tutorial_05.png` | 05 — Data association (clutter) |
| `tutorial_06.png` | 06 — Multi-target (GNN) |
| `tutorial_07.png` | 07 — Probabilistic data association |
| `tutorial_08.png` | 08 — Joint PDA |
| `tutorial_09.png` | 09 — Initiators & deleters |
| `tutorial_10.png` | 10 — Simulation & tracking stack |
| `tutorial_11.png` | Sampling — Particle resamplers |
| `tutorial_12.png` | 12 — Information filter |
| `tutorial_13.png` | 13 — GM-PHD |
| `tutorial_14.png` | 14 — ASD (accumulated state densities) |
| `tutorial_15.png` | 15 — KD-tree GNN data association |

## Regenerate

From the repo root:

```bash
bash tools/scripts/capture_wasm_tutorial_screenshots.sh
```

Requires [Trunk](https://trunkrs.dev/), Python 3, and Playwright (installed automatically by the script if missing).

## Deep links (local `trunk serve` or `dist/`)

Open a tutorial and run to the final step in one load:

```text
http://localhost:8080/?tutorial=05&finish=1
http://localhost:8080/?tutorial=11&finish=1
```

When stepping completes, the page title becomes `stonesoup-viewer-ready` (for automation).

## Python reference plots

Viewer-style matplotlib PNGs for the same catalogue (plot panel only): [../python/README.md](../python/README.md). Regenerate with `bash tools/scripts/capture_python_tutorial_screenshots.sh`.

## Tutorial 11 vs Stone Soup (Python)

The upstream notebook [`ResamplingTutorial.py`](https://github.com/dstl/Stone-Soup/blob/a4336b920a799cfe0a77ecb05867c5deeb371c7a/docs/tutorials/sampling/ResamplingTutorial.py) is mostly **CDF / resampler explanation** plots. The Rust viewer matches the tutorial’s **“Example in Stone Soup”** resampler calls (same sequence as `tools/python/export_lot5_resampling.py`): scatter **index → 1-D state** after each demo step.

Compare graphics:

```bash
python3 tools/python/plot_lot5_resampling_reference.py   # screenshot/python/tutorial_11_step_*.png
bash tools/scripts/capture_wasm_tutorial11_steps.sh      # screenshot/wasm/tutorial_11_steps/step_*.png
python3 verification/scripts/compare_lot5_graphics.py    # numeric (index, state) table for all 6 steps
```
