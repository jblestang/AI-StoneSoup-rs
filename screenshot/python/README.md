# Python / Stone Soup — tutorial final states

Matplotlib captures of the **same viewer trails** as the Rust WASM catalogue (`?tutorial=XX&finish=1`), produced by stepping scenarios in `stonesoup-scenarios` and plotting with `tools/python/plot_viewer_trail.py`.

Plot panel only (no egui catalogue sidebar). Colors and layers match `apps/stonesoup-viewer/src/app.rs`.

| File | Catalogue |
|------|-----------|
| `tutorial_01.png` … `tutorial_29.png` | Registry entries **01–29** (tutorials + OOSM/filter examples) |
| `tutorial_11_step_*.png` | Resampling tutorial — per resampler case |
| `tutorial_11_all_cases.png` | Resampling — 2×3 grid of all cases |

## Regenerate

From the repo root:

```bash
bash tools/scripts/capture_python_tutorial_screenshots.sh
```

Requires Rust, Python 3, and Matplotlib (installed automatically if missing). The script builds and runs `dump_viewer_trail` once per tutorial.

## Compare with WASM

```bash
bash tools/scripts/capture_wasm_tutorial_screenshots.sh
bash tools/scripts/capture_python_tutorial_screenshots.sh
```

WASM PNGs include the catalogue UI; Python PNGs are the tracking plot only. Trails and final-step geometry should align.
