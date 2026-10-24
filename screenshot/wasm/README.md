# WASM viewer — tutorial final states

PNG captures of the **Stone Soup Rust** egui viewer (`apps/stonesoup-viewer`) running as **WebAssembly**, each tutorial stepped to completion.

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
```

When stepping completes, the page title becomes `stonesoup-viewer-ready` (for automation).
