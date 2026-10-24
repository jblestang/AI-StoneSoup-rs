# Stone Soup — Rust port (work in progress)

Rust framework for target tracking and state estimation, porting [Stone Soup](https://github.com/dstl/Stone-Soup) with strict core crates and a shared **egui** viewer (desktop + WebAssembly).

**Upstream pin:** Stone Soup `v1.9.1` @ `a4336b920a799cfe0a77ecb05867c5deeb371c7a` — see [docs/UPSTREAM.md](docs/UPSTREAM.md).

## What works today (Lot 1)

- Strict **`no_std`** core: 2D nearly-constant velocity model, linear-Gaussian measurements, Kalman predict/update (Joseph form optional), RMSE, covariance ellipse helper.
- Scenario **Tutorial 01 — Kalman filter** with fixtures exported from Python (seed 1991) and parity test (±1e-6 on state mean vs Stone Soup).
- **`stonesoup_sim::generate_lot1_tutorial(1991)`** regenerates truth + measurements with the same NumPy/RandomKit stream as Stone Soup (not SplitMix64).
- **Viewer** (`apps/stonesoup-viewer`): catalogue entry, play/pause/step/reset, 2D plot (truth, detections, track, covariance ellipse), fixture seed **1991**, export results JSON to clipboard.
- **Headless** (`apps/stonesoup-headless`): run Lot 1 without GUI.
- **No-panic** certification for Kalman + Lot 1 scenario path (native harness).
- **WASM** build via Trunk (`apps/stonesoup-viewer/dist/`).

**Coverage (inventory at pinned SHA):** 271 tracked rows — components **~5/198** validated or adapted for Lot 1; scenarios **1/73** validated. See [docs/PORTING_MATRIX.csv](docs/PORTING_MATRIX.csv).

## Commands

```bash
# Tests (includes Python-oracle parity via Rust test)
cargo test --workspace --locked

# Strict core on embedded / WASM targets
cargo check -p stonesoup-core --no-default-features --target thumbv7em-none-eabihf --locked
cargo check -p stonesoup-scenarios --no-default-features --target wasm32-unknown-unknown --locked

# Clippy (strict crates)
cargo clippy -p stonesoup-core -p stonesoup-sim -p stonesoup-scenarios --no-deps --locked -- -D warnings

# Native viewer
cargo run -p stonesoup-viewer --release --locked

# WASM viewer (from app dir; unset NO_COLOR if Trunk errors)
cd apps/stonesoup-viewer && unset NO_COLOR && trunk build --release

Final-state UI captures (tutorials 01–13): see [screenshot/wasm/](screenshot/wasm/README.md). Regenerate with `bash tools/scripts/capture_wasm_tutorial_screenshots.sh`.

# No-panic harness (see docs/NO_PANIC.md)
cargo run --manifest-path verification/no-panic-harness/Cargo.toml --profile no-panic-test --locked

# Headless Lot 1 (stdout RMSE)
cargo run -p stonesoup-headless --release --locked -- lot1

# Aggregated verification
cargo run -p xtask -- verify
```

### Python oracle (dev only)

```bash
pip install -e git+https://github.com/dstl/Stone-Soup.git@v1.9.1#egg=stonesoup
python3 tools/python/export_lot1_fixtures.py
pytest verification/python_parity/test_lot1_kalman.py
```

## Layout

| Path | Role |
|------|------|
| `crates/stonesoup-core` | Types, models, Kalman, metrics (`no_std`) |
| `crates/stonesoup-sim` | Truth path, sensor, RNG (`no_std`) |
| `crates/stonesoup-scenarios` | Scenario registry + Lot 1 |
| `crates/stonesoup-io` | JSON fixtures (host `std`) |
| `apps/stonesoup-viewer` | egui / eframe / egui_plot |
| `verification/` | Fixtures, no-panic harness, embedded smoke |
| `tools/python/` | Stone Soup fixture export |

## Toolchain

Rust **1.88.0** (see `rust-toolchain.toml`) — required by egui 0.29 / transitive ICU crates.

## Next

[docs/NEXT_STEPS.md](docs/NEXT_STEPS.md) — Lot 2 (EKF/UKF/particle), extended no-panic coverage, browser smoke automation.
