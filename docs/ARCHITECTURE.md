# Architecture

## Workspace map

| Crate / app | Profile | Depends on |
|-------------|---------|------------|
| `stonesoup-core` | `#![no_std]`, `#![forbid(unsafe_code)]` | `libm` only |
| `stonesoup-sim` | same strict contract | `stonesoup-core` |
| `stonesoup-scenarios` | strict for autonomous scenarios | core + sim |
| `stonesoup-io` | host `std`, serde | core (not used by strict graph) |
| `stonesoup-viewer` | `std`, eframe 0.29, egui_plot 0.29 | scenarios (strict) |
| `verification/*`, `xtask`, `tools/` | host | strict crates |

No edge from core → io, viewer, or Python.

## Design decisions (Lot 1)

1. **Linear algebra:** fixed-size `Mat<R,C>` / `VecN<N>` with fallible `get`/`set` instead of nalgebra in the strict path (predictable errors, no hidden panics from deps). Nalgebra remains a candidate for host-only or audited subsets later.
2. **State order:** `[x, vx, y, vy]` (SI), matching Tutorial 01 mapping `(0, 2)` for position measurements.
3. **Kalman covariance update:** default **standard** form to match Stone Soup Tutorial 01; Joseph form available via `KalmanUpdater::with_joseph`.
4. **Time:** `Timestamp` = nanoseconds from scenario start; uniform 1 s steps in Lot 1.
5. **RNG:** `SplitMix64` + Box–Muller in sim — **not** NumPy-compatible; parity uses **exported measurements**, not matched noise draws.
6. **Viewer:** simulation stepping in `Lot1KalmanScenario::step()`; UI only reads snapshots and trails (no RNG in render path).

## Data flow (Lot 1)

```
fixtures (Python export) → MEASUREMENTS/TRUTH/TRACK_REF (embedded)
                         → Kalman loop in stonesoup-scenarios
                         → snapshots → stonesoup-viewer (native / WASM)
```
