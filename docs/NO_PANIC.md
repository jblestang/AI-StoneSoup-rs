# No-panic verification (strict crates)

## Claimed scope (current)

| Target | Profile | Status |
|--------|---------|--------|
| `x86_64-unknown-linux-gnu` | `no-panic-test` profile, `panic = "unwind"` | Partial harness links |
| `thumbv7em-none-eabihf` | `cargo check` strict crates | Compiles |
| `wasm32-unknown-unknown` | `cargo check` scenarios | Compiles |

**Certified entry points (native, `#[no_panic]` harness):**

- `ConstantVelocity2d::new`, `transition_matrix`, `process_noise`
- `LinearGaussianMeasurement::position_2d`
- `GaussianState::new_from_array`
- `RmseAccumulator::push`, `rmse`
- `stonesoup_scenarios::tutorial_cov`

**Not yet certified (linker plugin still reports panic paths):**

- `KalmanPredictor::predict`, `KalmanUpdater::update` (`Mat::mul` / transpose chain)
- `Lot1KalmanScenario::tutorial_defaults` / `run_to_end` (depends on Kalman)

Root cause under investigation: `no-panic` treats some optimised matrix loops as potentially panicking despite fallible `get`/`set`. Workaround tracked in Lot 1.1 — do **not** claim full strict-crate panic freedom until Kalman paths link clean.

## How to run

```bash
cargo run -p stonesoup-no-panic-harness --profile no-panic-test --locked
./verification/scripts/check_no_panic.sh
```

## Negative control

`verification/mutations/README.md` — manual mutation procedure (insert `unwrap()` in core); harness / link step must fail before merge when full Kalman coverage is enabled.

## Limits

- GUI / WASM / host tools are out of scope.
- `panic = "abort"` on embedded targets is a deployment choice, not a proof.
- Native x86_64 certification does not automatically transfer to ARM or WASM.
