# No-panic verification (strict crates)

## Claimed scope (Lot 1)

| Target | Profile | Status |
|--------|---------|--------|
| `x86_64-unknown-linux-gnu` | `no-panic-test`, `panic = "unwind"` | **Harness links** (Kalman + Lot 1 scenario) |
| `thumbv7em-none-eabihf` | `cargo check` strict crates | Compiles |
| `wasm32-unknown-unknown` | `cargo check` scenarios | Compiles |

**Certified entry points (native harness, `#[no_panic]`):**

- `exercise_kalman_once` — full predict/update on tutorial-sized state
- `exercise_lot1_scenario` — `Lot1KalmanScenario::run_to_end` + RMSE accumulator

Core/scenario functions used on these paths are annotated with `#[no_panic]` when built with feature `no-panic-cert` (enabled only for the standalone harness crate).

## How to run

```bash
cargo run --manifest-path verification/no-panic-harness/Cargo.toml --profile no-panic-test --locked
./verification/scripts/check_no_panic.sh   # wrapper (optional)
./verification/scripts/mutation_no_panic.sh
```

The harness lives **outside** the root workspace (`workspace.exclude`) so normal `cargo test --workspace` does not require the no-panic linker plugin.

## Negative control

`verification/scripts/mutation_no_panic.sh` injects a deliberate `unwrap()` into `KalmanUpdater::update` and expects a link failure with `ERROR[no-panic]`.

## Limits

- GUI / WASM / host tools are out of scope.
- `panic = "abort"` on embedded targets is a deployment choice, not a proof.
- Native x86_64 certification does not automatically transfer to ARM or WASM.
