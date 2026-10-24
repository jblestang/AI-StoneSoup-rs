# Panic mutation witness (manual)

Before enabling full Kalman coverage in `stonesoup-no-panic-harness`:

1. On a throwaway branch, add `let _ = Some(0).unwrap();` inside `exercise_kalman_once` in the no-panic harness (or run `./verification/scripts/mutation_no_panic.sh`).
2. Run `cargo run -p stonesoup-no-panic-harness --profile no-panic-test --locked`.
3. Expect **link failure** with `ERROR[no-panic]`.
4. Revert the mutation — harness must link again.

Do not commit the mutation to `main`.
