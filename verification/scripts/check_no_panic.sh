#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
cargo build -p stonesoup-no-panic-harness --profile no-panic-test --locked
BIN="$ROOT/target/no-panic-test/stonesoup-no-panic-harness"
if command -v no-panic-validate >/dev/null 2>&1; then
  no-panic-validate "$BIN"
else
  echo "no-panic-validate not installed; ran linked harness only: $BIN"
  "$BIN"
fi
