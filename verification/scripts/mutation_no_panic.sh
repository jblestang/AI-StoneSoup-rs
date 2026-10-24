#!/usr/bin/env bash
# Negative control: removing one #[no_panic] cert from the Kalman call graph must break the harness link.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
target="$root/crates/stonesoup-core/src/filter/kalman.rs"
backup="$(mktemp)"
cp "$target" "$backup"
cleanup() { cp "$backup" "$target"; rm -f "$backup"; }
trap cleanup EXIT

perl -i -0777 -pe 's/#\[cfg_attr\(feature = "no-panic-cert", no_panic::no_panic\)\]\n    pub fn predict\(/    pub fn predict(/s' "$target"

log="$(mktemp)"
cargo run --manifest-path "$root/verification/no-panic-harness/Cargo.toml" --profile no-panic-test --locked >"$log" 2>&1 || true

if rg -q 'Running.*stonesoup-no-panic-harness' "$log"; then
  echo "mutation_no_panic: harness linked after stripping predict cert (unexpected)"
  cat "$log"
  rm -f "$log"
  exit 1
fi

if rg -q 'ERROR\[no-panic\]' "$log"; then
  echo "mutation_no_panic: detected expected link failure"
  rm -f "$log"
  exit 0
fi

echo "mutation_no_panic: build failed but without no-panic marker:"
cat "$log"
rm -f "$log"
exit 1
