#!/usr/bin/env bash
# Ensure ScenarioRegistry entries match docs/PORTING_MATRIX.csv tutorial row.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
csv="$root/docs/PORTING_MATRIX.csv"
sha="$(grep -m1 'a4336b920a799cfe0a77ecb05867c5deeb371c7a' "$root/crates/stonesoup-scenarios/src/registry.rs" || true)"
if [[ -z "$sha" ]]; then
  echo "catalogue: upstream SHA missing from ScenarioRegistry"
  exit 1
fi
if ! grep -q '01_KalmanFilterTutorial.py' "$csv"; then
  echo "catalogue: tutorial 01 missing from PORTING_MATRIX.csv"
  exit 1
fi
if ! grep -qE 'Lot1KalmanTutorial|01_KalmanFilterTutorial' "$csv"; then
  echo "catalogue: Lot 1 row not found in matrix"
  exit 1
fi
for tutorial in 02_ExtendedKalmanFilterTutorial.py 03_UnscentedKalmanFilterTutorial.py 04_ParticleFilter.py 05_DataAssociation-Clutter.py 06_DataAssociation-MultiTargetTutorial.py 07_PDATutorial.py 08_JPDATutorial.py; do
  if ! grep -q "$tutorial" "$csv"; then
    echo "catalogue: $tutorial missing from PORTING_MATRIX.csv"
    exit 1
  fi
done
echo "catalogue: registry SHA and Lot 1–4 matrix rows present"
