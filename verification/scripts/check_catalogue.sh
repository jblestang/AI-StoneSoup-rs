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
for tutorial in 02_ExtendedKalmanFilterTutorial.py 03_UnscentedKalmanFilterTutorial.py 04_ParticleFilter.py 05_DataAssociation-Clutter.py 06_DataAssociation-MultiTargetTutorial.py 07_PDATutorial.py 08_JPDATutorial.py 09_Initiators_\&_Deleters.py 10_Simulation_\&_Tracking_Components.py; do
  if ! grep -q "$tutorial" "$csv"; then
    echo "catalogue: $tutorial missing from PORTING_MATRIX.csv"
    exit 1
  fi
done
if ! grep -q 'ResamplingTutorial.py' "$csv"; then
  echo "catalogue: ResamplingTutorial.py missing from PORTING_MATRIX.csv"
  exit 1
fi
if ! grep -q 'InformationFilterTutorial.py' "$csv"; then
  echo "catalogue: InformationFilterTutorial.py missing from PORTING_MATRIX.csv"
  exit 1
fi
if ! grep -q 'GMPHDTutorial.py' "$csv"; then
  echo "catalogue: GMPHDTutorial.py missing from PORTING_MATRIX.csv"
  exit 1
fi
if ! grep -q 'ASDFilter.py' "$csv"; then
  echo "catalogue: ASDFilter.py missing from PORTING_MATRIX.csv"
  exit 1
fi
if ! grep -q 'KDTree_Tutorial.py' "$csv"; then
  echo "catalogue: KDTree_Tutorial.py missing from PORTING_MATRIX.csv"
  exit 1
fi
if ! grep -q '01_SingleSensorManagement.py' "$csv"; then
  echo "catalogue: 01_SingleSensorManagement.py missing from PORTING_MATRIX.csv"
  exit 1
fi
if ! grep -q '02_MultiSensorManagement.py' "$csv"; then
  echo "catalogue: 02_MultiSensorManagement.py missing from PORTING_MATRIX.csv"
  exit 1
fi
echo "catalogue: registry SHA and matrix rows present (tutorials 01–10 + resampling + IF + GM-PHD + ASD + KD-tree + single + multi sensor mgmt)"
