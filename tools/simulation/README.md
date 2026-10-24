# Standalone Stone Soup simulations

## Dual rotating radar network (`dual_radar_clutter_network.py`)

Co-located **two rotating bearing–range radars** on one platform, fused with a single **MultiTargetTracker** (GNN + EKF, 3-point initiation, 8 s coast delete).

| Radar | Rotation | Max range |
|-------|----------|-----------|
| A | 6 s (10 RPM) | 400 km |
| B | 3 s (20 RPM) | 200 km |

With co-located circular coverage, the **B disk is 25% of A’s area** (close to a nominal 20% overlap spec). Four truth targets sit at **80 km** on the cardinal bearings so both radars rotate through them.

**Clutter:** Poisson `ClutterModel` with `clutter_rate=0.52` (tuned to ~**10%** clutter among plots for this geometry; re-run may vary slightly). Simulation stops after **250 plots per radar** (500 total cap).

**Note:** Call `sensor.act(time)` without resetting `sensor.timestamp` first, so dwell centres advance between steps.

**Run:**

```bash
python3 tools/simulation/dual_radar_clutter_network.py
```

**Output:** `verification/reports/dual_radar_clutter_stats.json` (plot counts, clutter fraction, truth detection coverage, track/false-track counts, durations).

Track–truth labelling uses Stone Soup `TrackToTruth` with a 2 km association threshold and default consecutive-confirm rules.
