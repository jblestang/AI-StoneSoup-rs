# Numerics conventions

## Floating point

- Reference type: **`f64`** everywhere in strict Lot 1.
- Non-finite inputs → `Error::NonFiniteInput`.
- Singular 2×2 innovation → `Error::SingularInnovation` (no silent regularisation).

## Kalman (linear)

Matches Stone Soup `KalmanPredictor` / `KalmanUpdater` for linear models:

- Predict: `x = F x`, `P = F P Fᵀ + Q`.
- Update: `S = H P Hᵀ + R`, `K = P Hᵀ S⁻¹`, standard `P = (I - K H) P` unless Joseph enabled.

**Tolerance vs Python oracle:** absolute **1e-6** on posterior mean per step (Tutorial 01 fixture set).

## Process / measurement noise

- `ConstantVelocity2d`: block-diagonal Q from 1D CV blocks (same exponents as Stone Soup `ConstantVelocity`).
- Measurement: `R = diag(5, 5)` in Tutorial 01.

## Ellipse display

- Position covariance block indices `(0,0), (0,2), (2,2)`.
- Viewer uses Mahalanobis scale **2.0** for display (not a calibrated 95% χ² gate yet).

## Angles / UKF / particles

Not in scope for Lot 1 — documented when Lot 2 lands.
