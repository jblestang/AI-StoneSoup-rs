use stonesoup_core::Error;

use crate::numpy_random::NumpyRng;

/// NumPy `multivariate_normal` SVD transform for Lot 1 process noise `Q` (seed stream compatible).
const LOT1_A_Q: [[f64; 4]; 4] = [
    [-0.1187916679626401, -0.2219642619541573, -0.0, -0.0],
    [-0.0, -0.0, -0.1187916679626401, -0.2219642619541573],
    [-0.050549048352273925, 0.027053029685161423, 0.0, 0.0],
    [0.0, 0.0, -0.050549048352273925, 0.027053029685161423],
];

/// NumPy SVD transform for Lot 1 measurement noise `R = diag(5, 5)`.
#[allow(clippy::excessive_precision)] // literals copied from NumPy reference output
const LOT1_A_R: [[f64; 2]; 2] = [[2.2360679774997898, 0.0], [0.0, 2.2360679774997898]];

pub(crate) fn z_dot_a_col<const N: usize>(
    z: [f64; N],
    a: [[f64; N]; N],
    col: usize,
) -> Result<f64, Error> {
    let mut s = 0.0;
    for j in 0..N {
        s += z[j] * a[j][col];
    }
    if !s.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(s)
}

/// `sum_j a[row][j] * z[j]` — NumPy `multivariate_normal` uses `mean + A @ z`.
pub(crate) fn z_dot_a_row<const N: usize>(
    z: [f64; N],
    a: [[f64; N]; N],
    row: usize,
) -> Result<f64, Error> {
    let mut s = 0.0;
    for j in 0..N {
        s += a[row][j] * z[j];
    }
    if !s.is_finite() {
        return Err(Error::NonFiniteInput);
    }
    Ok(s)
}

/// Sample `N(0, Q)` for Tutorial 01 using NumPy `RandomState.multivariate_normal` semantics.
pub fn mvn4_lot1_q(rng: &mut NumpyRng) -> Result<[f64; 4], Error> {
    let z = [
        rng.standard_normal()?,
        rng.standard_normal()?,
        rng.standard_normal()?,
        rng.standard_normal()?,
    ];
    Ok([
        z_dot_a_col(z, LOT1_A_Q, 0)?,
        z_dot_a_col(z, LOT1_A_Q, 1)?,
        z_dot_a_col(z, LOT1_A_Q, 2)?,
        z_dot_a_col(z, LOT1_A_Q, 3)?,
    ])
}

/// Sample `N(0, R)` for Tutorial 01 measurement noise.
pub fn mvn2_lot1_r(rng: &mut NumpyRng) -> Result<[f64; 2], Error> {
    let z = [rng.standard_normal()?, rng.standard_normal()?];
    Ok([z_dot_a_col(z, LOT1_A_R, 0)?, z_dot_a_col(z, LOT1_A_R, 1)?])
}

pub fn add_vec4(a: [f64; 4], b: [f64; 4]) -> Result<[f64; 4], Error> {
    let mut out = [0.0; 4];
    for i in 0..4 {
        let v = a[i] + b[i];
        if !v.is_finite() {
            return Err(Error::NonFiniteInput);
        }
        out[i] = v;
    }
    Ok(out)
}

pub fn mat4_vec4(m: stonesoup_core::Mat<4, 4>, v: [f64; 4]) -> Result<[f64; 4], Error> {
    let out = stonesoup_core::linalg::mat_vec_mul(m, stonesoup_core::VecN(v))?;
    Ok([out.get(0)?, out.get(1)?, out.get(2)?, out.get(3)?])
}

pub fn measure_position(h: stonesoup_core::Mat<2, 4>, state: [f64; 4]) -> Result<[f64; 2], Error> {
    let z = stonesoup_core::linalg::mat_vec_mul(h, stonesoup_core::VecN(state))?;
    Ok([z.get(0)?, z.get(1)?])
}

/// `np.random.multivariate_normal(mean, cov, size=n)` using [`NumpyRng`].
pub fn multivariate_normal4_batch(
    rng: &mut NumpyRng,
    mean: [f64; 4],
    cov: &[[f64; 4]; 4],
    n: usize,
) -> Result<alloc::vec::Vec<[f64; 4]>, Error> {
    extern crate alloc;
    use alloc::vec::Vec;

    use crate::sym4::mvn_factor_a;

    let a = mvn_factor_a(cov)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let z = [
            rng.standard_normal()?,
            rng.standard_normal()?,
            rng.standard_normal()?,
            rng.standard_normal()?,
        ];
        out.push([
            mean[0] + z_dot_a_row(z, a, 0)?,
            mean[1] + z_dot_a_row(z, a, 1)?,
            mean[2] + z_dot_a_row(z, a, 2)?,
            mean[3] + z_dot_a_row(z, a, 3)?,
        ]);
    }
    Ok(out)
}

pub fn mat4_to_array(m: stonesoup_core::Mat<4, 4>) -> Result<[[f64; 4]; 4], Error> {
    let mut a = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = m.get(r, c)?;
        }
    }
    Ok(a)
}
