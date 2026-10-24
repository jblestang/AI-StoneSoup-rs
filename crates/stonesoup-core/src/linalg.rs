#![allow(clippy::arithmetic_side_effects)]

pub mod dynamic;
pub mod fixed;

use crate::error::Error;
use libm::{fabs, sqrt};

#[inline]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn is_finite(x: f64) -> bool {
    x.is_finite()
}

#[inline]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn require_finite(x: f64) -> Result<(), Error> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(Error::NonFiniteInput)
    }
}

/// Row-major `rows x cols` matrix stored inline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat<const R: usize, const C: usize>(pub [[f64; C]; R]);

impl<const R: usize, const C: usize> Mat<R, C> {
    pub const fn zeros() -> Self {
        Self([[0.0; C]; R])
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn get(&self, r: usize, c: usize) -> Result<f64, Error> {
        self.0
            .get(r)
            .and_then(|row| row.get(c))
            .copied()
            .ok_or(Error::InvalidDimension)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn set(&mut self, r: usize, c: usize, v: f64) -> Result<(), Error> {
        require_finite(v)?;
        let row = self.0.get_mut(r).ok_or(Error::InvalidDimension)?;
        let cell = row.get_mut(c).ok_or(Error::InvalidDimension)?;
        *cell = v;
        Ok(())
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn transpose(self) -> Result<Mat<C, R>, Error> {
        let mut out = Mat::<C, R>::zeros();
        for i in 0..R {
            for j in 0..C {
                out.set(j, i, self.get(i, j)?)?;
            }
        }
        Ok(out)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn add(self, other: Self) -> Result<Self, Error> {
        let mut out = Self::zeros();
        for i in 0..R {
            for j in 0..C {
                let v = self.get(i, j)? + other.get(i, j)?;
                require_finite(v)?;
                out.set(i, j, v)?;
            }
        }
        Ok(out)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn sub(self, other: Self) -> Result<Self, Error> {
        let mut out = Self::zeros();
        for i in 0..R {
            for j in 0..C {
                let v = self.get(i, j)? - other.get(i, j)?;
                require_finite(v)?;
                out.set(i, j, v)?;
            }
        }
        Ok(out)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn scale(self, s: f64) -> Result<Self, Error> {
        require_finite(s)?;
        let mut out = Self::zeros();
        for i in 0..R {
            for j in 0..C {
                let v = self.get(i, j)? * s;
                require_finite(v)?;
                out.set(i, j, v)?;
            }
        }
        Ok(out)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn mul<const K: usize>(self, other: Mat<C, K>) -> Result<Mat<R, K>, Error> {
        let mut out = Mat::<R, K>::zeros();
        for i in 0..R {
            for j in 0..K {
                let mut sum = 0.0;
                for k in 0..C {
                    sum += self.get(i, k)? * other.get(k, j)?;
                }
                require_finite(sum)?;
                out.set(i, j, sum)?;
            }
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VecN<const N: usize>(pub [f64; N]);

impl<const N: usize> VecN<N> {
    pub const fn zeros() -> Self {
        Self([0.0; N])
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn get(&self, i: usize) -> Result<f64, Error> {
        self.0.get(i).copied().ok_or(Error::InvalidDimension)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn set(&mut self, i: usize, v: f64) -> Result<(), Error> {
        require_finite(v)?;
        let cell = self.0.get_mut(i).ok_or(Error::InvalidDimension)?;
        *cell = v;
        Ok(())
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn from_slice(slice: &[f64]) -> Result<Self, Error> {
        if slice.len() != N {
            return Err(Error::InvalidDimension);
        }
        let mut out = Self::zeros();
        for (i, &v) in slice.iter().enumerate() {
            require_finite(v)?;
            out.set(i, v)?;
        }
        Ok(out)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn add(self, other: Self) -> Result<Self, Error> {
        let mut out = Self::zeros();
        for i in 0..N {
            let v = self.get(i)? + other.get(i)?;
            require_finite(v)?;
            out.set(i, v)?;
        }
        Ok(out)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn sub(self, other: Self) -> Result<Self, Error> {
        let mut out = Self::zeros();
        for i in 0..N {
            let v = self.get(i)? - other.get(i)?;
            require_finite(v)?;
            out.set(i, v)?;
        }
        Ok(out)
    }

    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    #[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
    pub fn scale(self, s: f64) -> Result<Self, Error> {
        require_finite(s)?;
        let mut out = Self::zeros();
        for i in 0..N {
            let v = self.get(i)? * s;
            require_finite(v)?;
            out.set(i, v)?;
        }
        Ok(out)
    }
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mat_vec_mul<const R: usize, const C: usize>(
    m: Mat<R, C>,
    v: VecN<C>,
) -> Result<VecN<R>, Error> {
    let mut out = VecN::zeros();
    for i in 0..R {
        let mut sum = 0.0;
        for j in 0..C {
            sum += m.get(i, j)? * v.get(j)?;
        }
        require_finite(sum)?;
        out.set(i, sum)?;
    }
    Ok(out)
}

/// Invert 2×2 matrix; returns error if singular or non-finite.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn inv2(m: Mat<2, 2>) -> Result<Mat<2, 2>, Error> {
    fixed::inv2(m)
}

/// Cholesky factorization of 2×2 SPD matrix; returns lower triangular L with A ≈ L Lᵀ.
pub fn cholesky2(a: Mat<2, 2>) -> Result<Mat<2, 2>, Error> {
    let mut l = Mat::<2, 2>::zeros();
    let c00 = a.get(0, 0)?;
    let c10 = a.get(1, 0)?;
    let c11 = a.get(1, 1)?;
    if c00 <= 0.0 {
        return Err(Error::InvalidCovariance);
    }
    let l00 = sqrt(c00);
    require_finite(l00)?;
    l.set(0, 0, l00)?;
    let l10 = c10 / l00;
    require_finite(l10)?;
    l.set(1, 0, l10)?;
    let rem = c11 - l10 * l10;
    if rem <= 0.0 {
        return Err(Error::InvalidCovariance);
    }
    let l11 = sqrt(rem);
    require_finite(l11)?;
    l.set(1, 1, l11)?;
    Ok(l)
}

/// Cholesky factorization of 4×4 SPD matrix; returns lower triangular L with A ≈ L Lᵀ.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn cholesky4(a: Mat<4, 4>) -> Result<Mat<4, 4>, Error> {
    let mut l = Mat::<4, 4>::zeros();
    for i in 0..4 {
        for j in 0..=i {
            let mut sum = a.get(i, j)?;
            for k in 0..j {
                sum -= l.get(i, k)? * l.get(j, k)?;
            }
            require_finite(sum)?;
            if i == j {
                if sum <= 0.0 {
                    return Err(Error::InvalidCovariance);
                }
                let diag = sqrt(sum);
                require_finite(diag)?;
                l.set(i, j, diag)?;
            } else {
                let denom = l.get(j, j)?;
                if fabs(denom) < 1e-18 {
                    return Err(Error::NumericalFailure);
                }
                let v = sum / denom;
                require_finite(v)?;
                l.set(i, j, v)?;
            }
        }
    }
    Ok(l)
}

/// Solve A x = b for SPD A using Cholesky.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn solve_spd4(a: Mat<4, 4>, b: VecN<4>) -> Result<VecN<4>, Error> {
    let l = cholesky4(a)?;
    let mut y = VecN::<4>::zeros();
    for i in 0..4 {
        let mut sum = b.get(i)?;
        for k in 0..i {
            sum -= l.get(i, k)? * y.get(k)?;
        }
        let denom = l.get(i, i)?;
        if fabs(denom) < 1e-18 {
            return Err(Error::NumericalFailure);
        }
        let v = sum / denom;
        require_finite(v)?;
        y.set(i, v)?;
    }
    let mut x = VecN::<4>::zeros();
    for i in (0..4).rev() {
        let mut sum = y.get(i)?;
        for k in (i + 1)..4 {
            sum -= l.get(k, i)? * x.get(k)?;
        }
        let denom = l.get(i, i)?;
        if fabs(denom) < 1e-18 {
            return Err(Error::NumericalFailure);
        }
        let v = sum / denom;
        require_finite(v)?;
        x.set(i, v)?;
    }
    Ok(x)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn identity4() -> Result<Mat<4, 4>, Error> {
    fixed::identity4()
}

/// 2×2 symmetric covariance ellipse semi-axes at Mahalanobis scale `conf`.
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn ellipse_from_cov2(c00: f64, c01: f64, c11: f64, conf: f64) -> Result<(f64, f64), Error> {
    require_finite(c00)?;
    require_finite(c01)?;
    require_finite(c11)?;
    require_finite(conf)?;
    if conf <= 0.0 {
        return Err(Error::InvalidParameter);
    }
    let trace = c00 + c11;
    let det = c00 * c11 - c01 * c01;
    if det <= 0.0 {
        return Err(Error::InvalidCovariance);
    }
    let disc = trace * trace / 4.0 - det;
    let temp = libm::sqrt(if disc > 0.0 { disc } else { 0.0 });
    require_finite(temp)?;
    let l1 = trace / 2.0 + temp;
    let l2 = trace / 2.0 - temp;
    if l1 <= 0.0 || l2 <= 0.0 {
        return Err(Error::InvalidCovariance);
    }
    let s1 = conf * libm::sqrt(l1);
    let s2 = conf * libm::sqrt(l2);
    require_finite(s1)?;
    require_finite(s2)?;
    Ok((s1, s2))
}
