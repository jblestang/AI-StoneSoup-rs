#![allow(clippy::arithmetic_side_effects)]

use crate::error::Error;
use libm::{fabs, sqrt};

#[inline]
pub fn is_finite(x: f64) -> bool {
    x.is_finite()
}

#[inline]
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

    pub fn get(&self, r: usize, c: usize) -> Result<f64, Error> {
        self.0
            .get(r)
            .and_then(|row| row.get(c))
            .copied()
            .ok_or(Error::InvalidDimension)
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) -> Result<(), Error> {
        require_finite(v)?;
        let row = self.0.get_mut(r).ok_or(Error::InvalidDimension)?;
        let cell = row.get_mut(c).ok_or(Error::InvalidDimension)?;
        *cell = v;
        Ok(())
    }

    pub fn transpose(self) -> Result<Mat<C, R>, Error> {
        let mut out = Mat::<C, R>::zeros();
        for i in 0..R {
            for j in 0..C {
                out.set(j, i, self.get(i, j)?)?;
            }
        }
        Ok(out)
    }

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

    pub fn get(&self, i: usize) -> Result<f64, Error> {
        self.0.get(i).copied().ok_or(Error::InvalidDimension)
    }

    pub fn set(&mut self, i: usize, v: f64) -> Result<(), Error> {
        require_finite(v)?;
        let cell = self.0.get_mut(i).ok_or(Error::InvalidDimension)?;
        *cell = v;
        Ok(())
    }

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

    pub fn add(self, other: Self) -> Result<Self, Error> {
        let mut out = Self::zeros();
        for i in 0..N {
            let v = self.get(i)? + other.get(i)?;
            require_finite(v)?;
            out.set(i, v)?;
        }
        Ok(out)
    }

    pub fn sub(self, other: Self) -> Result<Self, Error> {
        let mut out = Self::zeros();
        for i in 0..N {
            let v = self.get(i)? - other.get(i)?;
            require_finite(v)?;
            out.set(i, v)?;
        }
        Ok(out)
    }

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
pub fn inv2(m: Mat<2, 2>) -> Result<Mat<2, 2>, Error> {
    let a = m.get(0, 0)?;
    let b = m.get(0, 1)?;
    let c = m.get(1, 0)?;
    let d = m.get(1, 1)?;
    let det = a * d - b * c;
    if fabs(det) < 1e-18 {
        return Err(Error::SingularInnovation);
    }
    let inv_det = 1.0 / det;
    require_finite(inv_det)?;
    let mut out = Mat::<2, 2>::zeros();
    out.set(0, 0, d * inv_det)?;
    out.set(0, 1, -b * inv_det)?;
    out.set(1, 0, -c * inv_det)?;
    out.set(1, 1, a * inv_det)?;
    Ok(out)
}

/// Cholesky factorization of 4×4 SPD matrix; returns lower triangular L with A ≈ L Lᵀ.
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

pub fn identity4() -> Result<Mat<4, 4>, Error> {
    let mut m = Mat::<4, 4>::zeros();
    for i in 0..4 {
        m.set(i, i, 1.0)?;
    }
    Ok(m)
}

/// 2×2 symmetric covariance ellipse semi-axes at Mahalanobis scale `conf`.
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
