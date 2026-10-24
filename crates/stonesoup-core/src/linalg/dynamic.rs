//! Small heap matrices for kernel filters (AKKF), `no_std` + `alloc`.

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::identity_op,
    clippy::erasing_op
)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::Error;
use crate::linalg::require_finite;

/// Row-major square matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct DynMat {
    pub n: usize,
    pub data: Vec<f64>,
}

impl DynMat {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n],
        }
    }

    pub fn identity(n: usize) -> Result<Self, Error> {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m.set(i, i, 1.0)?;
        }
        Ok(m)
    }

    fn idx(&self, r: usize, c: usize) -> Result<usize, Error> {
        if r >= self.n || c >= self.n {
            return Err(Error::InvalidDimension);
        }
        Ok(r * self.n + c)
    }

    pub fn get(&self, r: usize, c: usize) -> Result<f64, Error> {
        let i = self.idx(r, c)?;
        Ok(self.data[i])
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) -> Result<(), Error> {
        require_finite(v)?;
        let i = self.idx(r, c)?;
        self.data[i] = v;
        Ok(())
    }

    pub fn transpose(&self) -> Result<Self, Error> {
        let mut out = Self::zeros(self.n);
        for r in 0..self.n {
            for c in 0..self.n {
                out.set(c, r, self.get(r, c)?)?;
            }
        }
        Ok(out)
    }

    pub fn add(&self, other: &Self) -> Result<Self, Error> {
        if self.n != other.n {
            return Err(Error::InvalidDimension);
        }
        let mut out = Self::zeros(self.n);
        for i in 0..self.data.len() {
            let v = self.data[i] + other.data[i];
            require_finite(v)?;
            out.data[i] = v;
        }
        Ok(out)
    }

    pub fn sub(&self, other: &Self) -> Result<Self, Error> {
        if self.n != other.n {
            return Err(Error::InvalidDimension);
        }
        let mut out = Self::zeros(self.n);
        for i in 0..self.data.len() {
            let v = self.data[i] - other.data[i];
            require_finite(v)?;
            out.data[i] = v;
        }
        Ok(out)
    }

    pub fn scale(&self, s: f64) -> Result<Self, Error> {
        require_finite(s)?;
        let mut out = Self::zeros(self.n);
        for i in 0..self.data.len() {
            let v = self.data[i] * s;
            require_finite(v)?;
            out.data[i] = v;
        }
        Ok(out)
    }

    pub fn mul(&self, other: &Self) -> Result<Self, Error> {
        if self.n != other.n {
            return Err(Error::InvalidDimension);
        }
        let n = self.n;
        let mut out = Self::zeros(n);
        for r in 0..n {
            for c in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += self.get(r, k)? * other.get(k, c)?;
                }
                require_finite(sum)?;
                out.set(r, c, sum)?;
            }
        }
        Ok(out)
    }

    /// Moore–Penrose pseudoinverse via regularised normal equations (matches `numpy.linalg.pinv` on well-conditioned Gram matrices).
    pub fn pinv(&self) -> Result<Self, Error> {
        let n = self.n;
        if n == 0 {
            return Err(Error::InvalidDimension);
        }
        let at = self.transpose()?;
        let ata = at.mul(self)?;
        let mut reg = DynMat::identity(n)?;
        reg = reg.scale(1e-12)?;
        let ata = ata.add(&reg)?;
        let ata_inv = ata.inv()?;
        let pinv = ata_inv.mul(&at)?;
        Ok(pinv)
    }

    /// Gauss–Jordan inversion for moderate `n` (AKKF particle count).
    pub fn inv(&self) -> Result<Self, Error> {
        let n = self.n;
        let mut aug = vec![0.0; n * 2 * n];
        for r in 0..n {
            for c in 0..n {
                aug[r * 2 * n + c] = self.get(r, c)?;
            }
            aug[r * 2 * n + n + r] = 1.0;
        }
        for col in 0..n {
            let mut pivot = col;
            let mut best = libm::fabs(aug[pivot * 2 * n + col]);
            for r in (col + 1)..n {
                let v = libm::fabs(aug[r * 2 * n + col]);
                if v > best {
                    best = v;
                    pivot = r;
                }
            }
            if best < 1e-18 {
                return Err(Error::NumericalFailure);
            }
            if pivot != col {
                for c in 0..(2 * n) {
                    let i0 = col * 2 * n + c;
                    let i1 = pivot * 2 * n + c;
                    aug.swap(i0, i1);
                }
            }
            let div = aug[col * 2 * n + col];
            for c in 0..(2 * n) {
                let i = col * 2 * n + c;
                aug[i] /= div;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = aug[r * 2 * n + col];
                if factor == 0.0 {
                    continue;
                }
                for c in 0..(2 * n) {
                    let i = r * 2 * n + c;
                    aug[i] -= factor * aug[col * 2 * n + c];
                }
            }
        }
        let mut out = Self::zeros(n);
        for r in 0..n {
            for c in 0..n {
                out.set(r, c, aug[r * 2 * n + n + c])?;
            }
        }
        Ok(out)
    }
}

pub fn mat_vec(m: &DynMat, v: &[f64]) -> Result<Vec<f64>, Error> {
    if m.n != v.len() {
        return Err(Error::InvalidDimension);
    }
    let mut out = vec![0.0; m.n];
    for r in 0..m.n {
        let mut sum = 0.0;
        for c in 0..m.n {
            sum += m.get(r, c)? * v[c];
        }
        require_finite(sum)?;
        out[r] = sum;
    }
    Ok(out)
}
