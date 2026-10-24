//! Quadratic polynomial kernel Gram matrices (Stone Soup `QuadraticKernel`, `p=2`).

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::Error;
use crate::linalg::dynamic::DynMat;
use crate::linalg::require_finite;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraticKernel {
    pub c: f64,
    pub ialpha: f64,
}

impl QuadraticKernel {
    pub const fn tutorial_akkf() -> Self {
        Self { c: 1.0, ialpha: 1.0 }
    }

    fn pair(&self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        if a.len() != b.len() {
            return Err(Error::InvalidDimension);
        }
        let mut dot = 0.0;
        for i in 0..a.len() {
            dot += a[i] * b[i];
        }
        let v = dot / self.ialpha + self.c;
        require_finite(v)?;
        let k = v * v;
        require_finite(k)?;
        Ok(k)
    }

    fn col_vec(cols: &[f64], dim: usize, n: usize, i: usize) -> Result<Vec<f64>, Error> {
        if cols.len() != dim * n || i >= n {
            return Err(Error::InvalidDimension);
        }
        let mut v = vec![0.0; dim];
        for d in 0..dim {
            v[d] = cols[d * n + i];
        }
        Ok(v)
    }

    /// Gram matrix for column-major particles (`dim` × `n` layout).
    pub fn gram_columns(&self, dim: usize, cols: &[f64], n: usize) -> Result<DynMat, Error> {
        if cols.len() != dim * n {
            return Err(Error::InvalidDimension);
        }
        let mut k = DynMat::zeros(n);
        for i in 0..n {
            let ai = Self::col_vec(cols, dim, n, i)?;
            for j in 0..n {
                let bj = Self::col_vec(cols, dim, n, j)?;
                k.set(i, j, self.pair(&ai, &bj)?)?;
            }
        }
        Ok(k)
    }

    /// Kernel vector against one fixed column vector (`dim` elements).
    pub fn vector_columns(
        &self,
        dim: usize,
        cols: &[f64],
        n: usize,
        fixed: &[f64],
    ) -> Result<Vec<f64>, Error> {
        if cols.len() != dim * n || fixed.len() != dim {
            return Err(Error::InvalidDimension);
        }
        let mut out = vec![0.0; n];
        for i in 0..n {
            let ai = Self::col_vec(cols, dim, n, i)?;
            out[i] = self.pair(&ai, fixed)?;
        }
        Ok(out)
    }
}

/// Gram matrix from two column-major particle sets.
pub fn quadratic_gram(
    kernel: &QuadraticKernel,
    dim: usize,
    left: &[f64],
    right: &[f64],
    n: usize,
) -> Result<DynMat, Error> {
    if left.len() != dim * n || right.len() != dim * n {
        return Err(Error::InvalidDimension);
    }
    let mut k = DynMat::zeros(n);
    for i in 0..n {
        let ai = QuadraticKernel::col_vec(left, dim, n, i)?;
        for j in 0..n {
            let bj = QuadraticKernel::col_vec(right, dim, n, j)?;
            k.set(i, j, kernel.pair(&ai, &bj)?)?;
        }
    }
    Ok(k)
}
