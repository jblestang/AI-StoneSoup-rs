//! Fully unrolled fixed-size ops for certified Kalman (literal indices only).

use super::{require_finite, Mat, VecN};
use crate::error::Error;

#[inline]
fn add4(a: f64, b: f64) -> Result<f64, Error> {
    let v = a + b;
    require_finite(v)?;
    Ok(v)
}

#[inline]
fn sub4(a: f64, b: f64) -> Result<f64, Error> {
    let v = a - b;
    require_finite(v)?;
    Ok(v)
}

#[inline]
fn mulf(a: f64, b: f64) -> Result<f64, Error> {
    let v = a * b;
    require_finite(v)?;
    Ok(v)
}

macro_rules! dot4 {
    ($a:expr, $b:expr, $r:literal, $c:literal) => {{
        let mut s = 0.0;
        s = add4(s, mulf($a.0[$r][0], $b.0[0][$c])?)?;
        s = add4(s, mulf($a.0[$r][1], $b.0[1][$c])?)?;
        s = add4(s, mulf($a.0[$r][2], $b.0[2][$c])?)?;
        s = add4(s, mulf($a.0[$r][3], $b.0[3][$c])?)?;
        s
    }};
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn transpose4(m: Mat<4, 4>) -> Result<Mat<4, 4>, Error> {
    let mut out = Mat::<4, 4>::zeros();
    out.set(0, 0, m.0[0][0])?;
    out.set(0, 1, m.0[1][0])?;
    out.set(0, 2, m.0[2][0])?;
    out.set(0, 3, m.0[3][0])?;
    out.set(1, 0, m.0[0][1])?;
    out.set(1, 1, m.0[1][1])?;
    out.set(1, 2, m.0[2][1])?;
    out.set(1, 3, m.0[3][1])?;
    out.set(2, 0, m.0[0][2])?;
    out.set(2, 1, m.0[1][2])?;
    out.set(2, 2, m.0[2][2])?;
    out.set(2, 3, m.0[3][2])?;
    out.set(3, 0, m.0[0][3])?;
    out.set(3, 1, m.0[1][3])?;
    out.set(3, 2, m.0[2][3])?;
    out.set(3, 3, m.0[3][3])?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mul4x4(a: Mat<4, 4>, b: Mat<4, 4>) -> Result<Mat<4, 4>, Error> {
    let mut out = Mat::<4, 4>::zeros();
    out.set(0, 0, dot4!(a, b, 0, 0))?;
    out.set(0, 1, dot4!(a, b, 0, 1))?;
    out.set(0, 2, dot4!(a, b, 0, 2))?;
    out.set(0, 3, dot4!(a, b, 0, 3))?;
    out.set(1, 0, dot4!(a, b, 1, 0))?;
    out.set(1, 1, dot4!(a, b, 1, 1))?;
    out.set(1, 2, dot4!(a, b, 1, 2))?;
    out.set(1, 3, dot4!(a, b, 1, 3))?;
    out.set(2, 0, dot4!(a, b, 2, 0))?;
    out.set(2, 1, dot4!(a, b, 2, 1))?;
    out.set(2, 2, dot4!(a, b, 2, 2))?;
    out.set(2, 3, dot4!(a, b, 2, 3))?;
    out.set(3, 0, dot4!(a, b, 3, 0))?;
    out.set(3, 1, dot4!(a, b, 3, 1))?;
    out.set(3, 2, dot4!(a, b, 3, 2))?;
    out.set(3, 3, dot4!(a, b, 3, 3))?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mat4_vec4(m: Mat<4, 4>, v: VecN<4>) -> Result<VecN<4>, Error> {
    let mut out = VecN::<4>::zeros();
    let r0 = add4(
        add4(
            add4(mulf(m.0[0][0], v.0[0])?, mulf(m.0[0][1], v.0[1])?)?,
            mulf(m.0[0][2], v.0[2])?,
        )?,
        mulf(m.0[0][3], v.0[3])?,
    )?;
    let r1 = add4(
        add4(
            add4(mulf(m.0[1][0], v.0[0])?, mulf(m.0[1][1], v.0[1])?)?,
            mulf(m.0[1][2], v.0[2])?,
        )?,
        mulf(m.0[1][3], v.0[3])?,
    )?;
    let r2 = add4(
        add4(
            add4(mulf(m.0[2][0], v.0[0])?, mulf(m.0[2][1], v.0[1])?)?,
            mulf(m.0[2][2], v.0[2])?,
        )?,
        mulf(m.0[2][3], v.0[3])?,
    )?;
    let r3 = add4(
        add4(
            add4(mulf(m.0[3][0], v.0[0])?, mulf(m.0[3][1], v.0[1])?)?,
            mulf(m.0[3][2], v.0[2])?,
        )?,
        mulf(m.0[3][3], v.0[3])?,
    )?;
    out.set(0, r0)?;
    out.set(1, r1)?;
    out.set(2, r2)?;
    out.set(3, r3)?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn add4x4(a: Mat<4, 4>, b: Mat<4, 4>) -> Result<Mat<4, 4>, Error> {
    let mut out = Mat::<4, 4>::zeros();
    out.set(0, 0, add4(a.0[0][0], b.0[0][0])?)?;
    out.set(0, 1, add4(a.0[0][1], b.0[0][1])?)?;
    out.set(0, 2, add4(a.0[0][2], b.0[0][2])?)?;
    out.set(0, 3, add4(a.0[0][3], b.0[0][3])?)?;
    out.set(1, 0, add4(a.0[1][0], b.0[1][0])?)?;
    out.set(1, 1, add4(a.0[1][1], b.0[1][1])?)?;
    out.set(1, 2, add4(a.0[1][2], b.0[1][2])?)?;
    out.set(1, 3, add4(a.0[1][3], b.0[1][3])?)?;
    out.set(2, 0, add4(a.0[2][0], b.0[2][0])?)?;
    out.set(2, 1, add4(a.0[2][1], b.0[2][1])?)?;
    out.set(2, 2, add4(a.0[2][2], b.0[2][2])?)?;
    out.set(2, 3, add4(a.0[2][3], b.0[2][3])?)?;
    out.set(3, 0, add4(a.0[3][0], b.0[3][0])?)?;
    out.set(3, 1, add4(a.0[3][1], b.0[3][1])?)?;
    out.set(3, 2, add4(a.0[3][2], b.0[3][2])?)?;
    out.set(3, 3, add4(a.0[3][3], b.0[3][3])?)?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn sub4x4(a: Mat<4, 4>, b: Mat<4, 4>) -> Result<Mat<4, 4>, Error> {
    let mut out = Mat::<4, 4>::zeros();
    out.set(0, 0, sub4(a.0[0][0], b.0[0][0])?)?;
    out.set(0, 1, sub4(a.0[0][1], b.0[0][1])?)?;
    out.set(0, 2, sub4(a.0[0][2], b.0[0][2])?)?;
    out.set(0, 3, sub4(a.0[0][3], b.0[0][3])?)?;
    out.set(1, 0, sub4(a.0[1][0], b.0[1][0])?)?;
    out.set(1, 1, sub4(a.0[1][1], b.0[1][1])?)?;
    out.set(1, 2, sub4(a.0[1][2], b.0[1][2])?)?;
    out.set(1, 3, sub4(a.0[1][3], b.0[1][3])?)?;
    out.set(2, 0, sub4(a.0[2][0], b.0[2][0])?)?;
    out.set(2, 1, sub4(a.0[2][1], b.0[2][1])?)?;
    out.set(2, 2, sub4(a.0[2][2], b.0[2][2])?)?;
    out.set(2, 3, sub4(a.0[2][3], b.0[2][3])?)?;
    out.set(3, 0, sub4(a.0[3][0], b.0[3][0])?)?;
    out.set(3, 1, sub4(a.0[3][1], b.0[3][1])?)?;
    out.set(3, 2, sub4(a.0[3][2], b.0[3][2])?)?;
    out.set(3, 3, sub4(a.0[3][3], b.0[3][3])?)?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn identity4() -> Result<Mat<4, 4>, Error> {
    let mut m = Mat::<4, 4>::zeros();
    m.set(0, 0, 1.0)?;
    m.set(1, 1, 1.0)?;
    m.set(2, 2, 1.0)?;
    m.set(3, 3, 1.0)?;
    Ok(m)
}

macro_rules! dot24 {
    ($a:expr, $b:expr, $r:literal, $c:literal) => {{
        let mut s = 0.0;
        s = add4(s, mulf($a.0[$r][0], $b.0[0][$c])?)?;
        s = add4(s, mulf($a.0[$r][1], $b.0[1][$c])?)?;
        s = add4(s, mulf($a.0[$r][2], $b.0[2][$c])?)?;
        s = add4(s, mulf($a.0[$r][3], $b.0[3][$c])?)?;
        s
    }};
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mul2x4_4x4(a: Mat<2, 4>, b: Mat<4, 4>) -> Result<Mat<2, 4>, Error> {
    let mut out = Mat::<2, 4>::zeros();
    out.set(0, 0, dot24!(a, b, 0, 0))?;
    out.set(0, 1, dot24!(a, b, 0, 1))?;
    out.set(0, 2, dot24!(a, b, 0, 2))?;
    out.set(0, 3, dot24!(a, b, 0, 3))?;
    out.set(1, 0, dot24!(a, b, 1, 0))?;
    out.set(1, 1, dot24!(a, b, 1, 1))?;
    out.set(1, 2, dot24!(a, b, 1, 2))?;
    out.set(1, 3, dot24!(a, b, 1, 3))?;
    Ok(out)
}

macro_rules! dot24_42 {
    ($a:expr, $b:expr, $r:literal, $c:literal) => {{
        let mut s = 0.0;
        s = add4(s, mulf($a.0[$r][0], $b.0[0][$c])?)?;
        s = add4(s, mulf($a.0[$r][1], $b.0[1][$c])?)?;
        s = add4(s, mulf($a.0[$r][2], $b.0[2][$c])?)?;
        s = add4(s, mulf($a.0[$r][3], $b.0[3][$c])?)?;
        s
    }};
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mul2x4_4x2(a: Mat<2, 4>, b: Mat<4, 2>) -> Result<Mat<2, 2>, Error> {
    let mut out = Mat::<2, 2>::zeros();
    out.set(0, 0, dot24_42!(a, b, 0, 0))?;
    out.set(0, 1, dot24_42!(a, b, 0, 1))?;
    out.set(1, 0, dot24_42!(a, b, 1, 0))?;
    out.set(1, 1, dot24_42!(a, b, 1, 1))?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn add2x2(a: Mat<2, 2>, b: Mat<2, 2>) -> Result<Mat<2, 2>, Error> {
    let mut out = Mat::<2, 2>::zeros();
    out.set(0, 0, add4(a.0[0][0], b.0[0][0])?)?;
    out.set(0, 1, add4(a.0[0][1], b.0[0][1])?)?;
    out.set(1, 0, add4(a.0[1][0], b.0[1][0])?)?;
    out.set(1, 1, add4(a.0[1][1], b.0[1][1])?)?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mul4x4_4x2(a: Mat<4, 4>, b: Mat<4, 2>) -> Result<Mat<4, 2>, Error> {
    let mut out = Mat::<4, 2>::zeros();
    out.set(0, 0, dot24_42!(a, b, 0, 0))?;
    out.set(0, 1, dot24_42!(a, b, 0, 1))?;
    out.set(1, 0, dot24_42!(a, b, 1, 0))?;
    out.set(1, 1, dot24_42!(a, b, 1, 1))?;
    out.set(2, 0, dot24_42!(a, b, 2, 0))?;
    out.set(2, 1, dot24_42!(a, b, 2, 1))?;
    out.set(3, 0, dot24_42!(a, b, 3, 0))?;
    out.set(3, 1, dot24_42!(a, b, 3, 1))?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mul4x2_2x2(a: Mat<4, 2>, b: Mat<2, 2>) -> Result<Mat<4, 2>, Error> {
    let mut out = Mat::<4, 2>::zeros();
    out.set(
        0,
        0,
        add4(mulf(a.0[0][0], b.0[0][0])?, mulf(a.0[0][1], b.0[1][0])?)?,
    )?;
    out.set(
        0,
        1,
        add4(mulf(a.0[0][0], b.0[0][1])?, mulf(a.0[0][1], b.0[1][1])?)?,
    )?;
    out.set(
        1,
        0,
        add4(mulf(a.0[1][0], b.0[0][0])?, mulf(a.0[1][1], b.0[1][0])?)?,
    )?;
    out.set(
        1,
        1,
        add4(mulf(a.0[1][0], b.0[0][1])?, mulf(a.0[1][1], b.0[1][1])?)?,
    )?;
    out.set(
        2,
        0,
        add4(mulf(a.0[2][0], b.0[0][0])?, mulf(a.0[2][1], b.0[1][0])?)?,
    )?;
    out.set(
        2,
        1,
        add4(mulf(a.0[2][0], b.0[0][1])?, mulf(a.0[2][1], b.0[1][1])?)?,
    )?;
    out.set(
        3,
        0,
        add4(mulf(a.0[3][0], b.0[0][0])?, mulf(a.0[3][1], b.0[1][0])?)?,
    )?;
    out.set(
        3,
        1,
        add4(mulf(a.0[3][0], b.0[0][1])?, mulf(a.0[3][1], b.0[1][1])?)?,
    )?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mul2x4_4x1(h: Mat<2, 4>, v: VecN<4>) -> Result<VecN<2>, Error> {
    let mut out = VecN::<2>::zeros();
    let r0 = add4(
        add4(
            add4(mulf(h.0[0][0], v.0[0])?, mulf(h.0[0][1], v.0[1])?)?,
            mulf(h.0[0][2], v.0[2])?,
        )?,
        mulf(h.0[0][3], v.0[3])?,
    )?;
    let r1 = add4(
        add4(
            add4(mulf(h.0[1][0], v.0[0])?, mulf(h.0[1][1], v.0[1])?)?,
            mulf(h.0[1][2], v.0[2])?,
        )?,
        mulf(h.0[1][3], v.0[3])?,
    )?;
    out.set(0, r0)?;
    out.set(1, r1)?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mul4x2_2x1(k: Mat<4, 2>, v: VecN<2>) -> Result<VecN<4>, Error> {
    let mut out = VecN::<4>::zeros();
    out.set(0, add4(mulf(k.0[0][0], v.0[0])?, mulf(k.0[0][1], v.0[1])?)?)?;
    out.set(1, add4(mulf(k.0[1][0], v.0[0])?, mulf(k.0[1][1], v.0[1])?)?)?;
    out.set(2, add4(mulf(k.0[2][0], v.0[0])?, mulf(k.0[2][1], v.0[1])?)?)?;
    out.set(3, add4(mulf(k.0[3][0], v.0[0])?, mulf(k.0[3][1], v.0[1])?)?)?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn mul4x2_2x4(k: Mat<4, 2>, h: Mat<2, 4>) -> Result<Mat<4, 4>, Error> {
    let mut out = Mat::<4, 4>::zeros();
    out.set(
        0,
        0,
        add4(mulf(k.0[0][0], h.0[0][0])?, mulf(k.0[0][1], h.0[1][0])?)?,
    )?;
    out.set(
        0,
        1,
        add4(mulf(k.0[0][0], h.0[0][1])?, mulf(k.0[0][1], h.0[1][1])?)?,
    )?;
    out.set(
        0,
        2,
        add4(mulf(k.0[0][0], h.0[0][2])?, mulf(k.0[0][1], h.0[1][2])?)?,
    )?;
    out.set(
        0,
        3,
        add4(mulf(k.0[0][0], h.0[0][3])?, mulf(k.0[0][1], h.0[1][3])?)?,
    )?;
    out.set(
        1,
        0,
        add4(mulf(k.0[1][0], h.0[0][0])?, mulf(k.0[1][1], h.0[1][0])?)?,
    )?;
    out.set(
        1,
        1,
        add4(mulf(k.0[1][0], h.0[0][1])?, mulf(k.0[1][1], h.0[1][1])?)?,
    )?;
    out.set(
        1,
        2,
        add4(mulf(k.0[1][0], h.0[0][2])?, mulf(k.0[1][1], h.0[1][2])?)?,
    )?;
    out.set(
        1,
        3,
        add4(mulf(k.0[1][0], h.0[0][3])?, mulf(k.0[1][1], h.0[1][3])?)?,
    )?;
    out.set(
        2,
        0,
        add4(mulf(k.0[2][0], h.0[0][0])?, mulf(k.0[2][1], h.0[1][0])?)?,
    )?;
    out.set(
        2,
        1,
        add4(mulf(k.0[2][0], h.0[0][1])?, mulf(k.0[2][1], h.0[1][1])?)?,
    )?;
    out.set(
        2,
        2,
        add4(mulf(k.0[2][0], h.0[0][2])?, mulf(k.0[2][1], h.0[1][2])?)?,
    )?;
    out.set(
        2,
        3,
        add4(mulf(k.0[2][0], h.0[0][3])?, mulf(k.0[2][1], h.0[1][3])?)?,
    )?;
    out.set(
        3,
        0,
        add4(mulf(k.0[3][0], h.0[0][0])?, mulf(k.0[3][1], h.0[1][0])?)?,
    )?;
    out.set(
        3,
        1,
        add4(mulf(k.0[3][0], h.0[0][1])?, mulf(k.0[3][1], h.0[1][1])?)?,
    )?;
    out.set(
        3,
        2,
        add4(mulf(k.0[3][0], h.0[0][2])?, mulf(k.0[3][1], h.0[1][2])?)?,
    )?;
    out.set(
        3,
        3,
        add4(mulf(k.0[3][0], h.0[0][3])?, mulf(k.0[3][1], h.0[1][3])?)?,
    )?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn i_minus_kh(k: Mat<4, 2>, h: Mat<2, 4>) -> Result<Mat<4, 4>, Error> {
    let kh = mul4x2_2x4(k, h)?;
    let i = identity4()?;
    sub4x4(i, kh)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn transpose2x4(m: Mat<2, 4>) -> Result<Mat<4, 2>, Error> {
    let mut out = Mat::<4, 2>::zeros();
    out.set(0, 0, m.0[0][0])?;
    out.set(0, 1, m.0[1][0])?;
    out.set(1, 0, m.0[0][1])?;
    out.set(1, 1, m.0[1][1])?;
    out.set(2, 0, m.0[0][2])?;
    out.set(2, 1, m.0[1][2])?;
    out.set(3, 0, m.0[0][3])?;
    out.set(3, 1, m.0[1][3])?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
/// Inverse of a 4×4 matrix (Gauss–Jordan with partial pivoting, fully unrolled).
pub fn inv4(m: Mat<4, 4>) -> Result<Mat<4, 4>, Error> {
    let mut a = [
        [m.0[0][0], m.0[0][1], m.0[0][2], m.0[0][3]],
        [m.0[1][0], m.0[1][1], m.0[1][2], m.0[1][3]],
        [m.0[2][0], m.0[2][1], m.0[2][2], m.0[2][3]],
        [m.0[3][0], m.0[3][1], m.0[3][2], m.0[3][3]],
    ];
    let mut inv = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    // column 0
    let mut pivot = 0;
    let mut best = libm::fabs(a[0][0]);
    {
        let v = libm::fabs(a[1][0]);
        if v > best {
            best = v;
            pivot = 1;
        }
    }
    {
        let v = libm::fabs(a[2][0]);
        if v > best {
            best = v;
            pivot = 2;
        }
    }
    {
        let v = libm::fabs(a[3][0]);
        if v > best {
            best = v;
            pivot = 3;
        }
    }
    if best < 1e-18 {
        return Err(Error::SingularInnovation);
    }
    if pivot == 1 {
        {
            let (ra, rb) = (a[0], a[1]);
            a[0] = rb;
            a[1] = ra;
        }
        {
            let (ia, ib) = (inv[0], inv[1]);
            inv[0] = ib;
            inv[1] = ia;
        }
    }
    if pivot == 2 {
        {
            let (ra, rb) = (a[0], a[2]);
            a[0] = rb;
            a[2] = ra;
        }
        {
            let (ia, ib) = (inv[0], inv[2]);
            inv[0] = ib;
            inv[2] = ia;
        }
    }
    if pivot == 3 {
        {
            let (ra, rb) = (a[0], a[3]);
            a[0] = rb;
            a[3] = ra;
        }
        {
            let (ia, ib) = (inv[0], inv[3]);
            inv[0] = ib;
            inv[3] = ia;
        }
    }
    let div = a[0][0];
    a[0][0] /= div;
    inv[0][0] /= div;
    a[0][1] /= div;
    inv[0][1] /= div;
    a[0][2] /= div;
    inv[0][2] /= div;
    a[0][3] /= div;
    inv[0][3] /= div;
    {
        let factor = a[1][0];
        if factor != 0.0 {
            a[1][0] -= factor * a[0][0];
            inv[1][0] -= factor * inv[0][0];
            a[1][1] -= factor * a[0][1];
            inv[1][1] -= factor * inv[0][1];
            a[1][2] -= factor * a[0][2];
            inv[1][2] -= factor * inv[0][2];
            a[1][3] -= factor * a[0][3];
            inv[1][3] -= factor * inv[0][3];
        }
    }
    {
        let factor = a[2][0];
        if factor != 0.0 {
            a[2][0] -= factor * a[0][0];
            inv[2][0] -= factor * inv[0][0];
            a[2][1] -= factor * a[0][1];
            inv[2][1] -= factor * inv[0][1];
            a[2][2] -= factor * a[0][2];
            inv[2][2] -= factor * inv[0][2];
            a[2][3] -= factor * a[0][3];
            inv[2][3] -= factor * inv[0][3];
        }
    }
    {
        let factor = a[3][0];
        if factor != 0.0 {
            a[3][0] -= factor * a[0][0];
            inv[3][0] -= factor * inv[0][0];
            a[3][1] -= factor * a[0][1];
            inv[3][1] -= factor * inv[0][1];
            a[3][2] -= factor * a[0][2];
            inv[3][2] -= factor * inv[0][2];
            a[3][3] -= factor * a[0][3];
            inv[3][3] -= factor * inv[0][3];
        }
    }
    // column 1
    let mut pivot = 1;
    let mut best = libm::fabs(a[1][1]);
    {
        let v = libm::fabs(a[2][1]);
        if v > best {
            best = v;
            pivot = 2;
        }
    }
    {
        let v = libm::fabs(a[3][1]);
        if v > best {
            best = v;
            pivot = 3;
        }
    }
    if best < 1e-18 {
        return Err(Error::SingularInnovation);
    }
    if pivot == 0 {
        {
            let (ra, rb) = (a[1], a[0]);
            a[1] = rb;
            a[0] = ra;
        }
        {
            let (ia, ib) = (inv[1], inv[0]);
            inv[1] = ib;
            inv[0] = ia;
        }
    }
    if pivot == 2 {
        {
            let (ra, rb) = (a[1], a[2]);
            a[1] = rb;
            a[2] = ra;
        }
        {
            let (ia, ib) = (inv[1], inv[2]);
            inv[1] = ib;
            inv[2] = ia;
        }
    }
    if pivot == 3 {
        {
            let (ra, rb) = (a[1], a[3]);
            a[1] = rb;
            a[3] = ra;
        }
        {
            let (ia, ib) = (inv[1], inv[3]);
            inv[1] = ib;
            inv[3] = ia;
        }
    }
    let div = a[1][1];
    a[1][0] /= div;
    inv[1][0] /= div;
    a[1][1] /= div;
    inv[1][1] /= div;
    a[1][2] /= div;
    inv[1][2] /= div;
    a[1][3] /= div;
    inv[1][3] /= div;
    {
        let factor = a[0][1];
        if factor != 0.0 {
            a[0][0] -= factor * a[1][0];
            inv[0][0] -= factor * inv[1][0];
            a[0][1] -= factor * a[1][1];
            inv[0][1] -= factor * inv[1][1];
            a[0][2] -= factor * a[1][2];
            inv[0][2] -= factor * inv[1][2];
            a[0][3] -= factor * a[1][3];
            inv[0][3] -= factor * inv[1][3];
        }
    }
    {
        let factor = a[2][1];
        if factor != 0.0 {
            a[2][0] -= factor * a[1][0];
            inv[2][0] -= factor * inv[1][0];
            a[2][1] -= factor * a[1][1];
            inv[2][1] -= factor * inv[1][1];
            a[2][2] -= factor * a[1][2];
            inv[2][2] -= factor * inv[1][2];
            a[2][3] -= factor * a[1][3];
            inv[2][3] -= factor * inv[1][3];
        }
    }
    {
        let factor = a[3][1];
        if factor != 0.0 {
            a[3][0] -= factor * a[1][0];
            inv[3][0] -= factor * inv[1][0];
            a[3][1] -= factor * a[1][1];
            inv[3][1] -= factor * inv[1][1];
            a[3][2] -= factor * a[1][2];
            inv[3][2] -= factor * inv[1][2];
            a[3][3] -= factor * a[1][3];
            inv[3][3] -= factor * inv[1][3];
        }
    }
    // column 2
    let mut pivot = 2;
    let mut best = libm::fabs(a[2][2]);
    {
        let v = libm::fabs(a[3][2]);
        if v > best {
            best = v;
            pivot = 3;
        }
    }
    if best < 1e-18 {
        return Err(Error::SingularInnovation);
    }
    if pivot == 0 {
        {
            let (ra, rb) = (a[2], a[0]);
            a[2] = rb;
            a[0] = ra;
        }
        {
            let (ia, ib) = (inv[2], inv[0]);
            inv[2] = ib;
            inv[0] = ia;
        }
    }
    if pivot == 1 {
        {
            let (ra, rb) = (a[2], a[1]);
            a[2] = rb;
            a[1] = ra;
        }
        {
            let (ia, ib) = (inv[2], inv[1]);
            inv[2] = ib;
            inv[1] = ia;
        }
    }
    if pivot == 3 {
        {
            let (ra, rb) = (a[2], a[3]);
            a[2] = rb;
            a[3] = ra;
        }
        {
            let (ia, ib) = (inv[2], inv[3]);
            inv[2] = ib;
            inv[3] = ia;
        }
    }
    let div = a[2][2];
    a[2][0] /= div;
    inv[2][0] /= div;
    a[2][1] /= div;
    inv[2][1] /= div;
    a[2][2] /= div;
    inv[2][2] /= div;
    a[2][3] /= div;
    inv[2][3] /= div;
    {
        let factor = a[0][2];
        if factor != 0.0 {
            a[0][0] -= factor * a[2][0];
            inv[0][0] -= factor * inv[2][0];
            a[0][1] -= factor * a[2][1];
            inv[0][1] -= factor * inv[2][1];
            a[0][2] -= factor * a[2][2];
            inv[0][2] -= factor * inv[2][2];
            a[0][3] -= factor * a[2][3];
            inv[0][3] -= factor * inv[2][3];
        }
    }
    {
        let factor = a[1][2];
        if factor != 0.0 {
            a[1][0] -= factor * a[2][0];
            inv[1][0] -= factor * inv[2][0];
            a[1][1] -= factor * a[2][1];
            inv[1][1] -= factor * inv[2][1];
            a[1][2] -= factor * a[2][2];
            inv[1][2] -= factor * inv[2][2];
            a[1][3] -= factor * a[2][3];
            inv[1][3] -= factor * inv[2][3];
        }
    }
    {
        let factor = a[3][2];
        if factor != 0.0 {
            a[3][0] -= factor * a[2][0];
            inv[3][0] -= factor * inv[2][0];
            a[3][1] -= factor * a[2][1];
            inv[3][1] -= factor * inv[2][1];
            a[3][2] -= factor * a[2][2];
            inv[3][2] -= factor * inv[2][2];
            a[3][3] -= factor * a[2][3];
            inv[3][3] -= factor * inv[2][3];
        }
    }
    // column 3 (no pivot search — only one row left)
    let best = libm::fabs(a[3][3]);
    if best < 1e-18 {
        return Err(Error::SingularInnovation);
    }
    let div = a[3][3];
    a[3][0] /= div;
    inv[3][0] /= div;
    a[3][1] /= div;
    inv[3][1] /= div;
    a[3][2] /= div;
    inv[3][2] /= div;
    a[3][3] /= div;
    inv[3][3] /= div;
    {
        let factor = a[0][3];
        if factor != 0.0 {
            a[0][0] -= factor * a[3][0];
            inv[0][0] -= factor * inv[3][0];
            a[0][1] -= factor * a[3][1];
            inv[0][1] -= factor * inv[3][1];
            a[0][2] -= factor * a[3][2];
            inv[0][2] -= factor * inv[3][2];
            a[0][3] -= factor * a[3][3];
            inv[0][3] -= factor * inv[3][3];
        }
    }
    {
        let factor = a[1][3];
        if factor != 0.0 {
            a[1][0] -= factor * a[3][0];
            inv[1][0] -= factor * inv[3][0];
            a[1][1] -= factor * a[3][1];
            inv[1][1] -= factor * inv[3][1];
            a[1][2] -= factor * a[3][2];
            inv[1][2] -= factor * inv[3][2];
            a[1][3] -= factor * a[3][3];
            inv[1][3] -= factor * inv[3][3];
        }
    }
    {
        let factor = a[2][3];
        if factor != 0.0 {
            a[2][0] -= factor * a[3][0];
            inv[2][0] -= factor * inv[3][0];
            a[2][1] -= factor * a[3][1];
            inv[2][1] -= factor * inv[3][1];
            a[2][2] -= factor * a[3][2];
            inv[2][2] -= factor * inv[3][2];
            a[2][3] -= factor * a[3][3];
            inv[2][3] -= factor * inv[3][3];
        }
    }
    let mut out = Mat::<4, 4>::zeros();
    out.set(0, 0, inv[0][0])?;
    out.set(0, 1, inv[0][1])?;
    out.set(0, 2, inv[0][2])?;
    out.set(0, 3, inv[0][3])?;
    out.set(1, 0, inv[1][0])?;
    out.set(1, 1, inv[1][1])?;
    out.set(1, 2, inv[1][2])?;
    out.set(1, 3, inv[1][3])?;
    out.set(2, 0, inv[2][0])?;
    out.set(2, 1, inv[2][1])?;
    out.set(2, 2, inv[2][2])?;
    out.set(2, 3, inv[2][3])?;
    out.set(3, 0, inv[3][0])?;
    out.set(3, 1, inv[3][1])?;
    out.set(3, 2, inv[3][2])?;
    out.set(3, 3, inv[3][3])?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn symmetrize4(m: Mat<4, 4>) -> Result<Mat<4, 4>, Error> {
    let mut out = Mat::<4, 4>::zeros();
    out.set(0, 0, mulf(0.5, add4(m.0[0][0], m.0[0][0])?)?)?;
    out.set(0, 1, mulf(0.5, add4(m.0[0][1], m.0[1][0])?)?)?;
    out.set(0, 2, mulf(0.5, add4(m.0[0][2], m.0[2][0])?)?)?;
    out.set(0, 3, mulf(0.5, add4(m.0[0][3], m.0[3][0])?)?)?;
    out.set(1, 0, mulf(0.5, add4(m.0[1][0], m.0[0][1])?)?)?;
    out.set(1, 1, mulf(0.5, add4(m.0[1][1], m.0[1][1])?)?)?;
    out.set(1, 2, mulf(0.5, add4(m.0[1][2], m.0[2][1])?)?)?;
    out.set(1, 3, mulf(0.5, add4(m.0[1][3], m.0[3][1])?)?)?;
    out.set(2, 0, mulf(0.5, add4(m.0[2][0], m.0[0][2])?)?)?;
    out.set(2, 1, mulf(0.5, add4(m.0[2][1], m.0[1][2])?)?)?;
    out.set(2, 2, mulf(0.5, add4(m.0[2][2], m.0[2][2])?)?)?;
    out.set(2, 3, mulf(0.5, add4(m.0[2][3], m.0[3][2])?)?)?;
    out.set(3, 0, mulf(0.5, add4(m.0[3][0], m.0[0][3])?)?)?;
    out.set(3, 1, mulf(0.5, add4(m.0[3][1], m.0[1][3])?)?)?;
    out.set(3, 2, mulf(0.5, add4(m.0[3][2], m.0[2][3])?)?)?;
    out.set(3, 3, mulf(0.5, add4(m.0[3][3], m.0[3][3])?)?)?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn inv2(m: Mat<2, 2>) -> Result<Mat<2, 2>, Error> {
    let a = m.0[0][0];
    let b = m.0[0][1];
    let c = m.0[1][0];
    let d = m.0[1][1];
    let det = a * d - b * c;
    if libm::fabs(det) < 1e-18 {
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

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn sub_vec2(a: VecN<2>, b: VecN<2>) -> Result<VecN<2>, Error> {
    let mut out = VecN::<2>::zeros();
    out.set(0, sub4(a.0[0], b.0[0])?)?;
    out.set(1, sub4(a.0[1], b.0[1])?)?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn add_vec4(a: VecN<4>, b: VecN<4>) -> Result<VecN<4>, Error> {
    let mut out = VecN::<4>::zeros();
    out.set(0, add4(a.0[0], b.0[0])?)?;
    out.set(1, add4(a.0[1], b.0[1])?)?;
    out.set(2, add4(a.0[2], b.0[2])?)?;
    out.set(3, add4(a.0[3], b.0[3])?)?;
    Ok(out)
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn validate_gaussian(mean: VecN<4>, cov: Mat<4, 4>) -> Result<(), Error> {
    require_finite(mean.0[0])?;
    require_finite(mean.0[1])?;
    require_finite(mean.0[2])?;
    require_finite(mean.0[3])?;
    require_finite(cov.0[0][0])?;
    require_finite(cov.0[0][1])?;
    require_finite(cov.0[0][2])?;
    require_finite(cov.0[0][3])?;
    require_finite(cov.0[1][0])?;
    require_finite(cov.0[1][1])?;
    require_finite(cov.0[1][2])?;
    require_finite(cov.0[1][3])?;
    require_finite(cov.0[2][0])?;
    require_finite(cov.0[2][1])?;
    require_finite(cov.0[2][2])?;
    require_finite(cov.0[2][3])?;
    require_finite(cov.0[3][0])?;
    require_finite(cov.0[3][1])?;
    require_finite(cov.0[3][2])?;
    require_finite(cov.0[3][3])?;
    Ok(())
}

#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn transpose4x2(m: Mat<4, 2>) -> Result<Mat<2, 4>, Error> {
    let mut out = Mat::<2, 4>::zeros();
    out.set(0, 0, m.0[0][0])?;
    out.set(0, 1, m.0[1][0])?;
    out.set(0, 2, m.0[2][0])?;
    out.set(0, 3, m.0[3][0])?;
    out.set(1, 0, m.0[0][1])?;
    out.set(1, 1, m.0[1][1])?;
    out.set(1, 2, m.0[2][1])?;
    out.set(1, 3, m.0[3][1])?;
    Ok(out)
}
