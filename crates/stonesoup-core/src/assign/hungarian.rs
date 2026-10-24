//! Minimal assignment (Hungarian) for GNN — mirrors ``scipy.optimize.linear_sum_assignment``.

#![allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use crate::Error;

const INF: f64 = 1e100;

/// Min-cost assignment for a rectangular `rows` × `cols` cost matrix (row-major).
/// Writes chosen column index for each row into `col_for_row` (length `rows`).
#[cfg_attr(feature = "no-panic-cert", no_panic::no_panic)]
pub fn linear_sum_assignment_min(
    cost: &[f64],
    rows: usize,
    cols: usize,
    col_for_row: &mut [usize],
) -> Result<(), Error> {
    if rows == 0 || cols == 0 || rows > col_for_row.len() || cost.len() < rows * cols {
        return Err(Error::InvalidDimension);
    }
    let n = rows.max(cols);
    if n > 32 {
        return Err(Error::CapacityExceeded);
    }
    let mut a = [0.0f64; 32 * 32];
    for i in 0..rows {
        for j in 0..cols {
            a[i * n + j] = cost[i * cols + j];
        }
        for j in cols..n {
            a[i * n + j] = INF;
        }
    }
    for i in rows..n {
        for j in 0..n {
            a[i * n + j] = 0.0;
        }
    }
    let mut u = [0.0f64; 32];
    let mut v = [0.0f64; 32];
    let mut p = [0usize; 32];
    let mut way = [0usize; 32];
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = [INF; 32];
        let mut used = [false; 32];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = INF;
            let mut j1 = 0usize;
            for j in 1..=n {
                if used[j] {
                    continue;
                }
                let cur = a[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    for j in 1..=n {
        if p[j] > 0 && p[j] <= rows {
            col_for_row[p[j] - 1] = j - 1;
        }
    }
    Ok(())
}
