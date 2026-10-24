//! Static 2-D :math:`k`-d tree for detection indexing (Stone Soup ``scipy.spatial.KDTree`` subset).

#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::too_many_arguments
)]

use crate::error::Error;
use crate::types::Detection;

pub const MAX_KDTREE_POINTS: usize = 48;

#[derive(Debug, Clone)]
struct Node {
    point_index: usize,
    left: Option<usize>,
    right: Option<usize>,
    axis: u8,
}

/// 2-D kd-tree over detection positions ``(x, y)``.
#[derive(Debug, Clone)]
pub struct KdTree2D {
    nodes: [Option<Node>; MAX_KDTREE_POINTS * 2],
    n_nodes: usize,
    n_points: usize,
    points: [[f64; 2]; MAX_KDTREE_POINTS],
}

impl KdTree2D {
    pub fn build(detections: &[Detection]) -> Result<Self, Error> {
        let n = detections.len();
        if n == 0 {
            return Ok(Self {
                nodes: [const { None }; MAX_KDTREE_POINTS * 2],
                n_nodes: 0,
                n_points: 0,
                points: [[0.0; 2]; MAX_KDTREE_POINTS],
            });
        }
        if n > MAX_KDTREE_POINTS {
            return Err(Error::CapacityExceeded);
        }
        let mut points = [[0.0; 2]; MAX_KDTREE_POINTS];
        for (i, d) in detections.iter().enumerate() {
            let x = d.vector.get(0)?;
            let y = d.vector.get(1)?;
            points[i] = [x, y];
        }
        let mut order: [usize; MAX_KDTREE_POINTS] = core::array::from_fn(|i| i);
        let mut tree = Self {
            nodes: [const { None }; MAX_KDTREE_POINTS * 2],
            n_nodes: 0,
            n_points: n,
            points,
        };
        if n > 0 {
            tree.build_rec(0, n, &mut order, 0)?;
        }
        Ok(tree)
    }

    fn build_rec(
        &mut self,
        start: usize,
        end: usize,
        order: &mut [usize; MAX_KDTREE_POINTS],
        depth: usize,
    ) -> Result<usize, Error> {
        if start >= end {
            return Err(Error::InvalidDimension);
        }
        let axis = (depth % 2) as u8;
        let mid = start + (end - start) / 2;
        order[start..end].sort_by(|&a, &b| {
            let pa = self.points[a][axis as usize];
            let pb = self.points[b][axis as usize];
            pa.partial_cmp(&pb).unwrap_or(core::cmp::Ordering::Equal)
        });
        let pivot = order[mid];
        let node_idx = self.n_nodes;
        if node_idx >= self.nodes.len() {
            return Err(Error::CapacityExceeded);
        }
        self.n_nodes += 1;
        let mut left = None;
        let mut right = None;
        if mid > start {
            left = Some(self.build_rec(start, mid, order, depth + 1)?);
        }
        if mid + 1 < end {
            right = Some(self.build_rec(mid + 1, end, order, depth + 1)?);
        }
        self.nodes[node_idx] = Some(Node {
            point_index: pivot,
            left,
            right,
            axis,
        });
        Ok(node_idx)
    }

    fn dist_sq(a: [f64; 2], b: [f64; 2]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        dx * dx + dy * dy
    }

    /// Indices of up to ``k`` neighbours within ``max_dist`` (Stone Soup ``KDTree.query``).
    pub fn query_knn(
        &self,
        center: [f64; 2],
        k: usize,
        max_dist: f64,
        out: &mut [usize],
    ) -> Result<usize, Error> {
        if self.n_points == 0 {
            return Ok(0);
        }
        let max_d2 = max_dist * max_dist;
        let mut best = [(usize::MAX, f64::INFINITY); MAX_KDTREE_POINTS];
        let mut n_best = 0usize;
        if let Some(ref root) = self.nodes[0] {
            self.query_rec(root, center, max_d2, k, &mut best, &mut n_best)?;
        }
        best[..n_best].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        let take = k.min(n_best);
        let mut n_out = 0usize;
        for i in 0..take {
            if best[i].1 <= max_d2 && best[i].0 != usize::MAX {
                if n_out >= out.len() {
                    return Err(Error::CapacityExceeded);
                }
                out[n_out] = best[i].0;
                n_out += 1;
            }
        }
        Ok(n_out)
    }

    fn query_rec(
        &self,
        node: &Node,
        target: [f64; 2],
        max_d2: f64,
        k: usize,
        best: &mut [(usize, f64); MAX_KDTREE_POINTS],
        n_best: &mut usize,
    ) -> Result<(), Error> {
        let idx = node.point_index;
        let p = self.points[idx];
        let d2 = Self::dist_sq(p, target);
        if d2 <= max_d2 {
            Self::insert_best(idx, d2, k, best, n_best);
        }
        let axis = node.axis as usize;
        let diff = target[axis] - p[axis];
        let (first, second) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };
        if let Some(i) = first {
            if let Some(n) = self.nodes.get(i).and_then(|o| o.as_ref()) {
                self.query_rec(n, target, max_d2, k, best, n_best)?;
            }
        }
        if diff * diff <= max_d2 {
            if let Some(i) = second {
                if let Some(n) = self.nodes.get(i).and_then(|o| o.as_ref()) {
                    self.query_rec(n, target, max_d2, k, best, n_best)?;
                }
            }
        }
        Ok(())
    }

    fn insert_best(
        idx: usize,
        d2: f64,
        k: usize,
        best: &mut [(usize, f64); MAX_KDTREE_POINTS],
        n_best: &mut usize,
    ) {
        if *n_best < k {
            best[*n_best] = (idx, d2);
            *n_best += 1;
            return;
        }
        let mut worst = 0usize;
        for i in 1..*n_best {
            if best[i].1 > best[worst].1 {
                worst = i;
            }
        }
        if d2 < best[worst].1 {
            best[worst] = (idx, d2);
        }
    }
}
