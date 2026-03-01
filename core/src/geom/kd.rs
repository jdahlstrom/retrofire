use alloc::vec::Vec;
use core::ops::Range;

use crate::math::{Point2, pt2};

pub struct KdTree<T> {
    pub nodes: Vec<Node>,
    pub points: Vec<T>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Node {
    pub range: (usize, usize),
    pub bounds: (Point2, Point2),
    pub depth: u32,
    // Only if non-leaf
    pub split: Option<Split>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Split {
    pub children: (usize, usize),
    pub split_axis: usize,
    pub split_pos: f32,
}

impl<B> KdTree<Point2<B>> {
    /// Builds a k-d tree from points.
    pub fn new(mut pts: Vec<Point2<B>>) -> Self {
        let bounds = Self::bounds(&pts, 0..pts.len());
        Self::with_bounds(pts, bounds)
    }

    /// Builds a k-d tree from points, with given bounds.
    ///
    /// This is useful if you already know the bounds of the input points.
    /// The bounds must be a true upper bound, but no tight upper bound. If the
    /// bounds are not an upper bound, i.e. if at least one point is outside
    /// the bounds, the tree may not work correctly.
    pub fn with_bounds(
        points: Vec<Point2<B>>,
        (low, high): (Point2<B>, Point2<B>),
    ) -> Self {
        let n_pts = points.len();
        let root = Node {
            range: (0, n_pts),
            bounds: (low.to(), high.to()),
            split: None,
            depth: 0,
        };
        let mut tree = Self {
            points,
            // Heuristic
            nodes: Vec::with_capacity(n_pts / 2),
        };
        tree.build_tree(root, !0);
        tree
    }

    /// Rebuilds the tree after points have been added.
    pub fn rebuild(&mut self) {
        let n_pts = self.points.len();
        //let (low, high) = Self::bounds(&self.points, 0..n_pts);
        let root = Node {
            range: (0, n_pts),
            bounds: self.nodes[0].bounds,
            split: None,
            depth: 0,
        };
        self.nodes.clear();
        self.build_tree(root, !0);
    }

    /// Returns the point in the tree that is nearest to a point.
    ///
    /// If the tree is empty, returns `None`.
    pub fn nearest(&self, point: Point2<B>) -> Option<Point2<B>> {
        let mut d2_to_first = point.distance_sqr(self.points.first()?);
        self.nearest_in(&*self.root()?, point, &mut d2_to_first)
    }

    /// Returns the nearest point in the tree that is within a given radius
    /// from a point, if one exists.
    ///
    /// Returns `None` if there is no such point.
    pub fn nearest_within(
        &self,
        point: Point2<B>,
        radius: f32,
    ) -> Option<Point2<B>> {
        let mut r2 = radius * radius;
        self.nearest_in(&*self.root()?, point, &mut r2)
    }

    pub fn node(&self, pt: Point2<B>) -> Option<&Node> {
        let mut curr_node = self.root()?;

        if !Self::in_bounds(pt, curr_node) {
            return None;
        }

        loop {
            // Here we should always be in bounds of curr_node
            assert!(Self::in_bounds(pt, curr_node));

            let Some(Split { children: (c1, c2), .. }) = curr_node.split else {
                return Some(curr_node);
            };

            let c1 = &self.nodes[c1];
            if Self::in_bounds(pt, c1) {
                curr_node = c1;
            } else {
                curr_node = &self.nodes[c2];
                assert!(Self::in_bounds(pt, curr_node));
            }
        }
    }

    fn in_bounds(pt: Point2<B>, node: &Node) -> bool {
        let (lo, hi) = node.bounds;
        lo.x() <= pt.x()
            && pt.x() <= hi.x()
            && lo.y() <= pt.y()
            && pt.y() <= hi.y()
    }

    pub fn root(&self) -> Option<&Node> {
        self.nodes.first() // TODO always a root node even if empty
    }

    fn dist_to_rect(pt: Point2, (lo, hi): (Point2, Point2)) -> f32 {
        let p_to_l = lo - pt;
        let p_to_u = lo - hi;

        let x_in_bounds = p_to_l.x() >= 0.0 && p_to_u.x() <= 0.0;
        let y_in_bounds = p_to_l.y() >= 0.0 && p_to_u.y() <= 0.0;
        if x_in_bounds && y_in_bounds {
            // inside
            0.0
        } else if y_in_bounds {
            // left or right edge is closest
            p_to_l.x().abs().min(p_to_u.x().abs()).powi(2)
        } else if x_in_bounds {
            // top or bottom edge is closest
            p_to_l.y().abs().min(p_to_u.y().abs()).powi(2)
        } else {
            // one of the vertices is closest
            pt.distance_sqr(&lo)
                .min(pt.distance_sqr(&hi))
                .min(pt.distance_sqr(&pt2(lo.x(), hi.y())))
                .min(pt.distance_sqr(&pt2(hi.x(), lo.y())))
        }
    }

    fn nearest_in(
        &self,
        node: &Node,
        point: Point2<B>,
        curr_min_d2: &mut f32,
    ) -> Option<Point2<B>> {
        let d2 = Self::dist_to_rect(point.to(), node.bounds);

        if d2 > *curr_min_d2 {
            // Cannot be found here!
            return None;
        }
        let Some(inner) = node.split else {
            // Leaf, loop over points
            let mut nearest = None;
            for pt in &self.points[node.range.0..node.range.1] {
                let d2 = point.distance_sqr(pt);
                if d2 < *curr_min_d2 {
                    nearest = Some(*pt);
                    *curr_min_d2 = d2;
                }
            }
            return nearest;
        };
        let (lo_i, hi_i) = inner.children;
        let lo = self.nearest_in(&self.nodes[lo_i], point, curr_min_d2);
        let hi = self.nearest_in(&self.nodes[hi_i], point, curr_min_d2);

        if let (Some(lo), Some(hi)) = (lo, hi) {
            if point.distance_sqr(&lo) < point.distance_sqr(&hi) {
                Some(lo)
            } else {
                Some(hi)
            }
        } else {
            lo.or(hi)
        }
    }

    fn bounds(pts: &[Point2<B>], rg: Range<usize>) -> (Point2<B>, Point2<B>) {
        let [mut min_x, mut min_y] = [f32::INFINITY; 2];
        let [mut max_x, mut max_y] = [f32::NEG_INFINITY; 2];
        for p in &pts[rg] {
            min_x = min_x.min(p.x());
            max_x = max_x.max(p.x());
            min_y = min_y.min(p.y());
            max_y = max_y.max(p.y());
        }

        (pt2(min_x, min_y), pt2(max_x, max_y))
    }

    /// Builds a (sub)tree with the given node as the root.
    fn build_tree(&mut self, root: Node, last_split_axis: usize) -> usize {
        let (low, upp) = root.bounds;
        let (start, end) = root.range;

        let node_i = self.nodes.len();
        self.nodes.push(root);
        if end - start <= 32 {
            // Leaf node, we're done
            return node_i;
        }

        let dims = upp - low;
        let split_axis = if dims.x() >= dims.y() {
            0 // More wide than tall -> side by side, with a vertical divider
        } else {
            1 // More tall than wide -> top and bottom, with a horizontal divider
        };
        let pts = &mut self.points[start..end];

        // TODO if root's parent's split_axis equals this, no need to sort
        if split_axis != last_split_axis {
            pts.sort_by(|a, b| {
                a[split_axis].partial_cmp(&b[split_axis]).unwrap()
            });
        }

        let median_idx;
        let split_pos;

        if false && pts.len() % 2 == 0 {
            // FIXME
            // two medians
            median_idx = pts.len() / 2;
            // pts has at least four elements, median_idx >= 1
            split_pos = pts[median_idx - 1][split_axis]
                .midpoint(pts[median_idx][split_axis]);
        } else {
            median_idx = pts.len() / 2;
            split_pos = pts[median_idx][split_axis];
        }

        let [low_bds, hi_bds] = Self::split(&root, split_pos, split_axis);

        let lo_node = Node {
            range: (start, start + median_idx),
            bounds: low_bds,
            split: None,
            depth: root.depth + 1,
        };
        let lo_i = self.build_tree(lo_node, split_axis);

        let hi_node = Node {
            range: (start + median_idx, end),
            bounds: hi_bds,
            split: None,
            depth: root.depth + 1,
        };
        let hi_i = self.build_tree(hi_node, split_axis);

        self.nodes[node_i].split = Some(Split {
            children: (lo_i, hi_i),
            split_axis,
            split_pos,
        });
        node_i
    }

    /// Splits node bounds into two bounds, either at x=split_pos or y=split_pos,
    /// depending on split_axis.
    fn split(
        node: &Node,
        split_pos: f32,
        split_axis: usize,
    ) -> [(Point2, Point2); 2] {
        let (lo, hi) = node.bounds;

        let (lo_bds, hi_bds) = if split_axis == 0 {
            ((lo, pt2(split_pos, hi.y())), (pt2(split_pos, lo.y()), hi))
        } else {
            ((lo, pt2(hi.x(), split_pos)), (pt2(lo.x(), split_pos), hi))
        };
        [lo_bds, hi_bds]
    }
}

/* 3D stuff...
let mut bbox = BBox::default();
bbox.extend(points);

let [w, h, d] = bbox.extents().0;
let max_extent = if w > h && w > d {
    w
} else if h h < d {
    h
} else {
    d
};*/

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::math::pt2;

    use super::*;

    #[test]
    fn empty_tree() {
        let tree = KdTree::<Point2>::new(Vec::new());

        assert!(tree.nodes.is_empty());
        assert!(tree.points.is_empty());
    }

    #[test]
    fn single_point() {
        let tree = KdTree::<Point2>::new(vec![pt2(0.0, 0.0)]);

        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.points.len(), 1);

        assert!(matches!(
            tree.nodes[0],
            Node {
                range: (0, 1),
                depth: 0,
                split: None,
                ..
            }
        ));
        assert_eq!(tree.points[0], pt2(0.0, 0.0));
    }

    #[test]
    fn two_points() {
        let tree = KdTree::<Point2>::new(vec![pt2(0.0, 0.0), pt2(1.0, 0.0)]);

        assert_eq!(tree.nodes.len(), 1);
        assert_eq!(tree.points.len(), 2);

        assert!(matches!(
            tree.nodes[0],
            Node {
                range: (0, 2),
                depth: 0,
                split: None,
                ..
            }
        ));
        assert_eq!(tree.points[0], pt2(0.0, 0.0));
        assert_eq!(tree.points[1], pt2(1.0, 0.0));
    }

    #[test]
    fn many_horizontal() {
        let tree = KdTree::<Point2>::new(vec![
            pt2(0.0, 0.0),
            pt2(1.0, 0.0),
            pt2(2.0, 0.0),
            pt2(3.0, 0.0),
            pt2(4.0, 0.0),
            pt2(5.0, 0.0),
            pt2(6.0, 0.0),
            pt2(7.0, 0.0),
            pt2(8.0, 0.0),
            pt2(9.0, 0.0),
            pt2(10.0, 0.0),
            //
            // pt2(1.0, 1.0),
            // pt2(2.0, -2.0),
            // pt2(3.0, 1.0),
        ]);

        assert_eq!(tree.nodes, []);

        //assert_eq!(tree.points, []);
    }
}
