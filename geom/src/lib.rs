#![no_std]

extern crate alloc;
extern crate core;
#[cfg(feature = "std")]
extern crate std;

use re::geom::Edge;
use re::math::Point2;

pub mod io;
pub mod solids;

use std::dbg;

/*pub fn triangulate(poly: &[Point2]) -> Vec<Tri<Point2>> {
    let mut res = Vec::new();

    for pt in poly {}

    res
}
*/

/// Returns the intersection point of two 2D line segments, if any.
pub fn intersect(a: Edge<Point2>, b: Edge<Point2>) -> Option<Point2> {
    // Solve the following linear system for t and u:
    //
    // t (a.1 - a.0) - u (b.1 - b.0) = b.0 - a.0

    let d = (a.1 - a.0).perp_dot(b.1 - b.0);
    let t = (b.0 - a.0).perp_dot(b.1 - b.0);
    let u = (b.0 - a.0).perp_dot(a.1 - a.0);

    // Segments are parallel if the denominator is zero
    if d == 0.0 {
        // TODO the collinear case where s = t = 0
        return None;
    }

    // Segments intersect if 0 ≤ t / d ≤ 1 and 0 ≤ u / d ≤ 1
    let (min, max) = if d < 0.0 { (d, 0.0) } else { (0.0, d) };
    if min <= t && t <= max && min <= u && u <= max {
        return Some(a.0 + t / d * (a.1 - a.0));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use re::math::pt2;

    #[test]
    fn intersect_start_points() {
        let a = Edge(pt2::<_, ()>(0.0, 0.0), pt2(1.0, 0.0));
        let b = Edge(pt2::<_, ()>(0.0, 0.0), pt2(0.0, 1.0));

        assert_eq!(intersect(a, b), Some(pt2(0.0, 0.0)));
    }

    #[test]
    fn intersect_end_points() {
        let a = Edge(pt2::<_, ()>(0.0, 0.0), pt2(1.0, 0.0));
        let b = Edge(pt2::<_, ()>(1.0, 1.0), pt2(1.0, 0.0));

        assert_eq!(intersect(a, b), Some(pt2(1.0, 0.0)));
    }

    #[test]
    fn intersect_cross() {
        let a = Edge(pt2::<_, ()>(0.0, 0.0), pt2(1.0, 1.0));
        let b = Edge(pt2::<_, ()>(1.0, 0.0), pt2(0.0, 1.0));

        assert_eq!(intersect(a, b), Some(pt2(0.5, 0.5)));
    }

    #[test]
    fn intersect_no_intersection() {
        let a = Edge(pt2::<_, ()>(0.0, 0.0), pt2(1.0, 0.0));
        let b = Edge(pt2::<_, ()>(0.0, 1.0), pt2(2.0, 0.0));

        assert_eq!(intersect(a, b), None);
    }

    #[test]
    fn intersect_degenerate() {
        let p: Point2 = pt2(2.0, 3.0);
        let q: Point2 = pt2(-1.0, 4.0);

        // TODO Counts as overlapping
        //assert_eq!(intersect(Edge(p, p), Edge(p, p)), Some(p));

        assert_eq!(intersect(Edge(p, p), Edge(q, q)), None);
    }
}
