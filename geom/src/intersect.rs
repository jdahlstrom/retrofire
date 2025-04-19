//! Intersection tests of geometric primitives.

use re::geom::Edge;
use re::math::Point2;

/// Returns the intersection point of two 2D line segments, if any.
pub fn intersect(a: Edge<Point2>, b: Edge<Point2>) -> Option<Point2> {
    // Given the parametric representation of a and b:
    //
    //   p = a0 + t (a1 - a0)
    //   q = b0 + u (b1 - b0)
    //
    // Equate p and q, giving
    //
    //   a0 + t (a1 - a0) =  b0 + u (b1 - b0)
    //
    // Reorder to get the linear system:
    //
    //   t (a1 - a0) - u (b1 - b0) = b0 - a0
    //
    // Solve for t and u using Cramer's rule, giving
    //
    //        (b0 - a0)⟂ · (b1 - b0)         (b0 - a0)⟂ · (a1 - a0)
    //   t = ------------------------,  u = ------------------------ .
    //        (a1 - a0)⟂ · (b1 - b0)         (a1 - a0)⟂ · (b1 - b0)

    let d = (a.1 - a.0).perp_dot(b.1 - b.0);
    let t = (b.0 - a.0).perp_dot(b.1 - b.0);
    let u = (b.0 - a.0).perp_dot(a.1 - a.0);

    // Segments are parallel if the denominator is zero
    if d == 0.0 {
        // TODO the collinear case where t = u = 0
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
