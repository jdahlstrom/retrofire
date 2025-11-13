use core::ops::RangeBounds;

use retrofire_core::{
    geom::{Plane3, Ray},
    math::{ApproxEq, Point3, Vec3},
    render::scene::BBox,
};

/// Trait for finding intersection points of geometric objects.
pub trait Intersect<T> {
    /// The result of an intersection test.
    type Result;

    /// Finds the point(s) where `self` and another object intersect, if any.
    ///
    /// It is implementation defined whether this method returns all the
    /// intersection points or, for instance, only the closest one.
    fn intersect(&self, other: &T) -> Self::Result;
}

impl<B> Intersect<Plane3<B>> for Ray<Point3<B>> {
    type Result = Option<(f32, Point3<B>)>;

    /// Returns the unique intersection point of `self` and a plane,
    /// or `None` if they do not intersect.
    ///
    /// If an intersection point exists, returns `Some((t, point))`, where
    /// `point` is the intersection point and  `t` is the ray parameter value
    /// such that `self.orig + t * self.dir == point`.
    fn intersect(&self, p: &Plane3<B>) -> Self::Result {
        let Self(orig, dir) = self;

        let num = p.signed_dist(*orig);
        if num.approx_eq(&0.0) {
            // Origin point coincident with the plane
            return Some((0.0, *orig));
        }

        let denom = dir.dot(&p.normal().to());
        if denom.approx_eq(&0.0) {
            // Ray parallel with but not coincident with the plane
            // (or dir is a zero vector) -> no intersection
            return None;
        }

        let t = -num / denom;
        if t < 0.0 {
            // Ray points away from the plane, intersection "behind" it
            return None;
        }

        Some((t, *orig + t * *dir))
    }
}

/* TODO needs cart->bary
impl<A, B> Intersect<Tri<Vertex3<A, B>>> for Ray<Point3<B>> {
    type Result = Option<(f32, Point3<B>)>;

    /// Returns the unique intersection point of `self` and a plane,
    /// or `None` if they do not intersect.
    ///
    /// If an intersection point exists, returns `Some((t, point))`, where
    /// `point` is the intersection point and  `t` is the ray parameter value
    /// such that `self.orig + t * self.dir == point`.
    fn intersect(&self, tri: &Tri<Vertex3<A, B>>) -> Self::Result {
        let (u, pt) = self.intersect(&tri.plane())?;

        let vs = tri
            .0
            .each_ref()
            // TODO projection plane selection
            .map(|v| pt2::<f32, ()>(v.pos.x(), v.pos.y()))
            .map(|p| vertex(p, ()));

        let p = pt2::<f32, ()>(pt.x(), pt.y());

        let [s, t, r] = Tri(vs).cart_to_bary(&p).0;

        (s >= 0.0 && t >= 0.0 && r >= 0.0).then_some((u, pt))
    }
}*/

impl<B: Default> Intersect<BBox<B>> for Ray<Point3<B>> {
    type Result = Option<(f32, Point3<B>)>; // Only closest for now

    /// Returns the nearest intersection point of `self` and a box,
    /// or `None` if they do not intersect.
    ///
    /// If an intersection point exists, returns `Some((t, point))`, where
    /// `point` is the intersection point and  `t` is the ray parameter value
    /// such that `self.orig + t * self.dir == point`.
    fn intersect(&self, bbox @ &BBox(l, u): &BBox<B>) -> Self::Result {
        #[rustfmt::skip]
        let planes: [Plane3<B>; 6] = [
            (l, -Vec3::X), (u, Vec3::X),
            (l, -Vec3::Y), (u, Vec3::Y),
            (l, -Vec3::Z), (u, Vec3::Z),
        ]
        .map(|(p, n)| Plane3::from_point_and_normal(p, n));

        // TODO only need to consider planes "on the same side" as self.orig,
        //      in which case the first intersection found is the closest one
        let mut res = None;
        for plane in &planes {
            if let pt @ Some((t, p)) = self.intersect(plane)
                && bbox.contains(&p)
                && res.is_none_or(|(u, _)| t < u)
            {
                res = pt;
            }
        }
        res
    }
}

/* TODO if needed
pub trait InRange<R: RangeBounds<Self>>: PartialOrd {
    fn in_range(&self, range: R) -> bool {
        range.contains(self)
    }
}
impl<T: PartialOrd, R: RangeBounds<T>> InRange<R> for T {}
*/

/* TODO
impl<B: Debug + Default> Intersect<Edge<Point2<B>>> for Edge<Point2<B>> {
    type Result = Option<Point2<B>>;

    fn intersect(&self, other: &Edge<Point2<B>>) -> Self::Result {
        // p = o1 + t1*d1
        //
        // px = 1 o1x + t1 d1x
        // py = 1 o1y + t1 d1y
        //
        // o1 + t1*d1 = o2 + t2*d2
        //
        // 1 o1x + t1 d1x = 1 o2x + t2 d2x
        // 1 o1y + t1 d1y = 1 o2x + t2 d2x
        //
        // t1*d1 - t2*d2 = o2 - o1
        //
        // t1 d1x - t2 d2x = o2x - o1x
        // t1 d1y - t2 d2y = o2y - o1y
        //
        // (d1x  -d2x) (t1) = o2 - o1
        // (d1y  -d2y) (t2)
        //
        // M = (d1x  -d2x)
        //     (d1y  -d2y)
        //
        // M t = o2 - o1
        // t = M^-1 (o2 - o1)

        let Edge(s, t) = *self;
        let Edge(o, p) = *other;

        // Check if the endpoints of other are on the same side of the line of self
        let st = t - s;
        let so = o - s;
        let sp = p - s;
        if st.perp_dot(so) * st.perp_dot(sp) > 0.0 {
            // eprintln!("{o:?} and {p:?} on the same side of {s:?}-{t:?}");
            // s
            //  \   o-------p
            //   \
            //    t
            return None;
        }
        // Check if the endpoints of self are on the same side of the line of other
        let op = p - o;
        let os = s - o;
        let ot = t - o;
        if op.perp_dot(os) * op.perp_dot(ot) > 0.0 {
            // eprintln!("{s:?} and {t:?} on the same side of {o:?}-{p:?}");
            // o
            //  \   s-------t
            //   \
            //    p
            return None;
        }
        // After the checks above the only cases left are:
        // * edges cross at exactly one point
        // * edges are collinear, with either shared point(s) or not.

        let m: Mat2<B> = mat![
            st.x(), -op.x();
            st.y(), -op.y();
        ];

        let Some(m_inv) = m.checked_inverse() else {
            if s.x().in_range(o.x()..=p.x()) && s.y().in_range(o.y()..=p.y()) {
                // s between o and p
                // o--s--t--p or o--s--p--t
                return Some(s);
            }
            if o.x().in_range(s.x()..=t.x()) && o.y().in_range(s.y()..=t.y()) {
                // o between s and t
                // s--o--p--t or s--o--t--p
                return Some(o);
            }
            // st and op collinear but disjoint
            // s----t  o----p
            return None;
        };

        let [t1, t2] = m_inv.apply(&so).0;

        assert!(
            t1.in_range(0.0..=1.0) && t2.in_range(0.0..=1.0),
            "0 < t1, t2 < 1 due to earlier checks"
        );

        Some(self.0 + t1 * st)
    }
}*/

#[cfg(test)]
mod tests {
    use retrofire_core::assert_approx_eq;
    use retrofire_core::math::{Linear, pt3, vec3};

    use super::*;

    fn plane() -> Plane3 {
        Plane3::from_point_and_normal(pt3(1.0, 2.0, 3.0), vec3(0.0, 1.0, 0.0))
    }

    #[test]
    fn ray_towards_plane_has_intersection() {
        let p = plane();

        // Outside
        let r = Ray(pt3(0.0, 3.0, 0.0), vec3(1.0, -1.0, 1.0));
        assert_approx_eq!(r.intersect(&p), Some((1.0, pt3(1.0, 2.0, 1.0))));

        // Inside
        let r = Ray(pt3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0));
        assert_eq!(r.intersect(&p), Some((1.0, pt3(1.0, 2.0, 1.0))));
    }
    #[test]
    fn ray_origin_coincident_with_plane_has_intersection() {
        let p = plane();
        let r = Ray(pt3(0.0, 2.0, 0.0), vec3(1.0, -1.0, 1.0));

        assert_eq!(r.intersect(&p), Some((0.0, pt3(0.0, 2.0, 0.0))));
    }
    #[test]
    fn ray_coincident_with_plane_has_intersection() {
        let p = plane();
        let r = Ray(pt3(0.0, 2.0, 0.0), vec3(1.0, 0.0, 1.0));

        assert_eq!(r.intersect(&p), Some((0.0, pt3(0.0, 2.0, 0.0))));
    }
    #[test]
    fn ray_parallel_with_plane_no_intersection() {
        let p = plane();
        let r = Ray(pt3(0.0, 3.0, 0.0), vec3(1.0, 0.0, 1.0));

        assert_eq!(r.intersect(&p), None);
    }
    #[test]
    fn ray_points_away_from_plane_no_intersection() {
        let p = plane();

        // Outside
        let r = Ray(pt3(0.0, 3.0, 0.0), vec3(1.0, 1.0, 1.0));
        assert_eq!(r.intersect(&p), None);

        // Inside
        let r = Ray(pt3(0.0, 1.0, 0.0), vec3(1.0, -1.0, 1.0));
        assert_eq!(r.intersect(&p), None);
    }
    #[test]
    fn degenerate_ray_only_intersects_if_coincident() {
        let p = plane();

        let r = Ray(pt3(0.0, 3.0, 0.0), Vec3::zero());
        assert_eq!(r.intersect(&p), None);

        let r = Ray(pt3(0.0, 1.0, 0.0), Vec3::zero());
        assert_eq!(r.intersect(&p), None);

        let r = Ray(pt3(0.0, 2.0, 0.0), Vec3::zero());
        assert_eq!(r.intersect(&p), Some((0.0, pt3(0.0, 2.0, 0.0))));
    }

    /*#[test]
    fn edge_edge_intersection() {
        //   D   B
        //    \ /
        //    / \
        //   A   C
        let ab = Edge(pt2::<f32, ()>(0.0, 0.0), pt2(1.0, 1.0));
        let cd = Edge(pt2(1.0, 0.0), pt2(0.0, 1.0));

        assert_eq!(ab.intersect(&cd), Some(pt2(0.5, 0.5)));
        assert_eq!(cd.intersect(&ab), Some(pt2(0.5, 0.5)));
    }
    #[test]
    fn edge_edge_no_intersection() {
        //       C
        //  A---B \
        //         D
        let ab = Edge(pt2::<_, ()>(0.0, 0.0), pt2(1.0, 0.0));
        let cd = Edge(pt2(1.0, 1.0), pt2(2.0, -1.0));

        assert_eq!(ab.intersect(&cd), None);
        assert_eq!(cd.intersect(&ab), None);
    }
    #[test]
    fn edge_edge_parallel_disjoint() {
        //       C----D
        //    A----B
        let ab = Edge(pt2::<_, ()>(0.0, 0.0), pt2(2.0, 0.0));
        let cd = Edge(pt2(1.0, 1.0), pt2(3.0, 1.0));

        assert_eq!(ab.intersect(&cd), None);
        assert_eq!(cd.intersect(&ab), None);
    }
    #[test]
    fn edge_edge_collinear_disjoint() {
        //    D----C   A----B
        let ab = Edge(pt2::<_, ()>(0.0, 0.0), pt2(1.0, 0.0));
        let cd = Edge(pt2(-1.0, 0.0), pt2(-2.0, 0.0));

        assert_eq!(ab.intersect(&cd), None);
        assert_eq!(cd.intersect(&ab), None);
    }
    #[test]
    fn edge_edge_parallel_shared_endpoint() {
        //    A----BC----D
        let ab = Edge(pt2::<_, ()>(0.0, 0.0), pt2(1.0, 0.0));
        let cd = Edge(pt2(1.0, 0.0), pt2(2.0, 0.0));

        assert_eq!(ab.intersect(&cd), Some(pt2(1.0, 0.0)));
        assert_eq!(cd.intersect(&ab), Some(pt2(1.0, 0.0)));
    }
    #[test]
    fn edge_edge_nonparallel_shared_endpoint() {
        //            D
        //           /
        //    A----BC
        let e1 = Edge(pt2::<_, ()>(0.0, 0.0), pt2(1.0, 0.0));
        let e2 = Edge(pt2(1.0, 0.0), pt2(2.0, 1.0));

        assert_eq!(e1.intersect(&e2), Some(pt2(1.0, 0.0)));
        assert_eq!(e2.intersect(&e1), Some(pt2(1.0, 0.0)));
    }
    #[test]
    fn edge_edge_collinear_coincident() {
        //    A---C---B---D
        let ab = Edge(pt2::<_, ()>(0.0, 0.0), pt2(1.0, 0.0));
        let cd = Edge(pt2(0.5, 0.0), pt2(2.0, 0.0));

        assert_eq!(ab.intersect(&cd), Some(pt2(0.5, 0.0)));
        assert_eq!(cd.intersect(&ab), Some(pt2(0.5, 0.0)));
    }*/
}
