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
}
