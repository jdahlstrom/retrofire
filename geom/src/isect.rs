use retrofire_core::{
    geom::{Plane3, Ray, Ray3, Sphere},
    math::{ApproxEq, Point3},
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

type RayIntersect3<B> = Option<(f32, Point3<B>)>;

impl<B> Intersect<Plane3<B>> for Ray3<B> {
    type Result = RayIntersect3<B>;

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
        if t.approx_le(&0.0) {
            // Ray points away from the plane, intersection "behind" it
            return None;
        }

        Some((t, *orig + t * *dir))
    }
}

impl<B: Default> Intersect<BBox<B>> for Ray3<B> {
    type Result = RayIntersect3<B>; // Only closest for now

    /// Returns the nearest intersection point of `self` and a box,
    /// or `None` if they do not intersect.
    ///
    /// If an intersection point exists, returns `Some((t, point))`, where
    /// `point` is the intersection point and  `t` is the ray parameter value
    /// such that `self.orig + t * self.dir == point`.
    fn intersect(&self, bbox @ &BBox(l, u): &BBox<B>) -> Self::Result {
        let Ray(o, d) = self;

        // Fast path for clearly nonintersecting cases
        if (o.x() < l.x() && d.x() < 0.0)
            || (o.x() > u.x() && d.x() > 0.0)
            || (o.y() < l.y() && d.y() < 0.0)
            || (o.y() > u.y() && d.y() > 0.0)
            || (o.z() < l.z() && d.z() < 0.0)
            || (o.z() > u.z() && d.z() > 0.0)
            || bbox.is_empty()
        {
            return None;
        }

        let planes: [Plane3<B>; 6] = [
            Plane3::new(1.0, 0.0, 0.0, l.x()),
            Plane3::new(1.0, 0.0, 0.0, u.x()),
            Plane3::new(0.0, 1.0, 0.0, l.y()),
            Plane3::new(0.0, 1.0, 0.0, u.y()),
            Plane3::new(0.0, 0.0, 1.0, l.z()),
            Plane3::new(0.0, 0.0, 1.0, u.z()),
        ];

        // TODO only need to consider planes "on the same side" as self.orig,
        //      in which case the first intersection found is the closest one
        let mut res = None;
        for plane in &planes {
            if let pt @ Some((t, p)) = self.intersect(plane)
                // FIXME may be false due to rounding errors
                && bbox.contains(&p)
                && res.is_none_or(|(u, _)| t < u)
            {
                res = pt;
            }
        }
        res
    }
}

#[cfg(feature = "std")]
impl<B> Intersect<Sphere<B>> for Ray3<B> {
    type Result = RayIntersect3<B>; // Only closest for now

    /// Returns the intersection point of `self` and a sphere closest to the
    /// origin of `self`, or `None` if they do not intersect.
    ///
    /// # Examples
    /// ```
    /// ```
    fn intersect(&self, &Sphere(center, r): &Sphere<B>) -> Self::Result {
        let &Ray(orig, dir) = self;

        //             > r, no intersection
        // If |C - C'| = r, one    -"-
        //             < r, two    -"-
        //           _______
        //          /   |r  \
        //         |    C    |
        //          \___|___/
        //             _|
        //   O--------+-C'----> d
        //
        //
        // Nb. this is the same as the discriminant check later
        //     Is this even a fast path?
        let center_dir = center - orig;
        let c_prime = center_dir.vector_project(&dir);
        if (c_prime - center_dir).len_sqr() > r * r {
            return None;
        }

        // Find point p = (x, y, z)
        //
        // Sphere equation:
        //   (x-cx)^2 + (y-cy)^2 + (z-cz)^2 = r^2
        // or in vector form
        //   (p-c) · (p-c) = r^2
        //
        // Ray equation:
        //   p = o + t·d
        //
        // Intersection:
        //
        //   Substitute ray equation to sphere equation:
        //   (o + t·d - c) · (o + t·d - c) = r^2
        //
        //   Multiply out
        //   o (o + td - c) + t·d (o + td - c) - c (o + td - c) = r^2
        //
        //   Distribute
        //   o·o + o·td - o·c + o·td + td·td - td·c - c·o - c·td + c·c = r^2
        //
        //   Reorder
        //   td·td + o·td + o·td - td·c + o·o - o·c - c·o + c·c - r^2 = 0
        //
        //   Factor out t's
        //   (d·d) t^2 + (o·d + o·d - c·d) t + o·o - 2o·c + c·c - r^2 = 0
        //
        // Solve quadratic equation:
        //   (d·d) t^2 + 2(o - c)·d t + (o - c)·(o - c) - r^2 = 0

        let a = dir.len_sqr(); // >= 0
        let b = -2.0 * center_dir.dot(&dir); // center_dir = c - o
        let c = center_dir.len_sqr() - r * r;

        if a.approx_eq(&0.0) {
            // Only possible if dir == 0
            return None;
        }

        let discr = b * b - 4.0 * a * c;
        // Handled at the top
        debug_assert!(discr >= 0.0, "this should not happen");

        use retrofire_core::math::float::f32;
        let sqrt = f32::sqrt(discr);
        // sqrt >= 0.0, thus t0 <= t1 always
        let (t0, t1) = (-b - sqrt, -b + sqrt);
        let t = if t1 < 0.0 {
            // sphere is behind ray
            //dbg!(t0, t1, a, b, c, discr);
            return None;
        } else if t0 < 0.0 {
            // ray origin is inside sphere
            t1
        } else {
            t0.min(t1)
        };
        let t = t / (2.0 * a);
        Some((t, orig + t * dir))
    }
}

#[cfg(test)]
mod tests {
    use retrofire_core::math::{Linear, Vec3, pt3, vec3};

    use super::*;

    mod ray_plane {
        use super::*;

        const PLANE: Plane3<()> = Plane3::new(0.0, 1.0, 0.0, 2.0);

        #[test]
        fn ray_towards_plane_has_intersection() {
            // Outside
            let r = Ray(pt3(0.0, 3.0, 0.0), vec3(1.0, -1.0, 1.0));
            assert_eq!(r.intersect(&PLANE), Some((1.0, pt3(1.0, 2.0, 1.0))));

            // Inside
            let r = Ray(pt3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0));
            assert_eq!(r.intersect(&PLANE), Some((1.0, pt3(1.0, 2.0, 1.0))));
        }
        #[test]
        fn ray_origin_on_plane_has_intersection() {
            let r = Ray(pt3(0.0, 2.0, 0.0), vec3(1.0, -1.0, 1.0));
            assert_eq!(r.intersect(&PLANE), Some((0.0, pt3(0.0, 2.0, 0.0))));
        }
        #[test]
        fn ray_coincident_with_plane_has_intersection() {
            let r = Ray(pt3(0.0, 2.0, 0.0), vec3(1.0, 0.0, 1.0));
            assert_eq!(r.intersect(&PLANE), Some((0.0, pt3(0.0, 2.0, 0.0))));
        }
        #[test]
        fn ray_parallel_with_plane_no_intersection() {
            let r = Ray(pt3(0.0, 3.0, 0.0), vec3(1.0, 0.0, 1.0));
            assert_eq!(r.intersect(&PLANE), None);
        }
        #[test]
        fn ray_points_away_from_plane_no_intersection() {
            // Outside
            let r = Ray(pt3(0.0, 3.0, 0.0), vec3(1.0, 1.0, 1.0));
            assert_eq!(r.intersect(&PLANE), None);

            // Inside
            let r = Ray(pt3(0.0, 1.0, 0.0), vec3(1.0, -1.0, 1.0));
            assert_eq!(r.intersect(&PLANE), None);
        }
        #[test]
        fn degenerate_ray_only_intersects_if_coincident() {
            let r = Ray(pt3(0.0, 3.0, 0.0), Vec3::zero());
            assert_eq!(r.intersect(&PLANE), None);

            let r = Ray(pt3(0.0, 1.0, 0.0), Vec3::zero());
            assert_eq!(r.intersect(&PLANE), None);

            let r = Ray(pt3(0.0, 2.0, 0.0), Vec3::zero());
            assert_eq!(r.intersect(&PLANE), Some((0.0, pt3(0.0, 2.0, 0.0))));
        }
    }

    mod ray_bbox {
        use super::*;

        const BBOX: BBox<()> = BBox(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

        #[test]
        fn parallel_simple() {
            //      +----+
            // x--> |    |
            //      +----+
            let ray = Ray(pt3(0.0, 0.0, -2.0), vec3(0.0, 0.0, 1.0));
            assert_eq!(ray.intersect(&BBOX), Some((1.0, pt3(0.0, 0.0, -1.0))));
        }
        #[test]
        fn diagonal_simple() {
            //
            //      +----+
            //     ^|    |
            //    / +----+
            //   x
            let ray = Ray(pt3(-1.5, 0.0, -2.0), vec3(1.0, 0.0, 1.0));
            assert_eq!(ray.intersect(&BBOX), Some((1.0, pt3(-0.5, 0.0, -1.0))));
        }
        #[test]
        fn parallel_intersect_at_vertex() {
            //   x--> ,_____.
            //       /     /|
            //      /_____/ |
            //      |     | /
            //      |_____|/
            let ray = Ray(pt3(1.0, 1.0, -2.0), vec3(0.0, 0.0, 1.0));
            assert_eq!(ray.intersect(&BBOX), Some((1.0, pt3(1.0, 1.0, -1.0))));
        }
        #[test]
        fn parallel_intersect_at_edge() {
            //        ,_____.
            //  x--> /     /|
            //      /_____/ |
            //      |     | /
            //      |_____|/
            //
            let ray = Ray(pt3(0.0, 1.0, -2.0), vec3(0.0, 0.0, 1.0));
            assert_eq!(ray.intersect(&BBOX), Some((1.0, pt3(0.0, 1.0, -1.0))));
        }
        #[test]
        fn ray_starts_inside() {
            //  +--^----+
            //  |  |    |
            //  |  x    |
            //  +-------+
            let ray = Ray(pt3(0.0, 0.0, -0.5), vec3(0.0, 1.0, 0.0));
            assert_eq!(ray.intersect(&BBOX), Some((1.0, pt3(0.0, 1.0, -0.5))));
        }
        #[test]
        fn ray_starts_on_side_plane() {
            // Points away
            //    +-----+
            // <--x     |
            //    +-----+
            let ray = Ray(pt3(0.0, 0.0, -1.0), vec3(0.0, 0.0, -1.0));
            assert_eq!(ray.intersect(&BBOX), Some((0.0, pt3(0.0, 0.0, -1.0))));
            // Points inside
            //    +-----+
            //    x-->  |
            //    +-----+
            let ray = Ray(pt3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0));
            assert_eq!(ray.intersect(&BBOX), Some((0.0, pt3(0.0, 0.0, -1.0))));
        }
        #[test]
        fn no_intersection() {
            // Diagonal ray
            //   ^
            //  / +----+
            // x  |    |
            //    +----+
            let ray = Ray(pt3(0.0, 0.0, -2.5), vec3(0.0, 1.0, 1.0));
            assert_eq!(ray.intersect(&BBOX), None);

            // Parallel but offset ray
            // x--->
            //    +----+
            //    |    |
            //    +----+
            let ray = Ray(pt3(0.0, 1.5, -2.0), vec3(0.0, 0.0, 1.0));
            assert_eq!(ray.intersect(&BBOX), None);
        }
        #[test]
        fn opposite_direction() {
            //  +----+
            //  |    |  x--->
            //  +----+
            let ray = Ray(pt3(0.0, 0.0, 2.0), vec3(0.0, 0.0, 1.0));
            assert_eq!(ray.intersect(&BBOX), None);
        }
        #[test]
        fn zero_length_ray() {
            let ray = Ray(pt3(0.0, 0.0, -2.0), vec3(0.0, 0.0, 0.0));
            assert_eq!(ray.intersect(&BBOX), None);
        }
        #[test]
        fn empty_box() {
            let empty = BBox::<()>(pt3(-1.0, -1.0, 1.0), pt3(1.0, 1.0, -1.0));
            let ray = Ray(pt3(0.0, 0.0, -2.0), vec3(0.0, 0.0, 1.0));
            assert_eq!(ray.intersect(&empty), None);
        }
    }

    // TODO until sqrt has a fallback
    #[cfg(feature = "std")]
    mod ray_sphere {
        use super::*;

        const SPHERE: Sphere = Sphere(pt3(0.0, 0.0, 1.0), 2.0);

        #[test]
        fn ray_passes_through_sphere() {
            let ray: Ray3 = Ray(pt3(0.0, 0.0, -3.0), vec3(0.0, 0.0, 2.0));
            assert_eq!(
                ray.intersect(&SPHERE),
                Some((1.0, pt3(0.0, 0.0, -1.0)))
            );
        }
        #[test]
        fn ray_tangent_to_sphere() {
            let ray: Ray3 = Ray(pt3(0.0, 2.0, -3.0), vec3(0.0, 0.0, 2.0));
            assert_eq!(ray.intersect(&SPHERE), Some((2.0, pt3(0.0, 2.0, 1.0))));
        }

        #[test]
        fn ray_origin_inside_sphere() {
            let ray: Ray3 = Ray(pt3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 2.0));
            assert_eq!(ray.intersect(&SPHERE), Some((1.5, pt3(0.0, 0.0, 3.0))));

            let ray: Ray3 = Ray(pt3(0.0, 0.0, 2.0), vec3(0.0, 0.0, 2.0));
            assert_eq!(ray.intersect(&SPHERE), Some((0.5, pt3(0.0, 0.0, 3.0))));
        }

        #[test]
        fn no_intersection() {
            let ray: Ray3 = Ray(pt3(0.0, 0.0, -3.0), vec3(0.0, 0.0, -1.0));
            assert_eq!(ray.intersect(&SPHERE), None);

            let ray: Ray3 = Ray(pt3(0.0, 0.0, -3.0), vec3(0.0, 1.0, 1.0));
            assert_eq!(ray.intersect(&SPHERE), None);
        }
    }
}
