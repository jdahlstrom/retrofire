use core::fmt::Debug;

#[cfg(feature = "std")] // TODO separate fp feature for geom
use retrofire_core::geom::Sphere;
use retrofire_core::{
    geom::{Plane3, Ray, Ray3},
    math::{ApproxEq, Point3, vec3},
    render::scene::BBox,
};

/// Trait for calculating whether and at which points two objects intersect.
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

        // TODO checking two very unlikely conditions

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

impl<B: Debug + Default> Intersect<BBox<B>> for Ray3<B> {
    type Result = RayIntersect3<B>; // Only closest for now

    /// Returns the nearest intersection point of `self` and a box,
    /// or `None` if they do not intersect.
    ///
    /// If an intersection point exists, returns `Some((t, point))`, where
    /// `point` is the intersection point and  `t` is the ray parameter value
    /// such that `self.orig + t * self.dir == point`.
    fn intersect(&self, bbox: &BBox<B>) -> Self::Result {
        let &BBox(low, upp) = bbox;
        let Ray(orig, dir) = *self;

        // Ray equation:
        // p(t) = O + d·t
        //   x(t) = Ox + dx·t
        //   y(t) = Oy + dy·t
        //   z(t) = Oz + dz·t
        //
        // Plane equations:
        // x = lx, x = ux
        // y = ly, x = uy
        // z = lz, x = uz
        //
        // For each slab, ie. pair of parallel planes:
        // Substitute eg.
        // x_l = Ox + x_d·t0
        // x_u = Ox + x_d·t1
        //
        // t0 = (x_l - x_O) / x_d    | x_d=0 iff ray parallel with planes
        // t1 = (x_u - x_O) / x_d
        //
        // Same for y and z slabs

        if bbox.is_empty() {
            return None;
        }

        let r_d = vec3(1.0 / dir.x(), 1.0 / dir.y(), 1.0 / dir.z());
        let low = (low - orig) * r_d;
        let upp = (upp - orig) * r_d;

        let near = low.zip_map(upp, |l, u| l.min(u));
        let far = low.zip_map(upp, |l, u| l.max(u));

        let near_t = near[0].max(near[1]).max(near[2]);
        let far_t = far[0].min(far[1]).min(far[2]);

        if far_t.is_infinite() {
            return None;
        }

        if near_t > far_t || far_t < 0.0 {
            // ---max---min--- (misses the box) or
            // ---min---max---0---> (box behind ray)
            return None;
        }
        let t = if near_t >= 0.0 {
            // ---0---min---max---> (hits box)
            near_t
        } else {
            // ---min---0---max---> (inside the box, unlikely)
            far_t
        };
        Some((t, orig + t * dir))
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
        //          /       \
        //         /         \
        //        |     C--r--|
        //         \    |    /
        //          \___|___/
        //             _|
        //   O--------+-C'----> d
        //
        // Find point P = (x, y, z) given sphere (C, r) and ray (O, d)
        //
        // Sphere equation:
        //   (x - c_x)² + (y - c_y)² + (z - c_z)² = r²
        // or in vector form
        //   (P - C) · (P - C) = r²
        //
        // Ray equation:
        //   P = O + t·d
        //
        // Intersection:
        //
        //   Substitute ray equation to sphere equation:
        //   (o + t·d - c) · (o + t·d - c) = r²
        //
        //   Multiply out
        //   o (o + td - c) + t·d (o + td - c) - c (o + td - c) = r²
        //
        //   Distribute
        //   o·o + o·td - o·c + o·td + td·td - td·c - c·o - c·td + c·c = r²
        //
        //   Reorder
        //   td·td + o·td + o·td - td·c + o·o - o·c - c·o + c·c - r² = 0
        //
        //   Factor out t's
        //   (d·d) t² + (o·d + o·d - c·d) t + o·o - 2o·c + c·c - r² = 0
        //
        // Solve quadratic equation:
        //   (d·d) t²  +  2(o - c)·d t  +  (o - c)² - r²  =  0
        //
        //   t = (-b ± √(b² - 4ac)) / 2a

        let c_to_o = orig - center;
        let a = dir.len_sqr(); // >= 0
        let b = 2.0 * c_to_o.dot(&dir);
        let c = c_to_o.len_sqr() - r * r;

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            // the line of the ray does not hit the sphere
            return None;
        }

        use retrofire_core::math::float::f32;
        let sqrt = f32::sqrt(discriminant);
        // sqrt >= 0.0, thus t0 <= t1 always
        let (t0, t1) = (-b - sqrt, -b + sqrt);
        let t = if t0 >= 0.0 {
            // ray hits both points
            t0
        } else if t1 >= 0.0 {
            // ray origin is inside sphere
            t1
        } else {
            // sphere is behind ray
            return None;
        };
        let t = t / (2.0 * a);
        Some((t, orig + t * dir))
    }
}

#[cfg(test)]
mod tests {
    use retrofire_core::math::{Linear, Vec3, pt3};

    use super::*;

    mod ray_plane {
        use super::*;

        const PLANE: Plane3<()> = Plane3::new(0.0, 1.0, 0.0, 2.0);

        #[test]
        #[ignore]
        fn ray_plane_xxx() {
            let r = Ray::<Point3>(
                pt3(-3.308549, 6.2584567, -3.351655),
                vec3(3.308549, -6.2584567, 3.351655),
            );
            let bbox = BBox(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

            assert_eq!(r.intersect(&bbox), Some((0.0, pt3(0.0, 0.0, 0.0))));
        }

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
        #[ignore]
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
        #[ignore]
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
        fn sphere_behind_ray() {
            let ray: Ray3 = Ray(pt3(0.0, 0.0, -3.0), vec3(0.0, 0.0, -1.0));
            assert_eq!(ray.intersect(&SPHERE), None);
        }

        #[test]
        fn ray_misses_sphere() {
            let ray: Ray3 = Ray(pt3(0.0, 0.0, -3.0), vec3(0.0, 1.0, 1.0));
            assert_eq!(ray.intersect(&SPHERE), None);
        }
    }
}
