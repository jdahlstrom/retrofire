use retrofire_core::{
    geom::{Plane3, Ray},
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

        // TODO checking two very unlikely conditions

        let denom = dir.dot(&p.normal().to());
        let num = p.signed_dist(*orig);
        if num.approx_eq(&0.0) {
            // Origin point coincident with the plane
            return Some((0.0, *orig));
        }

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

impl<B: Default> Intersect<BBox<B>> for Ray<Point3<B>> {
    type Result = Option<(f32, Point3<B>)>; // Only closest for now

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

        let mut res = None;

        // for (p, a, b) in [
        //     (0, 1, 2),
        //     (1, 1, 2),
        //     (2, 0, 2),
        //     (3, 0, 2),
        //     (4, 0, 1),
        //     (5, 0, 1),
        // ] {
        //     if let pt @ Some((t, p)) = self.intersect(&planes[p])
        //         && (l[a]..=u[a]).contains(&p[a])
        //         && (l[b]..=u[b]).contains(&p[b])
        //         && res.is_none_or(|(u, _)| t < u)
        //     {
        //         res = pt;
        //     }
        // }
        // res

        // X planes
        if let pt @ Some((_, p)) = self.intersect(&planes[0])
            && (l[1]..=u[1]).contains(&p[1])
            && (l[2]..=u[2]).contains(&p[2])
        {
            res = pt;
        }
        if let pt @ Some((t, p)) = self.intersect(&planes[1])
            && (l[1]..=u[1]).contains(&p[1])
            && (l[2]..=u[2]).contains(&p[2])
            && res.is_none_or(|(u, _)| t < u)
        {
            res = pt;
        }

        // Y planes
        if let pt @ Some((t, p)) = self.intersect(&planes[2])
            && (l[0]..=u[0]).contains(&p[0])
            && (l[2]..=u[2]).contains(&p[2])
            && res.is_none_or(|(u, _)| t < u)
        {
            res = pt;
        }
        if let pt @ Some((t, p)) = self.intersect(&planes[3])
            && (l[0]..=u[0]).contains(&p[0])
            && (l[2]..=u[2]).contains(&p[2])
            && res.is_none_or(|(u, _)| t < u)
        {
            res = pt;
        }

        // Z planes
        if let pt @ Some((t, p)) = self.intersect(&planes[4])
            && (l[1]..u[1]).contains(&p[1])
            && (l[0]..u[0]).contains(&p[0])
            && res.is_none_or(|(u, _)| t < u)
        {
            res = pt;
        }
        if let pt @ Some((t, p)) = self.intersect(&planes[5])
            && (l[1]..=u[1]).contains(&p[1])
            && (l[0]..=u[0]).contains(&p[0])
            && res.is_none_or(|(u, _)| t < u)
        {
            res = pt;
        }
        res
        // for plane in &planes {
        //     if let pt @ Some((t, p)) = self.intersect(plane)
        //         && bbox.contains(&p)
        //         && res.is_none_or(|(u, _)| t < u)
        //     {
        //         res = pt;
        //     }
        // }
        // res
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
}
