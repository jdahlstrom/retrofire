use retrofire_core::{
    geom::{Edge, Line2, Plane3, Ray},
    mat,
    math::{ApproxEq, Mat2, Point2, Point3, Vec3, pt2},
    render::scene::BBox,
};

use LineIntersect::*;

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
                && bbox.contains(&p)
                && res.is_none_or(|(u, _)| t < u)
            {
                res = pt;
            }
        }
        res
    }
}

//
// 2D intersection
//

pub type RayIntersect2<B> = Option<(f32, Point2<B>)>;

#[derive(Copy, Clone, PartialEq)]
pub enum LineIntersect<B> {
    /// Unique intersection point.
    Point(Point2<B>),
    /// Line is coincident with intersecting object.
    Coincident,
}

impl<B> Intersect<Self> for Line2<B> {
    type Result = Option<LineIntersect<B>>;

    /// Computes the intersection point of `self` and another 2-line.
    ///
    /// If the lines are parallel but not coincident, returns `None`. Otherwise,
    /// if the lines are coincident, returns `Some(Coincident)`. Otherwise,
    /// returns `Some(Point(p))`, where `p` is the unique intersection point.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{
    ///     assert_approx_eq, geom::Ray, math::{pt2, vec2},
    /// };
    /// use retrofire_geom::isect::{
    ///     Intersect, Line2, LineIntersect::*,
    /// };
    ///
    /// let horiz = <Line2>::from(Ray(pt2(0.0, 2.0), vec2(1.0, 0.0)));
    /// let vert = <Line2>::from(Ray(pt2(3.0, 0.0), vec2(0.0, 1.0)));
    /// assert_eq!(horiz.intersect(&vert), Some(Point(pt2(3.0, 2.0))));
    ///
    /// let horiz2 = <Line2>::from(Ray(pt2(0.0, 3.0), vec2(1.0, 0.0)));
    /// assert_eq!(horiz.intersect(&horiz2), None);
    ///
    /// assert_eq!(horiz.intersect(&horiz), Some(Coincident));
    ///
    ///
    /// ```
    fn intersect(&self, other: &Self) -> Self::Result {
        // TODO only works if normalized
        if self.0 == other.0 {
            return Some(Coincident);
        }

        let [a, b, c] = self.0.0;
        let [d, e, f] = other.0.0;

        // Otherwise, solve the system of equations for x and y:
        //    ax + by = c     // self
        //    dx + ey = f     // other
        //
        // Write in matrix form and solve:
        //    (a b) (x) = (c)
        //    (d e) (y)   (f)
        //
        //               -1
        //    (x) = (a b)   (c)
        //    (y)   (d e)   (f)
        let abde: Mat2<B> = mat![
            a, b;
            d, e;
        ];
        let inv_abde = abde.checked_inverse()?;
        let res = inv_abde.apply(&pt2(c, f));
        Some(Point(res))
    }
}

impl<B> Intersect<Line2<B>> for Ray<Point2<B>> {
    type Result = RayIntersect2<B>;

    // Returns the intersection point of self and a line.
    fn intersect(&self, line: &Line2<B>) -> Self::Result {
        // TODO if degenerate ray, could return if ray origin on edge

        // First find intersection of lines
        let ray_line = Line2::from(*self);
        let isect = ray_line.intersect(&line)?;

        let Self(orig, dir) = self;
        match isect {
            Point(pt) => {
                // Check if the point lies in the correct half-line
                let t = (pt - *orig).scalar_project(dir);
                (t >= 0.0).then_some((t, pt))
            }
            Coincident => {
                // Ray lies on the line, the closest common point
                // is simply the origin point
                Some((0.0, *orig))
            }
        }
    }
}

impl<B> Intersect<Edge<Point2<B>>> for Ray<Point2<B>> {
    type Result = RayIntersect2<B>;

    /// Finds the intersection of `self` and an edge.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{
    ///     assert_approx_eq, geom::{Edge, Ray}, math::{pt2, vec2, Point2},
    /// };
    /// use retrofire_geom::isect::Intersect;
    ///
    /// //      ^
    /// //      3    O
    /// //      |     \
    /// //      |      v
    /// //   E==1=======X==E
    /// //      |
    /// // <----+-------------->
    /// let ray: Ray<Point2> = Ray(pt2(2.0, 3.0), vec2(1.0, -2.0));
    /// let edge = Edge(pt2(-1.0, 1.0), pt2(4.0, 1.0));
    ///
    /// let (t, point) = ray.intersect(&edge).unwrap();
    /// assert_eq!(t, 1.0);
    /// assert_approx_eq!(point, pt2(3.0, 1.0));
    ///
    /// //     O
    /// //       \
    /// // E=====E v
    /// //
    /// let edge = Edge(pt2(-1.0, 1.0), pt2(2.0, 1.0));
    /// assert_eq!(ray.intersect(&edge), None);
    /// ```
    // // TODO check that these are handled by actual unit tests
    // // E=====E  <---O
    // let ray: Ray<Point2> = Ray(pt2(4.0, 1.0), vec2(-1.0, 0.0));
    // assert_eq!(ray.intersect(&edge), Some((2.0, pt2(2.0, 1.0))));
    //
    // // E==O--E-->
    // let ray: Ray<Point2> = Ray(pt2(1.0, 1.0), vec2(1.0, 0.0));
    // assert_eq!(ray.intersect(&edge), Some((0.0, pt2(1.0, 1.0))));
    //
    // // E=====E  O--->
    // let ray: Ray<Point2> = Ray(pt2(4.0, 1.0), vec2(1.0, 0.0));
    // assert_eq!(ray.intersect(&edge), None);
    fn intersect(&self, edge: &Edge<Point2<B>>) -> Self::Result {
        // Compute ray-line intersection
        let (t, pt) = self.intersect(&Line2::from(*edge))?;

        // Ray intersects the line of edge, but still have to check
        // whether the point is between edge endpoints
        let u = (pt - edge.0).scalar_project(&(edge.1 - edge.0));
        if u.approx_in(0.0..1.0) {
            return Some((t, pt));
        }

        // If ray is coincident with the edge, ray-line gives ray.0 as the
        // intersection point. If ray.0 is outside the edge, there are two cases:
        //   e-----e   pt--->
        // where there is no intersection, and
        //   e-----e   <---pt
        // where the intersection point is the closest edge endpoint.

        if !self.0.approx_eq(&pt) {
            return None;
        }

        let t0 = (edge.0 - pt).scalar_project(&self.1);
        if t0.approx_le(&0.0) {
            return None;
        }
        let t1 = (edge.1 - pt).scalar_project(&self.1);

        Some(if t0 <= t1 { (t0, edge.0) } else { (t1, edge.1) })

        /*let Point(pt) = isect else {
            // Coincident
            let mut t0 = (e0 - orig).scalar_project(&dir);
            let mut t1 = (e1 - orig).scalar_project(&dir);

            if t0 > t1 {
                swap(&mut t0, &mut t1);
                swap(&mut e0, &mut e1);
            }
            return if t1 < 0.0 {
                // e-----e  o---->
                None
            } else if t0 < 0.0 {
                // e--o--->--e or e---o--e-->
                Some((0.0, orig))
            } else {
                // 0.0 <= t0
                // o---e-->--e or o--->  e-----e
                Some((t0, e0))
            };
        };*/
    }
}

impl<B> Intersect<Self> for Edge<Point2<B>> {
    type Result = Option<Point2<B>>;

    /// Returns the intersection point of `self` and another edge.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{
    ///     assert_approx_eq,
    ///     geom::Edge,
    ///     math::{pt2, Point2}
    /// };
    /// use retrofire_geom::isect::Intersect;
    ///
    /// //
    /// // O1
    /// //   \
    /// //    \
    /// //     \
    /// // E1---X----E2
    /// //       \
    /// //        O2
    /// let edge: Edge<Point2> = Edge(pt2(0.0, 1.0), pt2(4.0, 1.0));
    /// let other = Edge(pt2(0.0, 3.0), pt2(3.0, 0.0));
    ///
    /// assert_approx_eq!(edge.intersect(&other), Some(pt2(2.0, 1.0)));
    ///
    /// ```
    fn intersect(&self, edge: &Self) -> Self::Result {
        let ray = Ray(self.0, self.1 - self.0);
        let (t, pt) = ray.intersect(edge)?;
        assert!(t.approx_gt(&0.0));
        (t <= 1.0).then_some(pt)
    }
}

#[cfg(test)]
mod tests {
    use retrofire_core::math::{Linear, pt3, vec3};

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
            // x--> +----+
            //      |    |
            //      +----+
            let ray = Ray(pt3(0.0, 0.0, -2.0), vec3(1.0, 1.0, 1.0));
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
            let ray = Ray(pt3(0.0, 0.0, -2.0), vec3(-1.0, 0.0, 1.0));
            assert_eq!(ray.intersect(&BBOX), Some((1.0, pt3(-1.0, 0.0, -1.0))));
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

    // TODO 2D tests from stash
}
