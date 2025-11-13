use core::fmt::{Debug, Formatter};

use retrofire_core::{
    geom::{Edge, Line2, Plane3, Ray, Ray2, Ray3},
    mat,
    math::{ApproxEq, Mat2, Point2, Point3, pt2, vec3},
    render::scene::BBox,
};

#[cfg(feature = "std")]
use retrofire_core::geom::Sphere;

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

pub type RayIntersect3<B> = Option<(f32, Point3<B>)>;
pub type RayIntersect2<B> = Option<(f32, Point2<B>)>;

#[derive(Copy, Clone, PartialEq)]
pub enum LineIntersect<B> {
    /// Unique intersection point.
    Point(Point2<B>),
    /// Line is coincident with the intersecting object.
    Coincident,
}

//
// Inherent impls
//

impl<B> LineIntersect<B> {
    /// Returns the intersection point if `self` is a `LineIntersect::Point`,
    /// `None` otherwise.
    pub fn point(&self) -> Option<Point2<B>> {
        match self {
            LineIntersect::Point(p) => Some(*p),
            LineIntersect::Coincident => None,
        }
    }
}

//
// Trait impls
//

impl<B: Debug + Default> Debug for LineIntersect<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Point(p) => write!(f, "Point({:?})", p),
            Self::Coincident => f.write_str("Coincident"),
        }
    }
}

//
// 3D Intersect impls
//

impl<B> Intersect<Plane3<B>> for Ray3<B> {
    type Result = RayIntersect3<B>;

    /// Returns the unique intersection point of `self` and a plane,
    /// or `None` if they do not intersect.
    ///
    /// If an intersection point exists, returns `Some((t, point))`, where
    /// `point` is the intersection point and  `t` is the ray parameter value
    /// such that `self.orig + t * self.dir == point`.
    fn intersect(&self, p: &Plane3<B>) -> Self::Result {
        let &Self(orig, dir) = self;

        // Plane equation:
        //   ax + by + cz = d
        //   let n = (a, b, c), P = (x, y, z)
        //   n·P = d
        //
        // Ray equation:
        //   P = O + t·v
        //
        // Substitute ray eqn to plane eqn
        //   n·(O + t·v) = d
        //   n·O + t·n·v = d
        //   t·n·v = d - n·O  // divide by n·v (a scalar)
        //   t = (d - n·O) / n·v
        //     = (-n·O + d) / n·v
        //     = - (n·O - d) / n·v
        //
        // Or in homogeneous coordinates:
        //   let n = (a, b, c, -d), O = (o_x, o_y, o_z, 1)
        //   t = - n·O / n·v
        //
        // Length of n does not matter, normalization would cancel out anyway.
        // n·O is positive if the point is outside the plane, negative if inside.
        // n·v is positive if pointing towards the same hemisphere as the plane
        // normal, negative otherwise.
        //
        // Cases where an intersection exists:
        // * if n·O = 0, the ray origin lies on the plane.
        // * if n·O < 0 and n·v > 0, the ray is inside and points towards the plane.
        // * if n·O > 0 and n·v < 0, the ray is outside and points towards the plane.
        //
        // Cases where it does not exist:
        // * if n·O != 0 and n·v = 0, the ray is parallel but not coincident with the plane.
        // * if n·O < 0 and n·v < 0, the ray is inside and points away from the plane.
        // * if n·O > 0 and n·v > 0, the ray is outside and points away from the plane.

        let [a, b, c, d] = p.coeffs();
        let n = vec3(a, b, c);

        let num = orig.to_vec().dot(&n) + d;
        let denom = dir.dot(&n);

        if num >= 0.0 && denom < 0.0 || num <= 0.0 && denom > 0.0 {
            // Ray points towards the plane or origin lies on the plane
            let t = -num / denom;
            Some((t, orig + t * dir))
        } else if num == 0.0 && denom == 0.0 {
            // Ray coincident with the plane, unlikely
            Some((0.0, orig))
        } else {
            None
        }
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
        //   p(t) = O + d·t
        //   x(t) = Ox + dx·t
        //   y(t) = Oy + dy·t
        //   z(t) = Oz + dz·t
        //
        // Plane equations:
        //   x = lx, x = ux
        //   y = ly, x = uy
        //   z = lz, x = uz
        //
        // For each slab, ie. pair of parallel planes:
        //   Substitute eg.
        //   x_l = Ox + x_d·t0
        //   x_u = Ox + x_d·t1
        //
        //   t0 = (x_l - x_O) / x_d    | x_d=0 iff ray parallel with planes
        //   t1 = (x_u - x_O) / x_d
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

//
// 2D intersection
//

impl<B> Intersect<Self> for Line2<B> {
    type Result = Option<LineIntersect<B>>;

    /// Computes the intersection point of `self` and another 2-line.
    ///
    /// If the lines are parallel but not coincident, returns `None`. Otherwise,
    /// if the lines are coincident, returns `Some(LineIntersect::Coincident)`.
    /// Otherwise, returns `Some(LineIntersect::Point(p))`, where `p` is the
    /// unique intersection point.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{
    ///     assert_approx_eq, geom::{Ray, Line2}, math::{pt2, vec2},
    /// };
    /// use retrofire_geom::{Intersect, isect::LineIntersect::*};
    ///
    /// let horiz = Line2::<()>::from(Ray(pt2(0.0, 2.0), vec2(1.0, 0.0)));
    /// let vert = Line2::<()>::from(Ray(pt2(3.0, 0.0), vec2(0.0, 1.0)));
    /// assert_eq!(horiz.intersect(&vert), Some(Point(pt2(3.0, 2.0))));
    ///
    /// let horiz2 = Line2::<()>::from(Ray(pt2(0.0, 3.0), vec2(1.0, 0.0)));
    /// assert_eq!(horiz.intersect(&horiz2), None);
    ///
    /// assert_eq!(horiz.intersect(&horiz), Some(Coincident));
    ///
    ///
    /// ```
    fn intersect(&self, other: &Self) -> Self::Result {
        let [a, b, c] = self.coeffs();
        let [d, e, f] = other.coeffs();

        // Solve the system of equations for x and y:
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
        match abde.checked_inverse() {
            Some(inv) => {
                let res = inv.apply(&pt2(c, f));
                Some(LineIntersect::Point(res))
            }
            None if [a, b, c].approx_eq(&[d, e, f]) => {
                Some(LineIntersect::Coincident)
            }
            None => None,
        }
    }
}

impl<B> Intersect<Line2<B>> for Ray2<B> {
    type Result = RayIntersect2<B>;

    // Returns the intersection point of self and a line.
    fn intersect(&self, line: &Line2<B>) -> Self::Result {
        // TODO if degenerate ray, could return if ray origin on edge

        // First find intersection of lines
        let ray_line = Line2::from(*self);
        let isect = ray_line.intersect(line)?;

        let Self(orig, dir) = self;
        match isect {
            LineIntersect::Point(pt) => {
                // Check if the point lies in the correct half-line
                let t = (pt - *orig).dot(dir);
                (t >= 0.0).then_some((t / dir.len_sqr(), pt))
            }
            LineIntersect::Coincident => {
                // Ray lies on the line, the closest common point
                // is simply the origin point
                Some((0.0, *orig))
            }
        }
    }
}

impl<B> Intersect<Edge<Point2<B>>> for Ray2<B> {
    type Result = RayIntersect2<B>;

    /// Returns the intersection of `self` and an edge, or `None` if there is
    /// no intersection point.
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
        let e01 = edge.1 - edge.0;
        let u = (pt - edge.0).dot(&e01);
        if u.approx_in(0.0..e01.len_sqr()) {
            return Some((t, pt));
        }

        // If ray is coincident with the edge, ray-line gives ray.0 as the
        // intersection point. If ray.0 is outside the edge, there are two cases:
        //   e-----e   r--->
        // where there is no intersection, and
        //   e-----e   <---r
        // where the intersection point is the closest edge endpoint.
        if !self.0.approx_eq(&pt) {
            return None;
        }
        let t0 = (edge.0 - pt).dot(&self.1);
        if t0.approx_le(&0.0) {
            return None;
        }
        let t1 = (edge.1 - pt).dot(&self.1);
        if t0 <= t1 {
            Some((t0 / self.1.len_sqr(), edge.0))
        } else {
            Some((t1 / self.1.len_sqr(), edge.1))
        }
    }
}

impl<B> Intersect<Self> for Edge<Point2<B>> {
    type Result = Option<Point2<B>>;

    /// Returns the intersection point of `self` and another edge, or `None`
    /// if the edges do not intersect.
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
        match ray.intersect(edge) {
            Some((t, pt)) if t <= 1.0 => Some(pt),
            _ => None,
        }
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
    // TODO 2D tests from stash
}
