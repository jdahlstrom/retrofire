use core::fmt::{Debug, Formatter};

use re::geom::{Edge, Plane, Plane3, Ray, Tri, Vertex3, vertex};
use re::math::{
    Apply, ApproxEq, Linear, Mat2x2, Point2, Point3, Vec2, Vec3,
    mat::RealToReal, pt2, vec3,
};
use re::render::scene::BBox;

use re::mat;

use LineIntersect::*;

pub trait Intersect<T> {
    type Result;

    fn intersect(&self, other: &T) -> Self::Result;
}

type RayIntersect2<B> = Option<(f32, Point2<B>)>;
type RayIntersect3<B> = Option<(f32, Point3<B>)>;

// TODO move to core::geom, same methods as Plane
#[derive(Copy, Clone, PartialEq)]
pub struct Line2<B>(Vec3<B>);

impl<B> Line2<B> {
    /// Two-dimensional line, given by the line equation ax + by = c.
    pub fn new(a: f32, b: f32, c: f32) -> Self {
        // Normalize
        let (a, b, c) = if c != 0.0 {
            if a != 0.0 && b != 0.0 {
                // General affine: does not cross the origin
                //   a·x + b·y = c  | div by c
                //   a'·x + b'·y = 1
                //   (y = -a'·x/b' + 1/b')
                (a / c, b / c, -1.0)
            } else if a != 0.0 && b == 0.0 {
                // b = 0: vertical
                //   a·x + 0·y = c
                //   a'·x = 1
                //   (x = 1/a')
                (a / c, 0.0, -1.0)
            } else if a == 0.0 && b != 0.0 {
                // a = 0: horizontal
                //   0·x + b·y = c
                //   b'·y = 1
                //   y = b'
                (0.0, b / c, -1.0)
            } else {
                panic!("degenerate line (0, 0, {c})")
            }
        } else {
            // c = 0
            if a != 0.0 && b != 0.0 {
                // General linear: crosses the origin
                //   a·x + b·y = 0  | div by b
                //   a'·x + y = 0
                //   (y = -a'·x)
                (a / b, 1.0, 0.0)
            } else if a != 0.0 && b == 0.0 {
                // b = 0: coincident with x-axis
                //   a·x + 0·y = 0
                //   x = 0
                (1.0, 0.0, 0.0)
            } else if a == 0.0 && b != 0.0 {
                // a = 0: coincident with y-axis
                //   0·x + b·y = 0
                //   b·y = 0
                //   y = 0
                (0.0, 1.0, 0.0)
            } else {
                panic!("degenerate line (0, 0, 0)")
            }
        };
        Self(vec3(a, b, c))
    }
}

impl<B> Debug for Line2<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        // ax + by = c
        let [a, b, c] = self.0.0;

        f.write_str("Line(")?;
        if b != 0.0 {
            // by = -ax + c  <=>  y = -a/b x + c/b
            let ab = -a / b; // slope
            let cb = c / b; // y intercept
            let pm = if cb >= 0.0 { '+' } else { '-' };
            if ab == 0.0 {
                write!(f, "y =")?;
            } else if ab == 1.0 {
                write!(f, "y = x {pm}")?;
            } else if ab == -1.0 {
                write!(f, "y = -x {pm}")?;
            } else {
                write!(f, "y = {ab}·x {pm}")?;
            }
            write!(f, " {}", cb.abs())?;
        } else if a != 0.0 {
            // ax = -by + c  <=> x = c/a
            write!(f, "x = {}", c / a)?;
        } else {
            f.write_str("<degenerate>")?;
        }
        f.write_str(")")
    }
}

impl<B> From<Edge<Point2<B>>> for Line2<B> {
    fn from(Edge(p, q): Edge<Point2<B>>) -> Self {
        Self::from(Ray(p, q - p))
    }
}
impl<B> From<Ray<Point2<B>>> for Line2<B> {
    /// Computes the intersection point of a ray and a line.
    /// # Panics
    /// If the ray is zero-length.
    ///
    /// # Examples
    /// ```
    /// use re::{geom::Ray, math::{pt2, vec2, vec3}};
    /// use retrofire_geom::isect::Line2;
    ///
    /// let line = Line2::<()>::from(Ray(pt2(0.0, 0.0), vec2(1.0, 1.0)));
    /// assert_eq!(line.0, vec3(-1.0, 1.0, 0.0).normalize());
    /// ```
    fn from(Ray(pt, dir): Ray<Point2<B>>) -> Self {
        assert!(!dir.approx_eq(&Vec2::zero())); // TODO vec::zero() and point::origin()?

        let n = dir.perp();
        let [a, b] = n.0;
        let c = pt.to_vec().dot(&n);

        // At least one of a, b != 0, if n != *0*
        // If c = 0, line crosses the origin
        Self::new(a, b, -c)
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum LineIntersect<B> {
    Point(Point2<B>),
    Coincident,
}

impl<B> Intersect<Self> for Line2<B> {
    type Result = Option<LineIntersect<B>>;

    /// Computes the intersection point of `self` and another 2-line.
    ///
    /// # Examples
    /// ```
    /// use re::assert_approx_eq;
    /// use re::geom::Ray;
    /// use re::math::{pt2, vec2};
    ///
    /// use retrofire_geom::isect::{Intersect, Line2, LineIntersect::*};
    ///
    /// let horiz = Line2::<()>::from(Ray(pt2(0.0, 2.0), vec2(1.0, 0.0)));
    /// let vert = Line2::from(Ray(pt2(3.0, 0.0), vec2(0.0, 1.0)));
    /// assert_eq!(horiz.intersect(&vert), Some(Point(pt2(3.0, 2.0))));
    ///
    /// let horiz2 = Line2::from(Ray(pt2(0.0, 3.0), vec2(1.0, 0.0)));
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
        //               -1
        //    (x) = (a b)   (c)
        //    (y)   (d e)   (f)

        let abde: Mat2x2<RealToReal<2, B, B>> = mat![
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

    // 2D ray–line intersection point.
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

    /// 2D Ray–edge intersection point.
    ///
    /// # Examples
    /// ```
    /// use re::assert_approx_eq;
    /// use re::geom::{Edge, Ray};
    /// use re::math::{pt2, vec2, Point2};
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
    ///
    /// ``ìgnore
    /// // TODO check that these are handled by actual unit tests
    /// // E=====E  <---O
    /// let ray: Ray<Point2> = Ray(pt2(4.0, 1.0), vec2(-1.0, 0.0));
    /// assert_eq!(ray.intersect(&edge), Some((2.0, pt2(2.0, 1.0))));
    ///
    /// // E==O--E-->
    /// let ray: Ray<Point2> = Ray(pt2(1.0, 1.0), vec2(1.0, 0.0));
    /// assert_eq!(ray.intersect(&edge), Some((0.0, pt2(1.0, 1.0))));
    ///
    /// // E=====E  O--->
    /// let ray: Ray<Point2> = Ray(pt2(4.0, 1.0), vec2(1.0, 0.0));
    /// assert_eq!(ray.intersect(&edge), None);
    ///
    /// ```
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

    /// 2D edge–edge intersection point.
    ///
    /// # Examples
    /// ```
    /// use re::assert_approx_eq;
    /// use re::geom::Edge;
    /// use re::math::{pt2, Point2};
    /// use retrofire_geom::Intersect;
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

impl<B> Intersect<Plane3<B>> for Ray<Point3<B>> {
    type Result = RayIntersect3<B>;

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
            // Ray points away from the plane, intersection behind it
            return None;
        }

        Some((t, *orig + t * *dir))
    }
}

impl<A, B> Intersect<Tri<Vertex3<A, B>>> for Ray<Point3<B>> {
    type Result = Option<(f32, Point3<B>)>;

    fn intersect(&self, tri: &Tri<Vertex3<A, B>>) -> Self::Result {
        let (_u, pt) = self.intersect(&tri.plane())?;

        let _vs = tri
            .0
            .each_ref()
            // TODO projection plane selection
            .map(|v| pt2::<f32, ()>(v.pos.x(), v.pos.y()))
            .map(|p| vertex(p, ()));

        let _p = pt2::<f32, ()>(pt.x(), pt.y());

        //let [s, t, r] = Tri(vs).cart_to_bary(&PLANE).0;
        //(s >= 0.0 && t >= 0.0 && r >= 0.0).then_some((u, pt))

        todo!("needs cart-to-bary")
    }
}

impl<B: Default> Intersect<BBox<B>> for Ray<Point3<B>> {
    type Result = RayIntersect3<B>; // Only closest for now

    fn intersect(&self, bbox @ &BBox(l, u): &BBox<B>) -> Self::Result {
        let planes: [Plane3<B>; 6] = [
            // Normal directions don't matter here
            Plane::new(1.0, 0.0, 0.0, l.x()),
            Plane::new(1.0, 0.0, 0.0, u.x()),
            Plane::new(0.0, 1.0, 0.0, l.y()),
            Plane::new(0.0, 1.0, 0.0, u.y()),
            Plane::new(0.0, 0.0, 1.0, l.z()),
            Plane::new(0.0, 0.0, 1.0, u.z()),
        ];
        planes
            .iter()
            .filter_map(|pl| self.intersect(pl))
            .filter(|(_, pt)| bbox.contains(pt))
            .min_by(|(t, _), (u, _)| t.total_cmp(&u))
    }
}

impl<B> Debug for LineIntersect<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Point(p) => write!(f, "{p:?}"),
            Coincident => f.write_str("coincident"),
        }
    }
}

#[cfg(test)]
mod tests {
    use re::assert_approx_eq;
    use re::math::{pt3, vec2, vec3};

    use super::*;

    const BBOX: BBox<()> = BBox(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    const PLANE: Plane3<()> = Plane::new(0.0, 1.0, 0.0, 2.0);

    fn pt2(x: f32, y: f32) -> Point2 {
        super::pt2(x, y)
    }

    mod ray_plane_3d {
        use super::*;
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
        fn ray_origin_coincident_with_plane_has_intersection() {
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
    }
    //
    // 3D ray - tri
    //

    mod ray_bbox_3d {
        use super::*;
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

    mod line_line_2d {
        use super::*;
        #[test]
        fn crossing() {
            let l1 = Line2::from(Edge(pt2(-1.0, -1.0), pt2(-5.0, -1.0)));
            let l2 = Line2::from(Edge(pt2(0.0, 2.0), pt2(2.0, 3.0)));

            assert_eq!(l1.intersect(&l2), Some(Point(pt2(-6.0, -1.0))));
        }
        #[test]
        fn coincident() {
            let l1 = Line2::from(Edge(pt2(-1.0, -1.0), pt2(-2.0, -3.0)));
            let l2 = Line2::from(Edge(pt2(0.0, 1.0), pt2(1.0, 3.0)));

            assert_eq!(l1.intersect(&l2), Some(Coincident));
        }
        #[test]
        fn parallel_disjoint() {
            let l1 = Line2::from(Ray(pt2(0.0, -1.0), vec2(1.0, 1.0)));
            let l2 = Line2::from(Ray(pt2(0.0, 1.0), vec2(1.0, 1.0)));

            assert_eq!(l1.intersect(&l2), None);
        }
    }
    //
    // 2D ray - line
    //

    #[ignore]
    #[test]
    fn ray_line_crossing() {
        todo!()
    }

    //
    // 2D ray - edge
    //

    #[ignore]
    #[test]
    fn ray_edge_crossing() {
        todo!()
    }

    mod edge_edge_2d {
        use super::*;
        #[test]
        fn intersection() {
            //   D   B
            //    \ /
            //    / \
            //   A   C
            let ab = Edge(pt2(0.0, 0.0), pt2(1.0, 1.0));
            let cd = Edge(pt2(1.0, 0.0), pt2(0.0, 1.0));

            assert_eq!(ab.intersect(&cd), Some(pt2(0.5, 0.5)));
            assert_eq!(cd.intersect(&ab), Some(pt2(0.5, 0.5)));
        }
        #[test]
        fn no_intersection() {
            //       C
            //  A---B \
            //         D
            let ab = Edge(pt2(0.0, 0.0), pt2(1.0, 0.0));
            let cd = Edge(pt2(1.0, 1.0), pt2(2.0, -1.0));

            // CD crosses the AB line
            assert_eq!(ab.intersect(&cd), None);
            // AB fully on one side of CD line
            assert_eq!(cd.intersect(&ab), None);
        }
        #[test]
        fn parallel_disjoint() {
            //       C----D
            //    A----B
            let ab = Edge(pt2(0.0, 0.0), pt2(2.0, 0.0));
            let cd = Edge(pt2(1.0, 1.0), pt2(3.0, 1.0));

            assert_eq!(ab.intersect(&cd), None);
            assert_eq!(cd.intersect(&ab), None);
        }
        #[test]
        fn collinear_disjoint() {
            //    A----B   C----D
            let ab = Edge(pt2(-2.0, 0.0), pt2(-1.0, 0.0));
            let cd = Edge(pt2(1.0, 0.0), pt2(2.0, 0.0));

            assert_eq!(ab.intersect(&cd), None);
            assert_eq!(cd.intersect(&ab), None);
        }
        #[test]
        fn parallel_shared_endpoint() {
            //    A----BC----D
            let ab = Edge(pt2(0.0, 0.0), pt2(1.0, 0.0));
            let cd = Edge(pt2(1.0, 0.0), pt2(2.0, 0.0));

            assert_eq!(ab.intersect(&cd), Some(pt2(1.0, 0.0)));
            assert_eq!(cd.intersect(&ab), Some(pt2(1.0, 0.0)));
        }
        #[test]
        fn nonparallel_shared_endpoint() {
            //            D
            //           /
            //    A----BC
            let e1 = Edge(pt2(0.0, 0.0), pt2(1.0, 0.0));
            let e2 = Edge(pt2(1.0, 0.0), pt2(2.0, 1.0));

            assert_approx_eq!(e1.intersect(&e2), Some(pt2(1.0, 0.0)));
            assert_approx_eq!(e2.intersect(&e1), Some(pt2(1.0, 0.0)));
        }
        #[test]
        fn collinear_coincident() {
            //    A---C---B---D
            let [a, b] = [pt2(0.0, 0.0), pt2(1.0, 0.0)];
            let [c, d] = [pt2(0.5, 0.0), pt2(2.0, 0.0)];
            let ab = Edge(a, b);
            let cd = Edge(c, d);

            assert_eq!(ab.intersect(&cd), Some(c));
            assert_eq!(cd.intersect(&ab), Some(c));
        }
    }
}
