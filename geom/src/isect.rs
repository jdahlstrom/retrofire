use core::{
    fmt::{Debug, Formatter, Write},
    ops::RangeBounds,
};

use std::{dbg, eprintln};

use re::geom::{Edge, Plane, Plane3, Ray, Tri, Vertex3, vertex};
use re::math::{
    Apply, ApproxEq, Linear, Mat2x2, Point2, Point3, Vec2, Vec3,
    mat::RealToReal, pt2, pt3, vec3,
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

// TODO invariant nonzero
// TODO decide if normalized form
// TODO move to core::geom, same methods as Plane
#[derive(Copy, Clone, PartialEq)]
pub struct Line2<B>(pub Vec3<B>);

impl<B> Debug for Line2<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        // ax + by = c
        let [a, b, c] = self.0.0;

        f.write_str("Line(")?;
        if b != 0.0 {
            // by = -ax + c  <=>  y = -a/b x + c/b
            let ab = -a / b; // slope
            let cb = c / b; // y intersect
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
    ///
    /// # Examples
    /// ```
    /// use re::{geom::Ray, math::{pt2, vec2, vec3}};
    /// use retrofire_geom::isect::Line2;
    ///
    /// let line = Line2::<()>::from(Ray(pt2(0.0, 0.0), vec2(1.0, 1.0)));
    /// assert_eq!(line.0, vec3(-1.0, 1.0, 0.0).normalize());
    /// ```
    fn from(Ray(p, d): Ray<Point2<B>>) -> Self {
        assert!(!d.approx_eq(&Vec2::zero())); // TODO vec::zero() and point::origin()?

        let n = d.perp(); // TODO imprecise without fp...
        let [a, b] = n.0;
        let c = p.to_vec().dot(&n);

        if c != 0.0 {
            Line2(vec3(a / c, b / c, 1.0))
        } else if b != 0.0 {
            Line2(vec3(a / b, 1.0, 0.0))
        } else if a != 0.0 {
            Line2(vec3(1.0, 0.0, 0.0))
        } else {
            panic!("degenerate line")
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub enum LineIntersect<B> {
    Point(Point2<B>),
    Coincident,
}

impl<B> Intersect<Self> for Line2<B> {
    type Result = Option<LineIntersect<B>>;

    /// Computes the intersection point of two 2-lines.
    ///
    /// # Examples
    /// ```
    /// use re::assert_approx_eq;
    /// use re::geom::Ray;
    /// use re::math::{pt2, vec2};
    ///
    /// use retrofire_geom::isect::{Intersect, Line2};
    ///
    /// let horiz = Line2::<()>::from(Ray(pt2(0.0, 2.0), vec2(1.0, 0.0)));
    /// let vert = Line2::from(Ray(pt2(3.0, 0.0), vec2(0.0, 1.0)));
    /// assert_approx_eq!(horiz.intersect(&vert), Some(pt2(3.0, 2.0)));
    ///
    /// let horiz2 = Line2::from(Ray(pt2(0.0, 3.0), vec2(1.0, 0.0)));
    /// assert_approx_eq!(horiz.intersect(&horiz2), None);
    ///
    /// assert_eq!(horiz.intersect(&horiz), Some(pt2(0.0, 2.0)));
    ///
    ///
    /// ```
    fn intersect(&self, other: &Self) -> Self::Result {
        let [a, b, c] = dbg!(self).0.0;
        let [d, e, f] = dbg!(other).0.0;

        dbg!([a, b, c]);
        dbg!([d, e, f]);

        if self.0 == other.0 {
            return Some(Coincident);
        }

        /*
        Otherwise, solve the system of equations for x and y:
            ax + by = c     // self
            dx + ey = f     // other

        Write in matrix form and solve:
            (a b) (x) = (c)
            (d e) (y)   (f)
                       -1
            (x) = (a b)   (c)
            (y)   (d e)   (f)
        */
        let abde: Mat2x2<RealToReal<2, B, B>> = mat![
            a, b;
            d, e;
        ];
        let i_abde = abde.checked_inverse()?;

        let res = i_abde.apply(&pt2(c, f));

        Some(Point(res))
    }
}

impl<B> Intersect<Edge<Point2<B>>> for Ray<Point2<B>> {
    type Result = RayIntersect2<B>;

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
    /// //      |      \
    /// //      |        v
    /// //   A==1=========X==B
    /// //      |
    /// // <-1--+----2----4--5->
    /// let ray: Ray<Point2> = Ray(pt2(2.0, 3.0), vec2(1.0, -1.0));
    /// let edge = Edge(pt2(-1.0, 1.0), pt2(5.0, 1.0));
    ///
    /// let (t, point) = ray.intersect(&edge).unwrap();
    /// assert_eq!(t, 2.0);
    /// assert_approx_eq!(point, pt2(4.0, 1.0));
    ///
    /// ```
    fn intersect(&self, edge: &Edge<Point2<B>>) -> Self::Result {
        // Compute ray-line intersection

        // TODO if degenerate ray, could return if ray origin on edge
        let ray_l = Line2::from(*self);
        let edge_l = Line2::from(*edge);

        let pt = ray_l.intersect(&edge_l)?;

        let Point(pt) = pt else {
            // Coincident

            let e0 = (edge.0 - self.0).scalar_project(&self.1);
            let e1 = (edge.1 - self.0).scalar_project(&self.1);

            if e0 < 0.0 && e1 < 0.0 {
                //   e------e   r---->
                return None;
            } else if 0.0 <= e0 && e0 <= e1 {
                //  r-----> e0----e1
                return Some((e0, edge.0));
            } else if 0.0 <= e1 && e1 <= e0 {
                //  r-----> e1----e0
                return Some((e1, edge.1));
            } else if e0 < 0.0 && 0.0 <= e1 {
                // e0---r--->--e1 or e0---r--e1-->
                return Some((0.0, self.0));
            } else if e1 < 0.0 && 0.0 <= e0 {
                // e1---r--->--e0 or e1---r--e0-->
                return Some((0.0, self.0));
            } else {
                unreachable!("???")
            }
        };

        let t = (pt - self.0).scalar_project(&self.1);

        eprintln!("line-line intersection: t={t} pt={:?}", pt.0);

        if t < -1e-6 {
            return None;
        }

        //Ray intersects the line of edge, but still have to check
        // the point is between edge.0 and edge.1

        let u = (pt - edge.0).scalar_project(&(edge.1 - edge.0));

        eprintln!("line-line intersection: t={t} pt={:?} u={u}", pt.0);

        if u < -1e-6 || u > 1.0 {
            return None;
        }

        Some((t, pt))
    }
}

impl<B> Intersect<Self> for Edge<Point2<B>> {
    type Result = Option<Point2<B>>;

    ///
    /// # Examples
    /// ```
    /// use re::assert_approx_eq;
    /// use re::geom::Edge;
    /// use re::math::{pt2, Point2};
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

        assert!(t >= -1e-6);

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
        if t < 0.0 {
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

        //let [s, t, r] = Tri(vs).cart_to_bary(&p).0;
        //(s >= 0.0 && t >= 0.0 && r >= 0.0).then_some((u, pt))

        todo!("needs cart-to-bary")
    }
}

impl<B: Default> Intersect<BBox<B>> for Ray<Point3<B>> {
    type Result = RayIntersect3<B>; // Only closest for now

    fn intersect(&self, bbox @ &BBox(l, u): &BBox<B>) -> Self::Result {
        #[rustfmt::skip]
        let planes: [Plane3<B>; 6] = [
            Plane::new(-1.0,  0.0,  0.0, l.x()),
            Plane::new( 1.0,  0.0,  0.0, u.x()),
            Plane::new( 0.0, -1.0,  0.0, l.y()),
            Plane::new( 0.0,  1.0,  0.0, u.y()),
            Plane::new( 0.0,  0.0, -1.0, l.z()),
            Plane::new( 0.0,  0.0,  1.0, u.z()),
        ];

        planes
            .iter()
            .filter_map(|p| self.intersect(p))
            .filter(|i| bbox.contains(&i.1))
            .min_by(|a, b| a.0.total_cmp(&b.0))
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

/*pub trait InRange<R: RangeBounds<Self>>: PartialOrd {
    fn in_range(&self, range: R) -> bool {
        range.contains(self)
    }
}
impl<T: PartialOrd, R: RangeBounds<T>> InRange<R> for T {}*/

#[cfg(test)]
mod tests {
    use re::assert_approx_eq;
    use re::math::{pt3, vec2, vec3};

    use super::*;

    fn plane() -> Plane3 {
        Plane3::from_point_and_normal(pt3(1.0, 2.0, 3.0), vec3(0.0, 1.0, 0.0))
    }

    //
    // 3D ray - plane
    //

    #[test]
    fn ray_towards_plane_has_intersection() {
        let p = plane();

        // Outside
        let r = Ray(pt3(0.0, 3.0, 0.0), vec3(1.0, -1.0, 1.0));
        assert_eq!(r.intersect(&p), Some((1.0, pt3(1.0, 2.0, 1.0))));

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

    fn pt2(x: f32, y: f32) -> Point2 {
        super::pt2(x, y)
    }

    //
    // 3D ray - tri
    //

    //
    // 3D ray - bbox
    //

    #[test]
    fn ray_bbox() {
        todo!()
    }

    //
    // 2D line - line
    //

    #[test]
    fn line_line_crossing() {
        let l1 = Line2::from(Edge(pt2(-1.0, -1.0), pt2(-5.0, -1.0)));
        let l2 = Line2::from(Edge(pt2(0.0, 2.0), pt2(2.0, 3.0)));

        assert_eq!(l1.intersect(&l2), Some(Point(pt2(-6.0, -1.0))));
    }
    #[test]
    fn line_line_coincident() {
        let l1 = Line2::from(Edge(pt2(-1.0, -1.0), pt2(-2.0, -3.0)));
        let l2 = Line2::from(Edge(pt2(0.0, 1.0), pt2(1.0, 3.0)));

        assert_eq!(l1.intersect(&l2), Some(Coincident));
    }
    #[test]
    fn line_line_disjoint() {
        let l1 = Line2::from(Ray(pt2(0.0, -1.0), vec2(1.0, 1.0)));
        let l2 = Line2::from(Ray(pt2(0.0, 1.0), vec2(1.0, 1.0)));

        assert_eq!(l1.intersect(&l2), None);
    }

    //
    // 2D ray - line
    //

    #[test]
    fn ray_line_crossing() {
        todo!()
    }

    //
    // 2D ray - edge
    //

    #[test]
    fn ray_edge_crossing() {
        todo!()
    }

    //
    // 2D edge - edge
    //

    #[test]
    fn edge_edge_intersection() {
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
    fn edge_edge_no_intersection() {
        //       C
        //  A---B \
        //         D
        let ab = Edge(pt2(0.0, 0.0), pt2(1.0, 0.0));
        let cd = Edge(pt2(1.0, 1.0), pt2(2.0, -1.0));

        assert_eq!(ab.intersect(&cd), None);
        assert_eq!(cd.intersect(&ab), None);
    }
    #[test]
    fn edge_edge_parallel_disjoint() {
        //       C----D
        //    A----B
        let ab = Edge(pt2(0.0, 0.0), pt2(2.0, 0.0));
        let cd = Edge(pt2(1.0, 1.0), pt2(3.0, 1.0));

        assert_eq!(ab.intersect(&cd), None);
        assert_eq!(cd.intersect(&ab), None);
    }
    #[test]
    fn edge_edge_collinear_disjoint() {
        //    D----C   A----B
        let ab = Edge(pt2(0.0, 0.0), pt2(1.0, 0.0));
        let cd = Edge(pt2(-1.0, 0.0), pt2(-2.0, 0.0));

        assert_eq!(ab.intersect(&cd), None);
        assert_eq!(cd.intersect(&ab), None);
    }
    #[test]
    fn edge_edge_parallel_shared_endpoint() {
        //    A----BC----D
        let ab = Edge(pt2(0.0, 0.0), pt2(1.0, 0.0));
        let cd = Edge(pt2(1.0, 0.0), pt2(2.0, 0.0));

        assert_eq!(ab.intersect(&cd), Some(pt2(1.0, 0.0)));
        assert_eq!(cd.intersect(&ab), Some(pt2(1.0, 0.0)));
    }
    #[test]
    fn edge_edge_nonparallel_shared_endpoint() {
        //            D
        //           /
        //    A----BC
        let e1 = Edge(pt2(0.0, 0.0), pt2(1.0, 0.0));
        let e2 = Edge(pt2(1.0, 0.0), pt2(2.0, 1.0));

        assert_approx_eq!(e1.intersect(&e2), Some(pt2(1.0, 0.0)));
        assert_approx_eq!(e2.intersect(&e1), Some(pt2(1.0, 0.0)));
    }
    #[test]
    fn edge_edge_collinear_coincident() {
        //    A---C---B---D
        let [a, b] = [pt2(0.0, 0.0), pt2(1.0, 0.0)];
        let [c, d] = [pt2(0.5, 0.0), pt2(2.0, 0.0)];
        let ab = Edge(a, b);
        let cd = Edge(c, d);

        assert_eq!(ab.intersect(&cd), Some(c));
        assert_eq!(cd.intersect(&ab), Some(c));
    }
}
