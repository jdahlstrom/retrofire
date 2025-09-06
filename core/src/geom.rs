//! Basic geometric primitives.

use alloc::vec::Vec;

use crate::math::{
    Affine, Lerp, Linear, Mat4x4, Parametric, Point2, Point3, Vec2, Vec3, Vec4,
    Vector, mat::RealToReal, space::Real, vec3,
};
use crate::render::Model;

pub use mesh::Mesh;

pub mod mesh;

/// Vertex with a position and arbitrary other attributes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Vertex<P, A> {
    pub pos: P,
    pub attrib: A,
}

/// Two-dimensional vertex type.
pub type Vertex2<A, B = Model> = Vertex<Point2<B>, A>;

/// Three-dimensional vertex type.
pub type Vertex3<A, B = Model> = Vertex<Point3<B>, A>;

/// Triangle, defined by three vertices.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct Tri<V>(pub [V; 3]);

/// Plane, defined by the four parameters of the plane equation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct Plane<V>(pub(crate) V);

/// Plane embedded in 3D space, splitting the space into two half-spaces.
pub type Plane3<B = ()> = Plane<Vector<[f32; 4], Real<3, B>>>;

/// A ray, or a half line, composed of an initial point and a direction vector.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Ray<T: Affine>(pub T, pub T::Diff);

/// A curve composed of a chain of line segments.
///
/// The polyline is represented as a list of points, or vertices, with each
/// pair of consecutive vertices sharing an edge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Polyline<T>(pub Vec<T>);

/// A line segment between two vertices.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Edge<T>(pub T, pub T);

/// A surface normal in 3D.
// TODO Use distinct type rather than alias
pub type Normal3 = Vec3;
/// A surface normal in 2D.
pub type Normal2 = Vec2;

/// Polygon winding order.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum Winding {
    /// Clockwise winding.
    Cw,
    /// Counter-clockwise winding.
    #[default]
    Ccw,
}

/// Creates a `Vertex` with the give position and attribute values.
pub const fn vertex<P, A>(pos: P, attrib: A) -> Vertex<P, A> {
    Vertex { pos, attrib }
}

/// Creates a `Tri` with the given vertices.
pub const fn tri<V>(a: V, b: V, c: V) -> Tri<V> {
    Tri([a, b, c])
}

// Inherent impls

impl<B> Plane<Vector<[f32; 4], Real<3, B>>> {
    /// The x = 0 coordinate plane.
    pub const YZ: Self = Self(Vector::new([1.0, 0.0, 0.0, 0.0]));

    /// The y = 0 coordinate plane.
    pub const XZ: Self = Self(Vector::new([0.0, 1.0, 1.0, 0.0]));

    /// The z = 0 coordinate plane.
    pub const XY: Self = Self(Vector::new([0.0, 0.0, 1.0, 0.0]));
}

impl<A, B> Tri<Vertex2<A, B>> {
    /// Given a triangle ABC, returns the vectors AB, AC.
    pub fn tangents(&self) -> [Vec2<B>; 2] {
        let [a, b, c] = &self.0;
        [b.pos - a.pos, c.pos - a.pos]
    }

    /// Returns the winding order of `self`.
    pub fn winding(&self) -> Winding {
        let [t, u] = self.tangents();
        if t.perp_dot(u) < 0.0 {
            Winding::Cw
        } else {
            Winding::Ccw
        }
    }

    /// Returns the signed area of `self`.
    ///
    /// The area is positive if
    pub fn signed_area(&self) -> f32 {
        let [t, u] = self.tangents();
        t.perp_dot(u) / 2.0
    }

    /// Returns the (positive) area of `self`.
    pub fn area(&self) -> f32 {
        self.signed_area().abs()
    }
}

impl<A, B> Tri<Vertex3<A, B>> {
    /// Given a triangle ABC, returns the edges [AB, BC, CA].
    pub fn edges(&self) -> [Edge<Vertex3<A, B>>; 3]
    where
        Vertex3<A, B>: Clone,
    {
        let [a, b, c] = &self.0;
        [
            Edge(a.clone(), b.clone()),
            Edge(b.clone(), c.clone()),
            Edge(c.clone(), a.clone()),
        ]
    }

    /// Given a triangle ABC, returns the vectors [AB, AC].
    ///
    /// # Examples
    /// ```
    ///
    /// ```
    pub fn tangents(&self) -> [Vec3<B>; 2] {
        let [a, b, c] = &self.0;
        [b.pos - a.pos, c.pos - a.pos]
    }

    /// Returns the normal vector of `self`.
    ///
    /// The result is normalized to unit length.
    ///
    /// # Examples
    /// ```
    /// ```
    pub fn normal(&self) -> Normal3 {
        let [t, u] = self.tangents();
        // TODO normal with basis
        t.cross(&u).normalize().to()
    }

    /// Returns the plane that `self` lies on.
    ///
    /// # Examples
    /// ```
    /// let tri = Tri([
    ///     vertex(vec3(0.0, 2.0, 0.0), ()),
    ///     vertex(vec3(1.0, 2.0, 0.0), ()),
    ///     vertex(vec3(0.0, 2.0, 1.0), ())
    /// ]);
    /// assert_eq!(tri.plane(), Plane(vec4(0.0, 1.0, 0.0, 2.0)));
    /// ```
    pub fn plane(&self) -> Plane3<B> {
        Plane::from_points(self.0.each_ref().map(|v| v.pos))
    }

    /// Returns the winding order of `self`, as projected to the XY plane.
    // TODO is this 3D version meaningful/useful enough?
    pub fn winding(&self) -> Winding {
        let z = self.normal().z();
        if z < 0.0 { Winding::Cw } else { Winding::Ccw }
    }

    /// Returns the signed area of `self`.
    pub fn signed_area(&self) -> f32 {
        let [t, u] = self.tangents();
        t.cross(&u).len() / 2.0
    }
    /// Returns the area of `self`.
    pub fn area(&self) -> f32 {
        self.signed_area().abs()
    }
}

impl<B> Plane3<B> {
    /// Creates a plane given three points on the plane.
    ///
    /// # Panics
    /// If the points are collinear or nearly so.
    pub fn from_points([a, b, c]: [Point3<B>; 3]) -> Self {
        let n = (b - a).cross(&(c - a)).to();
        Self::from_point_and_normal(a, n)
    }
    /// Creates a plane given a point on the plane and a normal.
    ///
    /// `n` does not have to be normalized.
    ///
    /// # Panics
    /// If `n` is non-finite or nearly zero-length.
    pub fn from_point_and_normal(pt: Point3<B>, n: Normal3) -> Self {
        let n = n.normalize();
        // For example, if pt = (0, 1, 0) and n = (0, 1, 0), d has to be -1
        // to satisfy the plane equation n_x + n_y + n_z + d = 0
        let d = -pt.to_vec().dot(&n.to());
        Plane(Vector::new([n.x(), n.y(), n.z(), d]))
    }

    /// Returns the normal vector of this plane.
    ///
    /// The normal returned is unit length.
    #[inline]
    pub fn normal(&self) -> Normal3 {
        let [x, y, z, _] = self.0.0;
        vec3(x, y, z)
    }

    /*
        p = (a ,b, c, w)

        ax + by + cz + w = 0

        x = ox + t*dx
        y = oy + t*dy
        z = oz + t*dz

        a(ox + t*dx) + b(oy + t*dy) + c(oz + t*dz) + w(ow + t*dw) = 0

        p dot (o + t*d) = p dot o + p dot t*d = p dot o + t p dot d

        t = -(p dot o) / (p dot d)

        - or -

        aox + at*dx + boy + bt*dy + coz + ct*dz + w ow + wt*dw = 0

        at*dx + bt*dy + ct*dz + wt*dw = -(aox + boy + coz + w ow)

        t (a dx + b dy + c dz + w dw) = -(a ox + b oy + c oz + w ow)

        t = -(a ox + b oy + c oz + w ow) / (a dx + b dy + c dz + w dw)

          = - p dot o / p dot d


    */

    /// Returns the perpendicular projection of a point on the plane.
    ///
    /// In other words, returns the point on the plane closest to `pt`.
    pub fn project(&self, pt: Point3<B>) -> Point3<B> {
        // t = -(plane dot orig) / (plane dot dir)
        // In this special case plane dot dir == 1

        let p = self.0;
        // TODO use to_homog once committed
        let o = Vector::new([pt.x(), pt.y(), pt.z(), 1.0]);
        let d = self.normal().to();
        let t = -p.dot(&o);

        pt + t * d
    }

    /// Returns whether a point is in the half-space that the normal points away from.
    pub fn is_inside(&self, pt: Point3<B>) -> bool {
        // TODO use to_homog once committed
        let pt: Vec4 = [pt.x(), pt.y(), pt.z(), 1.0].into();
        self.0.dot(&pt.to()) <= 0.0
    }

    /// Returns an orthonormal affine basis on the plane.
    ///
    /// The y-axis of the basis is the normal vector; the x- and z-axes are
    /// two arbitrary orthogonal unit vectors tangent to the plane. The origin
    /// point is the point on the plane closest to the origin.
    pub fn basis<T>(&self) -> Mat4x4<RealToReal<3, B, T>> {
        let y = self.normal();

        let x = if y.x().abs() < y.y().abs() && y.x().abs() < y.z().abs() {
            Vec3::X
        } else {
            Vec3::Z
        };
        let z = x.cross(&y).normalize();
        let x = y.cross(&z);

        let t = self.0[3] * y;
        [
            [x.x(), y.x(), z.x(), t.x()],
            [x.y(), y.y(), z.y(), t.y()],
            [x.z(), y.z(), z.z(), t.z()],
            [0.0, 0.0, 0.0, 1.0],
        ]
        .into()
    }
}

impl<T> Polyline<T> {
    pub fn new(verts: impl IntoIterator<Item = T>) -> Self {
        Self(verts.into_iter().collect())
    }

    /// Returns an iterator over the line segments of `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{
    ///     geom::{Polyline, Edge},
    ///     math::{pt2, Point2}
    /// };
    ///
    /// let points = [pt2(0.0, 0.0), pt2(1.0, 1.0), pt2(2.0, 1.0)];
    ///
    /// let pl = Polyline::<Point2>::new(points);
    /// let mut edges = pl.edges();
    ///
    /// assert_eq!(edges.next(), Some(Edge(points[0], points[1])));
    /// assert_eq!(edges.next(), Some(Edge(points[1], points[2])));
    /// assert_eq!(edges.next(), None);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = Edge<T>> + '_
    where
        T: Clone,
    {
        self.0
            .windows(2)
            .map(|e| Edge(e[0].clone(), e[1].clone()))
    }
}

// Local trait impls

impl<T> Parametric<T> for Ray<T>
where
    T: Affine<Diff: Linear<Scalar = f32>>,
{
    fn eval(&self, t: f32) -> T {
        self.0.add(&self.1.mul(t))
    }
}

impl<T: Lerp + Clone> Parametric<T> for Polyline<T> {
    /// Returns the point on `self` at `t`.
    ///
    /// If the number of vertices in `self` is `n`, the vertex at index
    /// `k` < `n` corresponds to `t` = `k` / (`n` - 1). Intermediate values
    /// of `t` are linearly interpolated between the two closest vertices.
    /// Values `t` < 0 and `t` > 1 are clamped to 0 and 1 respectively.
    ///
    /// # Panics
    /// If `self` has no vertices.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{
    ///     geom::{Polyline, Edge},
    ///     math::{pt2, Point2, Parametric}
    /// };
    ///
    /// let pl = Polyline::<Point2>(
    ///     vec![pt2(0.0, 0.0), pt2(1.0, 2.0), pt2(2.0, 1.0)]);
    ///
    /// assert_eq!(pl.eval(0.0), pl.0[0]);
    /// assert_eq!(pl.eval(0.5), pl.0[1]);
    /// assert_eq!(pl.eval(1.0), pl.0[2]);
    ///
    /// // Values not corresponding to a vertex are interpolated:
    /// assert_eq!(pl.eval(0.25), pt2(0.5, 1.0));
    /// assert_eq!(pl.eval(0.75), pt2(1.5, 1.5));
    ///
    /// // Values of t outside 0.0..=1.0 are clamped:
    /// assert_eq!(pl.eval(-1.23), pl.eval(0.0));
    /// assert_eq!(pl.eval(7.68), pl.eval(1.0));
    /// ```
    fn eval(&self, t: f32) -> T {
        assert!(self.0.len() > 0, "cannot eval an empty polyline");

        let max = self.0.len() as f32 - 1.0;
        let i = 0.0.lerp(&max, t.clamp(0.0, 1.0));
        let t_rem = i % 1.0;
        let i = i as usize;

        if i == max as usize {
            self.0[i].clone()
        } else {
            let p0 = &self.0[i];
            let p1 = &self.0[i + 1];
            p0.lerp(p1, t_rem)
        }
    }
}

impl<P: Lerp, A: Lerp> Lerp for Vertex<P, A> {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        vertex(
            self.pos.lerp(&other.pos, t),
            // TODO Normals shouldn't be lerped
            self.attrib.lerp(&other.attrib, t),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;
    use crate::math::*;
    use alloc::vec;

    use super::*;

    type Pt<const N: usize> = Point<[f32; N], Real<N>>;

    fn tri<const N: usize>(
        a: Pt<N>,
        b: Pt<N>,
        c: Pt<N>,
    ) -> Tri<Vertex<Pt<N>, ()>> {
        Tri([a, b, c].map(|p| vertex(p, ())))
    }

    #[test]
    fn triangle_winding_2_cw() {
        let tri = tri(pt2(-1.0, -1.0), pt2(1.0, 1.0), pt2(1.0, 0.0));
        assert_eq!(tri.winding(), Winding::Cw);
    }
    #[test]
    fn triangle_winding_2_ccw() {
        let tri = tri(pt2(-2.0, 0.0), pt2(1.0, 0.0), pt2(0.0, 1.0));
        assert_eq!(tri.winding(), Winding::Ccw);
    }
    #[test]
    fn triangle_winding_3_cw() {
        let tri =
            tri(pt3(0.0, 0.0, 0.0), pt3(1.0, 0.0, 0.0), pt3(0.0, 1.0, 0.0));
        assert_eq!(tri.winding(), Winding::Ccw);
    }
    #[test]
    fn triangle_winding_3_ccw() {
        let tri =
            tri(pt3(0.0, 0.0, 0.0), pt3(1.0, 0.0, 0.0), pt3(0.0, 1.0, 0.0));
        assert_eq!(tri.winding(), Winding::Ccw);
    }

    #[test]
    fn triangle_area_2() {
        let tri = tri(pt2(-1.0, 0.0), pt2(2.0, 0.0), pt2(2.0, 1.0));
        assert_eq!(tri.area(), 1.5);
    }
    #[test]
    fn triangle_area_3() {
        // base = 3, height = 2
        let tri = tri(
            pt3(-1.0, 0.0, -1.0),
            pt3(2.0, 0.0, -1.0),
            pt3(0.0, 0.0, 1.0),
        );
        assert_eq!(tri.area(), 3.0);
    }

    #[test]
    fn triangle_plane() {
        let tri = tri(
            pt3(-1.0, -2.0, -1.0),
            pt3(2.0, -2.0, -1.0),
            pt3(0.0, -2.0, 1.0),
        );
        assert_eq!(tri.plane(), Plane([0.0, -1.0, 0.0, -2.0].into()));
    }

    #[test]
    fn plane_from_points() {
        let p = Plane3::<()>::from_points([
            pt3(1.0, 0.0, 0.0),
            pt3(0.0, 1.0, 0.0),
            pt3(0.0, 0.0, 1.0),
        ]);

        assert_eq!(p.normal(), vec3(1.0, 1.0, 1.0).normalize());
        assert_eq!(p.0[3], -f32::sqrt(1.0 / 3.0));
    }
    #[test]
    #[should_panic]
    fn plane_from_collinear_points_panics() {
        Plane3::<()>::from_points([
            pt3(1.0, 2.0, 3.0),
            pt3(-2.0, -4.0, -6.0),
            pt3(0.5, 1.0, 1.5),
        ]);
    }
    #[test]
    #[should_panic]
    fn plane_from_zero_normal_panics() {
        Plane3::<()>::from_point_and_normal(
            pt3(1.0, 2.0, 3.0),
            vec3(0.0, 0.0, 0.0),
        );
    }
    #[test]
    fn plane_from_point_and_normal() {
        let p = Plane3::<()>::from_point_and_normal(
            pt3(1.0, 2.0, -3.0),
            vec3(0.0, 0.0, 12.3),
        );
        assert_eq!(p.normal(), vec3(0.0, 0.0, 1.0));
        assert_eq!(p.0[3], 3.0);
    }
    #[test]
    fn plane_is_point_inside_xz() {
        let p: Plane3 =
            Plane::from_point_and_normal(pt3(1.0, 2.0, 3.0), Vec3::Y);

        // Inside
        assert!(p.is_inside(pt3(0.0, 0.0, 0.0)));
        // Coincident=inside
        assert!(p.is_inside(pt3(0.0, 2.0, 0.0)));
        assert!(p.is_inside(pt3(1.0, 2.0, 3.0)));
        // Outside
        assert!(!p.is_inside(pt3(0.0, 3.0, 0.0)));
        assert!(!p.is_inside(pt3(1.0, 3.0, 3.0)));
    }
    #[test]
    fn plane_is_point_inside_neg_xz() {
        let p: Plane3 =
            Plane::from_point_and_normal(pt3(1.0, 2.0, 3.0), -Vec3::Y);

        // Outside
        assert!(!p.is_inside(pt3(0.0, 0.0, 0.0)));
        // Coincident=inside
        assert!(p.is_inside(pt3(0.0, 2.0, 0.0)));
        assert!(p.is_inside(pt3(1.0, 2.0, 3.0)));
        // Inside
        assert!(p.is_inside(pt3(0.0, 3.0, 0.0)));
        assert!(p.is_inside(pt3(1.0, 3.0, 3.0)));
    }
    #[test]
    fn plane_is_point_inside_diagonal() {
        let p: Plane3 =
            Plane::from_point_and_normal(pt3(0.0, 1.0, 0.0), splat(1.0));

        // Inside
        assert!(p.is_inside(pt3(0.0, 0.0, 0.0)));
        assert!(p.is_inside(pt3(-1.0, 1.0, -1.0)));
        // Coincident=inside
        assert!(p.is_inside(pt3(0.0, 1.0, 0.0)));
        // Outside
        assert!(!p.is_inside(pt3(0.0, 2.0, 0.0)));
        assert!(!p.is_inside(pt3(1.0, 1.0, 1.0)));
        assert!(!p.is_inside(pt3(1.0, 0.0, 1.0)));
    }

    #[test]
    fn plane_project_point() {
        let p: Plane3 =
            Plane3::from_point_and_normal(pt3(0.0, 2.0, 0.0), Vec3::Y);

        assert_eq!(p.project(pt3(5.0, 10.0, -3.0)), pt3(5.0, 2.0, -3.0));
        assert_eq!(p.project(pt3(5.0, -10.0, -3.0)), pt3(5.0, 2.0, -3.0));
    }

    #[test]
    fn polyline_eval_f32() {
        let pl = Polyline(vec![0.0, 1.0, -0.5]);

        assert_eq!(pl.eval(-5.0), 0.0);
        assert_eq!(pl.eval(0.00), 0.0);
        assert_eq!(pl.eval(0.25), 0.5);
        assert_eq!(pl.eval(0.50), 1.0);
        assert_eq!(pl.eval(0.75), 0.25);
        assert_eq!(pl.eval(1.00), -0.5);
        assert_eq!(pl.eval(5.00), -0.5);
    }

    #[test]
    #[should_panic]
    fn empty_polyline_eval() {
        Polyline::<f32>(vec![]).eval(0.5);
    }

    #[test]
    fn singleton_polyline_eval() {
        let pl = Polyline(vec![3.14]);
        assert_eq!(pl.eval(0.0), 3.14);
        assert_eq!(pl.eval(1.0), 3.14);
    }
}
