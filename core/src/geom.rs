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

/// A closed curve composed of a chain of line segments.
///
/// The polygon is represented as a list of points, or vertices, with each pair
/// of consecutive vertices, as well as the first and last vertex, sharing an edge.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Polygon<T>(pub Vec<T>);

/// A line segment between two vertices.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Edge<T>(pub T, pub T);

/// A surface normal in 3D.
// TODO Use distinct type rather than alias
pub type Normal3 = Vec3;
/// A surface normal in 2D.
pub type Normal2 = Vec2;

/// Polygon winding order.
///
/// The triangle *ABC* below has clockwise winding, while
/// the triangle *DEF* has counter-clockwise winding.
///
/// ```text
///     B            F
///    / \          / \
///   /   \        /   \
///  /     \      /     \
/// A-------C    D-------E
///    Cw           Ccw
/// ```
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
    /// The area is positive if `self` is wound counter-clockwise.
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
    /// use retrofire_core::geom::{Tri, Plane3, vertex};
    /// use retrofire_core::math::pt3;
    ///
    /// let tri = Tri([
    ///     vertex(pt3::<f32, ()>(0.0, 0.0, 2.0), ()),
    ///     vertex(pt3(1.0, 0.0, 2.0), ()),
    ///     vertex(pt3(0.0, 1.0, 2.0), ())
    /// ]);
    /// assert_eq!(tri.plane(), Plane3::new(0.0, 0.0, 1.0, -2.0));
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
    #[cfg(feature = "fp")]
    pub fn signed_area(&self) -> f32 {
        let [t, u] = self.tangents();
        t.cross(&u).len() / 2.0
    }
    /// Returns the area of `self`.
    #[cfg(feature = "fp")]
    pub fn area(&self) -> f32 {
        self.signed_area().abs()
    }
}

impl<B> Plane3<B> {
    /// The x = 0 coordinate plane.
    pub const YZ: Self = Self::new(1.0, 0.0, 0.0, 0.0);

    /// The y = 0 coordinate plane.
    pub const XZ: Self = Self::new(0.0, 1.0, 0.0, 0.0);

    /// The z = 0 coordinate plane.
    pub const XY: Self = Self::new(0.0, 0.0, 1.0, 0.0);

    /// Creates a new plane with the given coefficients.
    ///
    // TODO not normalized because const
    // The coefficients are normalized to
    //
    // (a', b', c', d') = (a, b, c, d) / |(a, b, c)|.
    ///
    /// The returned plane satisfies the plane equation
    ///
    /// *ax* + *by* + *cz* + *d* = 0,
    ///
    /// or equivalently
    ///
    /// *ax* + *by* + *cz* = -*d*.
    ///
    /// Note the sign of the *d* coefficient.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{geom::Plane3, math::{pt3, vec3}};
    ///
    /// let p = <Plane3>::new(0.0, 0.0, 1.0, -2.0);
    /// assert_eq!(p.normal(), vec3(0.0, 0.0, 1.0));
    /// assert_eq!(p.offset(), 2.0);
    ///
    /// let p = <Plane3>::new(0.0, -2.0, 0.0, 1.0);
    /// assert_eq!(p.normal(), vec3(0.0, -1.0, 0.0));
    /// assert_eq!(p.offset(), -0.5);
    /// ```
    #[inline]
    pub const fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Plane(Vector::new([a, b, c, d]))
    }

    /// Creates a plane given three points on the plane.
    ///
    /// # Panics
    /// If the points are collinear or nearly so.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{geom::Plane3, math::{pt3, vec3}};
    ///
    /// let p = <Plane3>::from_points([
    ///     pt3(0.0, 0.0, 2.0),
    ///     pt3(1.0, 0.0, 2.0),
    ///     pt3(0.0, 1.0, 2.0),
    /// ]);
    /// assert_eq!(p, <Plane3>::new(0.0, 0.0, 1.0, -2.0));
    /// assert_eq!(p.normal(), vec3(0.0, 0.0, 1.0));
    /// assert_eq!(p.offset(), 2.0);
    ///
    /// ```
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
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{geom::Plane3, math::{Vec3, pt3, vec3}};
    ///
    /// let p = <Plane3>::from_point_and_normal(pt3(1.0, 2.0, 3.0), Vec3::Z);
    /// assert_eq!(p, <Plane3>::new(0.0, 0.0, 1.0, -3.0));
    /// assert_eq!(p.normal(), Vec3::Z);
    /// assert_eq!(p.offset(), 3.0);
    ///
    /// ```
    pub fn from_point_and_normal(pt: Point3<B>, n: Normal3) -> Self {
        let n = n.normalize();
        // For example, if pt = (0, 1, 0) and n = (0, 1, 0), d has to be -1
        // to satisfy the plane equation n_x + n_y + n_z + d = 0
        let d = -pt.to_vec().dot(&n.to());
        Self::new(n.x(), n.y(), n.z(), d)
    }

    /// Returns the normal vector of `self`.
    ///
    /// The normal returned is unit length.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{geom::Plane3, math::Vec3};
    ///
    /// assert_eq!(<Plane3>::XY.normal(), Vec3::Z);
    /// assert_eq!(<Plane3>::YZ.normal(), Vec3::X);
    #[inline]
    pub fn normal(&self) -> Normal3 {
        let [a, b, c, _] = self.0.0;
        vec3(a, b, c).normalize()
    }

    /// Returns the signed distance of `self` from the origin.
    ///
    /// This distance is negative if the origin is [*outside*][Self::is_inside]
    /// the plane and positive if the origin is *inside* the plane.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{geom::Plane3, math::{Vec3, pt3}};
    ///
    /// assert_eq!(<Plane3>::new(0.0, 1.0, 0.0, 3.0).offset(), -3.0);
    /// assert_eq!(<Plane3>::new(0.0, 2.0, 0.0, 6.0).offset(), -3.0);
    /// assert_eq!(<Plane3>::new(0.0, -1.0, 0.0, -3.0).offset(), 3.0);
    /// ```
    #[cfg(feature = "fp")]
    #[inline]
    pub fn offset(&self) -> f32 {
        let [a, b, c, d] = self.0.0;
        return -d / <Vec3>::new([a, b, c]).len();
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

    /// Returns the perpendicular projection of a point on `self`.
    ///
    /// In other words, returns *P'*, the point on the plane closest to *P*.
    ///
    /// ```text
    ///          ^        P
    ///         /         ·
    ///        /          ·
    ///       / · · · · · P'
    ///      /           ·
    ///     /           ·
    ///    O------------------>
    /// ```
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::Plane3;
    /// use retrofire_core::math::{Point3, pt3};
    ///
    /// let pt: Point3 = pt3(1.0, 2.0, -3.0);
    ///
    /// assert_eq!(<Plane3>::XZ.project(pt), pt3(1.0, 0.0, -3.0));
    /// assert_eq!(<Plane3>::XY.project(pt), pt3(1.0, 2.0, 0.0));
    ///
    /// assert_eq!(<Plane3>::new(0.0, 0.0, 1.0, 2.0).project(pt), pt3(1.0, 2.0, -2.0));
    /// assert_eq!(<Plane3>::new(0.0, 0.0, 2.0, 2.0).project(pt), pt3(1.0, 2.0, -1.0));
    /// ```
    pub fn project(&self, pt: Point3<B>) -> Point3<B> {
        // t = -(plane dot orig) / (plane dot dir)
        // In this special case plane dot dir == 1
        let d = self.normal().to();
        let t = -self.signed_dist(pt);
        pt + t * d
    }

    /// Returns the signed distance of a point to `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::Plane3;
    /// use retrofire_core::math::{Point3, pt3, Vec3};
    ///
    /// let pt: Point3 = pt3(1.0, 2.0, -3.0);
    ///
    /// assert_eq!(<Plane3>::XZ.signed_dist(pt), 2.0);
    /// assert_eq!(<Plane3>::XY.signed_dist(pt), -3.0);
    ///
    /// let p = <Plane3>::new(-2.0, 0.0, 0.0, 4.0);
    /// assert_eq!(p.normal(), -Vec3::X);
    /// assert_eq!(p.offset(), -2.0);
    /// assert_eq!(p.signed_dist(pt), 1.0);
    /// ```
    #[cfg(feature = "fp")]
    #[inline]
    pub fn signed_dist(&self, pt: Point3<B>) -> f32 {
        // TODO use to_homog once committed
        let pt = Vector::new([pt.x(), pt.y(), pt.z(), 1.0]);
        self.0.dot(&pt)
    }

    /// Returns whether a point is in the half-space that the normal of `self`
    /// points away from.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::Plane3;
    /// use retrofire_core::math::{Point3, pt3};
    ///
    /// let pt: Point3 = pt3(1.0, 2.0, -3.0);
    ///
    /// assert!(!<Plane3>::XZ.is_inside(pt));
    /// assert!(<Plane3>::XY.is_inside(pt));
    /// ```
    // TODO "plane.is_inside(point)" reads wrong
    #[cfg(feature = "fp")]
    #[inline]
    pub fn is_inside(&self, pt: Point3<B>) -> bool {
        self.signed_dist(pt) <= 0.0
    }

    /// Returns an orthonormal affine basis on `self`.
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
    /// Creates a new polyline from an iterator of vertex points.
    pub fn new(verts: impl IntoIterator<Item = T>) -> Self {
        Self(verts.into_iter().collect())
    }

    /// Returns an iterator over the line segments of `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Polyline, Edge};
    /// use retrofire_core::math::{pt2, Point2};
    ///
    /// let pts: [Point2; _] = [pt2(0.0, 0.0), pt2(1.0, 1.0), pt2(2.0, 1.0)];
    ///
    /// let pline = Polyline::new(pts);
    /// let mut edges = pline.edges();
    ///
    /// assert_eq!(edges.next(), Some(Edge(&pts[0], &pts[1])));
    /// assert_eq!(edges.next(), Some(Edge(&pts[1], &pts[2])));
    /// assert_eq!(edges.next(), None);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = Edge<&T>> + '_ {
        self.0.windows(2).map(|e| Edge(&e[0], &e[1]))
    }
}

impl<T> Polygon<T> {
    /// Creates a new polygon from an iterator of vertex points.
    pub fn new(verts: impl IntoIterator<Item = T>) -> Self {
        Self(verts.into_iter().collect())
    }

    /// Returns an iterator over the edges of `self`.
    ///
    /// Given a polygon ABC...XYZ, returns the edges AB, BC, ..., XY, YZ, ZA.
    /// If `self` has zero or one vertices, returns an empty iterator.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Polygon, Edge};
    /// use retrofire_core::math::{Point2, pt2};
    ///
    /// let pts: [Point2; _] = [pt2(0.0, 0.0), pt2(1.0, 1.0), pt2(2.0, 1.0)];
    ///
    /// let poly = Polygon::new(pts);
    /// let mut edges = poly.edges();
    ///
    /// assert_eq!(edges.next(), Some(Edge(&pts[0], &pts[1])));
    /// assert_eq!(edges.next(), Some(Edge(&pts[1], &pts[2])));
    /// assert_eq!(edges.next(), Some(Edge(&pts[2], &pts[0])));
    /// assert_eq!(edges.next(), None);
    /// ```
    pub fn edges(&self) -> impl Iterator<Item = Edge<&T>> + '_ {
        let last_first = if let [f, .., l] = &self.0[..] {
            Some(Edge(l, f))
        } else {
            None
        };
        self.0
            .windows(2)
            .map(|e| Edge(&e[0], &e[1]))
            .chain(last_first)
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
    /// Returns the point on `self` at *t*.
    ///
    /// If the number of vertices in `self` is *n* > 1, the vertex at index
    /// *k* < *n* corresponds to `t` = *k* / (*n* - 1). Intermediate values
    /// of *t* are linearly interpolated between the two closest vertices.
    /// Values *t* < 0 and *t* > 1 are clamped to 0 and 1 respectively.
    /// A polyline with a single vertex returns the value of that vertex
    /// for any value of *t*.
    ///
    /// # Panics
    /// If `self` has no vertices.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Polyline, Edge};
    /// use retrofire_core::math::{pt2, Point2, Parametric};
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
        assert!(!self.0.is_empty(), "cannot eval an empty polyline");

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
    #[cfg(feature = "fp")]
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
