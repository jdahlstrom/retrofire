//! Basic geometric primitives.
//!
//! Includes vertices, polygons, planes, rays, and more.

use alloc::vec::Vec;
use core::fmt::{self, Debug, Formatter};

use crate::math::{
    Affine, ApproxEq, Lerp, Linear, Mat4, Parametric, Point, Point2, Point3,
    Vec2, Vec3, Vector,
    space::{Hom, Real},
    vec::dot,
    vec2, vec3,
};
use crate::render::Model;

/// Vertex with a position and arbitrary other attributes.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Vertex<P, A> {
    pub pos: P,
    pub attrib: A,
}

/// Two-dimensional vertex type.
pub type Vertex2<A, B = Model> = Vertex<Point2<B>, A>;

/// Three-dimensional vertex type.
pub type Vertex3<A, B = Model> = Vertex<Point3<B>, A>;

/// Triangle, defined by three vertices.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct Tri<V>(pub [V; 3]);

/// Plane, defined by the four parameters of the plane equation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct Plane<V>(pub(crate) V);

/// Plane embedded in 3D space, splitting the space into two half-spaces.
pub type Plane3<B = ()> = Plane<Vector<[f32; 4], Hom<3, B>>>;

/// A ray, or a half line, composed of an initial point and a direction vector.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Ray<T: Affine>(pub T, pub T::Diff);

pub type Ray2<B = ()> = Ray<Point2<B>>;
pub type Ray3<B = ()> = Ray<Point3<B>>;

/// A curve composed of a chain of line segments.
///
/// The polyline is represented as a list of points, or vertices, with each
/// pair of consecutive vertices sharing an edge.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Polyline<T>(pub Vec<T>);

/// A closed curve composed of a chain of line segments.
///
/// The polygon is represented as a list of points, or vertices, with each pair
/// of consecutive vertices, as well as the first and last vertex, sharing an edge.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Polygon<T>(pub Vec<T>);

/// A line segment between two vertices.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Edge<T>(pub T, pub T);

#[derive(Copy, Clone, PartialEq)]
pub struct Sphere<B = ()>(pub Point3<B>, pub f32);

// TODO Line3?
#[derive(Copy, Clone, PartialEq)]
pub struct Line2<B = ()>(Vector<[f32; 3], Hom<2, B>>);

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

/// Creates a [`Vertex`] with the give position and attribute values.
#[inline]
pub const fn vertex<P, A>(pos: P, attrib: A) -> Vertex<P, A> {
    Vertex { pos, attrib }
}

/// Creates a [`Tri`] with the given vertices.
#[inline]
pub const fn tri<V>(a: V, b: V, c: V) -> Tri<V> {
    Tri([a, b, c])
}

//
// Inherent impls
//

impl<V> Tri<V> {
    /// Given a triangle ABC, returns the edges [AB, BC, CA].
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Tri, Edge};
    /// use retrofire_core::math::{Point2, pt2};
    ///
    /// let pts: [Point2; _] = [pt2(-1.0, 0.0), pt2(2.0, 0.0), pt2(1.0, 2.0)];
    /// let tri = Tri(pts);
    ///
    /// let [e0, e1, e2] = tri.edges();
    /// assert_eq!(e0, Edge(&pts[0], &pts[1]));
    /// assert_eq!(e1, Edge(&pts[1], &pts[2]));
    /// assert_eq!(e2, Edge(&pts[2], &pts[0]));
    ///
    /// ```
    #[inline]
    pub fn edges(&self) -> [Edge<&V>; 3] {
        let [a, b, c] = &self.0;
        [Edge(a, b), Edge(b, c), Edge(c, a)]
    }

    /// Returns `self` with each vertex mapped with a function.
    #[inline]
    pub fn map<U>(self, mut f: impl FnMut(V) -> U) -> Tri<U> {
        let [a, b, c] = self.0;
        Tri([f(a), f(b), f(c)])
    }
}

impl<P: Affine, A> Tri<Vertex<P, A>> {
    /// Given a triangle ABC, returns the vectors [AB, AC].
    #[inline]
    pub fn tangents(&self) -> [P::Diff; 2] {
        let [a, b, c] = &self.0;
        [b.pos.sub(&a.pos), c.pos.sub(&a.pos)]
    }

    /// Returns the geometric center, or "balance point", of `self`.
    ///
    /// The centroid is simply the average of the three vertex positions.
    pub fn centroid(&self) -> P
    where
        P::Diff: Linear<Scalar = f32>,
    {
        let [ab, ac] = self.tangents();
        self.0[0].pos.add(&ab.add(&ac).mul(1.0 / 3.0))
    }
}

impl<A, B> Tri<Vertex2<A, B>> {
    /// Returns the winding order of `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Tri, vertex, Winding};
    /// use retrofire_core::math::pt2;
    ///
    /// let mut tri = Tri([
    ///     vertex(pt2::<_, ()>(0.0, 0.0), ()),
    ///     vertex(pt2(0.0, 3.0), ()),
    ///     vertex(pt2(4.0, 0.0), ()),
    /// ]);
    /// assert_eq!(tri.winding(), Winding::Cw);
    ///
    /// tri.0.swap(1, 2);
    /// assert_eq!(tri.winding(), Winding::Ccw);
    /// ```
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
    /// The area is positive *iff* `self` is wound counter-clockwise.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Tri, vertex};
    /// use retrofire_core::math::pt2;
    ///
    /// let tri = Tri([
    ///     vertex(pt2::<_, ()>(0.0, 0.0), ()),
    ///     vertex(pt2(0.0, 3.0), ()),
    ///     vertex(pt2(4.0, 0.0), ()),
    /// ]);
    /// assert_eq!(tri.signed_area(), -6.0);
    /// ```
    pub fn signed_area(&self) -> f32 {
        let [t, u] = self.tangents();
        t.perp_dot(u) / 2.0
    }

    /// Returns the (positive) area of `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{vertex, Tri};
    /// use retrofire_core::math::pt2;
    ///
    /// let tri = Tri([
    ///     vertex(pt2::<_, ()>(0.0, 0.0), ()),
    ///     vertex(pt2(0.0, 3.0), ()),
    ///     vertex(pt2(4.0, 0.0), ()),
    /// ]);
    /// assert_eq!(tri.area(), 6.0);
    /// ```
    pub fn area(&self) -> f32 {
        self.signed_area().abs()
    }
}

impl<A, B> Tri<Vertex3<A, B>> {
    /// Returns the normal vector of `self`.
    ///
    /// The result is normalized to unit length. If self is degenerate and
    /// has no normal, returns a zero vector.
    ///
    /// # Examples
    /// ```
    /// use core::f32::consts::SQRT_2;
    ///
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::geom::{Tri, vertex};
    /// use retrofire_core::math::{pt3, vec3};
    ///
    /// // Triangle lying in a 45° angle
    /// let tri = Tri([
    ///     vertex(pt3::<_, ()>(0.0, 0.0, 0.0), ()),
    ///     vertex(pt3(0.0, 3.0, 3.0), ()),
    ///     vertex(pt3(4.0, 0.0,0.0), ()),
    /// ]);
    /// assert_approx_eq!(tri.normal(), vec3(0.0, SQRT_2 / 2.0, -SQRT_2 / 2.0));
    /// ```
    pub fn normal(&self) -> Normal3 {
        let [t, u] = self.tangents();
        // TODO normal with basis
        t.cross(&u).normalize_or_zero().to()
    }

    /// Returns the plane that `self` lies on.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Tri, Plane3, vertex};
    /// use retrofire_core::math::{pt3, Vec3};
    ///
    /// let tri = Tri([
    ///     vertex(pt3::<f32, ()>(0.0, 0.0, 2.0), ()),
    ///     vertex(pt3(1.0, 0.0, 2.0), ()),
    ///     vertex(pt3(0.0, 1.0, 2.0), ())
    /// ]);
    /// assert_eq!(tri.plane().normal(), Vec3::Z);
    /// assert_eq!(tri.plane().offset(), 2.0);
    /// ```
    pub fn plane(&self) -> Plane3<B> {
        let [a, b, c] = &self.0;
        let [p, q, r] = [a.pos, b.pos, c.pos];
        Plane::from_points(p, q, r)
    }

    /// Returns the winding order of `self`, as projected to the XY plane.
    // TODO is this 3D version meaningful/useful enough?
    pub fn winding(&self) -> Winding {
        // TODO better way to xyz->xy...
        let [u, v] = self.tangents();
        let ([ux, uy, _], [vx, vy, _]) = (u.0, v.0);
        let z = vec2::<_, ()>(ux, uy).perp_dot(vec2(vx, vy));
        if z < 0.0 { Winding::Cw } else { Winding::Ccw }
    }

    /// Returns the area of `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{tri, vertex};
    /// use retrofire_core::math::pt3;
    ///
    /// let tri = tri(
    ///     vertex(pt3::<_, ()>(0.0, 0.0, 0.0), ()),
    ///     vertex(pt3(4.0, 0.0, 0.0), ()),
    ///     vertex(pt3(0.0, 3.0, 0.0), ()),
    /// );
    /// assert_eq!(tri.area(), 6.0);
    /// ```
    pub fn area(&self) -> f32 {
        let [t, u] = self.tangents();
        t.cross(&u).len() / 2.0
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
    /// The vector (a, b, c) must be a unit vector. The returned plane
    /// satisfies the plane equation
    ///
    /// *ax* + *by* + *cz* = *d*,
    ///
    /// or equivalently
    ///
    /// *ax* + *by* + *cz* - *d* = 0.
    ///
    /// Note the sign of the *d* coefficient.
    ///
    /// The coefficients (a, b, c) make up a vector normal to the plane,
    /// and d is the plane's offset from the origin.
    ///
    /// # Panics
    /// If the vector (a, b, c) is not unit length.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{geom::Plane3, math::Vec3};
    ///
    /// let p = <Plane3>::new(1.0, 0.0, 0.0, -2.0);
    /// assert_eq!(p.normal(), Vec3::X);
    /// assert_eq!(p.offset(), -2.0);
    ///
    /// ```
    #[inline]
    pub const fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        // TODO This method can't itself normalize because const
        assert!(
            (a * a + b * b + c * c - 1.0).abs() < 1e-6,
            "non-unit normal"
        );
        Self(Vector::new([a, b, c, -d]))
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
    /// let p = <Plane3>::from_points(
    ///     pt3(0.0, 0.0, 2.0),
    ///     pt3(1.0, 0.0, 2.0),
    ///     pt3(0.0, 1.0, 2.0),
    /// );
    /// assert_eq!(p.normal(), vec3(0.0, 0.0, 1.0));
    /// assert_eq!(p.offset(), 2.0);
    ///
    /// ```
    pub fn from_points(a: Point3<B>, b: Point3<B>, c: Point3<B>) -> Self {
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
    /// assert_eq!(p.normal(), Vec3::Z);
    /// assert_eq!(p.offset(), 3.0);
    ///
    /// ```
    pub fn from_point_and_normal(pt: Point3<B>, n: Normal3) -> Self {
        let mut n = n.to().normalize().to_hom();
        n[3] = -n.dot(&pt.to_hom());
        Self(n)
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
        let n = vec3(a, b, c);
        debug_assert!(n.len_sqr().approx_eq(&1.0));
        n
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
    /// assert_eq!(<Plane3>::new(0.0, 1.0, 0.0, 3.0).offset(), 3.0);
    /// assert_eq!(<Plane3>::new(1.0, 0.0, 0.0, -3.0).offset(), -3.0);
    /// ```
    #[inline]
    pub fn offset(&self) -> f32 {
        -self.0[3]
    }

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
    /// assert_eq!(<Plane3>::new(0.0, 0.0, 1.0, 2.0).project(pt), pt3(1.0, 2.0, 2.0));
    /// ```
    pub fn project(&self, pt: Point3<B>) -> Point3<B> {
        // The vector that projects pt on the plane is parallel with the plane
        // normal and its length is the distance of pt from the plane.
        let t = -self.signed_dist(pt);
        let dir = self.normal();
        pt + t * dir.to()
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
    /// let p = <Plane3>::new(-1.0, 0.0, 0.0, 2.0);
    /// assert_eq!(p.signed_dist(pt), -3.0);
    /// ```
    #[inline]
    pub fn signed_dist(&self, pt: Point3<B>) -> f32 {
        self.0.dot(&pt.to_hom())
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
    #[inline]
    pub fn is_inside(&self, pt: Point3<B>) -> bool {
        self.signed_dist(pt) <= 0.0
    }

    /// Returns an orthonormal affine basis on `self`.
    ///
    /// The y-axis of the basis is the normal vector; the x- and z-axes are
    /// two arbitrary orthogonal unit vectors tangent to the plane. The origin
    /// point is the point on the plane closest to the origin.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::geom::Plane3;
    /// use retrofire_core::math::{Point3, pt3, vec3};
    ///
    /// let p = <Plane3>::from_point_and_normal(
    ///     pt3(0.0,1.0,0.0),
    ///     vec3(0.0,1.0,1.0)
    /// );
    /// let m = p.basis::<()>();
    ///
    /// assert_approx_eq!(m.apply(&Point3::origin()), pt3(0.0, 0.5, 0.5));
    /// ```
    pub fn basis<F>(&self) -> Mat4<F, B> {
        let up = self.normal().to();

        let right: Vec3<B> =
            if up.x().abs() <= up.y().abs() && up.x().abs() <= up.z().abs() {
                Vec3::X
            } else {
                Vec3::Z
            };
        let fwd = right.cross(&up).normalize();
        let up = up.normalize();
        let right = up.cross(&fwd);

        let origin = self.offset() * up;

        Mat4::from_affine(right, up, fwd, origin.to_pt())
    }

    #[inline]
    pub fn coeffs(&self) -> [f32; 4] {
        self.0.0
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

    /// Returns the sum of the lengths of the edges using a custom metric.
    ///
    /// The function passed can be arbitrary; this method does not assume any
    /// actual metric properties such as positivity or triangle inequality.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Polyline, Edge};
    /// use retrofire_core::math::{pt2, Point2};
    ///
    /// let pts: [Point2; _] = [pt2(0.0, 0.0), pt2(1.0, 2.0), pt2(2.0, -3.0)];
    /// let pline = Polyline::new(pts);
    ///
    /// // The taxicab, or Manhattan, distance.
    /// fn taxicab(a: &Point2, b: &Point2) -> f32 {
    ///     let d = *b - *a;
    ///     d.x().abs() + d.y().abs()
    /// }
    ///
    /// assert_eq!(pline.len_by(taxicab), 9.0);
    /// ```
    pub fn len_by(&self, mut m: impl FnMut(&T, &T) -> f32) -> f32 {
        self.edges().map(|Edge(a, b)| m(a, b)).sum()
    }
}

impl<const N: usize, B> Polyline<Point<[f32; N], Real<N, B>>> {
    /// Returns the sum of the lengths of the edges of `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Polyline, Edge};
    /// use retrofire_core::math::{pt2, Point2};
    ///
    /// let pts: [Point2; _] = [pt2(0.0, 0.0), pt2(1.0, 0.0), pt2(1.0, -3.0)];
    /// let pline = Polyline::new(pts);
    ///
    /// assert_eq!(pline.len(), 4.0);
    /// ```
    pub fn len(&self) -> f32 {
        self.len_by(Point::distance)
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

impl<B> Line2<B> {
    /// Two-dimensional line, given by the line equation ax + by = c.
    ///
    /// # Panics
    /// If the vector (a, b) is not unit-length.
    pub fn new(a: f32, b: f32, c: f32) -> Self {
        // TODO This method can't itself normalize because const
        assert!((a * a + b * b - 1.0).abs() < 1e-6, "non-unit normal");
        Self(Vector::new([a, b, -c]))
    }

    /// Returns the unique line that crosses the given points.
    ///
    /// # Panics
    /// If the points coincide.
    pub fn from_points(p: Point2<B>, q: Point2<B>) -> Self {
        Edge(p, q).into()
    }

    /// Returns the slope and y-intercept of `self` if `self` is not vertical.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::{assert_approx_eq, geom::Line2, math::pt2};
    ///
    /// let line = <Line2>::from_points(pt2(-1.0, 1.0), pt2(1.0, 2.0));
    /// let (slope, y_intercept) = line.slope_intercept().unwrap();
    /// assert_approx_eq!(slope, 0.5);
    /// assert_approx_eq!(y_intercept, 1.5);
    /// ```
    pub fn slope_intercept(&self) -> Option<(f32, f32)> {
        // ax + by + c = 0
        let [a, b, c] = self.coeffs();

        (b != 0.0).then(|| {
            // by = -ax - c  <=>  y = -a/b x - c/b
            let m = -a / b; // slope
            let y0 = -c / b; // y intercept
            (m, y0)
        })
    }

    /// Returns
    pub fn normal(&self) -> Vec2<B> {
        vec2(self.0[0], self.0[1]).normalize()
    }
    /// Returns the signed distance of `self` from the origin.
    pub fn offset(&self) -> f32 {
        -self.0[2]
    }

    /// Returns the coefficients [a, b, c] of the line equation ax + by = c.
    pub const fn coeffs(&self) -> [f32; 3] {
        return self.0.0;
    }
}

//
// Local trait impls
//

impl<T> Parametric<T> for Ray<T>
where
    T: Affine<Diff: Linear<Scalar = f32>>,
{
    fn eval(&self, t: f32) -> T {
        self.0.add(&self.1.mul(t))
    }
}

impl<T: Lerp> Parametric<T> for Polyline<T> {
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
        let pts = &self.0;
        assert!(!pts.is_empty(), "cannot eval an empty polyline");

        let max = pts.len() - 1;
        let i = t.clamp(0.0, 1.0) * max as f32;
        let t_rem = i % 1.0;
        let i = i as usize;

        if i == max {
            pts[i].clone()
        } else {
            pts[i].lerp(&pts[i + 1], t_rem)
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

//
// Foreign trait impls
//

impl<B> Default for Plane3<B> {
    /// Returns the XZ coordinate plane.
    fn default() -> Self {
        Plane3::XZ
    }
}

impl<B: Debug + Default> Debug for Sphere<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Sphere")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl<B> Default for Sphere<B> {
    /// Returns a unit sphere, with the center at the origin and radius 1.
    fn default() -> Self {
        Self(Point3::origin(), 1.0)
    }
}

impl<B> Debug for Line2<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // ax + by + c = 0
        let [a, _, c] = self.coeffs();

        f.write_str("Line(")?;
        match self.slope_intercept() {
            Some((m, y0)) => {
                let pm = if y0 >= 0.0 { '+' } else { '-' };
                let ay0 = y0.abs();
                if m == 0.0 {
                    write!(f, "y = {y0}")
                } else if m == 1.0 {
                    write!(f, "y = x {pm} {ay0}")
                } else if m == -1.0 {
                    write!(f, "y = -x {pm} {ay0}")
                } else {
                    write!(f, "y = {m}·x {pm} {ay0}")
                }?;
            }
            None if a != 0.0 => write!(f, "x = {}", -c / a)?,
            None => unreachable!("invariant: a*a + b*b = 1"),
        }
        f.write_str(")")
    }
}

impl<B> From<Ray<Point2<B>>> for Line2<B> {
    /// Returns the line coincident with the given ray.
    ///
    /// # Panics
    /// If the ray is zero-length or approximately so.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::geom::{Line2, Ray};
    /// use retrofire_core::math::{pt2, vec2, vec3};
    ///
    /// //     ^
    /// //     |
    /// // ----2----O——————>-------
    /// //     |
    /// //     1
    /// //     |
    /// // <---+----1----2----3--->
    /// let line = <Line2>::from(Ray(pt2(1.0, 2.0), vec2(1.5, 0.0)));
    ///
    /// assert_eq!(line.normal(), vec2(0.0, 1.0));
    /// assert_eq!(line.offset(), 2.0);
    /// ```
    fn from(Ray(orig, dir): Ray<Point2<B>>) -> Self {
        let n = dir.perp().normalize();
        Self::new(n.x(), n.y(), dot(&orig.0, &n.0))
    }
}

impl<B> From<Edge<Point2<B>>> for Line2<B> {
    /// Returns the line coincident with the given edge.
    fn from(Edge(p, q): Edge<Point2<B>>) -> Self {
        Ray(p, q - p).into()
    }
}

#[cfg(test)]
mod tests {
    use crate::math::*;
    use crate::{assert_approx_eq, mat};
    use alloc::vec;
    use core::f32::consts::*;
    use std::format;

    use super::*;

    type Pt<const N: usize> = Point<[f32; N], Real<N>>;

    fn tri<const N: usize>(
        a: Pt<N>,
        b: Pt<N>,
        c: Pt<N>,
    ) -> Tri<Vertex<Pt<N>, ()>> {
        Tri([a, b, c]).map(|p| vertex(p, ()))
    }

    #[test]
    fn triangle_winding_2_cw() {
        let tri = tri(pt2(-1.0, 0.0), pt2(0.0, 1.0), pt2(1.0, -1.0));
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
            tri(pt3(-1.0, 0.0, 0.0), pt3(0.0, 1.0, 1.0), pt3(1.0, -1.0, 0.0));
        assert_eq!(tri.winding(), Winding::Cw);
    }
    #[test]
    fn triangle_winding_3_ccw() {
        let tri =
            tri(pt3(-1.0, 0.0, 0.0), pt3(1.0, 0.0, 0.0), pt3(0.0, 1.0, -1.0));
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
        assert_approx_eq!(tri.area(), 3.0);
    }

    #[test]
    fn triangle_plane() {
        let tri = tri(
            pt3(-1.0, -2.0, -1.0),
            pt3(2.0, -2.0, -1.0),
            pt3(0.0, -2.0, 1.0),
        );
        assert_approx_eq!(tri.plane().0, Plane3::new(0.0, -1.0, 0.0, 2.0).0);
    }

    #[test]
    fn plane_from_points() {
        let p = <Plane3>::from_points(
            pt3(1.0, 0.0, 0.0),
            pt3(0.0, 1.0, 0.0),
            pt3(0.0, 0.0, 1.0),
        );

        assert_approx_eq!(p.normal(), vec3(1.0, 1.0, 1.0).normalize());
        assert_approx_eq!(p.offset(), f32::sqrt(1.0 / 3.0));
    }
    #[test]
    #[should_panic]
    fn plane_from_collinear_points_panics() {
        <Plane3>::from_points(
            pt3(1.0, 2.0, 3.0),
            pt3(-2.0, -4.0, -6.0),
            pt3(0.5, 1.0, 1.5),
        );
    }
    #[test]
    #[should_panic]
    fn plane_from_zero_normal_panics() {
        <Plane3>::from_point_and_normal(
            pt3(1.0, 2.0, 3.0),
            vec3(0.0, 0.0, 0.0),
        );
    }
    #[test]
    fn plane_from_point_and_normal() {
        let p = <Plane3>::from_point_and_normal(
            pt3(1.0, 2.0, -3.0),
            vec3(0.0, 0.0, 12.3),
        );
        assert_approx_eq!(p.normal(), vec3(0.0, 0.0, 1.0));
        assert_approx_eq!(p.offset(), -3.0);
    }
    #[test]
    fn plane_is_point_inside_xz() {
        let p = <Plane3>::from_point_and_normal(pt3(1.0, 2.0, 3.0), Vec3::Y);

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
        let p = <Plane3>::from_point_and_normal(pt3(1.0, 2.0, 3.0), -Vec3::Y);

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
        let p = <Plane3>::from_point_and_normal(pt3(0.0, 1.0, 0.0), splat(1.0));

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
        let p = <Plane3>::from_point_and_normal(pt3(0.0, 2.0, 0.0), Vec3::Y);

        // Outside
        assert_approx_eq!(p.project(pt3(5.0, 10.0, -3.0)), pt3(5.0, 2.0, -3.0));
        // Coincident
        assert_approx_eq!(p.project(pt3(5.0, 2.0, -3.0)), pt3(5.0, 2.0, -3.0));
        // Inside
        assert_approx_eq!(
            p.project(pt3(5.0, -10.0, -3.0)),
            pt3(5.0, 2.0, -3.0)
        );
    }

    #[test]
    fn plane_basis() {
        let p = <Plane3>::new(0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, 2.0);

        let m = p.basis::<Plane3>();

        assert_approx_eq!(
            m,
            mat![
                1.0, 0.0, 0.0, 0.0;
                0.0, FRAC_1_SQRT_2, -FRAC_1_SQRT_2, SQRT_2;
                0.0, FRAC_1_SQRT_2, FRAC_1_SQRT_2, SQRT_2;
                0.0, 0.0, 0.0, 1.0;
            ]
        );
    }

    #[test]
    fn line_debug() {
        let mut l: Line2;

        l = Line2::new(1.0, 0.0, 0.0); // x = 0
        assert_eq!(format!("{:?}", l), "Line(x = 0)");

        l = Line2::from_points(pt2(2.0, 0.0), pt2(2.0, -1.0));
        assert_eq!(l.coeffs(), [1.0, 0.0, -2.0]);
        assert_eq!(format!("{:?}", l), "Line(x = 2)");

        l = Line2::new(1.0, 0.0, 2.0); // x = 2
        assert_eq!(format!("{:?}", l), "Line(x = 2)");

        l = Line2::new(0.0, 1.0, 0.0); // y = 0
        assert_eq!(format!("{:?}", l), "Line(y = 0)");

        l = Line2::from_points(pt2(0.0, -3.0), pt2(1.0, -3.0)); // y = -3
        assert_eq!(l.slope_intercept(), Some((0.0, -3.0)));
        assert_eq!(format!("{:?}", l), "Line(y = -3)");

        l = Line2::new(0.0, 1.0, -3.0); // y = -3
        assert_eq!(format!("{:?}", l), "Line(y = -3)");
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
        let pl = Polyline(vec![1.23]);
        assert_eq!(pl.eval(0.0), 1.23);
        assert_eq!(pl.eval(1.0), 1.23);
    }
}
