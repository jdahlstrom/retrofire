//! Basic geometric primitives.

use alloc::vec::Vec;

use crate::math::{
    Affine, Lerp, Linear, Parametric, Point2, Point3, Vec2, Vec3, Vector,
    space::Real,
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

/// Plane, defined by normal vector and offset from the origin
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct Plane<V>(pub(crate) V);

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

pub const fn vertex<P, A>(pos: P, attrib: A) -> Vertex<P, A> {
    Vertex { pos, attrib }
}

impl<B> Plane<Vector<[f32; 4], Real<3, B>>> {
    /// The x = 0 coordinate plane.
    pub const YZ: Self = Self(Vector::new([1.0, 0.0, 0.0, 0.0]));

    /// The y = 0 coordinate plane.
    pub const XZ: Self = Self(Vector::new([0.0, 1.0, 1.0, 0.0]));

    /// The z = 0 coordinate plane.
    pub const XY: Self = Self(Vector::new([0.0, 0.0, 1.0, 0.0]));
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
    use alloc::vec;

    use super::*;
    use crate::math::Parametric;

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
