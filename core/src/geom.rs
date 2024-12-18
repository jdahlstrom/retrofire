//! Basic geometric primitives.

use crate::math::{Point2, Point3, Vec2, Vec3};
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

/// A surface normal.
// TODO Use distinct type rather than alias
pub type Normal3 = Vec3;
pub type Normal2 = Vec2;

pub const fn vertex<P, A>(pos: P, attrib: A) -> Vertex<P, A> {
    Vertex { pos, attrib }
}
