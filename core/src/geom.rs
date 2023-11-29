//! Basic geometric primitives.

pub use mesh::Mesh;

use crate::math::Vec2;

pub mod mesh;

/// Vertex with a position and arbitrary other attributes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Vertex<P, A> {
    pub pos: P,
    pub attrib: A,
}

/// Triangle, defined by three vertices.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct Tri<V>(pub [V; 3]);

/// Plane, defined by normal vector and offset from the origin
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct Plane<V>(pub(crate) V);

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sprite<P, V> {
    /// Center point coordinates
    pub center: P,
    /// Horizontal and vertical extent
    pub size: Vec2,
    /// Corner vertices, with positions relative to `center`.
    pub verts: [V; 4],
}

pub const fn vertex<P, A>(pos: P, attrib: A) -> Vertex<P, A> {
    Vertex { pos, attrib }
}
