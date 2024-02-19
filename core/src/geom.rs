//! Basic geometric primitives.

use core::fmt::{self, Debug, Formatter};
pub use mesh::Mesh;

pub mod mesh;

/// Vertex with a position and arbitrary other attributes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Vertex<P, A> {
    pub pos: P,
    pub attrib: A,
}

/// Triangle, defined by three vertices.
#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(transparent)]
pub struct Tri<V>(pub [V; 3]);

/// Plane, defined by normal vector and offset from the origin
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct Plane<V>(pub(crate) V);

pub const fn vertex<P, A>(pos: P, attrib: A) -> Vertex<P, A> {
    Vertex { pos, attrib }
}

impl<V: Debug> Debug for Tri<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let delims = [", ", ", ", ")"];
        f.write_str("Tri(")?;
        for (v, d) in self.0.iter().zip(&delims) {
            v.fmt(f)?;
            f.write_str(d)?;
        }
        Ok(())
    }
}
