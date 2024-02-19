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
        for i in 0..3 {
            self.0[i].fmt(f)?;
            f.write_str(delims[i])?;
        }
        Ok(())
    }
}
