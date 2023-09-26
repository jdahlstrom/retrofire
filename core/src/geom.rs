/// Vertex with position and arbitrary other attributes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Vertex<P, A> {
    pub pos: P,
    pub attrib: A,
}

/// Triangle, consisting of three vertices.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Tri<V>(pub [V; 3]);

/// Plane, defined by normal vector and offset from the origin
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct Plane<V>(pub(crate) V);
