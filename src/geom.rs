use crate::math::vec::Vec4;

pub mod mesh;
pub mod solids;
pub mod bbox;

#[cfg(feature = "teapot")]
mod teapot;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vertex<A> {
    pub coord: Vec4,
    pub attr: A,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Face<VA, FA> {
    pub indices: [usize; 3],
    pub verts: [Vertex<VA>; 3],
    pub attr: FA,
}
