use crate::mesh::Vertex;

pub mod mesh;
pub mod solids;
pub mod bbox;

#[cfg(feature = "teapot")]
mod teapot;

pub struct LineSeg<VA>(pub [Vertex<VA>; 2]);

pub struct Polyline<VA>(pub Vec<Vertex<VA>>);
