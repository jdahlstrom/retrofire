use std::iter::FromIterator;

use math::mat::Mat4;
use math::transform::Transform;

use crate::mesh::Vertex;

pub mod mesh;
pub mod solids;
pub mod bbox;

#[cfg(feature = "teapot")]
mod teapot;

#[derive(Copy, Clone, Debug)]
pub struct LineSeg<VA>(pub [Vertex<VA>; 2]);

impl<VA> Transform for LineSeg<VA> {
    fn transform(&mut self, tf: &Mat4) {
        self.0[0].coord.transform(tf);
        self.0[1].coord.transform(tf);
    }
}

#[derive(Clone, Debug, Default)]
pub struct Polyline<VA>(pub Vec<Vertex<VA>>);

impl<VA: Copy> Polyline<VA> {
    pub fn edges<'a>(&'a self) ->
        impl Iterator<Item=[Vertex<VA>; 2]> + 'a
    {
        self.0.windows(2).map(|vs| [vs[0], vs[1]])
    }
}

impl<VA> FromIterator<Vertex<VA>> for Polyline<VA> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item=Vertex<VA>>
    {
        Self(Vec::from_iter(iter))
    }
}

impl<VA> Transform for Polyline<VA> {
    fn transform(&mut self, tf: &Mat4) {
        for v in &mut self.0 {
            v.coord.transform(tf);
        }
    }
}