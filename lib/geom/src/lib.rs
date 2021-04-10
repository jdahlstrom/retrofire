use crate::mesh::Vertex;
use math::transform::Transform;
use math::mat::Mat4;
use std::iter::FromIterator;

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

impl<VA> Polyline<VA> {
    pub fn edges(&self) -> impl Iterator<Item=&[Vertex<VA>]> {
        self.0.windows(2)
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
        for v in self.0.iter_mut() {
            v.coord.transform(tf);
        }
    }
}