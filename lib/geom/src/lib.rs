use std::array::IntoIter;
use std::iter::FromIterator;

use math::mat::Mat4;
use math::transform::Transform;
use math::vec::{dir, Vec4};

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

#[derive(Copy, Clone, Debug)]
pub struct Sprite<VA, FA = ()> {
    pub center: Vec4,
    pub width: f32,
    pub height: f32,
    pub vertex_attrs: [VA; 4],
    pub face_attr: FA,
}

impl<VA: Copy, FA> Sprite<VA, FA> {
    pub fn verts<'a>(&'a self) -> impl Iterator<Item=Vertex<VA>> + 'a {
        let (hw, hh) = (0.5 * self.width, 0.5 * self.height);
        IntoIter::new([(-hw, hh), (hw, hh), (hw, -hh), (-hw, -hh)])
            .zip(&self.vertex_attrs)
            .map(move |((w, h), &attr)| {
                let coord = self.center + dir(w, h, 0.0);
                Vertex { coord, attr }
            })
    }
}

impl<VA, FA> Transform for Sprite<VA, FA> {
    fn transform(&mut self, tf: &Mat4) {
        self.center.transform(tf);
    }
}
