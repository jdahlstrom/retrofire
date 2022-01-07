use std::iter::FromIterator;

use math::mat::Mat4;
use math::transform::Transform;
use math::vec::{dir, Vec4};

use crate::mesh2::Vertex;

pub mod mesh2;
pub mod solids;
pub mod bbox;

#[cfg(feature = "teapot")]
mod teapot;

#[derive(Copy, Clone, Debug)]
pub struct LineSeg<VA>(pub [Vertex<VA>; 2]);

impl<VA> Transform for LineSeg<VA> {
    fn transform_mut(&mut self, tf: &Mat4) {
        self.0[0].coord.transform_mut(tf);
        self.0[1].coord.transform_mut(tf);
    }
}

#[derive(Clone, Debug, Default)]
pub struct Polyline<VA>(pub Vec<Vertex<VA>>);

impl<VA: Copy> Polyline<VA> {
    pub fn edges<'a>(&'a self) -> impl Iterator<Item=LineSeg<VA>> + 'a {
        self.0.windows(2).map(|vs| LineSeg([vs[0], vs[1]]))
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
    fn transform_mut(&mut self, tf: &Mat4) {
        for v in &mut self.0 {
            v.coord.transform_mut(tf);
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Align {
    TopLeft, TopCenter, TopRight,
    Left, Center, Right,
    BottomLeft, BottomCenter, BottomRight,
}

impl Default for Align {
    fn default() -> Self {
        Self::Center
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Sprite<VA, FA = ()> {
    pub anchor: Vec4,
    pub align: Align,
    pub width: f32,
    pub height: f32,
    pub vertex_attrs: [VA; 4],
    pub face_attr: FA,
}

impl<VA: Copy, FA> Sprite<VA, FA> {
    pub fn verts<'a>(&'a self) -> impl Iterator<Item=Vertex<VA>> + 'a {
        let (x, y) = match self.align {
            Align::TopLeft => (0.0, 1.0),
            Align::TopCenter => (0.5, 1.0),
            Align::TopRight => (1.0, 1.0),
            Align::Left => (0.0, 0.5),
            Align::Center => (0.5, 0.5),
            Align::Right => (1.0, 0.5),
            Align::BottomLeft => (0.0, 0.0),
            Align::BottomCenter => (0.5, 0.0),
            Align::BottomRight => (1.0, 0.0),
        };
        [(x - 1.0, y), (x, y), (x, y - 1.0), (x - 1.0, y - 1.0)]
            .into_iter()
            .zip(&self.vertex_attrs)
            .map(move |((x, y), &attr)| {
                let coord = self.anchor + dir(x * self.width, y * self.height, 0.0);
                Vertex { coord, attr }
            })
    }
}

impl<VA, FA> Transform for Sprite<VA, FA> {
    fn transform_mut(&mut self, tf: &Mat4) {
        self.anchor.transform_mut(tf);
    }
}
