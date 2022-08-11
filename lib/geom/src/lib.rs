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
pub struct LineSeg<V>(pub [V; 2]);

impl<A> Transform for LineSeg<Vertex<A>> {
    fn transform_mut(&mut self, tf: &Mat4) {
        self.0[0].coord.transform_mut(tf);
        self.0[1].coord.transform_mut(tf);
    }
}

#[derive(Clone, Debug, Default)]
pub struct Polyline<V>(pub Vec<V>);

impl<V: Copy> Polyline<V> {
    pub fn edges<'a>(&'a self) -> impl Iterator<Item = LineSeg<V>> + 'a {
        self.0.windows(2).map(|vs| LineSeg([vs[0], vs[1]]))
    }
}

impl<V> FromIterator<V> for Polyline<V> {
    fn from_iter<I: IntoIterator<Item = V>>(iter: I) -> Self {
        Self(Vec::from_iter(iter))
    }
}

impl<A> Transform for Polyline<Vertex<A>> {
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
pub struct Sprite<V, A = ()> {
    pub verts: [V; 4],
    pub face_attr: A,
}

impl<VA: Copy, FA> Sprite<Vertex<VA>, FA> {
    pub fn new(anchor: Vec4, align: Align, w: f32, h: f32, vas: [VA; 4], fa: FA) -> Self {
        use Align::*;
        let (x, y) = match align {
            TopLeft => (0.0, 1.0),
            TopCenter => (0.5, 1.0),
            TopRight => (1.0, 1.0),
            Left => (0.0, 0.5),
            Center => (0.5, 0.5),
            Right => (1.0, 0.5),
            BottomLeft => (0.0, 0.0),
            BottomCenter => (0.5, 0.0),
            BottomRight => (1.0, 0.0),
        };
        Self {
            verts: [
                (x - 1.0, y, vas[0]),
                (x, y, vas[1]),
                (x, y - 1.0, vas[2]),
                (x - 1.0, y - 1.0, vas[3])
            ].map(move |(x, y, attr)| {
                let coord = anchor + dir(x * w, y * h, 0.0);
                Vertex { coord, attr }
            }),
            face_attr: fa,
        }
    }
}
