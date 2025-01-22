//! Render impls for primitives and related items.

use crate::{
    geom::{Edge, Tri, Vertex},
    math::Vary,
};

use super::{
    clip::ClipVert,
    raster::{line, tri_fill, Scanline, ScreenPt},
    Render,
};

pub trait Primitive<Vtx> {
    type Vertices: AsRef<[Vtx]>;
    type Mapped<U: Clone>: Primitive<U>;

    fn vertices(&self) -> Self::Vertices;
    fn map_vertices<U: Clone>(self, f: impl Fn(Vtx) -> U) -> Self::Mapped<U>;
}

impl<V: Clone> Primitive<V> for Tri<V> {
    type Vertices = [V; 3];
    type Mapped<U: Clone> = Tri<U>;

    fn vertices(&self) -> [V; 3] {
        self.0.clone()
    }
    fn map_vertices<U: Clone>(self, f: impl Fn(V) -> U) -> Tri<U> {
        Tri(self.0.map(f))
    }
}

impl<V: Clone> Primitive<V> for Edge<V> {
    type Vertices = [V; 2];
    type Mapped<U: Clone> = Edge<U>;

    fn vertices(&self) -> [V; 2] {
        [self.0.clone(), self.1.clone()]
    }
    fn map_vertices<U: Clone>(self, f: impl Fn(V) -> U) -> Edge<U> {
        Edge(f(self.0), f(self.1))
    }
}

impl<V: Vary> Render<V> for Tri<usize> {
    fn depth(Tri([a, b, c]): &Tri<ClipVert<V>>) -> f32 {
        (a.pos.z() + b.pos.z() + c.pos.z()) / 3.0
    }

    fn is_backface(Tri(vs): &Tri<Vertex<ScreenPt, V>>) -> bool {
        let v = vs[1].pos - vs[0].pos;
        let u = vs[2].pos - vs[0].pos;
        v[0] * u[1] - v[1] * u[0] > 0.0
    }

    fn rasterize<F: FnMut(Scanline<V>)>(
        scr: &Tri<Vertex<ScreenPt, V>>,
        scanline_fn: F,
    ) {
        tri_fill(scr.0.clone(), scanline_fn);
    }
}

impl<V: Vary> Render<V> for Edge<usize> {
    fn rasterize<F: FnMut(Scanline<V>)>(
        scr: &Edge<Vertex<ScreenPt, V>>,
        scanline_fn: F,
    ) {
        line(scr.vertices(), scanline_fn);
    }
}
