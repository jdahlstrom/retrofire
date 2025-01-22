//! Render impls for primitives and related items.

use crate::{
    geom::{Edge, Tri, Vertex},
    math::Vary,
};

use super::{
    clip::ClipVert,
    raster::{Scanline, ScreenPt, line, tri_fill},
};

pub trait Primitive {
    type Vertex;
    type Vertices: AsRef<[Self::Vertex]>;
    type Mapped<U: Clone>: Primitive<Vertex = U>;

    /// Returns the vertices of the primitive.
    fn vertices(&self) -> Self::Vertices;

    /// Returns a primitive with the vertices of `self` mapped
    fn map_vertices<U: Clone>(
        &self,
        f: impl Fn(Self::Vertex) -> U,
    ) -> Self::Mapped<U>;

    /// Returns the (average) depth of the primitive.
    fn depth<V: Clone>(_: &ToClip<Self, V>) -> f32 {
        f32::INFINITY
    }

    /// Returns whether the primitive is facing away from the camera.
    fn is_backface<V: Clone>(_: &ToScreen<Self, V>) -> bool {
        false
    }

    /// Rasterizes the argument by calling the function for each scanline.
    fn rasterize<V: Vary, F: FnMut(Scanline<V>)>(
        scr: &ToScreen<Self, V>,
        scanline_fn: F,
    );
}

pub type ToClip<P, V> = <P as Primitive>::Mapped<ClipVert<V>>;
pub type ToScreen<P, V> =
    <ToClip<P, V> as Primitive>::Mapped<Vertex<ScreenPt, V>>;

impl<Vtx: Clone> Primitive for Tri<Vtx> {
    type Vertex = Vtx;
    type Vertices = [Vtx; 3];
    type Mapped<U: Clone> = Tri<U>;

    fn vertices(&self) -> [Vtx; 3] {
        self.0.clone()
    }
    fn map_vertices<U: Clone>(&self, f: impl Fn(Vtx) -> U) -> Tri<U> {
        Tri(self.0.clone().map(f))
    }

    fn depth<V>(Tri([a, b, c]): &Tri<ClipVert<V>>) -> f32 {
        (a.pos.z() + b.pos.z() + c.pos.z()) / 3.0
    }

    fn is_backface<V>(Tri(vs): &Tri<Vertex<ScreenPt, V>>) -> bool {
        let v = vs[1].pos - vs[0].pos;
        let u = vs[2].pos - vs[0].pos;
        v[0] * u[1] - v[1] * u[0] > 0.0
    }
    fn rasterize<V: Vary, F: FnMut(Scanline<V>)>(
        scr: &Tri<Vertex<ScreenPt, V>>,
        scanline_fn: F,
    ) {
        tri_fill(scr.vertices(), scanline_fn);
    }
}

impl<Vtx: Clone> Primitive for Edge<Vtx> {
    type Vertex = Vtx;
    type Vertices = [Vtx; 2];
    type Mapped<U: Clone> = Edge<U>;

    fn vertices(&self) -> [Vtx; 2] {
        [self.0.clone(), self.1.clone()]
    }
    fn map_vertices<U: Clone>(&self, f: impl Fn(Vtx) -> U) -> Edge<U> {
        Edge(f(self.0.clone()), f(self.1.clone()))
    }
    fn rasterize<V: Vary, F: FnMut(Scanline<V>)>(
        scr: &Edge<Vertex<ScreenPt, V>>,
        scanline_fn: F,
    ) {
        line(scr.vertices(), scanline_fn);
    }
}
