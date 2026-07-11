use alloc::vec::Vec;
use core::{iter, mem::replace};

use crate::{
    geom::{Edge, Mesh, Polyline, Tri, Vertex, Vertex3, tri},
    math::{ProjVec3, Vary},
};

use super::{
    ClipVert, Indexed, Primitive, Render, TriFan, VertexShader,
    vertex_transform,
};

impl<V, P, A> Render<V, Self, Vertex<P, A>> for Tri<Vertex<P, A>>
where
    V: Vary + 'static,
    Vertex<P, A>: Clone,
{
    fn to_primitives<Shd, Uni>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> impl Iterator<Item = Tri<ClipVert<V>>>
    where
        Shd: VertexShader<Vertex<P, A>, Uni, Output = Vertex<ProjVec3, V>>,
        Uni: Copy,
    {
        iter::once(
            self.clone()
                .map(|v| ClipVert::new(shader.shade_vertex(v, uniform))),
        )
    }
}

impl<V, P, A> Render<V, Tri<Vertex<P, A>>, Vertex<P, A>>
    for Vec<Tri<Vertex<P, A>>>
where
    V: Vary + 'static,
    Vertex<P, A>: Clone,
{
    fn to_primitives<Shd, Uni>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> impl Iterator<Item = Tri<ClipVert<V>>>
    where
        Shd: VertexShader<Vertex<P, A>, Uni, Output = Vertex<ProjVec3, V>>,
        Uni: Copy,
    {
        self.iter().map(move |tri| {
            tri.clone()
                .map(move |v| ClipVert::new(shader.shade_vertex(v, uniform)))
        })
    }
}

impl<V, P, A> Render<V, Tri<Vertex<P, A>>, Vertex<P, A>>
    for TriFan<Vertex<P, A>>
where
    V: Vary + 'static,
    Vertex<P, A>: Clone,
{
    fn to_primitives<Shd, Uni>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> impl Iterator<Item = Tri<ClipVert<V>>>
    where
        Shd: VertexShader<Vertex<P, A>, Uni, Output = Vertex<ProjVec3, V>>,
        Uni: Copy,
    {
        let verts = vertex_transform(shader, uniform, &self.0);
        let mut it = verts.into_iter();

        let a = it.next().unwrap();
        let mut b = it.next().unwrap();
        iter::from_fn(move || {
            let c = it.next()?;
            let b = replace(&mut b, c.clone());
            Some(tri(a.clone(), b, c))
        })
    }
}

impl<V, P, A> Render<V, Edge<Vertex<P, A>>, Vertex<P, A>>
    for Polyline<Vertex<P, A>>
where
    V: Vary + 'static,
    Vertex<P, A>: Clone,
{
    fn to_primitives<Shd, Uni>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> impl Iterator<Item = Edge<ClipVert<V>>>
    where
        Shd: VertexShader<Vertex<P, A>, Uni, Output = Vertex<ProjVec3, V>>,
        Uni: Copy,
    {
        let mut it = self
            .0
            .iter()
            .cloned()
            .map(move |v| ClipVert::new(shader.shade_vertex(v, uniform)));
        // Collect to avoid lifetime bound on Uni

        let mut a = it.next().unwrap();
        iter::from_fn(move || {
            let b = it.next()?;
            let a = replace(&mut a, b.clone());
            Some(Edge(a.clone(), b))
        })
    }
}

impl<V, A, B> Render<V, Tri<usize>, Vertex3<A, B>> for Mesh<A, B>
where
    V: Vary + 'static,
    Vertex3<A, B>: Clone,
{
    fn to_primitives<Shd, Uni>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> impl Iterator<Item = Tri<ClipVert<V>>>
    where
        Shd: VertexShader<Vertex3<A, B>, Uni, Output = Vertex<ProjVec3, V>>,
        Uni: Copy,
    {
        let verts = vertex_transform(shader, uniform, &self.verts);
        self.faces
            .iter()
            .cloned()
            .map(move |prim| Tri::inline(prim, &verts))
    }
}

impl<'a, V, Prim, Vert> Render<V, Prim, Vert>
    for Indexed<&'a [Prim], &'a [Vert]>
where
    V: Vary + 'static,
    Prim: Primitive<V> + Clone,
    Vert: Clone,
{
    fn to_primitives<Shd, Uni>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> impl Iterator<Item = Prim::Clip>
    where
        Shd: VertexShader<Vert, Uni, Output = Vertex<ProjVec3, V>>,
        Uni: Copy,
    {
        let verts = vertex_transform(shader, uniform, self.verts);
        self.prims
            .iter()
            .cloned()
            .map(move |prim| Prim::inline(prim, &verts))
    }
}
