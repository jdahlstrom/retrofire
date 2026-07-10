use alloc::{vec, vec::Vec};

use crate::{
    geom::{Mesh, Tri, Vertex, Vertex3},
    math::{ProjVec3, Vary},
};

use super::{
    ClipVert, Indexed, Primitive, Render, VertexShader, primitive_assembly,
    vertex_transform,
};

impl<V, P, A> Render<V, Self, Vertex<P, A>> for Tri<Vertex<P, A>>
where
    V: Vary + 'static,
    Vertex<P, A>: Clone,
{
    type InClipSpace = Tri<ClipVert<V>>;

    fn primitives(&self) -> impl AsRef<[Self]> {
        [self.clone()]
    }

    fn vertices(&self) -> impl AsRef<[Vertex<P, A>]> {
        self.0.clone()
    }

    fn transform_to_clip<Shd, Uni: Copy>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> Self::InClipSpace
    where
        Shd: VertexShader<Vertex<P, A>, Uni, Output = Vertex<ProjVec3, V>>,
    {
        self.clone()
            .map(|v| ClipVert::new(shader.shade_vertex(v, uniform)))
    }

    fn primitive_assembly(this: &Self::InClipSpace) -> Vec<Self::InClipSpace> {
        vec![this.clone()]
    }
}

impl<V, P, A> Render<V, Tri<Vertex<P, A>>, Vertex<P, A>>
    for Vec<Tri<Vertex<P, A>>>
where
    V: Vary + 'static,
    Vertex<P, A>: Clone,
{
    type InClipSpace = Vec<Tri<ClipVert<V>>>;

    fn primitives(&self) -> impl AsRef<[Tri<Vertex<P, A>>]> {
        self
    }

    fn vertices(&self) -> impl AsRef<[Vertex<P, A>]> {
        unimplemented!();
        []
    }

    fn transform_to_clip<Shd, Uni: Copy>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> Self::InClipSpace
    where
        Shd: VertexShader<Vertex<P, A>, Uni, Output = Vertex<ProjVec3, V>>,
    {
        self.iter()
            .map(|tri| {
                tri.clone()
                    .map(|v| ClipVert::new(shader.shade_vertex(v, uniform)))
            })
            .collect()
    }

    fn primitive_assembly(this: &Self::InClipSpace) -> Self::InClipSpace {
        this.clone()
    }
}

impl<V, A, B> Render<V, Tri<usize>, Vertex3<A, B>> for Mesh<A, B>
where
    V: Vary + 'static,
    Vertex3<A, B>: Clone,
{
    type InClipSpace = Indexed<Vec<Tri<usize>>, Vec<ClipVert<V>>>;

    fn primitives(&self) -> impl AsRef<[Tri<usize>]> {
        &self.faces
    }
    fn vertices(&self) -> impl AsRef<[Vertex3<A, B>]> {
        &self.verts
    }

    fn transform_to_clip<Shd, Uni: Copy>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> Self::InClipSpace
    where
        Shd: VertexShader<Vertex3<A, B>, Uni, Output = Vertex<ProjVec3, V>>,
    {
        Indexed {
            prims: self.faces.clone(),
            verts: vertex_transform(shader, uniform, &self.verts),
        }
    }

    fn primitive_assembly(
        this: &Self::InClipSpace,
    ) -> Vec<<Tri<usize> as Primitive<V>>::Clip> {
        primitive_assembly(&this.prims, &this.verts).collect()
    }
}

impl<'a, V, Prim, Vert> Render<V, Prim, Vert>
    for Indexed<&'a [Prim], &'a [Vert]>
where
    V: Vary + 'static,
    Prim: Primitive<V> + Clone,
    Vert: Clone,
{
    type InClipSpace = Indexed<&'a [Prim], Vec<ClipVert<V>>>;

    fn primitives(&self) -> impl AsRef<[Prim]> {
        self.prims
    }
    fn vertices(&self) -> impl AsRef<[Vert]> {
        self.verts
    }

    fn transform_to_clip<Shd, Uni: Copy>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> Self::InClipSpace
    where
        Shd: VertexShader<Vert, Uni, Output = Vertex<ProjVec3, V>>,
    {
        Indexed {
            prims: self.prims,
            verts: vertex_transform(shader, uniform, &self.verts),
        }
    }

    fn primitive_assembly(
        this: &Self::InClipSpace,
    ) -> Vec<<Prim as Primitive<V>>::Clip> {
        primitive_assembly(this.prims, &this.verts).collect()
    }
}
