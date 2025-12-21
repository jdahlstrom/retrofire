//! Builder for setting up geometry for rendering.

use alloc::vec::Vec;
use core::borrow::Borrow;

use crate::geom::{Edge, Mesh, Tri, Vertex3};
use crate::math::{Mat4, Vary};

use super::{Clip, Context, Ndc, Render, Screen, Shader, Target};

/// A builder for rendering a chunk of geometry as a batch.
///
/// Several values must be assigned before the [`render`][Batch::render]
/// method can be called:
/// * [primitives][Batch::primitives]: A list of primitives, each a tuple
///   of indices into the list of vertices (TODO: handling oob)
/// * [vertices][Batch::vertices]: A list of vertices
/// * [shader][Batch::shader]: The combined vertex and fragment shader used
/// * [target][Batch::target]: The render target to render into
/// * [context][Batch::context]: The rendering context and settings used.
///   (TODO: optional?)
///
/// Additionally, setting the following values is optional:
/// * [uniform][Batch::uniform]: The uniform value passed to the vertex shader
/// * [viewport][Batch::viewport]: The matrix used for the NDC-to-screen transform.
// TODO Not correct right now due to method call ordering constraints
// A batch can be freely reused, for example to render several chunks of geometry
// using the same configuration, or several [instances] of the same geometry.
// [instances]: https://en.wikipedia.org/wiki/Geometry_instancing
#[derive(Clone, Debug, Default)]
pub struct Batch<Prim, Vtx, Uni, Shd, Tgt, Ctx> {
    pub prims: Vec<Prim>,
    pub verts: Vec<Vtx>,
    pub uniform: Uni,
    pub shader: Shd,
    pub viewport: Mat4<Ndc, Screen>,
    pub target: Tgt,
    pub ctx: Ctx,
}

macro_rules! update {
    ($($upd:ident)+ ; $self:ident $($rest:ident)+) => {{
        let Self { $($upd: _, )+ $($rest, )+ } = $self;
        Batch { $($upd, )+ $($rest, )+ }
    }};
}

impl Batch<(), (), (), (), (), Context> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<Prim, Vtx, Uni, Shd, Tgt, Ctx> Batch<Prim, Vtx, Uni, Shd, Tgt, Ctx> {
    /// Sets the primitives to be rendered.
    ///
    /// The primitives are copied into the batch.
    pub fn primitives<P: Clone>(
        self,
        prims: impl AsRef<[P]>,
    ) -> Batch<P, Vtx, Uni, Shd, Tgt, Ctx> {
        let prims = prims.as_ref().to_vec();
        update!(prims; self verts uniform shader viewport target ctx)
    }

    /// Sets the vertices to be rendered.
    ///
    /// The vertices are cloned into the batch.
    // TODO: Allow taking by reference to make cloning Batch cheap
    pub fn vertices<V: Clone>(
        self,
        verts: impl AsRef<[V]>,
    ) -> Batch<Prim, V, Uni, Shd, Tgt, Ctx> {
        let verts = verts.as_ref().to_vec();
        update!(verts; self prims uniform shader viewport target ctx)
    }

    /// Clones faces and vertices from a mesh to this batch.
    pub fn mesh<A: Clone, B>(
        self,
        mesh: &Mesh<A, B>,
    ) -> Batch<Tri<usize>, Vertex3<A, B>, Uni, Shd, Tgt, Ctx> {
        let prims = mesh.faces.clone();
        let verts = mesh.verts.clone();
        update!(verts prims; self uniform shader viewport target ctx)
    }

    /// Sets the uniform data to be passed to the vertex shaders.
    pub fn uniform<U: Copy>(
        self,
        uniform: U,
    ) -> Batch<Prim, Vtx, U, Shd, Tgt, Ctx> {
        update!(uniform; self verts prims shader viewport target ctx)
    }

    /// Sets the combined vertex and fragment shader.
    pub fn shader<V: Vary, U, S: Shader<Vtx, V, U>>(
        self,
        shader: S,
    ) -> Batch<Prim, Vtx, Uni, S, Tgt, Ctx> {
        update!(shader; self verts prims uniform viewport target ctx)
    }

    /// Sets the viewport matrix.
    pub fn viewport(self, viewport: Mat4<Ndc, Screen>) -> Self {
        update!(viewport; self verts prims uniform shader target ctx)
    }

    /// Sets the render target.
    // TODO what bound for T?
    pub fn target<T>(self, target: T) -> Batch<Prim, Vtx, Uni, Shd, T, Ctx> {
        update!(target; self verts prims uniform shader viewport ctx)
    }

    /// Sets the rendering context.
    pub fn context(
        self,
        ctx: &Context,
    ) -> Batch<Prim, Vtx, Uni, Shd, Tgt, &Context> {
        update!(ctx; self verts prims uniform shader viewport target)
    }
}

impl<Prim, Vtx, Uni, Shd, Tgt, Ctx> Batch<Prim, Vtx, Uni, Shd, Tgt, Ctx> {
    /// Renders this batch of geometry.
    #[rustfmt::skip]
    pub fn render<Var>(&mut self)
    where
        Var: Vary,
        Prim: Render<Var> + Clone,
        Vtx: Clone,
        Uni: Copy,
        [<Prim>::Clip]: Clip<Item= Prim::Clip>,
        Shd: Shader<Vtx, Var, Uni>,
        Tgt: Target,
        Ctx: Borrow<Context>
    {
        let Self {
            prims, verts, shader, uniform, viewport, target, ctx,
        } = self;

        super::render(
            prims, verts, shader, *uniform, *viewport,
            target, (*ctx).borrow(),
        );
    }
}

impl<Vtx, Uni, Shd, Tgt, Ctx> Batch<Edge<usize>, Vtx, Uni, Shd, Tgt, Ctx> {
    pub fn append(&mut self, other: Self) {
        let Batch { prims, verts, .. } = other;
        let n = self.verts.len();
        let prims = prims.into_iter().map(|e| Edge(e.0 + n, e.1 + n));

        self.verts.extend(verts);
        self.prims.extend(prims)
    }
}

impl<Vtx, Uni, Shd, Tgt, Ctx> Batch<Tri<usize>, Vtx, Uni, Shd, Tgt, Ctx> {
    pub fn append(&mut self, other: Self) {
        let Batch { prims, verts, .. } = other;
        let n = self.verts.len();
        let prims = prims.into_iter().map(|tri| tri.map(|i| i + n));

        self.verts.extend(verts);
        self.prims.extend(prims);
    }
}
