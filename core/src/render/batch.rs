//! Builder for setting up geometry for rendering.

use alloc::vec::Vec;
use core::borrow::Borrow;

use crate::{
    geom::{Mesh, Tri, Vertex3},
    math::{Lerp, mat::Mat4x4, vary::Vary},
};

use super::{Context, NdcToScreen, Shader, Target};

/// A builder for rendering a chunk of geometry as a batch.
///
/// Several values must be assigned before the [`render`][Batch::render]
/// method can be called:
/// * [faces][Batch::faces]: A list of triangles, each a triplet of indices
///   into the list of vertices (TODO: handling oob)
/// * [vertices][Batch::vertices]: A list of vertices
/// * [shader][Batch::shader]: The combined vertex and fragment shader used
/// * [target][Batch::target]: The render target to render into
/// * [context][Batch::context]: The rendering context and settings used. (TODO: optional?)
///
/// Additionally, setting the following values is optional:
/// * [uniform][Batch::uniform]: The uniform value passed to the vertex shader
/// * [viewport][Batch::viewport]: The matrix used for the NDC-to-screen transform.
// TODO Not correct right now due to method call ordering constraints
// A batch can be freely reused, for example to render several chunks of geometry
// using the same configuration, or several [instances] of the same geometry.
// [instances]: https://en.wikipedia.org/wiki/Geometry_instancing
#[derive(Clone, Debug, Default)]
pub struct Batch<Vtx, Uni, Shd, Tgt, Ctx> {
    faces: Vec<Tri<usize>>,
    verts: Vec<Vtx>,
    uniform: Uni,
    shader: Shd,
    viewport: Mat4x4<NdcToScreen>,
    target: Tgt,
    ctx: Ctx,
}

macro_rules! update {
    ($($upd:ident)+ ; $self:ident $($rest:ident)+) => {{
        let Self { $($upd: _, )+ $($rest, )+ } = $self;
        Batch { $($upd, )+ $($rest, )+ }
    }};
}

impl Batch<(), (), (), (), Context> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<Vtx, Uni, Shd, Tgt, Ctx> Batch<Vtx, Uni, Shd, Tgt, Ctx> {
    /// Sets the faces to be rendered.
    ///
    /// The faces are copied into the batch.
    pub fn faces(self, faces: impl AsRef<[Tri<usize>]>) -> Self {
        Self {
            faces: faces.as_ref().to_vec(),
            ..self
        }
    }

    /// Sets the vertices to be rendered.
    ///
    /// The vertices are cloned into the batch.
    // TODO: Allow taking by reference to make cloning Batch cheap
    pub fn vertices<V: Clone>(
        self,
        verts: impl AsRef<[V]>,
    ) -> Batch<V, Uni, Shd, Tgt, Ctx> {
        let verts = verts.as_ref().to_vec();
        update!(verts; self faces uniform shader viewport target ctx)
    }

    /// Clones faces and vertices from a mesh to this batch.
    pub fn mesh<A: Clone>(
        self,
        mesh: &Mesh<A>,
    ) -> Batch<Vertex3<A>, Uni, Shd, Tgt, Ctx> {
        let faces = mesh.faces.clone();
        let verts = mesh.verts.clone();
        update!(verts faces; self uniform shader viewport target ctx)
    }

    /// Sets the uniform data to be passed to the vertex shaders.
    pub fn uniform<U: Copy>(self, uniform: U) -> Batch<Vtx, U, Shd, Tgt, Ctx> {
        update!(uniform; self verts faces shader viewport target ctx)
    }

    /// Sets the combined vertex and fragment shader.
    pub fn shader<V: Vary, U, S: Shader<Vtx, V, U>>(
        self,
        shader: S,
    ) -> Batch<Vtx, Uni, S, Tgt, Ctx> {
        update!(shader; self verts faces uniform viewport target ctx)
    }

    /// Sets the viewport matrix.
    pub fn viewport(self, viewport: Mat4x4<NdcToScreen>) -> Self {
        update!(viewport; self verts faces uniform shader target ctx)
    }

    /// Sets the render target.
    pub fn target<T>(self, target: T) -> Batch<Vtx, Uni, Shd, T, Ctx> {
        update!(target; self verts faces uniform shader viewport ctx)
    }

    /// Sets the rendering context.
    pub fn context(self, ctx: &Context) -> Batch<Vtx, Uni, Shd, Tgt, &Context> {
        update!(ctx; self verts faces uniform shader viewport target)
    }
}

impl<Vtx: Clone, Uni: Copy, Shd, Tgt, Ctx> Batch<Vtx, Uni, Shd, &mut Tgt, Ctx>
where
    Ctx: Borrow<Context>,
    Tgt: Target,
{
    /// Renders this batch of geometry.
    #[rustfmt::skip]
    pub fn render<V: Lerp + Vary>(&mut self)
    where
        Shd: Shader<Vtx, V, Uni>,
    {

        let Self {
            faces, verts, shader, uniform, viewport, target, ctx,
        } = self;
        super::render(
            faces, verts, shader, *uniform, *viewport, *target,
            (*ctx).borrow(),
        );
    }
}
