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
pub struct Batch<Prims, Verts, Uni, Shd, Tgt, Ctx> {
    pub prims: Prims,
    pub verts: Verts,
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

impl<Prims, Verts, Uni, Shd, Tgt, Ctx> Batch<Prims, Verts, Uni, Shd, Tgt, Ctx> {
    /// Sets the primitives to be rendered.
    #[must_use]
    pub fn primitives<Ps: IntoIterator>(
        self,
        prims: Ps,
    ) -> Batch<Ps, Verts, Uni, Shd, Tgt, Ctx> {
        update!(prims; self verts uniform shader viewport target ctx)
    }

    /// Sets the vertices to be rendered.
    #[must_use]
    pub fn vertices<Vs: IntoIterator>(
        self,
        verts: Vs,
    ) -> Batch<Prims, Vs, Uni, Shd, Tgt, Ctx> {
        update!(verts; self prims uniform shader viewport target ctx)
    }

    /// Clones faces and vertices from a mesh to this batch.
    ///
    /// You can also create a new batch from a moved or borrowed mesh
    /// directly using the `From<Mesh>` or `From<&Mesh>` impls.
    #[must_use]
    pub fn mesh<A: Clone>(
        self,
        mesh: &Mesh<A>,
    ) -> Batch<&[Tri<usize>], &[Vertex3<A>], Uni, Shd, Tgt, Ctx> {
        let prims = &mesh.faces;
        let verts = &mesh.verts;
        update!(verts prims; self uniform shader viewport target ctx)
    }

    /// Sets the uniform data to be passed to the vertex shaders.
    #[must_use]
    pub fn uniform<U: Copy>(
        self,
        uniform: U,
    ) -> Batch<Prims, Verts, U, Shd, Tgt, Ctx> {
        update!(uniform; self verts prims shader viewport target ctx)
    }

    /// Sets the combined vertex and fragment shader.
    #[must_use]
    pub fn shader<Vtx, Var, U, S>(
        self,
        shader: S,
    ) -> Batch<Prims, Verts, Uni, S, Tgt, Ctx>
    where
        Var: Vary,
        S: Shader<Vtx, Var, U>,
        Verts: AsRef<[Vtx]>,
    {
        update!(shader; self verts prims uniform viewport target ctx)
    }

    /// Sets the viewport matrix.
    #[must_use]
    pub fn viewport(self, viewport: Mat4<Ndc, Screen>) -> Self {
        update!(viewport; self verts prims uniform shader target ctx)
    }

    /// Sets the render target.
    // TODO what bound for T?
    #[must_use]
    pub fn target<T>(self, target: T) -> Batch<Prims, Verts, Uni, Shd, T, Ctx> {
        update!(target; self verts prims uniform shader viewport ctx)
    }

    /// Sets the rendering context.
    #[must_use]
    pub fn context(
        self,
        ctx: &Context,
    ) -> Batch<Prims, Verts, Uni, Shd, Tgt, &Context> {
        update!(ctx; self verts prims uniform shader viewport target)
    }
}

impl<Prims, Verts, Uni, Shd, Tgt, Ctx> Batch<Prims, Verts, Uni, Shd, Tgt, Ctx> {
    /// Renders this batch of geometry.
    #[rustfmt::skip]
    pub fn render<Prim, Vtx, Var>(&mut self)
    where
        Var: Vary,
        Prim: Render<Var> + Clone,
        Vtx: Clone,

        Prims: AsRef<[Prim]>,
        Verts: AsRef<[Vtx]>,

        Uni: Copy,
        [Prim::Clip]: Clip<Item = Prim::Clip>,
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

impl<Vtx, Uni, Shd, Tgt, Ctx>
    Batch<Vec<Edge<usize>>, Vec<Vtx>, Uni, Shd, Tgt, Ctx>
{
    pub fn append(&mut self, other: Self) {
        let Batch { prims, verts, .. } = other;
        let n = self.verts.len();
        let prims = prims.into_iter().map(|e| Edge(e.0 + n, e.1 + n));

        self.verts.extend(verts);
        self.prims.extend(prims);
    }
}

impl<Vtx, Uni, Shd, Tgt, Ctx>
    Batch<Vec<Tri<usize>>, Vec<Vtx>, Uni, Shd, Tgt, Ctx>
{
    pub fn append(&mut self, other: Self) {
        let Batch { prims, verts, .. } = other;
        let n = self.verts.len();
        let prims = prims.into_iter().map(|tri| tri.map(|i| i + n));

        self.verts.extend(verts);
        self.prims.extend(prims);
    }
}

//
// Foreign trait impls
//

impl<A, B> From<Mesh<A, B>>
    for Batch<Vec<Tri<usize>>, Vec<Vertex3<A, B>>, (), (), (), Context>
{
    fn from(m: Mesh<A, B>) -> Self {
        Batch::new().primitives(m.faces).vertices(m.verts)
    }
}

impl<'a, A, B> From<&'a Mesh<A, B>>
    for Batch<&'a [Tri<usize>], &'a [Vertex3<A, B>], (), (), (), Context>
{
    fn from(m: &'a Mesh<A, B>) -> Self {
        Batch::new()
            .primitives(m.faces.as_slice())
            .vertices(m.verts.as_slice())
    }
}
