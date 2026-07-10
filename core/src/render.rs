//! Turning 3D geometry into raster images.
//!
//! This module constitutes the core 3D rendering pipeline of `retrofire`.
//! It contains code for [clipping][clip], [transforming, shading][shader],
//! [texturing][tex], [rasterizing][raster], and [outputting][target] basic
//! geometric shapes such as triangles.

use alloc::vec::Vec;
use core::fmt::Debug;

use crate::geom::Vertex;
use crate::math::{Mat4, ProjVec3, Vary};

use self::{clip::view_frustum, ctx::DepthSort};

pub(super) mod re_exports {
    pub use super::{
        batch::Batch,
        cam::Camera,
        clip::{Clip, ClipVert},
        ctx::Context,
        light::Light,
        prim::Primitive,
        raster::Frag,
        scene::{BBox, Obj},
        shader::{FragmentShader, VertexShader},
        target::{Colorbuf, Framebuf, Target},
        tex::{TexCoord, Texture, uv},
        text::Text,
    };

    #[cfg(feature = "stats")]
    pub use super::stats::Stats;
}
pub use re_exports::*;

pub mod batch;
pub mod cam;
pub mod clip;
pub mod ctx;
pub mod debug;
pub mod light;
pub mod prim;
pub mod raster;
pub mod scene;
pub mod shader;
pub mod target;
pub mod tex;
pub mod text;

#[cfg(feature = "stats")]
pub mod stats;

mod impls;

pub trait Render<Var: Vary + 'static, Prim: Primitive<Var>, Vert> {
    fn to_primitives<Shd, Uni: Copy>(
        &self,
        shader: &Shd,
        uniform: Uni,
    ) -> impl Iterator<Item = Prim::Clip>
    where
        Shd: VertexShader<Vert, Uni, Output = Vertex<ProjVec3, Var>>;
}

pub struct Indexed<Prims, Verts> {
    pub prims: Prims,
    pub verts: Verts,
}

pub struct TriFan<V>(pub Vec<V>);

pub struct TriStrip<V>(pub Vec<V>);

/// Alias for combined vertex+fragment shader types
pub trait Shader<Vtx, Var, Uni>:
    VertexShader<Vtx, Uni, Output = Vertex<ProjVec3, Var>>
    + FragmentShader<Var, Uni>
{
}

/// Model space coordinate basis.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Model;

/// World space coordinate basis.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct World;

/// View (camera) space coordinate basis.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct View;

/// NDC space coordinate basis (normalized device coordinates).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Ndc;

/// Screen space coordinate basis.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Screen;

/// Renders the given primitives into `target`.
pub fn render<Prim, Vert, Geom, Var, Uni, Shd>(
    geometry: &Geom,
    shader: &Shd,
    uniform: Uni,
    to_screen: Mat4<Ndc, Screen>,
    target: &mut impl Target,
    ctx: &Context,
) where
    // FIXME Primitive currently tacitly assumes indexed representation!
    Prim: Primitive<Var> + Clone,
    Vert: Clone,
    Geom: Render<Var, Prim, Vert>,
    Var: Vary + 'static,
    Uni: Copy,
    Shd: Shader<Vert, Var, Uni>,
{
    // 0. Setup

    // #[cfg(feature = "stats")]
    // let stats = Stats::start_call(
    //     geometry.primitives().as_ref().len(),
    //     geometry.vertices().as_ref().len(),
    // );

    // 1. Vertex shader: transform vertices to clip space
    //let verts = vertex_transform(shader, uniform, verts);

    //let geom_in_clip = geometry.transform_to_clip(shader, uniform);

    // 2. Primitive assembly: map vertex indices to actual vertices
    //let prims = primitive_assembly(prims, &verts);

    let prims = geometry.to_primitives(shader, uniform);

    // 3. Clipping: clip against the view frustum
    let clipped = Clip::clip(prims, &view_frustum::PLANES);

    // 4. Rasterize: Turn visible primitives to fragments
    #[allow(unused_variables)]
    let (prims_out, verts_out) = if let Some(d) = ctx.depth_sort {
        // Optional depth sorting for use cases such as transparency
        // Only materialize to a vector if depth sort is enabled
        let mut clipped = clipped.collect::<Vec<_>>();
        depth_sort::<Prim, _>(&mut clipped, d);
        rasterize::<Prim, _, _, _>(
            clipped, shader, uniform, to_screen, target, ctx,
        )
    } else {
        rasterize::<Prim, _, _, _>(
            clipped, shader, uniform, to_screen, target, ctx,
        )
    };

    // #[cfg(feature = "stats")]
    // {
    //     *ctx.stats.borrow_mut() += stats.finish_call(prims_out, verts_out);
    // }
}

fn rasterize<Prim, Shd, Var, Uni>(
    clipped: impl IntoIterator<Item = Prim::Clip>,
    shader: &Shd,
    uniform: Uni,
    to_screen: Mat4<Ndc, Screen>,
    target: &mut impl Target,
    ctx: &Context,
) -> (usize, usize)
where
    Prim: Primitive<Var>,
    Shd: FragmentShader<Var, Uni>,
    Var: Vary,
    Uni: Copy,
{
    let mut out = (0, 0);
    for prim in clipped {
        // Transform to screen space
        let prim = Prim::to_screen(prim, &to_screen);
        // Back/frontface culling
        // TODO This could also be done earlier, before or as part of clipping,
        //      but determining back/frontface is more difficult before z-div
        if ctx.face_cull(Prim::is_backface(&prim)) {
            continue;
        }

        // Log output stats after culling
        out.0 += 1;
        out.1 += 3; // TODO Get number of verts in prim somehow

        // 4. Fragment shader and rasterization
        Prim::rasterize(prim, |scanline| {
            // Convert to fragments, shade, and draw to target
            target.rasterize(scanline, shader, uniform, ctx);
        });
    }
    out
}

#[inline]
fn vertex_transform<'a, Shd, Vtx: Clone + 'a, Var: Vary, Uni: Copy>(
    shader: &Shd,
    uniform: Uni,
    verts: impl IntoIterator<Item = &'a Vtx>,
) -> Vec<ClipVert<Var>>
where
    Shd: VertexShader<Vtx, Uni, Output = Vertex<ProjVec3, Var>>,
{
    verts
        .into_iter()
        // TODO Pass vertex as ref to shader
        .map(|v| shader.shade_vertex(v.clone(), uniform))
        .map(ClipVert::new)
        .collect()
}

fn depth_sort<P: Primitive<V>, V: Vary>(prims: &mut [P::Clip], d: DepthSort) {
    prims.sort_unstable_by(|t, u| {
        let z = P::depth(t);
        let w = P::depth(u);
        if d == DepthSort::FrontToBack {
            z.total_cmp(&w)
        } else {
            w.total_cmp(&z)
        }
    });
}

impl<S, Vtx, Var, Uni> Shader<Vtx, Var, Uni> for S where
    S: VertexShader<Vtx, Uni, Output = Vertex<ProjVec3, Var>>
        + FragmentShader<Var, Uni>
{
}
