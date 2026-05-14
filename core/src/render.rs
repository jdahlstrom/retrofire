//! Turning 3D geometry into raster images.
//!
//! This module constitutes the core 3D rendering pipeline of `retrofire`.
//! It contains code for [clipping][clip], [transforming, shading][shader],
//! [texturing][tex], [rasterizing][raster], and [outputting][target] basic
//! geometric shapes such as triangles.

use alloc::vec::Vec;
use core::{fmt::Debug, ops::DerefMut};
#[cfg(feature = "std")]
use std::time::Instant;

use crate::geom::Vertex;
use crate::math::{
    Mat4, ProjVec3, Vary,
    mat::{RealToProj, RealToReal},
};

use self::{
    clip::{ClipVert, view_frustum},
    ctx::DepthSort,
    raster::Scanline,
};

pub(super) mod re_exports {
    pub use super::{
        batch::Batch,
        cam::Camera,
        clip::Clip,
        ctx::Context,
        raster::Frag,
        shader::{FragmentShader, VertexShader},
        stats::Stats,
        target::{Colorbuf, Framebuf, Target},
        tex::{TexCoord, Texture, uv},
        text::Text,
    };
}
pub use re_exports::*;

pub mod batch;
pub mod cam;
pub mod clip;
pub mod ctx;
pub mod debug;
pub mod prim;
pub mod raster;
pub mod scene;
pub mod shader;
pub mod stats;
pub mod target;
pub mod tex;
pub mod text;

/// Renderable geometric primitive.
pub trait Render<V: Vary> {
    /// The type of this primitive in clip space
    type Clip;

    /// The type for which `Clip` is implemented.
    type Clips: Clip<Item = Self::Clip> + ?Sized;

    /// The type of this primitive in screen space.
    type Screen;

    /// Maps the indexes of the argument to vertices.
    fn inline(ixd: Self, vs: &[ClipVert<V>]) -> Self::Clip;

    /// Returns the (average) depth of the argument.
    fn depth(_clip: &Self::Clip) -> f32 {
        f32::INFINITY
    }

    /// Returns whether the argument is facing away from the camera.
    fn is_backface(_: &Self::Screen) -> bool {
        false
    }

    /// Transforms the argument from NDC to screen space.
    fn to_screen(clip: Self::Clip, tf: &Mat4<Ndc, Screen>) -> Self::Screen;

    /// Rasterizes the argument by calling the function for each scanline.
    fn rasterize<F: FnMut(Scanline<V>)>(scr: Self::Screen, scanline_fn: F);
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

// Mapping from model space to world space.
pub type ModelToWorld = RealToReal<3, Model, World>;

// Mapping from world space to view space.
pub type WorldToView = RealToReal<3, World, View>;

/// Mapping from model space to view space.
pub type ModelToView = RealToReal<3, Model, View>;

/// Mapping from model space to view space.
pub type ModelToProj = RealToProj<Model>;

/// Mapping from view space to projective space.
pub type ViewToProj = RealToProj<View>;

/// Mapping from NDC space to screen space.
pub type NdcToScreen = RealToReal<3, Ndc, Screen>;

/// Alias for combined vertex+fragment shader types
pub trait Shader<Vtx, Var, Uni>:
    VertexShader<Vtx, Uni, Output = Vertex<ProjVec3, Var>> + FragmentShader<Var>
{
}
impl<S, Vtx, Var, Uni> Shader<Vtx, Var, Uni> for S where
    S: VertexShader<Vtx, Uni, Output = Vertex<ProjVec3, Var>>
        + FragmentShader<Var>
{
}

/// Renders the given primitives into `target`.
pub fn render<Prim, Vtx: Clone, Var, Uni: Copy, Shd>(
    prims: impl AsRef<[Prim]>,
    verts: impl AsRef<[Vtx]>,
    shader: &Shd,
    uniform: Uni,
    to_screen: Mat4<Ndc, Screen>,
    target: &mut impl Target,
    ctx: &Context,
) where
    Prim: Render<Var> + Clone,
    [<Prim>::Clip]: Clip<Item = Prim::Clip>,
    Var: Vary,
    Shd: Shader<Vtx, Var, Uni>,
{
    // 0. Preparations
    let verts = verts.as_ref();
    let prims = prims.as_ref();

    let mut stats = Stats::new();
    stats.calls = 1.0;
    stats.prims.i = prims.len();
    stats.verts.i = verts.len();
    #[cfg(feature = "std")]
    let start = Instant::now();

    // 1. Vertex shader: transform vertices to clip space
    let verts = vertex_transform(shader, uniform, verts);

    // 2. Primitive assembly: map vertex indices to actual vertices
    let prims: Vec<_> = primitive_assembly(prims, &verts);

    // 3. Clipping: clip against the view frustum
    let mut clipped = Vec::new();
    view_frustum::clip(&prims[..], &mut clipped);

    // Optional depth sorting for use cases such as transparency
    if let Some(d) = ctx.depth_sort {
        depth_sort::<Prim, _>(&mut clipped, d);
    }

    // 4. Rasterize: Turn visible primitives to fragments
    stats += rasterize::<Prim, _, _>(clipped, shader, to_screen, target, ctx);

    #[cfg(feature = "std")]
    {
        stats.render_time = start.elapsed();
    }

    *ctx.stats.borrow_mut() += stats.finish();
}

fn rasterize<Prim: Render<Var>, Shd: FragmentShader<Var>, Var: Vary>(
    clipped: Vec<Prim::Clip>,
    shader: &Shd,
    to_screen: Mat4<Ndc, Screen>,
    mut target: &mut impl Target,
    ctx: &Context,
) -> Stats {
    let mut stats = Stats::new();
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
        stats.prims.o += 1;
        stats.verts.o += 3; // TODO Get number of verts in prim somehow

        // 4. Fragment shader and rasterization
        Prim::rasterize(prim, |scanline| {
            // Convert to fragments, shade, and draw to target
            stats.frags += target
                .deref_mut()
                .rasterize(scanline, shader, ctx);
        });
    }
    stats
}

#[inline]
fn primitive_assembly<Prim: Render<Var> + Clone, Var: Vary>(
    prims: &[Prim],
    verts: &[ClipVert<Var>],
) -> Vec<Prim::Clip> {
    prims
        .iter()
        .cloned()
        .map(|prim| Prim::inline(prim, &verts))
        // Collect needed because clip takes a slice...
        .collect()
}

#[inline]
fn vertex_transform<Shd, Vtx: Clone, Var: Vary, Uni: Copy>(
    shader: &Shd,
    uniform: Uni,
    verts: &[Vtx],
) -> Vec<ClipVert<Var>>
where
    Shd: VertexShader<Vtx, Uni, Output = Vertex<ProjVec3, Var>>,
{
    verts
        // verts is borrowed, can't consume
        .iter()
        // TODO Pass vertex as ref to shader
        .cloned()
        .map(|v| shader.shade_vertex(v, uniform))
        .map(ClipVert::new)
        .collect()
}

fn depth_sort<P: Render<V>, V: Vary>(prims: &mut [P::Clip], d: DepthSort) {
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
