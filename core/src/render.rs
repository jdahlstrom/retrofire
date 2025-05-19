//! Turning 3D geometry into raster images.
//!
//! This module constitutes the core 3D rendering pipeline of `retrofire`.
//! It contains code for [clipping][clip], [transforming, shading][shader],
//! [texturing][tex], [rasterizing][raster], and [outputting][target] basic
//! geometric shapes such as triangles.

use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use crate::geom::Vertex;
use crate::math::{
    mat::{RealToProj, RealToReal},
    vary::ZDiv,
    vec::ProjVec4,
    vec3, Lerp, Mat4x4, Vary,
};

use {
    clip::{view_frustum, ClipVert},
    ctx::{DepthSort, FaceCull},
    raster::{Scanline, ScreenPt},
};

pub use {
    batch::Batch,
    cam::Camera,
    clip::Clip,
    ctx::Context,
    shader::{FragmentShader, VertexShader},
    stats::Stats,
    target::{Framebuf, Target},
    tex::{uv, TexCoord, Texture},
    text::Text,
};

pub mod batch;
pub mod cam;
pub mod clip;
pub mod ctx;
pub mod prim;
pub mod raster;
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
    fn to_screen(clip: Self::Clip, tf: &Mat4x4<NdcToScreen>) -> Self::Screen;

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
    VertexShader<Vtx, Uni, Output = Vertex<ProjVec4, Var>> + FragmentShader<Var>
{
}
impl<S, Vtx, Var, Uni> Shader<Vtx, Var, Uni> for S where
    S: VertexShader<Vtx, Uni, Output = Vertex<ProjVec4, Var>>
        + FragmentShader<Var>
{
}

/// Renders the given primitives into `target`.
pub fn render<Prim, Vtx: Clone, Var, Uni: Copy, Shd>(
    prims: impl AsRef<[Prim]>,
    verts: impl AsRef<[Vtx]>,
    shader: &Shd,
    uniform: Uni,
    to_screen: Mat4x4<NdcToScreen>,
    target: &mut impl Target,
    ctx: &Context,
) where
    Prim: Clone + Render<Var>,
    [<Prim>::Clip]: Clip<Item = Prim::Clip>,
    Var: Lerp + Vary,
    Shd: Shader<Vtx, Var, Uni>,
{
    // 0. Preparations
    let verts = verts.as_ref();
    let prims = prims.as_ref();

    let mut stats = Stats::start();
    stats.calls = 1.0;
    stats.prims.i = prims.len();
    stats.verts.i = verts.len();

    // 1. Vertex shader: transform vertices to clip space
    let verts: Vec<_> = verts
        // verts is borrowed, can't consume
        .iter()
        // TODO Pass vertex as ref to shader
        .cloned()
        .map(|v| shader.shade_vertex(v, uniform))
        .map(ClipVert::new)
        .collect();

    // 2. Primitive assembly: map vertex indices to actual vertices
    let prims: Vec<_> = prims
        .iter()
        .map(|tri| Prim::inline(tri.clone(), &verts))
        // Collect needed because clip takes a slice...
        .collect();

    // 3. Clipping: clip against the view frustum
    let mut clipped = vec![];
    view_frustum::clip(&prims[..], &mut clipped);

    // Optional depth sorting for use case such as transparency
    if let Some(d) = ctx.depth_sort {
        depth_sort::<Prim, _>(&mut clipped, d);
    }

    for prim in clipped {
        // Transform to screen space
        let prim = Prim::to_screen(prim, &to_screen);
        // Back/frontface culling
        // TODO This could also be done earlier, before or as part of clipping
        let bf = Prim::is_backface(&prim);
        match ctx.face_cull {
            Some(FaceCull::Back) if bf => continue,
            Some(FaceCull::Front) if !bf => continue,
            _ => {}
        }

        // Log output stats after culling
        stats.prims.o += 1;
        stats.verts.o += 3; // TODO Get number of verts in prim somehow

        // 4. Fragment shader and rasterization
        Prim::rasterize(prim, |scanline| {
            // Convert to fragments, shade, and draw to target
            stats.frags += target.rasterize(scanline, shader, ctx);
        });
    }
    *ctx.stats.borrow_mut() += stats.finish();
}

pub fn to_screen<V: ZDiv, const N: usize>(
    vs: [ClipVert<V>; N],
    tf: &Mat4x4<NdcToScreen>,
) -> [Vertex<ScreenPt, V>; N] {
    vs.map(|v| {
        let [x, y, _, w] = v.pos.0;
        // Perspective division (projection to the real plane)
        //
        // We use the screen-space z coordinate to store the reciprocal
        // of the original view-space depth. The interpolated reciprocal
        // is used in fragment processing for depth testing (larger values
        // are closer) and for perspective correction of the varyings.
        // TODO z_div could be space-aware
        let pos = vec3(x, y, 1.0).z_div(w);
        Vertex {
            // Viewport transform
            pos: tf.apply(&pos).to_pt(),
            // Perspective correction
            attrib: v.attrib.z_div(w),
        }
    })
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
