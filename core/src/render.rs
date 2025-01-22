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
    rgba,
    vary::ZDiv,
    vec::ProjVec4,
    vec3, Lerp, Mat4x4, Vary,
};

use {
    clip::{view_frustum, ClipVert},
    ctx::{DepthSort, FaceCull},
    raster::{Scanline, ScreenPt},
};

use crate::render::raster::line;
pub use {
    batch::Batch,
    cam::Camera,
    clip::Clip,
    ctx::Context,
    prim::Primitive,
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

pub type ToClip<P, V> = <P as Primitive<usize>>::Mapped<ClipVert<V>>;
pub type ToScreen<P, V> =
    <ToClip<P, V> as Primitive<ClipVert<V>>>::Mapped<Vertex<ScreenPt, V>>;

/// Renderable geometric primitive.
pub trait Render<V: Vary>: Primitive<usize> + Sized {
    /// Maps the indices of an indexed primitive to clip-space vertices.
    fn inline(ixd: Self, vs: &[ClipVert<V>]) -> ToClip<Self, V> {
        ixd.map_vertices(|i| vs[i].clone())
    }

    /// Returns the (average) depth of the primitive.
    fn depth(_: &ToClip<Self, V>) -> f32 {
        f32::INFINITY
    }

    /// Returns whether the primitive is facing away from the camera.
    fn is_backface(_: &ToScreen<Self, V>) -> bool {
        false
    }

    /// Transforms the argument from NDC to screen space.
    fn to_screen(
        clip: ToClip<Self, V>,
        tf: &Mat4x4<NdcToScreen>,
    ) -> ToScreen<Self, V> {
        clip.map_vertices(|v| {
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

    /// Rasterizes the argument by calling the function for each scanline.
    fn rasterize<F: FnMut(Scanline<V>)>(
        scr: &ToScreen<Self, V>,
        scanline_fn: F,
    );
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
    Prim: Render<Var> + Clone,
    [ToClip<Prim, Var>]: Clip<Item = ToClip<Prim, Var>>,
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
        stats.verts.o += prim.vertices().as_ref().len();

        // 4. Fragment shader and rasterization
        Prim::rasterize(&prim, |scanline| {
            // Convert to fragments, shade, and draw to target
            stats.frags += target.rasterize(scanline, shader, ctx);
        });

        let vs = prim.vertices();

        let _vs = [
            &vs.as_ref()[0],
            &vs.as_ref()[1],
            &vs.as_ref()[2],
            &vs.as_ref()[0],
        ];

        let ctx = &mut ctx.clone();
        ctx.color_write = true;

        let mut v0 = vs.as_ref()[0].clone();
        for v1 in vs.as_ref()[1..]
            .iter()
            .cloned()
            .chain([v0.clone()])
        {
            let mut edge = [v0.clone(), v1.clone()];
            edge[0].pos[2] += 0.005;
            edge[1].pos[2] += 0.005;
            line(edge, |sl| {
                target.rasterize(sl, &|_| rgba(0xFF, 0, 0, 0), ctx);
            });
            v0 = v1;
        }
    }
    *ctx.stats.borrow_mut() += stats.finish();
}

fn depth_sort<P: Render<V>, V: Vary>(
    prims: &mut [P::Mapped<ClipVert<V>>],
    d: DepthSort,
) {
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
