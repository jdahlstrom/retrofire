//! Turning 3D geometry into raster images.
//!
//! This module constitutes the core 3D rendering pipeline of `retrofire`.
//! It contains code for [clipping][clip], [transforming, shading][shader],
//! [texturing][tex], [rasterizing][raster], and [outputting][target] basic
//! geometric shapes such as triangles.

use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use crate::geom::{Tri, Vertex};
use crate::math::{
    mat::{Mat4x4, RealToProj, RealToReal},
    vary::Vary,
    vec::{vec3, ProjVec4},
    Lerp,
};

use clip::{view_frustum, Clip, ClipVert};
use ctx::{Context, DepthSort, FaceCull};
use raster::{tri_fill, ScreenPt};
use shader::{FragmentShader, VertexShader};
use stats::Stats;
use target::Target;

pub mod batch;
pub mod cam;
pub mod clip;
pub mod ctx;
pub mod raster;
pub mod shader;
pub mod stats;
pub mod target;
pub mod tex;

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

/// Renders the given triangles into `target`.
pub fn render<Vtx: Clone, Var: Lerp + Vary, Uni: Copy, Shd>(
    tris: impl AsRef<[Tri<usize>]>,
    verts: impl AsRef<[Vtx]>,
    shader: &Shd,
    uniform: Uni,
    to_screen: Mat4x4<NdcToScreen>,
    target: &mut impl Target,
    ctx: &Context,
) where
    Shd: Shader<Vtx, Var, Uni>,
{
    let verts = verts.as_ref();
    let tris = tris.as_ref();
    let mut stats = Stats::start();

    stats.calls = 1.0;
    stats.prims.i += tris.len();
    stats.verts.i += verts.len();

    // Vertex shader: transform vertices to clip space
    let verts: Vec<_> = verts
        .iter()
        // TODO Pass vertex as ref to shader
        .cloned()
        .map(|v| ClipVert::new(shader.shade_vertex(v, uniform)))
        .collect();

    // Map triangle vertex indices to actual vertices
    let tris: Vec<_> = tris
        .iter()
        .map(|Tri(vs)| Tri(vs.map(|i| verts[i].clone())))
        .collect();

    // Clip against the view frustum
    let mut clipped = vec![];
    tris.clip(&view_frustum::PLANES, &mut clipped);

    // Optional depth sorting for use case such as transparency
    if let Some(d) = ctx.depth_sort {
        depth_sort(&mut clipped, d);
    }

    for Tri(vs) in clipped {
        // Transform to screen space
        let vs = vs.map(|v| {
            let [x, y, _, w] = v.pos.0;
            // Perspective division (projection to the real plane)
            //
            // We use the screen-space z coordinate to store the reciprocal
            // of the original view-space depth. The interpolated reciprocal
            // is used in fragment processing for depth testing (larger values
            // are closer) and for perspective correction of the varyings.
            let pos = vec3(x, y, 1.0) / w; // TODO
            Vertex {
                // Viewport transform
                pos: to_screen.apply(&pos).to_pt(),
                // Perspective correction
                attrib: v.attrib.z_div(w),
            }
        });

        // Back/frontface culling
        //
        // TODO This could also be done earlier, before or as part of clipping
        match ctx.face_cull {
            Some(FaceCull::Back) if is_backface(&vs) => continue,
            Some(FaceCull::Front) if !is_backface(&vs) => continue,
            _ => {}
        }

        // Log output stats after culling
        stats.prims.o += 1;
        stats.verts.o += 3;

        // Fragment shader and rasterization
        tri_fill(vs, |scanline| {
            // Convert to fragments and shade
            stats.frags += target.rasterize(scanline, shader, ctx);
        });
    }
    *ctx.stats.borrow_mut() += stats.finish();
}

fn depth_sort<A>(tris: &mut [Tri<ClipVert<A>>], d: DepthSort) {
    tris.sort_unstable_by(|t, u| {
        let z = t.0[0].pos.z() + t.0[1].pos.z() + t.0[2].pos.z();
        let w = u.0[0].pos.z() + u.0[1].pos.z() + u.0[2].pos.z();
        if d == DepthSort::FrontToBack {
            z.total_cmp(&w)
        } else {
            w.total_cmp(&z)
        }
    });
}

fn is_backface<V>(vs: &[Vertex<ScreenPt, V>]) -> bool {
    let v = vs[1].pos - vs[0].pos;
    let u = vs[2].pos - vs[0].pos;
    v[0] * u[1] - v[1] * u[0] > 0.0
}
