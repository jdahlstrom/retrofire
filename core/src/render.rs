//! Turning 3D geometry into raster images.
//!
//! This module constitutes the core 3D rendering pipeline of `retrofire`.
//! It contains code for [clipping][clip], [transforming, shading][shader],
//! [texturing][tex], [rasterizing][raster], and [outputting][target] basic
//! geometric shapes such as triangles.

use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use clip::{view_frustum, Clip, ClipVec, ClipVert};
use ctx::{Context, DepthSort, FaceCull};
use raster::{tri_fill, Frag};
use shader::{FragmentShader, VertexShader};
use stats::Stats;
use target::Target;

use crate::geom::{Tri, Vertex};
use crate::math::mat::{RealToProj, RealToReal};
use crate::math::space::Real;
use crate::math::{Mat4x4, Vary, Vec3};

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

/// Mapping from model space to view space.
pub type ModelToView = RealToReal<3, Model, View>;

/// Mapping from model space to view space.
pub type ModelToProj = RealToProj<Model>;

/// Mapping from view space to projective space.
pub type ViewToProj = RealToProj<View>;

/// Mapping from NDC space to screen space.
pub type NdcToScreen = RealToReal<3, Ndc, Screen>;

/// Renders the given triangles into `target`.
pub fn render<Vtx: Clone, Var: Vary, Uni: Copy, Shd>(
    tris: impl AsRef<[Tri<usize>]>,
    verts: impl AsRef<[Vtx]>,
    shader: &Shd,
    uniform: Uni,
    viewport_tf: Mat4x4<NdcToScreen>,
    target: &mut impl Target,
    ctx: &Context,
) where
    Shd: VertexShader<Vtx, Uni, Output = Vertex<ClipVec, Var>>
        + FragmentShader<Frag<Var>>,
{
    let mut stats = Stats::start();

    stats.calls = 1.0;
    stats.prims.i += tris.as_ref().len();
    stats.verts.i += verts.as_ref().len();

    // Vertex shader: transform vertices to clip space
    let verts: Vec<_> = verts
        .as_ref()
        .iter()
        // TODO Pass vertex as ref to shader
        .cloned()
        .map(|v| ClipVert::new(shader.shade_vertex(v, uniform)))
        .collect();

    // Map triangle vertex indices to actual vertices
    let tris: Vec<_> = tris
        .as_ref()
        .iter()
        .map(|Tri(vs)| Tri(vs.map(|i| verts[i].clone())))
        .collect();

    // Clip against the view frustum
    let mut clipped = vec![];
    tris.clip(&view_frustum::PLANES, &mut clipped);

    if let Some(d) = ctx.depth_sort {
        depth_sort(&mut clipped, d);
    }

    for Tri(vs) in clipped {
        // Transform to screen space
        let vs = vs.map(|v| Vertex {
            pos: {
                // Perspective divide
                let pos = v.pos.project_to_real();
                // Viewport transform
                viewport_tf.apply(&pos.to())
            },
            attrib: v.attrib,
        });

        // Back/frontface culling
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
            stats.frags += target.rasterize(scanline, shader, &ctx);
        });
    }
    *ctx.stats.borrow_mut() += stats.finish();
}

fn depth_sort<A>(tris: &mut Vec<Tri<ClipVert<A>>>, d: DepthSort) {
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

fn is_backface<V>(vs: &[Vertex<Vec3<Real<3, Screen>>, V>]) -> bool {
    let v = vs[1].pos - vs[0].pos;
    let u = vs[2].pos - vs[0].pos;
    v[0] * u[1] - v[1] * u[0] > 0.0
}
