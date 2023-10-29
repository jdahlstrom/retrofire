//! Turning 3D geometry into raster images.
//!
//! This module constitutes the core 3D rendering pipeline of `retrofire`.
//! It contains code for [clipping][clip], [transforming, shading][shader],
//! [texturing][tex], [rasterizing][raster], and [outputting][target] basic
//! geometric shapes such as triangles.

use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use clip::{view_frustum, Clip};
use raster::{tri_fill, Frag};
use shader::{FragmentShader, VertexShader};
use stats::Stats;
use target::{Config, Target};

use crate::geom::{Tri, Vertex};
use crate::math::mat::{RealToProjective, RealToReal};
use crate::math::{Mat4x4, Vary};
use crate::render::clip::ClipVert;

pub mod clip;
pub mod raster;
pub mod shader;
pub mod stats;
pub mod target;
pub mod tex;

/// Model space coordinate basis.
#[derive(Copy, Clone, Debug, Default)]
pub struct Model;

/// World space coordinate basis.
#[derive(Copy, Clone, Debug, Default)]
pub struct World;

/// View (camera) space coordinate basis.
#[derive(Copy, Clone, Debug, Default)]
pub struct View;

/// NDC space coordinate basis (normalized device coordinates).
#[derive(Copy, Clone, Debug, Default)]
pub struct Ndc;

/// Screen space coordinate basis.
#[derive(Copy, Clone, Debug, Default)]
pub struct Screen;

/// Mapping from model space to view space.
pub type ModelToView = RealToReal<3, Model, View>;

/// Mapping from model space to view space.
pub type ModelToProjective = RealToProjective<Model>;

/// Mapping from view space to projective space.
pub type ViewToProjective = RealToProjective<View>;

/// Mapping from NDC space to screen space.
pub type NdcToScreen = RealToReal<3, Ndc, Screen>;

/// Renders the given triangles into `target`.
pub fn render<P, A, B, Sp, Sh, Uni, Tgt>(
    tris: impl AsRef<[Tri<usize>]>,
    verts: impl AsRef<[Vertex<P, A>]>,
    shader: &Sh,
    uni: Uni,
    vport_tf: Mat4x4<NdcToScreen>,
    target: &mut Tgt,
) -> Stats
where
    A: Clone + Debug,
    B: Vary</* TODO */ Diff = B> + Clone + Debug,
    Sh: VertexShader<Vertex<P, A>, Uni, Output = ClipVert<B>>
        + FragmentShader<Frag<B>>,
    Sp: Clone,
    Uni: Copy,
    Tgt: Target,
{
    let start = std::time::Instant::now();
    let mut stats = Stats::new();

    stats.calls = 1.0;
    stats.prims().i += tris.as_ref().len();
    stats.verts().i += verts.as_ref().len();

    // Vertex shader
    let verts: Vec<_> = verts
        .as_ref()
        .iter()
        // TODO Pass vertex as ref to shader
        .cloned()
        .map(|v| shader.shade_vertex(v, uni))
        .collect();

    let tris: Vec<_> = tris
        .as_ref()
        .iter()
        .map(|tri| Tri(tri.0.map(|i| verts[i].clone())))
        .collect();

    let mut clipped = vec![];
    tris.clip(&view_frustum::PLANES, &mut clipped);

    // TODO Optional depth sorting

    for tri in clipped {
        // TODO Backface culling

        stats.prims().o += 1;
        stats.verts().o += 3;

        let vs = tri.0.clone().map(|v| Vertex {
            pos: {
                // Perspective divide
                let pos = v.pos.project_to_real();
                // Viewport transform
                vport_tf.apply(&pos.to())
            },
            attrib: v.attrib,
        });

        // Fragment shader and rasterization
        tri_fill(vs, |scanline| {
            stats.frags().i += scanline.xs.len();
            // TODO count only frags that were actually output
            stats.frags().o += scanline.xs.len();
            target.rasterize(scanline, shader, Config::default());
        });
    }

    stats.time += start.elapsed();
    stats
}
