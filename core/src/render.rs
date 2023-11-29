//! Turning 3D geometry into raster images.
//!
//! This module constitutes the core 3D rendering pipeline of `retrofire`.
//! It contains code for [clipping][clip], [transforming, shading][shader],
//! [texturing][tex], [rasterizing][raster], and [outputting][target] basic
//! geometric shapes such as triangles.

use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use clip::{view_frustum, Clip, ClipVert};
use raster::{tri_fill, Frag};
use shader::{FragmentShader, VertexShader};
use stats::Stats;
use target::{Config, Target};

use crate::geom::{Tri, Vertex};
use crate::math::mat::{RealToProjective, RealToReal};
use crate::math::{Mat4x4, Vary};

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
pub fn render<P, A, V, Sh, U>(
    tris: impl AsRef<[Tri<usize>]>,
    verts: impl AsRef<[Vertex<P, A>]>,
    shader: &Sh,
    uniform: U,
    viewport_tf: Mat4x4<NdcToScreen>,
    target: &mut impl Target,
) -> Stats
where
    P: Clone,
    A: Clone + Debug,
    V: Vary + Debug,
    Sh: VertexShader<Vertex<P, A>, U, Output = ClipVert<V>>
        + FragmentShader<Frag<V>>,
    U: Copy,
{
    #[cfg(feature = "std")]
    let start = std::time::Instant::now();
    let mut stats = Stats::new();

    stats.calls = 1.0;
    stats.prims().i += tris.as_ref().len();
    stats.verts().i += verts.as_ref().len();

    // Vertex shader: transform vertices to clip space
    let verts: Vec<_> = verts
        .as_ref()
        .iter()
        // TODO Pass vertex as ref to shader
        .cloned()
        .map(|v| shader.shade_vertex(v, uniform))
        .collect();

    // TODO use outcodes to cull tris fully outside the frustum here

    // Map triangle vertex indices to actual vertices
    let tris: Vec<_> = tris
        .as_ref()
        .iter()
        .map(|Tri(vs)| Tri(vs.map(|i| verts[i].clone())))
        .collect();

    // Clip against the view frustum
    let mut clipped = vec![];
    tris.clip(&view_frustum::PLANES, &mut clipped);

    // TODO Optional depth sorting

    for Tri(vs) in clipped {
        // TODO Backface culling

        stats.prims().o += 1;
        stats.verts().o += 3;

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

        // Fragment shader and rasterization
        tri_fill(vs, |scanline| {
            stats.frags().i += scanline.xs.len();
            // TODO count only frags that were actually output
            stats.frags().o += scanline.xs.len();
            target.rasterize(scanline, shader, Config::default());
        });
    }
    #[cfg(feature = "std")]
    {
        stats.time += start.elapsed();
    }
    stats
}
