//! Turning 3D geometry into raster images.
//!
//! This module constitutes the core 3D rendering pipeline of `retrofire`.
//! It contains code for [clipping][clip], [transforming, shading][shader],
//! [texturing][tex], [rasterizing][raster], and [outputting][target] basic
//! geometric shapes such as triangles.

use alloc::{vec, vec::Vec};
use core::fmt::Debug;

use clip::{view_frustum, Clip, ClipVec};
use raster::{tri_fill, Frag};
use shader::{FragmentShader, VertexShader};
use target::{Config, Target};

use crate::geom::{Tri, Vertex};
use crate::math::mat::{RealToProjective, RealToReal};
use crate::math::{Linear, Mat4x4, Vec3};

pub mod clip;
pub mod raster;
pub mod shader;
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
pub fn render<A, B, Sh, Uni, Tgt>(
    tris: &[Tri<usize>],
    verts: &[Vertex<Vec3, A>],
    shader: &Sh,
    uni: Uni,
    vport_tf: Mat4x4<NdcToScreen>,
    target: &mut Tgt,
) where
    A: Clone + Debug,
    B: Linear<Scalar = f32> + Clone + Debug,
    Sh: VertexShader<Vertex<Vec3, A>, Uni, Output = Vertex<ClipVec, B>>
        + FragmentShader<Frag<B>>,
    Uni: Copy,
    Tgt: Target,
{
    // Vertex shader
    let verts: Vec<_> = verts
        .iter()
        // TODO Pass vertex as ref to shader
        .cloned()
        .map(|v| shader.shade_vertex(v, uni))
        .collect();

    let tris: Vec<_> = tris
        .iter()
        .map(|tri| Tri(tri.0.map(|i| verts[i].clone())))
        .collect();

    let mut clipped = vec![];
    tris.clip(&view_frustum::PLANES, &mut clipped);

    // TODO Optional depth sorting

    for tri in clipped {
        // TODO Backface culling

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
        tri_fill(vs, |s| target.rasterize(s, shader, Config::default()))
    }
}
