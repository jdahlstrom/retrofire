//! Fragment and vertex shaders.
//!
//! Shaders are functions that are used to customize vertex and fragment
//! handling during rendering.
//!
//! A *vertex shader* is responsible for transforming and projecting each
//! vertex in the rendered geometry, usually using a modelview matrix to move
//! the vertex to the view (camera) space and a projection matrix to transform
//! it to the clip space. Vertex shaders can also perform any other per-vertex
//! calculations and pass on the results as attributes of the output vertex.
//!
//! A *fragment shader* is used to compute the color of each individual pixel,
//! or fragment, drawn to the render target. The fragment shader receives as
//! input any vertex attributes interpolated across the primitive being
//! rasterized, such as color, texture coordinate, or normal vector.

use crate::{geom::Vertex, math::ProjVec3};

use super::Frag;

/// Trait for vertex shaders, used to transform vertices and perform other
/// per-vertex computations.
///
/// # Type parameters
/// * `In`: Type of the input vertex.
/// * `Uni`: Type of custom "uniform" (non-vertex-specific) data, such as
///   transform matrices, passed to the shader.
pub trait VertexShader<In, Uni> {
    /// The type of the output vertex.
    type VertexOut;

    /// Transforms `vertex` and does performs other per-vertex computations
    /// needed, outputting a new vertex of type `Self::Output`. Custom data
    /// that is not vertex-specific can be passed in the `uniform` parameter.
    ///
    /// # Panics
    /// `shade_vertex` should never panic.
    fn shade_vertex(&self, vertex: In, uniform: Uni) -> Self::VertexOut;
}

/// Trait for fragment shaders, used to compute the color of each individual
/// pixel, or fragment, rendered.
///
/// # Type parameters
/// * `Var`: The varying of the input fragment.
pub trait FragmentShader<Var, Uni> {
    /// The type of the output color.
    type FragmentOut;

    /// Computes the color of `frag`. Returns either `Some(color)`, or `None`
    /// if the fragment should be discarded.
    ///
    /// # Panics
    /// `shade_fragment` should never panic.
    fn shade_fragment(
        &self,
        frag: Frag<Var>,
        uniform: Uni,
    ) -> Option<Self::FragmentOut>;
}

impl<F, In, Out, Uni> VertexShader<In, Uni> for F
where
    F: Fn(In, Uni) -> Out,
{
    type VertexOut = Out;

    #[inline]
    fn shade_vertex(&self, vertex: In, uniform: Uni) -> Out {
        self(vertex, uniform)
    }
}

impl<C, F, Var, Uni> FragmentShader<Var, Uni> for F
where
    F: Fn(Frag<Var>, Uni) -> Option<C>,
{
    type FragmentOut = C;

    #[inline]
    fn shade_fragment(&self, frag: Frag<Var>, uniform: Uni) -> Option<C> {
        self(frag, uniform)
    }
}

pub fn new<Vs, Fs, Vtx, Var, Uni>(vs: Vs, fs: Fs) -> Shader<Vs, Fs>
where
    Vs: VertexShader<Vtx, Uni, VertexOut = Vertex<ProjVec3, Var>>,
    Fs: FragmentShader<Var, Uni>,
{
    Shader::new(vs, fs)
}

/// A type that composes a vertex and a fragment shader.
#[derive(Copy, Clone)]
pub struct Shader<Vs, Fs> {
    pub vertex_shader: Vs,
    pub fragment_shader: Fs,
}

impl<Vs, Fs> Shader<Vs, Fs> {
    /// Returns a new `Shader` with `vs` as the vertex shader
    /// and `fs` as the fragment shader.
    pub const fn new<In, Uni, Pos, Attr>(vs: Vs, fs: Fs) -> Self
    where
        Vs: VertexShader<In, Uni, VertexOut = Vertex<Pos, Attr>>,
        Fs: FragmentShader<Attr, Uni>,
    {
        Self {
            vertex_shader: vs,
            fragment_shader: fs,
        }
    }
}

impl<In, Vs, Fs, Uni> VertexShader<In, Uni> for Shader<Vs, Fs>
where
    Vs: VertexShader<In, Uni>,
{
    type VertexOut = Vs::VertexOut;

    #[inline]
    fn shade_vertex(&self, vertex: In, uniform: Uni) -> Self::VertexOut {
        self.vertex_shader.shade_vertex(vertex, uniform)
    }
}

impl<Vs, Fs, Var, Uni> FragmentShader<Var, Uni> for Shader<Vs, Fs>
where
    Fs: FragmentShader<Var, Uni>,
{
    type FragmentOut = Fs::FragmentOut;

    #[inline]
    fn shade_fragment(
        &self,
        frag: Frag<Var>,
        uni: Uni,
    ) -> Option<Self::FragmentOut> {
        self.fragment_shader.shade_fragment(frag, uni)
    }
}
