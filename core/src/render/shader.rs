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

use crate::geom::Vertex;
use crate::math::color::Color4;
use crate::math::Vec3;

/// A fragment, an individual "pixel" being drawn on the screen or other
/// render target.
// TODO Move to raster.rs once it's added
#[derive(Copy, Clone, Debug)]
pub struct Frag<V> {
    // Fragment position in the render target.
    pub pos: Vec3,
    // Interpolated vertex attributes aka varyings.
    pub var: V,
}

/// Trait for vertex shaders, used to transform vertices and perform other
/// per-vertex computations.
///
/// # Type parameters
/// * `In`: Type of the input vertex.
/// * `Uni`: Type of custom "uniform" (non-vertex-specific) data, such as
///     transform matrices, passed to the shader.
pub trait VertexShader<In, Uni> {
    /// The type of the output vertex.
    type Output;
    /// Transforms `vertex` and does performs other per-vertex computations
    /// needed, outputting a new vertex of type `Self::Output`. Custom data
    /// that is not vertex-specific can be passed in the `uniform` parameter.
    ///
    /// # Panics
    /// `shade_vertex` should never panic.
    fn shade_vertex(&self, vertex: In, uniform: Uni) -> Self::Output;
}

/// Trait for fragment shaders, used to compute the color of each individual
/// pixel, or fragment, rendered.
///
/// # Type parameters
/// * `F`: Type of the input fragment.
pub trait FragmentShader<F> {
    /// Computes the color of `frag`. Returns either `Some(color)`, or `None`
    /// if the fragment should be discarded.
    ///
    /// # Panics
    /// `shade_fragment` should never panic.
    fn shade_fragment(&self, frag: F) -> Option<Color4>;
}

impl<F, In, Out, Uni> VertexShader<In, Uni> for F
where
    F: Fn(In, Uni) -> Out,
{
    type Output = Out;

    fn shade_vertex(&self, vertex: In, uniform: Uni) -> Out {
        self(vertex, uniform)
    }
}

impl<F, Frag, Out> FragmentShader<Frag> for F
where
    F: Fn(Frag) -> Out,
    Out: Into<Option<Color4>>,
{
    fn shade_fragment(&self, frag: Frag) -> Option<Color4> {
        self(frag).into()
    }
}

/// A type that composes a vertex and a fragment shader.
#[derive(Copy, Clone)]
pub struct Shader<Vs, Fs> {
    pub vertex_shader: Vs,
    pub fragment_shader: Fs,
}

impl<In, Vs, Fs, Uni> VertexShader<In, Uni> for Shader<Vs, Fs>
where
    Vs: VertexShader<In, Uni>,
{
    type Output = Vs::Output;

    fn shade_vertex(&self, vertex: In, uniform: Uni) -> Self::Output {
        self.vertex_shader.shade_vertex(vertex, uniform)
    }
}

impl<Vs, Fs, F> FragmentShader<F> for Shader<Vs, Fs>
where
    Fs: FragmentShader<F>,
{
    fn shade_fragment(&self, frag: F) -> Option<Color4> {
        self.fragment_shader.shade_fragment(frag)
    }
}

impl<Vs, Fs> Shader<Vs, Fs> {
    /// Returns a new `Shader` with `vs` as the vertex shader
    /// and `fs` as the fragment shader.
    pub fn new<In, Uni, P, A>(vs: Vs, fs: Fs) -> Self
    where
        Vs: VertexShader<In, Uni, Output = Vertex<P, A>>,
        Fs: FragmentShader<Frag<A>>,
    {
        Self {
            vertex_shader: vs,
            fragment_shader: fs,
        }
    }
}
