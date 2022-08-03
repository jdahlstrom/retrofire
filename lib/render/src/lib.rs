use std::borrow::Cow;
use std::fmt::Debug;
use std::mem::replace;

use geom::mesh::{Face, Vertex};
use math::Linear;
use math::mat::Mat4;
use math::transform::Transform;
use math::vary::Vary;
pub use render::Render;
pub use stats::Stats;
use util::buf::Buffer;
use util::color::Color;

use crate::hsr::{clip, frontface, vertex_visibility, Visibility};
use crate::raster::{Fragment, line, Span, tri_fill};
use crate::shade::{FragmentShader, Shader};

mod hsr;
pub mod fx;
pub mod raster;
pub mod render;
pub mod scene;
pub mod shade;
pub mod stats;
pub mod text;


const CHUNK_SIZE: usize = 16;

pub struct Batch<P, V> {
    pub prims: Vec<P>,
    pub verts: Vec<Vertex<V>>,
}

impl<'a, P, V, U> Render<U, V, V> for Batch<P, V>
where
    P: Primitive<U> + Clone,
    V: Linear<f32> + Copy,
    U: Copy + Debug,
{
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<U, V, VtxOut=V>,
        R: Rasterize
    {
        st.stats.prims_in += self.prims.len();
        st.stats.verts_in += self.verts.len();

        //
        // 1. MVP transform, vertex shade
        //

        let mvp = &st.modelview * &st.projection;

        let mut vs: Vec<_> = self.verts.iter().map(|v| {
            let mut v = v.clone();
            v.coord.transform_mut(&mvp);
            shader.shade_vertex(v)
        }).collect();

        //
        // 2. Frustum culling and clipping, backface culling
        //

        let vis = vertex_visibility(vs.iter().map(|v| &v.coord));

        let prims: Cow<Vec<_>> = match vis {
            Visibility::Unclipped => Cow::Borrowed(&self.prims),
            Visibility::Clipped => Cow::Owned(self.prims.iter()
                // Clipping and culling a primitive
                // yields zero or more new vertices and primitives
                .flat_map(|prim| prim.clip(&mut vs))
                .collect()),
            Visibility::Hidden => return
        };

        // TODO Calculate which verts are part of visible prims

        for v in &mut vs {

            // 3. Perspective divide
            v.coord.x /= v.coord.w;
            v.coord.y /= v.coord.w;
            v.coord.z = v.coord.w;
            v.coord.w = 1.0;

            // 4. Viewport transform
            v.coord.transform_mut(&st.viewport);
        }

        //
        // Rasterization
        //

        for p in prims.iter() {
            p.rasterize(&vs, st, shader, raster);
        }

        st.stats.prims_out += prims.len();
        // st.stats.verts_out += .len(); TODO
    }
}

pub trait Primitive<U: Copy + Debug>: Sized + Debug {
    type Vertices: IntoIterator;

    fn vertices(&self) -> Self::Vertices;

    fn clip<V>(&self, verts: &mut Vec<Vertex<V>>) -> Vec<Self>
    where
        V: Linear<f32> + Copy;

    fn rasterize<S, R, V>(&self, verts: &[Vertex<V>], st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: FragmentShader<V, U>,
        R: Rasterize,
        V: Linear<f32> + Copy;
}


impl<U> Primitive<U> for Face<usize, U>
where
    U: Copy + Debug,
{
    type Vertices = [usize; 3];

    fn vertices(&self) -> Self::Vertices {
        self.verts
    }

    fn clip<V>(&self, verts: &mut Vec<Vertex<V>>) -> Vec<Self>
    where
        V: Linear<f32> + Copy,
    {
        let face_verts = self.verts.map(|i| verts[i]);

        let vis = vertex_visibility(face_verts.iter().map(|v| &v.coord));

        match vis {
            Visibility::Unclipped => {
                if frontface(&face_verts) {
                    vec![*self]
                } else {
                    vec![]
                }
            }
            Visibility::Clipped => {
                let mut verts_in = face_verts.to_vec();
                let mut verts_out = vec![];
                clip(&mut verts_in, &mut verts_out);

                if verts_out.is_empty() {
                    return vec![];
                }
                if !frontface(&[verts_out[0], verts_out[1], verts_out[2]]) {
                    // New faces are coplanar, if one is a backface then all are
                    return vec![];
                }

                let old_count = verts.len();
                verts.append(&mut verts_out);
                let new_count = verts.len();

                (old_count + 1..new_count - 1)
                    .map(|i| Face {
                        verts: [old_count, i, i + 1],
                        attr: self.attr,
                    })
                    .collect()
            }
            Visibility::Hidden => {
                vec![]
            }
        }
    }

    fn rasterize<S, R, V>(&self, verts: &[Vertex<V>], st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: FragmentShader<V, U>,
        R: Rasterize,
        V: Linear<f32> + Copy,
    {
        let vs = self.vertices()
            .map(|i| with_depth(verts[i]));

        // TODO Dispatching based on whether PC is enabled causes a major
        // TODO performance regression in checkers.rs but not in benchmarks.
        // TODO Should be investigated.
        if true /* st.options.perspective_correct */ {
            tri_fill(vs, self.attr, |span| {
                st.stats.pix_in += span.xs.1 - span.xs.0;
                st.stats.pix_out += raster.rasterize_span(shader, span);
            });
        } else {
            // TODO
        }

        if let Some(col) = st.options.wireframes {
            let [a, b, c] = vs;
            for edge in [[a, b], [b, c], [c, a]] {
                line(edge, col, |mut frag| {
                    // Avoid Z fighting
                    frag.varying.0 -= 0.01;
                    raster.rasterize_frag(
                        &mut |frag: Fragment<_, _>| Some(frag.uniform),
                        frag,
                    );
                });
            }
        }
    }
}

pub trait Rasterize {
    fn rasterize_span<V, U, S>(
        &mut self,
        shader: &mut S,
        span: Span<(f32, V), U>,
    ) -> usize
    where
        V: Linear<f32> + Copy,
        U: Copy,
        S: FragmentShader<V, U>
    {
        let Span { y, xs: (x0, x1), vs: (v0, v1), uni } = span;

        let mut vars =
            v0.vary(&v1, x1.saturating_sub(x0) as f32 / CHUNK_SIZE as f32)
                .map(|v| v.perspective_div());

        let mut v1 = vars.next().unwrap();
        let mut count = 0;
        for x in (x0..x1).step_by(CHUNK_SIZE) {
            let v0 = replace(&mut v1, vars.next().unwrap());

            count += self.rasterize_chunk(shader, Span {
                y,
                xs: (x, (x + CHUNK_SIZE).min(x1)),
                vs: (v0, v1),
                uni,
            });
        }
        count
    }

    fn rasterize_chunk<V, U, S>(
        &mut self,
        shader: &mut S,
        span: Span<(f32, V), U>,
    ) -> usize
    where
        V: Linear<f32> + Copy,
        U: Copy,
        S: FragmentShader<V, U>;

    fn rasterize_frag<V, U, S>(
        &mut self,
        shader: &mut S,
        frag: Fragment<(f32, V), U>,
    ) -> usize
    where
        V: Linear<f32> + Copy,
        U: Copy,
        S: FragmentShader<V, U>;
}

pub struct Raster<Test, Output> {
    pub test: Test,
    pub output: Output,
}

impl<Test, Output> Rasterize for Raster<Test, Output>
where
    Test: Fn(Fragment<f32>) -> bool,
    Output: FnMut(Fragment<(f32, Color)>),
{
    fn rasterize_chunk<V, U, S>(
        &mut self,
        shader: &mut S,
        span: Span<(f32, V), U>,
    ) -> usize
    where
        V: Linear<f32> + Copy,
        U: Copy,
        S: FragmentShader<V, U>
    {
        let Span { y, xs, vs, uni } = span;
        let vars = vs.0.vary(&vs.1, CHUNK_SIZE as f32);
        let mut count = 0;
        for (x, v) in (xs.0..xs.1).zip(vars) {
            count += self.rasterize_frag(shader, Fragment {
                coord: (x, y),
                varying: v,
                uniform: uni,
            });
        }
        count
    }
    fn rasterize_frag<V, U, S>(
        &mut self,
        shader: &mut S,
        frag: Fragment<(f32, V), U>,
    ) -> usize
    where
        V: Linear<f32> + Copy,
        U: Copy,
        S: FragmentShader<V, U>
    {
        let (z, v) = frag.varying;
        if (self.test)(frag.varying(z).uniform(())) {
            if let Some(col) = shader.shade_fragment(frag.varying(v)) {
                (self.output)(frag.varying((z, col)).uniform(()));
                return 1;
            }
        }
        0
    }
}

pub struct Framebuf<'a> {
    // TODO Support other color buffer formats
    pub color: Buffer<u8, &'a mut [u8]>,
    pub depth: &'a mut Buffer<f32>,
}

impl<'a> Rasterize for Framebuf<'a> {
    fn rasterize_chunk<V, U, S>(
        &mut self,
        shader: &mut S,
        span: Span<(f32, V), U>,
    ) -> usize
    where
        V: Linear<f32> + Copy,
        U: Copy,
        S: FragmentShader<V, U>,
    {
        let Span { y, xs, vs, uni } = span;

        let mut count = 0;
        let off = self.color.width() * y + xs.0;
        let end = self.color.width() * y + xs.1;
        let vars = vs.0.vary(&vs.1, CHUNK_SIZE as f32);

        let color_slice = &mut self.color.data_mut()[4 * off..4 * end];
        let depth_slice = &mut self.depth.data_mut()[off..end];

        color_slice.chunks_exact_mut(4)
            .zip(depth_slice)
            .zip(xs.0..xs.1)
            .zip(vars)
            .for_each(|(((col, depth), x), (z, var))| {
                if z < *depth {
                    let frag = Fragment {
                        coord: (x, y),
                        varying: var,
                        uniform: uni,
                    };
                    if let Some(c) = shader.shade_fragment(frag) {
                        let [_, r, g, b] = c.to_argb();
                        col[0] = b;
                        col[1] = g;
                        col[2] = r;
                        *depth = z;
                        count += 1;
                    }
                }
            });
        count
    }
    fn rasterize_frag<V, U, S>(
        &mut self,
        shader: &mut S,
        frag: Fragment<(f32, V), U>,
    ) -> usize
    where
        V: Linear<f32> + Copy,
        U: Copy,
        S: FragmentShader<V, U>
    {
        let (x, y) = frag.coord;
        let (z, v) = frag.varying;
        let offset = self.color.width() * y + x;

        let depth = &mut self.depth.data_mut()[offset];
        if z < *depth {
            let frag = frag.varying(v);
            if let Some(c) = shader.shade_fragment(frag) {
                let [_, r, g, b] = c.to_argb();
                let col = &mut self.color.data_mut()[4 * offset..][..4];
                col[0] = b;
                col[1] = g;
                col[2] = r;
                *depth = z;
                return 1;
            }
        }
        0
    }
}

#[derive(Default, Clone)]
pub struct State {
    pub modelview: Mat4,
    pub projection: Mat4,
    pub viewport: Mat4,
    pub options: Options,
    pub stats: Stats,
}

#[derive(Copy, Clone, Default)]
pub struct Options {
    pub perspective_correct: bool,
    pub depth_sort: bool,
    pub wireframes: Option<Color>,
    pub bounding_boxes: Option<Color>,
}

impl State {
    pub fn new() -> State {
        Self::default()
    }
}

#[inline]
fn with_depth<A>(v: Vertex<A>) -> Vertex<(f32, A)> {
    v.attr_with(|v| (v.coord.z, v.attr))
}
