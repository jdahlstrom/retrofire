use std::mem::replace;

use math::Linear;
use math::mat::Mat4;
pub use render::Render;
pub use stats::Stats;
use util::buf::Buffer;
use util::color::Color;

use crate::raster::*;
use crate::shade::Shader;
use crate::vary::Varying;

mod hsr;
pub mod fx;
pub mod raster;
pub mod render;
pub mod scene;
pub mod shade;
pub mod stats;
pub mod text;
pub mod vary;

pub trait Rasterize {
    #[inline]
    fn rasterize_span<VI, VO, U, S>(
        &mut self,
        shader: &mut S,
        span: Span<(f32, VO), U>
    ) -> usize
    where
        VO: Linear<f32> + Copy,
        U: Copy,
        S: Shader<U, VI, VO>
    {
        let Span { y, xs: (x0, x1), vs: (v0, v1), uni } = span;

        if x1 < x0 {
            // TODO Investigate, should not happen
            return 0;
        }

        let mut vars =
            Varying::between(v0, v1, (x1 - x0) as f32 * INV_CH_SIZE)
            .map(|v| v.perspective_div());

        let mut v1 = vars.next().unwrap();
        let mut x = x0;
        let mut count = 0;
        while x < x1 {
            let v0 = replace(&mut v1, vars.next().unwrap());

            count += self.rasterize_chunk(shader, Span {
                y,
                xs: (x, (x + CH_SIZE).min(x1)),
                vs: (v0, v1),
                uni,
            });

            x += CH_SIZE;
        }
        count
    }

    fn rasterize_chunk<VI, VO, U, S>(
        &mut self,
        shader: &mut S,
        span: Span<(f32, VO), U>,
    ) -> usize
    where
        VO: Linear<f32> + Copy,
        U: Copy,
        S: Shader<U, VI, VO>;
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
    #[inline]
    fn rasterize_chunk<VI, VO, U, S>(
        &mut self,
        shader: &mut S,
        span: Span<(f32, VO), U>,
    ) -> usize
    where
        VO: Linear<f32> + Copy,
        U: Copy,
        S: Shader<U, VI, VO>
    {
        let Span { y, xs, vs, uni } = span;
        let vars = Varying::between(vs.0, vs.1, CH_SIZE as f32);
        let mut count = 0;
        for (x, (z, v)) in (xs.0..xs.1).zip(vars) {
            let coord = (x, y);
            count += (self.test)(Fragment{
                coord,
                varying: z,
                uniform: ()
            }).then(|| {
                shader
                    .shade_fragment(Fragment{
                        coord,
                        varying: v,
                        uniform: uni
                    })
                    .map(|col| {
                        (self.output)(Fragment{
                            coord,
                            varying: (z, col),
                            uniform: ()
                        });
                    })
            }).is_some() as usize;
        }
        count
    }
}

pub struct Framebuf<'a> {
    // TODO Support other color buffer formats
    pub color: Buffer<u8, &'a mut [u8]>,
    pub depth: &'a mut Buffer<f32>,
}

const CH_SIZE: usize = 16;
const INV_CH_SIZE: f32 = 1.0 / CH_SIZE as f32;

impl<'a> Rasterize for Framebuf<'a> {
    #[inline]
    fn rasterize_chunk<VI, VO, U, S>(
        &mut self,
        shader: &mut S,
        span: Span<(f32, VO), U>,
    ) -> usize
    where
        VO: Linear<f32> + Copy,
        U: Copy,
        S: Shader<U, VI, VO>,
    {
        let Span { y, xs, vs, uni } = span;

        let mut count = 0;
        let off = self.color.width() * y + xs.0;
        let end = self.color.width() * y + xs.1;
        let vars = Varying::between(vs.0, vs.1, CH_SIZE as f32);

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
