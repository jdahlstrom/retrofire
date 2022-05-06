use math::mat::Mat4;
pub use stats::Stats;
pub use render::Render;
use util::buf::Buffer;
use util::color::Color;

use crate::raster::*;

mod hsr;
pub mod fx;
pub mod raster;
pub mod render;
pub mod scene;
pub mod shade;
pub mod stats;
pub mod text;
pub mod vary;

pub trait RasterOps {
    #[inline(always)]
    fn test<U>(&self, _: Fragment<f32, U>) -> bool { true }

    // TODO
    // fn blend(&mut self, _x: usize, _y: usize, c: Color) -> Color { c }

    fn output<U>(&mut self, _: Fragment<(f32, Color), U>);
}

pub struct Raster<Test, Output> {
    pub test: Test,
    pub output: Output,
}

impl<Test, Output> RasterOps for Raster<Test, Output>
where
    Test: Fn(Fragment<f32>) -> bool,
    Output: FnMut(Fragment<(f32, Color)>),
{
    #[inline(always)]
    fn test<U>(&self, frag: Fragment<f32, U>) -> bool {
        (self.test)(frag.uniform(()))
    }
    #[inline(always)]
    fn output<U>(&mut self, frag: Fragment<(f32, Color), U>) {
        (self.output)(frag.uniform(()));
    }
}


pub struct Framebuf<'a> {
    // TODO Support other color buffer formats
    pub color: Buffer<u8, &'a mut [u8]>,
    pub depth: &'a mut Buffer<f32>,
}

impl<'a> RasterOps for Framebuf<'a> {
    #[inline]
    fn test<U>(&self, f: Fragment<f32, U>) -> bool {
        f.varying < *self.depth.get(f.coord.0, f.coord.1)
    }
    #[inline]
    fn output<U>(&mut self, f: Fragment<(f32, Color), U>) {
        let ((x, y), (z, col)) = (f.coord, f.varying);
        let [_, r, g, b] = col.to_argb();
        let idx = 4 * (self.color.width() * y + x);
        let data = self.color.data_mut();
        data[idx + 0] = b;
        data[idx + 1] = g;
        data[idx + 2] = r;

        *self.depth.get_mut(x, y) = z;
    }
}

#[derive(Default, Clone)]
pub struct Renderer {
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

impl Renderer {
    pub fn new() -> Renderer {
        Self::default()
    }
}
