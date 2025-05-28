//! Render targets such as framebuffers.
//!
//! The typical render target comprises a color buffer, depth buffer,
//! and possible auxiliary buffers. Special render targets can be used,
//! for example, for visibility or occlusion computations.

use crate::math::{Color4, Vary};
use crate::util::{
    buf::{AsMutSlice2, MutSlice2},
    pixfmt::IntoPixel,
};

use super::{Context, FragmentShader, raster::Scanline, stats::Throughput};

/// Trait for types that can be used as render targets.
pub trait Target {
    /// Writes a single scanline into `self`.
    ///
    /// Returns count of fragments input and output.
    fn rasterize<V, Fs>(
        &mut self,
        scanline: Scanline<V>,
        frag_shader: &Fs,
        ctx: &Context,
    ) -> Throughput
    where
        V: Vary,
        Fs: FragmentShader<V>;
}

/// Framebuffer, combining a color (pixel) buffer and a depth buffer.
#[derive(Clone)]
pub struct Framebuf<Col, Dep> {
    pub color_buf: Col,
    pub depth_buf: Dep,
}

/// Color buffer with a specified pixel format.
pub struct Colorbuf<B, F> {
    pub buf: B,
    pub fmt: F,
}

impl<B, F: Default> Colorbuf<B, F> {
    pub fn new(buf: B) -> Self {
        Self { buf, fmt: F::default() }
    }
}

impl<T, B: AsMutSlice2<T>, F> AsMutSlice2<T> for Colorbuf<B, F> {
    fn as_mut_slice2(&mut self) -> MutSlice2<T> {
        self.buf.as_mut_slice2()
    }
}

impl<Col, Fmt, Dep> Target for Framebuf<Colorbuf<Col, Fmt>, Dep>
where
    Col: AsMutSlice2<u32>,
    Dep: AsMutSlice2<f32>,
    Color4: IntoPixel<u32, Fmt>,
{
    /// Rasterizes `scanline` into this framebuffer.
    fn rasterize<V, Fs>(
        &mut self,
        mut sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput
    where
        V: Vary,
        Fs: FragmentShader<V>,
    {
        let x0 = sl.xs.start;
        let x1 = sl.xs.end.max(x0);
        let cbuf_span = &mut self.color_buf.as_mut_slice2()[sl.y][x0..x1];
        let zbuf_span = &mut self.depth_buf.as_mut_slice2()[sl.y][x0..x1];

        let mut io = Throughput { i: x1 - x0, o: 0 };

        sl.fragments()
            .zip(cbuf_span)
            .zip(zbuf_span)
            .for_each(|((frag, curr_col), curr_z)| {
                let new_z = frag.pos.z();

                if ctx.depth_test(new_z, *curr_z) {
                    if let Some(new_col) = fs.shade_fragment(frag) {
                        if ctx.color_write {
                            io.o += 1;
                            // TODO Blending should happen here
                            *curr_col = new_col.into_pixel()
                        }
                        if ctx.depth_write {
                            *curr_z = new_z;
                        }
                    }
                }
            });
        io
    }
}

impl<Buf, Fmt> Target for Colorbuf<Buf, Fmt>
where
    Buf: AsMutSlice2<u32>,
    Color4: IntoPixel<u32, Fmt>,
{
    /// Rasterizes `scanline` into this `u32` color buffer.
    /// Does no z-buffering.
    fn rasterize<V, Fs>(
        &mut self,
        mut sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput
    where
        V: Vary,
        Fs: FragmentShader<V>,
    {
        let x0 = sl.xs.start;
        let x1 = sl.xs.end.max(x0);
        let mut io = Throughput { i: x1 - x0, o: 0 };
        let cbuf_span = &mut self.as_mut_slice2()[sl.y][x0..x1];

        sl.fragments()
            .zip(cbuf_span)
            .for_each(|(frag, c)| {
                if let Some(color) = fs.shade_fragment(frag) {
                    if ctx.color_write {
                        io.o += 1;
                        *c = color.into_pixel()
                    }
                }
            });
        io
    }
}
