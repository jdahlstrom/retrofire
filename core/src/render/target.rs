//! Render targets such as framebuffers.
//!
//! The typical render target comprises a color buffer, depth buffer,
//! and possible auxiliary buffers. Special render targets can be used,
//! for example, for visibility or occlusion computations.

use crate::math::{Color4, Vary};
use crate::util::{
    buf::AsMutSlice2,
    pixfmt::{Fmt, ToFmt},
};

use super::{raster::Scanline, stats::Throughput, Context, FragmentShader};

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
pub struct Framebuf<Col, Dep, Fmt> {
    pub color_buf: PixelBuf<Col, Fmt>,
    pub depth_buf: Dep,
}

#[derive(Clone)]
pub struct PixelBuf<Buf, Fmt>(pub Buf, pub Fmt);

impl<C, Z, F> Target for Framebuf<C, Z, F>
where
    C: AsMutSlice2<u32>,
    Z: AsMutSlice2<f32>,
    F: Fmt<u32> + Copy,
    Color4: ToFmt<u32, F>,
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
        let PixelBuf(col_buf, pix_fmt) = &mut self.color_buf;
        let cbuf_span = &mut col_buf.as_mut_slice2()[sl.y][x0..x1];
        let zbuf_span = &mut self.depth_buf.as_mut_slice2()[sl.y][x0..x1];

        let mut io = Throughput { i: x1 - x0, o: 0 };

        sl.fragments()
            .zip(cbuf_span)
            .zip(zbuf_span)
            .for_each(|((frag, curr_col), curr_z)| {
                let new_z = frag.pos.z();

                if !ctx.depth_test(new_z, *curr_z) {
                    return;
                };
                let Some(new_col) = fs.shade_fragment(frag) else {
                    return;
                };

                if ctx.color_write {
                    io.o += 1;
                    // TODO Blending should happen here
                    new_col.write(*pix_fmt, curr_col);
                }
                if ctx.depth_write {
                    *curr_z = new_z;
                }
            });
        io
    }
}

impl<B, F> Target for PixelBuf<B, F>
where
    B: AsMutSlice2<u32>,
    F: Fmt<u32> + Copy,
    Color4: ToFmt<u32, F>,
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
        let cbuf_span = &mut self.0.as_mut_slice2()[sl.y][x0..x1];

        sl.fragments()
            .zip(cbuf_span)
            .for_each(|(frag, curr)| {
                if let Some(new) = fs.shade_fragment(frag) {
                    if ctx.color_write {
                        io.o += 1;
                        new.write(self.1, curr);
                    }
                }
            });
        io
    }
}
