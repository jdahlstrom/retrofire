//! Render targets such as framebuffers.
//!
//! The typical render target comprises a color buffer, depth buffer,
//! and possible auxiliary buffers. Special render targets can be used,
//! for example, for visibility or occlusion computations.

use crate::{math::Vary, util::buf::AsMutSlice2};

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

impl<Col, Dep> Target for Framebuf<Col, Dep>
where
    Col: AsMutSlice2<u32>,
    Dep: AsMutSlice2<f32>,
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
                            *curr_col = new_col.to_argb_u32();
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

impl<Buf: AsMutSlice2<u32>> Target for Buf {
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
                        *c = color.to_argb_u32();
                    }
                }
            });
        io
    }
}
