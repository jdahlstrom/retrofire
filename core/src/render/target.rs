//! Render targets.
//!
//! The typical render target is a framebuffer, comprising a color buffer,
//! depth buffer, and possible auxiliary buffers. Special render targets can
//! be used, for example, for visibility or occlusion computations.

use crate::math::vary::Vary;
use crate::util::buf::AsMutSlice2;

use super::ctx::Context;
use super::raster::{Frag, Scanline};
use super::shader::FragmentShader;
use super::stats::Throughput;

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
        Fs: FragmentShader<Frag<V>>;
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
        sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput
    where
        V: Vary,
        Fs: FragmentShader<Frag<V>>,
    {
        let Scanline { y, xs, frags } = sl;

        let x0 = xs.start;
        let x1 = xs.end.max(xs.start);
        let cbuf_span = &mut self.color_buf.as_mut_slice2()[y][x0..x1];
        let zbuf_span = &mut self.depth_buf.as_mut_slice2()[y][x0..x1];

        let mut io = Throughput { i: x1 - x0, o: 0 };
        frags
            .zip(cbuf_span)
            .zip(zbuf_span)
            .for_each(|(((pos, var), c), z)| {
                let new_z = pos.z();
                if ctx.depth_test(new_z, *z) {
                    let frag = Frag { pos, var };
                    if let Some(new_c) = fs.shade_fragment(frag) {
                        if ctx.color_write {
                            // TODO Blending should happen here
                            io.o += 1;
                            *c = new_c.to_argb_u32();
                        }
                        if ctx.depth_write {
                            *z = new_z;
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
        sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput
    where
        V: Vary,
        Fs: FragmentShader<Frag<V>>,
    {
        let mut io = Throughput { i: sl.xs.len(), o: 0 };
        let cbuf_span = &mut self.as_mut_slice2()[sl.y][sl.xs];
        sl.frags
            .zip(cbuf_span)
            .for_each(|((pos, var), pix)| {
                let frag = Frag { pos, var };
                if let Some(color) = fs.shade_fragment(frag) {
                    if ctx.color_write {
                        io.o += 1;
                        *pix = color.to_argb_u32();
                    }
                }
            });
        io
    }
}
