//! Render targets.
//!
//! The typical render target is a framebuffer, comprising a color buffer,
//! depth buffer, and possible auxiliary buffers. Special render targets can
//! be used, for example, for visibility or occlusion computations.

use core::iter::zip;

use crate::math::vary::Vary;
use crate::util::buf::AsMutSlice2;

use super::ctx::Context;
use super::raster::{Frag, Scanline, Varyings};
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
        let Scanline { y, xs, vs, dv_dx } = sl;

        let mut io = Throughput { i: xs.len(), o: 0 };

        let mut rasterize_block =
            |cs: &mut [u32],
             zs: &mut [f32],
             v0: Varyings<V>,
             v1: Varyings<V>| {
                let u0 = v0.z_div(v0.0.z());
                let u1 = v1.z_div(v1.0.z());

                let dv_dx = u0.dv_dt(&u1, 1.0 / cs.len() as f32);
                let vs = u0.clone().vary(dv_dx, None);

                for ((c, z), (pos, var)) in zip(zip(cs, zs), vs) {
                    let frag = Frag { pos, var };

                    let true = ctx.depth_test(pos.z(), *z) else {
                        continue;
                    };
                    let Some(new_col) = fs.shade_fragment(frag) else {
                        continue;
                    };
                    if ctx.color_write {
                        // TODO Blending should happen here
                        io.o += 1;
                        *c = new_col.to_argb_u32();
                    }
                    if ctx.depth_write {
                        *z = pos.z();
                    }
                }
            };

        const BS: usize = 1;

        let x0 = xs.start;
        let x1 = xs.end.max(xs.start);

        // Needed to keep references alive
        let mut color_buf = self.color_buf.as_mut_slice2();
        let mut depth_buf = self.depth_buf.as_mut_slice2();

        let mut cbuf_chunks = color_buf[y][x0..x1].chunks_exact_mut(BS);
        let mut zbuf_chunks = depth_buf[y][x0..x1].chunks_exact_mut(BS);

        let mut v_iter = vs.start.vary(dv_dx, None);

        let Some(mut v0) = v_iter.next().clone() else {
            return io;
        };

        for (cb_chunk, zb_chunk) in zip(&mut cbuf_chunks, &mut zbuf_chunks) {
            let Some(v1) = v_iter.clone().nth(BS - 1) else {
                break;
            };
            rasterize_block(cb_chunk, zb_chunk, v0, v1.clone());

            _ = v_iter.nth(BS - 1); // TODO use advance_by once stable
            v0 = v1;
        }

        let cb_rem = cbuf_chunks.into_remainder();
        let zb_rem = zbuf_chunks.into_remainder();

        let Some(v0) = v_iter.next() else {
            return io;
        };

        rasterize_block(cb_rem, zb_rem, v0, vs.end);

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

        let x0 = sl.xs.start;
        let x1 = sl.xs.end.max(sl.xs.start);
        let cbuf_span = &mut self.as_mut_slice2()[sl.y][x0..x1];

        /*sl.vs
        .zip(cbuf_span)
        .for_each(|((pos, var), pix)| {
            if let Some(color) = fs.shade_fragment(Frag { pos, var }) {
                if ctx.color_write {
                    io.o += 1;
                    *pix = color.to_argb_u32();
                }
            }
        });*/
        io
    }
}
