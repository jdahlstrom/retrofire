//! Render targets such as framebuffers.
//!
//! The typical render target comprises a color buffer, depth buffer,
//! and possible auxiliary buffers. Special render targets can be used,
//! for example, for visibility or occlusion computations.

use core::cell::RefCell;

use crate::math::{Color3, Color4, Vary};
use crate::util::{AsMutSlice2, IntoPixel, MutSlice2};

#[cfg(feature = "alloc")]
use crate::util::Buf2;

use super::{Context, FragmentShader, raster::Scanline};

/// Trait for types that can be used as render targets.
pub trait Target {
    /// Writes a single scanline into `self`.
    ///
    /// Returns count of fragments input and output.
    fn rasterize<V: Vary, U: Copy, Fs: FragmentShader<V, U>>(
        &mut self,
        scanline: Scanline<V>,
        frag_shader: &Fs,
        uniform: U,
        ctx: &Context,
    );
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

impl<B: AsMutSlice2, F> AsMutSlice2 for Colorbuf<B, F> {
    type Elem = B::Elem;
    #[inline]
    fn as_mut_slice2(&mut self) -> MutSlice2<'_, Self::Elem> {
        self.buf.as_mut_slice2()
    }
}

impl<T: Target> Target for &mut T {
    #[inline]
    fn rasterize<V: Vary, U: Copy, Fs: FragmentShader<V, U>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        uni: U,
        ctx: &Context,
    ) {
        (*self).rasterize(sl, fs, uni, ctx);
    }
}
impl<T: Target> Target for &RefCell<T> {
    #[inline]
    fn rasterize<V: Vary, U: Copy, Fs: FragmentShader<V, U>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        uni: U,
        ctx: &Context,
    ) {
        RefCell::borrow_mut(self).rasterize(sl, fs, uni, ctx);
    }
}

impl<Col, Fmt, Dep> Target for Framebuf<Colorbuf<Col, Fmt>, Dep>
where
    Col: AsMutSlice2,
    Fmt: Copy,
    Dep: AsMutSlice2<Elem = f32>,
    Color4: IntoPixel<Col::Elem, Fmt>,
{
    /// Rasterizes `scanline` into this framebuffer.
    #[inline]
    fn rasterize<V: Vary, U: Copy, Fs: FragmentShader<V, U>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        uni: U,
        ctx: &Context,
    ) {
        let Self { color_buf, depth_buf } = self;
        let fmt = color_buf.fmt; // borrowck...
        let conv = |c: Color4| c.into_pixel(fmt);
        rasterize_fb(color_buf, depth_buf, sl, fs, uni, conv, ctx);
    }
}

impl<Buf, Fmt> Target for Colorbuf<Buf, Fmt>
where
    Buf: AsMutSlice2,
    Fmt: Copy,
    Color4: IntoPixel<Buf::Elem, Fmt>,
{
    /// Rasterizes `scanline` into this `u32` color buffer.
    /// Does no z-buffering.
    #[inline]
    fn rasterize<V: Vary, U: Copy, Fs: FragmentShader<V, U>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        uni: U,
        ctx: &Context,
    ) {
        let conv = |c: Color4| c.into_pixel(self.fmt);
        rasterize(&mut self.buf, sl, fs, uni, conv, ctx);
    }
}

#[cfg(feature = "alloc")]
impl Target for Buf2<Color4> {
    #[inline]
    fn rasterize<V: Vary, U: Copy, Fs: FragmentShader<V, U>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        uni: U,
        ctx: &Context,
    ) {
        rasterize(self, sl, fs, uni, |c| c, ctx);
    }
}

#[cfg(feature = "alloc")]
impl Target for Buf2<Color3> {
    #[inline]
    fn rasterize<V: Vary, U: Copy, Fs: FragmentShader<V, U>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        uni: U,
        ctx: &Context,
    ) {
        rasterize(self, sl, fs, uni, |c| c.to_rgb(), ctx);
    }
}

pub fn rasterize<B: AsMutSlice2, V: Vary, U: Copy>(
    buf: &mut B,
    mut sl: Scanline<V>,
    fs: &impl FragmentShader<V, U>,
    uni: U,
    mut conv: impl FnMut(Color4) -> B::Elem,
    ctx: &Context,
) {
    let x0 = sl.xs.start;
    let x1 = sl.xs.end.max(x0);

    #[cfg(feature = "stats")]
    let mut frags_out = 0;

    let cbuf_span = &mut buf.as_mut_slice2()[sl.y][x0..x1];

    sl.fragments()
        .zip(cbuf_span)
        .for_each(|(frag, curr_col)| {
            if let Some(new_col) = fs.shade_fragment(frag, uni)
                && ctx.color_write
            {
                #[cfg(feature = "stats")]
                {
                    frags_out += 1;
                }
                *curr_col = conv(new_col);
            }
        });
    #[cfg(feature = "stats")]
    {
        ctx.stats.borrow_mut().frags +=
            super::stats::Throughput { i: x1 - x0, o: frags_out };
    };
}

pub fn rasterize_fb<B: AsMutSlice2, V: Vary, U: Copy>(
    cbuf: &mut B,
    zbuf: &mut impl AsMutSlice2<Elem = f32>,
    mut sl: Scanline<V>,
    fs: &impl FragmentShader<V, U>,
    uni: U,
    mut conv: impl FnMut(Color4) -> B::Elem,
    ctx: &Context,
) {
    let x0 = sl.xs.start;
    let x1 = sl.xs.end.max(x0);
    let cbuf_span = &mut cbuf.as_mut_slice2()[sl.y][x0..x1];
    let zbuf_span = &mut zbuf.as_mut_slice2()[sl.y][x0..x1];

    #[cfg(feature = "stats")]
    let mut frags_out = 0;

    sl.fragments()
        .zip(cbuf_span)
        .zip(zbuf_span)
        .for_each(|((frag, curr_col), curr_z)| {
            let new_z = frag.pos.z();

            if ctx.depth_test(new_z, *curr_z)
                && let Some(new_col) = fs.shade_fragment(frag, uni)
            {
                if ctx.color_write {
                    #[cfg(feature = "stats")]
                    {
                        frags_out += 1;
                    }
                    // TODO Blending should happen here
                    *curr_col = conv(new_col);
                }
                if ctx.depth_write {
                    *curr_z = new_z;
                }
            }
        });
    #[cfg(feature = "stats")]
    {
        ctx.stats.borrow_mut().frags +=
            super::stats::Throughput { i: x1 - x0, o: frags_out };
    };
}
