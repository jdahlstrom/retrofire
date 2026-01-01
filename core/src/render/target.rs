//! Render targets such as framebuffers.
//!
//! The typical render target comprises a color buffer, depth buffer,
//! and possible auxiliary buffers. Special render targets can be used,
//! for example, for visibility or occlusion computations.

use core::cell::RefCell;

use crate::math::{Color3, Color4, Vary};
use crate::util::{
    buf::{AsMutSlice2, Buf2, MutSlice2},
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

impl<B: AsMutSlice2, F> AsMutSlice2 for Colorbuf<B, F> {
    type Elem = B::Elem;
    #[inline]
    fn as_mut_slice2(&mut self) -> MutSlice2<'_, Self::Elem> {
        self.buf.as_mut_slice2()
    }
}

impl<T: Target> Target for &mut T {
    fn rasterize<V: Vary, Fs: FragmentShader<V>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput {
        (*self).rasterize(sl, fs, ctx)
    }
}
impl<T: Target> Target for &RefCell<T> {
    fn rasterize<V: Vary, Fs: FragmentShader<V>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput {
        RefCell::borrow_mut(self).rasterize(sl, fs, ctx)
    }
}

impl<Col, Fmt, Dep> Target for Framebuf<Colorbuf<Col, Fmt>, Dep>
where
    Col: AsMutSlice2,
    Dep: AsMutSlice2<Elem = f32>,
    Color4: IntoPixel<Col::Elem, Fmt>,
{
    /// Rasterizes `scanline` into this framebuffer.
    fn rasterize<V: Vary, Fs: FragmentShader<V>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput {
        let Self { color_buf, depth_buf } = self;
        rasterize_fb(color_buf, depth_buf, sl, fs, Color4::into_pixel, ctx)
    }
}

impl<Buf, Fmt> Target for Colorbuf<Buf, Fmt>
where
    Buf: AsMutSlice2,
    Color4: IntoPixel<Buf::Elem, Fmt>,
{
    /// Rasterizes `scanline` into this `u32` color buffer.
    /// Does no z-buffering.
    fn rasterize<V: Vary, Fs: FragmentShader<V>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput {
        rasterize(&mut self.buf, sl, fs, Color4::into_pixel, ctx)
    }
}

impl Target for Buf2<Color4> {
    fn rasterize<V: Vary, Fs: FragmentShader<V>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput {
        rasterize(self, sl, fs, |c| c, ctx)
    }
}

impl Target for Buf2<Color3> {
    fn rasterize<V: Vary, Fs: FragmentShader<V>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput {
        rasterize(self, sl, fs, |c| c.to_rgb(), ctx)
    }
}

pub fn rasterize<B: AsMutSlice2, V: Vary>(
    buf: &mut B,
    mut sl: Scanline<V>,
    fs: &impl FragmentShader<V>,
    mut conv: impl FnMut(Color4) -> B::Elem,
    ctx: &Context,
) -> Throughput {
    let x0 = sl.xs.start;
    let x1 = sl.xs.end.max(x0);
    let mut io = Throughput { i: x1 - x0, o: 0 };
    let cbuf_span = &mut buf.as_mut_slice2()[sl.y][x0..x1];

    sl.fragments()
        .zip(cbuf_span)
        .for_each(|(frag, curr_col)| {
            if let Some(new_col) = fs.shade_fragment(frag)
                && ctx.color_write
            {
                io.o += 1;
                *curr_col = conv(new_col);
            }
        });
    io
}

pub fn rasterize_fb<B: AsMutSlice2, V: Vary>(
    cbuf: &mut B,
    zbuf: &mut impl AsMutSlice2<Elem = f32>,
    mut sl: Scanline<V>,
    fs: &impl FragmentShader<V>,
    mut conv: impl FnMut(Color4) -> B::Elem,
    ctx: &Context,
) -> Throughput {
    let x0 = sl.xs.start;
    let x1 = sl.xs.end.max(x0);
    let cbuf_span = &mut cbuf.as_mut_slice2()[sl.y][x0..x1];
    let zbuf_span = &mut zbuf.as_mut_slice2()[sl.y][x0..x1];

    let mut io = Throughput { i: x1 - x0, o: 0 };

    sl.fragments()
        .zip(cbuf_span)
        .zip(zbuf_span)
        .for_each(|((frag, curr_col), curr_z)| {
            let new_z = frag.pos.z();

            if ctx.depth_test(new_z, *curr_z)
                && let Some(new_col) = fs.shade_fragment(frag)
            {
                if ctx.color_write {
                    io.o += 1;
                    // TODO Blending should happen here
                    *curr_col = conv(new_col);
                }
                if ctx.depth_write {
                    *curr_z = new_z;
                }
            }
        });
    io
}
