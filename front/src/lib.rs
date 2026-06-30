//! Frontends for creating simple applications with `retrofire`.

extern crate alloc;
extern crate core;

use core::{cell::RefCell, time::Duration};

use retrofire_core::{
    math::{Color3, Color4},
    render::{Colorbuf, Context, Framebuf, tex::Atlas},
    util::{AsMutSlice2, Dims, IntoPixel, pnm::read_pnm},
};

#[cfg(feature = "minifb")]
pub mod minifb;

#[cfg(feature = "sdl2")]
pub mod sdl2;

#[no_std]
#[cfg(feature = "wasm")]
pub mod wasm;

pub static FONT_6X10: &[u8] = include_bytes!("../assets/font_6x10.pbm");

/// Returns a 6x10 bitmap font, e.g. for rendering debug messages.
pub fn font_6x10() -> Atlas<Color3> {
    let font = read_pnm(FONT_6X10).expect("font statically included");
    Atlas::grid(Dims(6, 10), font.into())
}

/// Per-frame state. The window run method passes an instance of `Frame`
/// to the callback function on every iteration of the main loop.
pub struct Frame<'a, Win, Buf> {
    /// Elapsed time since the start of the first frame.
    pub t: Duration,
    /// Elapsed time since the start of the previous frame.
    pub dt: Duration,
    /// Framebuffer in which to draw.
    pub buf: Buf,
    /// Reference to the window object.
    pub win: &'a mut Win,
    /// Rendering context and config.
    pub ctx: &'a mut Context,
}

impl<Win, Fmt, Cbuf, Zbuf>
    Frame<'_, Win, &RefCell<Framebuf<Colorbuf<Cbuf, Fmt>, Zbuf>>>
where
    Fmt: Copy,
    Cbuf: AsMutSlice2<Elem: Clone>,
    Zbuf: AsMutSlice2<Elem = f32>,
    Color4: IntoPixel<Cbuf::Elem, Fmt>,
{
    /// Clears the color buffer if [color clearing][Context::color_clear]
    /// is enabled and the depth buffer if [depth clearing][Context::depth_clear]
    /// is enabled.
    pub fn clear(&mut self) {
        if let Some(c) = self.ctx.color_clear {
            let cbuf = &mut self.buf.borrow_mut().color_buf;
            let fmt = cbuf.fmt;
            cbuf.as_mut_slice2().fill(c.into_pixel(fmt));
        }
        if let Some(z) = self.ctx.depth_clear {
            // Depth buffer contains reciprocal depth values
            // TODO Assumes depth format
            self.buf
                .borrow_mut()
                .depth_buf
                .as_mut_slice2()
                .fill(z.recip());
        }
    }
}
