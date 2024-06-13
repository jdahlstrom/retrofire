//! Frontends for creating simple applications with `retrofire`.

use std::time::Duration;

use retrofire_core::render::ctx::Context;

#[cfg(feature = "minifb")]
pub mod minifb;

#[cfg(feature = "sdl2")]
pub mod sdl2;

/// Per-frame state. The window run method passes an instance  of `Frame`
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
