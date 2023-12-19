//! Frontends for creating simple applications with `retrofire`.

use std::time::Duration;

use retrofire_core::render::stats::Stats;
use retrofire_core::render::target::Framebuf;
use retrofire_core::util::buf::MutSlice2;

#[cfg(feature = "minifb")]
pub mod minifb;

#[cfg(feature = "sdl2")]
pub mod sdl2;

/// Per-frame state. The window run method passes an instance  of `Frame`
/// to the callback function on every iteration of the main loop.
pub struct Frame<'a, Win> {
    /// Elapsed time since the start of the first frame.
    pub t: Duration,
    /// Elapsed time since the start of the previous frame.
    pub dt: Duration,
    /// Framebuffer in which to draw.
    pub buf: Framebuf<MutSlice2<'a, u32>, MutSlice2<'a, f32>>,
    /// Reference to the window object.
    pub win: &'a mut Win,
    /// Rendering statistics.
    pub stats: &'a mut Stats,
}
