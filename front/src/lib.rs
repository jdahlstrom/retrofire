//! Frontends for creating simple applications with `retrofire`.

use std::time::Duration;

use retrofire_core::render::stats::Stats;
use retrofire_core::render::target::Framebuf;
use retrofire_core::util::buf::MutSlice2;

#[cfg(feature = "minifb")]
pub mod minifb;

#[cfg(feature = "sdl2")]
pub mod sdl2;

/// Per-frame state. [`Window::run`][minifb::Window::run] passes an instance
/// of `Frame` to the callback function on every iteration of the main loop.
pub struct Frame<'a> {
    /// Time since the first frame.
    pub t: Duration,
    /// Time since the previous frame.
    pub dt: Duration,
    /// Framebuffer in which to draw.
    pub buf: Framebuf<MutSlice2<'a, u32>, MutSlice2<'a, f32>>,
    /// Rendering statistics.
    pub stats: &'a mut Stats,
}
