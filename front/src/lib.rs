#[cfg(feature = "minifb")]
pub mod minifb;

#[cfg(feature = "sdl2")]
pub mod sdl2;

pub struct Frame<'a> {
    // Time since first frame.
    pub t: Duration,
    // Time since last frame.
    pub dt: Duration,
    // Framebuffer in which to draw.
    pub buf: Framebuf<MutSlice2<'a, u32>, MutSlice2<'a, f32>>,
}
