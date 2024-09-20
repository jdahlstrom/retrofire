//! Frontends for creating simple applications with `retrofire`.

extern crate alloc;
extern crate core;

use core::time::Duration;

use retrofire_core::render::{ctx::Context, target::Framebuf};
use retrofire_core::util::buf::MutSlice2;

#[cfg(feature = "minifb")]
pub mod minifb;

#[cfg(feature = "sdl2")]
pub mod sdl2;

#[no_std]
#[cfg(feature = "wasm")]
pub mod wasm;

/// Per-frame state. The window run method passes an instance of `Frame`
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
    /// Rendering context and config.
    pub ctx: &'a mut Context,
}

pub mod dims {
    // Source for the names:
    // https://commons.wikimedia.org/wiki/File:Vector_Video_Standards8.svg

    use retrofire_core::util::Dims;

    // 5:4
    pub const SXGA_1280_1024: Dims = (1280, 1024);

    // 4:3
    #[allow(non_upper_case_globals)]
    pub const qVGA_320_240: Dims = (320, 240);
    pub const VGA_640_480: Dims = (640, 480);
    pub const SVGA_800_600: Dims = (800, 600);
    pub const XGA_1024_768: Dims = (1024, 768);
    pub const QVGA_1280_960: Dims = (1280, 960);
    pub const UXGA_1600_1200: Dims = (1600, 1200);
    pub const QXGA_2048_1536: Dims = (2048, 1536);

    // 16:10
    pub const WSXGAP_1680_1050: Dims = (1680, 1050);
    pub const WUXGA_1920_1200: Dims = (1920, 1200);
    pub const WQXGA_2560_1600: Dims = (2560, 1600);

    // 16:9
    pub const HD_1280_720: Dims = (720, 480);
    pub const WSXGA_1600_900: Dims = (1600, 900);
    pub const FHD_1920_1080: Dims = (1920, 1080);
    pub const QHD_2560_1440: Dims = (2560, 1440);
    pub const UHD_4K_3840_2160: Dims = (3840, 2160);

    // DCI ~17:9
    pub const DCI_2K_2048_1080: Dims = (2048, 1080);
    pub const DCI_4K_4096_2160: Dims = (4096, 2160);
}
