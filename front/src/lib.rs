//! Frontends for creating simple applications with `retrofire`.

extern crate alloc;
extern crate core;

use core::{cell::RefCell, time::Duration};

use retrofire_core::{
    math::Color4,
    render::{Colorbuf, Context, Framebuf},
    util::{buf::AsMutSlice2, pixfmt::IntoPixel},
};

#[cfg(feature = "minifb")]
pub mod minifb;

#[cfg(feature = "sdl2")]
pub mod sdl2;

#[no_std]
#[cfg(feature = "wasm")]
pub mod wasm;

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

#[allow(non_upper_case_globals)]
pub mod dims {
    // Source for the names:
    // https://commons.wikimedia.org/wiki/File:Vector_Video_Standards8.svg

    use retrofire_core::util::Dims;

    // 5:4
    pub const qSXGA_640_512: Dims = (640, 512);
    pub const SXGA_1280_1024: Dims = (1280, 1024);

    // 4:3
    pub const qVGA_320_240: Dims = (320, 240);
    pub const qSVGA_400_300: Dims = (400, 300);
    pub const qXGA_512_384: Dims = (512, 384);
    pub const VGA_640_480: Dims = (640, 480);
    pub const SVGA_800_600: Dims = (800, 600);
    pub const XGA_1024_768: Dims = (1024, 768);
    pub const QVGA_1280_960: Dims = (1280, 960);
    pub const UXGA_1600_1200: Dims = (1600, 1200);
    pub const QXGA_2048_1536: Dims = (2048, 1536);

    // 16:10
    pub const CGA_320_200: Dims = (320, 200);
    pub const MODE_13H: Dims = CGA_320_200;
    pub const QCGA_640_400: Dims = (640, 400);
    pub const qWXGA_640_400: Dims = (640, 640);
    pub const WXGA_1280_800: Dims = (1280, 800);
    pub const WXGAP_1440_900: Dims = (1440, 900);
    pub const WSXGAP_1680_1050: Dims = (1680, 1050);
    pub const WUXGA_1920_1200: Dims = (1920, 1200);
    pub const WQXGA_2560_1600: Dims = (2560, 1600);

    // 16:9
    // 640x360 = "qHD"?
    // 800x450 = qWSXGA?
    // 960x540 = qFHD
    pub const HD_1280_720: Dims = (1280, 720);
    pub const WSXGA_1600_900: Dims = (1600, 900);
    pub const FHD_1920_1080: Dims = (1920, 1080);
    pub const QHD_2560_1440: Dims = (2560, 1440);
    pub const UHD_4K_3840_2160: Dims = (3840, 2160);

    // DCI ~17:9
    pub const DCI_2K_2048_1080: Dims = (2048, 1080);
    pub const DCI_4K_4096_2160: Dims = (4096, 2160);

    // ~21:9
    pub const qUWFHD_1280_540: Dims = (1280, 540);
    pub const UWFHD_2560_1080: Dims = (2560, 1080);
    pub const UWQHD_3440_1440: Dims = (3440, 1440);
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
            self.buf
                .borrow_mut()
                .color_buf
                .as_mut_slice2()
                .fill(c.into_pixel());
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
