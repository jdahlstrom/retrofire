//! Frontend using the `sdl3` crate for window creation and event handling.
use core::{cell::RefCell, fmt, mem::replace, ops::ControlFlow};
use std::time::Instant;

use sdl3::{
    EventPump, IntegerOrSdlError, Sdl,
    event::Event,
    keyboard::Keycode,
    pixels::PixelFormat,
    render::{Texture, TextureValueError, WindowCanvas},
    video::{FullscreenType, Window as SdlWindow, WindowBuildError},
};

use retrofire_core::math::{Color4, Vary};
use retrofire_core::render::{
    Colorbuf, Context, FragmentShader, Stats, Target, raster::Scanline,
    stats::Throughput, target::rasterize_fb,
};
use retrofire_core::util::{
    Dims,
    buf::{AsMutSlice2, Buf2, MutSlice2},
    pixfmt::{IntoPixel, Rgb565, Rgba4444, Rgba8888},
};

use super::{Frame, dims};

/// Helper trait to support different pixel format types.
pub trait PixelFmt: Copy + Default {
    type Pixel: AsRef<[u8]> + Copy + Sized;
    const SDL_FMT: PixelFormat;

    fn encode<C: IntoPixel<Self::Pixel, Self>>(self, color: C) -> Self::Pixel {
        color.into_pixel_fmt(self)
    }
}

#[derive(Debug)]
pub struct Error(String);

/// A lightweight wrapper of an `SDL2` window.
pub struct Window<PF> {
    /// The SDL canvas.
    pub canvas: WindowCanvas,
    /// The SDL event pump.
    pub ev_pump: EventPump,
    /// Pending events.
    pub events: Vec<Event>,
    /// The width and height of the window.
    pub dims: Dims,
    /// Framebuffer pixel format.
    pub pixfmt: PF,
    /// Rendering context defaults.
    pub ctx: Context,
}

/// Builder for creating `Window`s.
pub struct Builder<'title, PF> {
    pub dims: Dims,
    pub title: &'title str,
    pub vsync: bool,
    pub hidpi: bool,
    pub fullscreen: bool,
    pub pixfmt: PF,
}

pub struct Framebuf<'a, PF: PixelFmt> {
    pub color_buf: Colorbuf<MutSlice2<'a, PF::Pixel>, PF>,
    pub depth_buf: MutSlice2<'a, f32>,
}

//
// Inherent impls
//

impl<'t, PF: PixelFmt> Builder<'t, PF> {
    /// Sets the width and height of the window, in pixels.
    pub fn dims(mut self, dims: Dims) -> Self {
        self.dims = dims;
        self
    }
    /// Sets the title of the window.
    pub fn title(mut self, title: &'t str) -> Self {
        self.title = title;
        self
    }
    /// Sets whether vertical sync is enabled.
    ///
    /// If true, the frame rate is synced to the monitor's refresh rate.
    pub fn vsync(mut self, enabled: bool) -> Self {
        self.vsync = enabled;
        self
    }
    /// Sets whether high-dpi
    ///
    /// If true, the physical resolution may be higher than the logical resolution.
    pub fn high_dpi(mut self, enabled: bool) -> Self {
        self.hidpi = enabled;
        self
    }
    /// Sets the fullscreen state of the window.
    pub fn fullscreen(mut self, fs: bool) -> Self {
        self.fullscreen = fs;
        self
    }
    /// Sets the framebuffer pixel format.
    ///
    /// Supported formats are [`Rgba8888`], [`Rgb565`], and [`Rgba4444`].
    pub fn pixel_fmt(mut self, fmt: PF) -> Self {
        self.pixfmt = fmt;
        self
    }

    /// Creates the window.
    pub fn build(self) -> Result<Window<PF>, Error> {
        let sdl = sdl3::init()?;
        let win = self.create_window(&sdl)?;

        self.set_mouse_mode(&sdl, &win);

        let canvas = self.create_canvas(win)?;
        let ev_pump = sdl.event_pump()?;
        let ctx = Context::default();

        Ok(Window {
            canvas,
            ev_pump,
            ctx,
            events: Vec::new(),
            dims: self.dims,
            pixfmt: self.pixfmt,
        })
    }

    fn create_window(&self, sdl: &Sdl) -> Result<SdlWindow, Error> {
        let Self {
            dims: (w, h),
            title,
            fullscreen,
            hidpi,
            ..
        } = *self;
        let mut bld = sdl.video()?.window(title, w, h);
        if hidpi {
            bld.high_pixel_density();
        }
        if fullscreen {
            bld.fullscreen();
        }
        Ok(bld.build()?)
    }

    fn create_canvas(&self, w: SdlWindow) -> Result<WindowCanvas, Error> {
        let canvas = w.into_canvas();
        if self.vsync {
            //let _ok = canvas.present();
            //TODO vsync? canvas = canvas.present_vsync();
        }
        Ok(canvas)
    }

    fn set_mouse_mode(&self, sdl: &Sdl, win: &SdlWindow) {
        let m = sdl.mouse();
        m.set_relative_mouse_mode(win, true);
        m.capture(true);
        m.show_cursor(true);
    }
}

impl<PF: PixelFmt<Pixel = [u8; N]>, const N: usize> Window<PF> {
    /// Returns a window builder.
    pub fn builder() -> Builder<'static, PF> {
        Builder::default()
    }

    /// Copies the texture to the frame buffer and updates the screen.
    pub fn present(&mut self, tex: &Texture) -> Result<(), Error> {
        self.canvas.copy(tex, None, None)?;
        self.canvas.present();
        Ok(())
    }

    /// Runs the main loop of the program.
    ///
    /// Invokes `frame_fn` on each iteration to compute and draw the next frame.
    ///
    /// The main loop stops and this function returns if:
    /// * the user closes the window via the GUI (e.g. a title bar button);
    /// * the Esc key is pressed; or
    /// * the callback returns [`ControlFlow::Break`][ControlFlow].
    pub fn run<F>(&mut self, mut frame_fn: F) -> Result<Stats, Error>
    where
        F: FnMut(&mut Frame<Self, &RefCell<Framebuf<PF>>>) -> ControlFlow<()>,
        Color4: IntoPixel<PF::Pixel, PF>,
    {
        let dims @ (w, h) = self.canvas.window().size_in_pixels();

        let tc = self.canvas.texture_creator();
        let mut tex = tc.create_texture_streaming(PF::SDL_FMT, w, h)?;

        let mut zbuf = Buf2::new(dims);
        let mut ctx = self.ctx.clone();

        let start = Instant::now();
        let mut last = Instant::now();
        'main: loop {
            self.events.clear();
            for e in self.ev_pump.poll_iter() {
                match e {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape), ..
                    } => break 'main,
                    e => self.events.push(e),
                }
            }

            let cf = tex.with_lock(None, |bytes, pitch| {
                let bytes = bytes.as_chunks_mut().0;
                let pitch = pitch / N;

                if let Some(z) = ctx.depth_clear {
                    // Z-buffer stores reciprocals
                    zbuf.fill(z.recip());
                }
                if let Some(c) = ctx.color_clear {
                    bytes.fill(self.pixfmt.encode(c));
                }

                let color_buf =
                    Colorbuf::new(MutSlice2::new(dims, pitch as u32, bytes));
                let buf = Framebuf {
                    color_buf,
                    depth_buf: zbuf.as_mut_slice2(),
                };

                let frame = &mut Frame {
                    t: start.elapsed(),
                    dt: replace(&mut last, Instant::now()).elapsed(),
                    buf: &RefCell::new(buf),
                    win: self,
                    ctx: &mut ctx,
                };
                frame_fn(frame)
            })?;

            self.present(&tex)?;
            ctx.stats.borrow_mut().frames += 1.0;

            if cf.is_break() {
                break;
            }
        }
        let stats = ctx.stats.into_inner();
        println!("{stats}");
        Ok(stats)
    }
}

//
// Trait impls
//

impl PixelFmt for Rgba8888 {
    type Pixel = [u8; 4];
    const SDL_FMT: PixelFormat = PixelFormat::RGBA32;
}
impl PixelFmt for Rgb565 {
    type Pixel = [u8; 2];
    const SDL_FMT: PixelFormat = PixelFormat::RGB565;
}
impl PixelFmt for Rgba4444 {
    type Pixel = [u8; 2];
    const SDL_FMT: PixelFormat = PixelFormat::RGBA4444;
}

impl<'a, PF, const N: usize> Target for Framebuf<'a, PF>
where
    PF: PixelFmt<Pixel = [u8; N]>,
    Color4: IntoPixel<PF::Pixel, PF>,
{
    fn rasterize<V: Vary, Fs: FragmentShader<V>>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput {
        rasterize_fb(
            &mut self.color_buf,
            &mut self.depth_buf,
            sl,
            fs,
            |c| c.into_pixel(),
            ctx,
        )
    }
}

impl<PF: PixelFmt> Default for Builder<'_, PF> {
    fn default() -> Self {
        Self {
            dims: dims::SVGA_800_600,
            title: "// retrofire application //",
            vsync: true,
            fullscreen: false,
            pixfmt: PF::default(),
            hidpi: false,
        }
    }
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

macro_rules! impl_from_error {
    ($($e:ty)+) => { $(
        impl From<$e> for Error {
            fn from(e: $e) -> Self { Self(e.to_string()) }
        }
    )+ };
}

impl_from_error! {
    String
    sdl3::Error
    WindowBuildError
    TextureValueError
    IntegerOrSdlError
}
