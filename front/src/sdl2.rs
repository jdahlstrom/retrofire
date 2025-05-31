//! Frontend using the `sdl2` crate for window creation and event handling.

use std::{fmt, mem::replace, ops::ControlFlow, time::Instant};

use sdl2::{
    EventPump, IntegerOrSdlError,
    event::Event,
    keyboard::Keycode,
    pixels::PixelFormatEnum,
    render::{Texture, TextureValueError, WindowCanvas},
    video::{FullscreenType, WindowBuildError},
};

use retrofire_core::math::{Color4, Vary};
use retrofire_core::render::{
    Context, FragmentShader, Target, raster::Scanline, stats::Throughput,
    target::Colorbuf,
};
use retrofire_core::util::{
    Dims,
    buf::{AsMutSlice2, Buf2, MutSlice2},
    pixfmt::{IntoPixel, Rgb565, Rgba4444, Rgba8888},
};

use super::{Frame, dims};

/// Helper trait to support different pixel format types.
pub trait PixelFmt: Default {
    type Pixel: AsRef<[u8]>;
    const INSTANCE: Self;
    const SDL_FMT: PixelFormatEnum;

    fn size() -> usize {
        size_of::<Self::Pixel>()
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
    /// The width and height of the window.
    pub dims: Dims,
    /// Rendering context defaults.
    pub ctx: Context,
    /// Framebuffer pixel format.
    pub pixfmt: PF,
}

/// Builder for creating `Window`s.
pub struct Builder<'title, PF> {
    pub dims: (u32, u32),
    pub title: &'title str,
    pub vsync: bool,
    pub fs: FullscreenType,
    pub pixfmt: PF,
}

pub struct Framebuf<'a, PF> {
    pub color_buf: Colorbuf<MutSlice2<'a, u8>, PF>,
    pub depth_buf: MutSlice2<'a, f32>,
}

//
// Inherent impls
//

impl<'t, PF: PixelFmt> Builder<'t, PF> {
    /// Sets the width and height of the window, in pixels.
    pub fn dims(mut self, w: u32, h: u32) -> Self {
        self.dims = (w, h);
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
    /// Sets the fullscreen state of the window.
    pub fn fullscreen(mut self, fs: FullscreenType) -> Self {
        self.fs = fs;
        self
    }
    /// Sets the framebuffer pixel format.
    ///
    /// Supported formats are [`Rgba8888`], [`Rgb565`], and [`Rgb4444].
    pub fn pixel_fmt(mut self, fmt: PF) -> Self {
        self.pixfmt = fmt;
        self
    }

    /// Creates the window.
    pub fn build(self) -> Result<Window<PF>, Error> {
        let Self { dims, title, vsync, fs, pixfmt } = self;

        let sdl = sdl2::init()?;

        let mut win = sdl
            .video()?
            .window(title, dims.0, dims.1)
            .build()?;

        win.set_fullscreen(fs)?;
        sdl.mouse().set_relative_mouse_mode(true);

        let mut canvas = win.into_canvas();
        if vsync {
            canvas = canvas.present_vsync();
        }
        let canvas = canvas.accelerated().build()?;

        let ev_pump = sdl.event_pump()?;

        let ctx = Context::default();

        let m = sdl.mouse();
        m.set_relative_mouse_mode(true);
        m.capture(true);
        m.show_cursor(true);

        Ok(Window {
            canvas,
            ev_pump,
            dims,
            ctx,
            pixfmt,
        })
    }
}

impl<PF: PixelFmt> Window<PF> {
    /// Returns a window builder.
    pub fn builder() -> Builder<'static, PF> {
        Builder::default()
    }

    /// Copies the texture to the frame buffer and updates the screen.
    pub fn present(&mut self, tex: &Texture) -> Result<(), Error> {
        self.canvas.copy(&tex, None, None)?;
        self.canvas.present();
        Ok(())
    }

    /// Runs the main loop of the program, invoking the callback on each
    /// iteration to compute and draw the next frame.
    ///
    /// The main loop stops and this function returns if:
    /// * the user closes the window via the GUI (e.g. a title bar button);
    /// * the Esc key is pressed; or
    /// * the callback returns [`ControlFlow::Break`][ControlFlow].
    pub fn run<F>(&mut self, mut frame_fn: F) -> Result<(), Error>
    where
        F: FnMut(&mut Frame<Self, Framebuf<PF>>) -> ControlFlow<()>,
        Color4: IntoPixel<PF::Pixel, PF>,
    {
        let (w, h) = self.dims;

        let tc = self.canvas.texture_creator();
        let mut tex = tc.create_texture_streaming(PF::SDL_FMT, w, h)?;

        let mut zbuf = Buf2::new(self.dims);
        let mut ctx = self.ctx.clone();

        let start = Instant::now();
        let mut last = Instant::now();
        'main: loop {
            for e in self.ev_pump.poll_iter() {
                match e {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape), ..
                    } => break 'main,
                    _ => (),
                }
            }

            let cf = tex.with_lock(None, |bytes, pitch| {
                if let Some(c) = ctx.depth_clear {
                    // Z-buffer stores reciprocals
                    zbuf.fill(c.recip());
                }
                if let Some(c) = ctx.color_clear {
                    let c: PF::Pixel = c.into_pixel_fmt(PF::INSTANCE);
                    bytes.chunks_exact_mut(PF::size()).for_each(|ch| {
                        ch.copy_from_slice(c.as_ref());
                    });
                }

                let color_buf = Colorbuf::new(MutSlice2::new(
                    (PF::size() as u32 * w, h),
                    pitch as u32,
                    bytes,
                ));
                let buf = Framebuf {
                    color_buf,
                    depth_buf: zbuf.as_mut_slice2(),
                };

                let frame = &mut Frame {
                    t: start.elapsed(),
                    dt: replace(&mut last, Instant::now()).elapsed(),
                    buf,
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
        println!("{}", ctx.stats.borrow());
        Ok(())
    }
}

//
// Trait impls
//

impl PixelFmt for Rgba8888 {
    type Pixel = [u8; 4];
    const INSTANCE: Self = Self;
    const SDL_FMT: PixelFormatEnum = PixelFormatEnum::RGBA32;
}
impl PixelFmt for Rgb565 {
    type Pixel = [u8; 2];
    const INSTANCE: Self = Self;
    const SDL_FMT: PixelFormatEnum = PixelFormatEnum::RGB565;
}
impl PixelFmt for Rgba4444 {
    type Pixel = [u8; 2];
    const INSTANCE: Self = Self;
    const SDL_FMT: PixelFormatEnum = PixelFormatEnum::RGBA4444;
}

impl<'a, PF, const N: usize> Target for Framebuf<'a, PF>
where
    PF: PixelFmt<Pixel = [u8; N]>,
    Color4: IntoPixel<PF::Pixel, PF>,
{
    fn rasterize<V: Vary, Fs: FragmentShader<V>>(
        &mut self,
        mut sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput {
        // TODO Lots of duplicate code

        let x0 = sl.xs.start;
        let x1 = sl.xs.end.max(x0);
        // TODO use as_chunks once stable
        let mut cbuf_span =
            &mut self.color_buf.buf[sl.y][PF::size() * x0..PF::size() * x1];
        let zbuf_span = &mut self.depth_buf.as_mut_slice2()[sl.y][x0..x1];

        let mut io = Throughput { i: x1 - x0, o: 0 };

        for (frag, z) in sl.fragments().zip(zbuf_span) {
            let c: &mut PF::Pixel;
            (c, cbuf_span) = cbuf_span.split_first_chunk_mut().unwrap();

            let new_z = frag.pos.z();
            if ctx.depth_test(new_z, *z) {
                if let Some(new_c) = fs.shade_fragment(frag) {
                    if ctx.color_write {
                        // TODO Blending should happen here
                        io.o += 1;
                        *c = new_c.into_pixel_fmt(PF::INSTANCE);
                    }
                    if ctx.depth_write {
                        *z = new_z;
                    }
                }
            }
        }
        io
    }
}

impl<PF: PixelFmt> Default for Builder<'_, PF> {
    fn default() -> Self {
        Self {
            dims: dims::SVGA_800_600,
            title: "// retrofire application //",
            vsync: true,
            fs: FullscreenType::Off,
            pixfmt: PF::INSTANCE,
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
    sdl2::Error
    WindowBuildError
    TextureValueError
    IntegerOrSdlError
}
