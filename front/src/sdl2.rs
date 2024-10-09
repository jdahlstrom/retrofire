//! Frontend using the `sdl2` crate for window creation and event handling.

use std::{fmt, mem::replace, ops::ControlFlow, time::Instant};

use sdl2::video::FullscreenType;
use sdl2::{
    event::Event,
    keyboard::Keycode,
    render::{Texture, TextureValueError, WindowCanvas},
    video::WindowBuildError,
    EventPump, IntegerOrSdlError,
};

use retrofire_core::math::Vary;
use retrofire_core::render::{
    ctx::Context, raster::Scanline, shader::FragmentShader, stats::Throughput,
    target::Target,
};
use retrofire_core::util::buf::{Buf2, MutSlice2};

use crate::{dims, Frame};

#[derive(Debug)]
pub struct Error(String);

/// A lightweight wrapper of an `SDL2` window.
pub struct Window {
    /// The SDL canvas.
    pub canvas: WindowCanvas,
    /// The SDL event pump.
    pub ev_pump: EventPump,
    /// The width and height of the window.
    pub dims: (u32, u32),
    /// Rendering context defaults.
    pub ctx: Context,
}

/// Builder for creating `Window`s.
pub struct Builder<'title> {
    pub dims: (u32, u32),
    pub title: &'title str,
    pub vsync: bool,
    pub fs: FullscreenType,
}

pub struct Framebuf<'a> {
    pub color_buf: MutSlice2<'a, u8>,
    pub depth_buf: MutSlice2<'a, f32>,
}

//
// Inherent impls
//

impl<'t> Builder<'t> {
    /// Sets the width and height of the window.
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
    /// If true, frame rate is tied to the monitor's refresh rate.
    pub fn vsync(mut self, enabled: bool) -> Self {
        self.vsync = enabled;
        self
    }
    /// Sets the fullscreen state of the window.
    pub fn fullscreen(mut self, fs: FullscreenType) -> Self {
        self.fs = fs;
        self
    }

    /// Creates the window.
    pub fn build(self) -> Result<Window, Error> {
        let Self { dims, title, vsync, fs } = self;

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

        Ok(Window { canvas, ev_pump, dims, ctx })
    }
}

impl Window {
    /// Returns a window builder.
    pub fn builder() -> Builder<'static> {
        Builder::default()
    }

    pub fn present(&mut self, tex: &Texture) -> Result<(), Error> {
        self.canvas.copy(&tex, None, None)?;
        self.canvas.present();
        Ok(())
    }

    /// Runs the main loop of the program, invoking the callback on each
    /// iteration to compute and draw the next frame.
    ///
    /// The main loop stops and this function returns if:
    /// * the user closes the window via the GUI (e.g. title bar close button);
    /// * the Esc key is pressed; or
    /// * the callback returns `ControlFlow::Break`.
    pub fn run<F>(&mut self, mut frame_fn: F) -> Result<(), Error>
    where
        F: FnMut(&mut Frame<Self, Framebuf>) -> ControlFlow<()>,
    {
        let (w, h) = self.dims;

        let tc = self.canvas.texture_creator();
        let mut tex = tc.create_texture_streaming(None, w, h)?;

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

            let cf = tex.with_lock(None, |cbuf, pitch| {
                if let Some(c) = ctx.depth_clear {
                    // Z-buffer stores reciprocals
                    zbuf.fill(c.recip());
                }
                if let Some(c) = ctx.color_clear {
                    let [r, g, b, a] = c.0;
                    cbuf.chunks_exact_mut(4).for_each(|ch| {
                        ch.copy_from_slice(&[b, g, r, a]);
                    });
                }

                let buf = Framebuf {
                    color_buf: MutSlice2::new((4 * w, h), pitch as u32, cbuf),
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

impl<'a> Target for Framebuf<'a> {
    fn rasterize<V, Fs>(
        &mut self,
        mut sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput
    where
        V: Vary,
        Fs: FragmentShader<V>,
    {
        // TODO Lots of duplicate code

        let x0 = sl.xs.start;
        let x1 = sl.xs.end.max(sl.xs.start);
        let cbuf_span =
            &mut self.color_buf.as_mut_slice2()[sl.y][4 * x0..4 * x1];
        let zbuf_span = &mut self.depth_buf.as_mut_slice2()[sl.y][x0..x1];

        let mut io = Throughput { i: x1 - x0, o: 0 };
        sl.fragments()
            .zip(cbuf_span.chunks_exact_mut(4))
            .zip(zbuf_span)
            .for_each(|((frag, c), z)| {
                let new_z = frag.pos.z();
                if ctx.depth_test(new_z, *z) {
                    if let Some(new_c) = fs.shade_fragment(frag) {
                        if ctx.color_write {
                            // TODO Blending should happen here
                            io.o += 1;
                            let [r, g, b, a] = new_c.0;
                            c.copy_from_slice(&[b, g, r, a]);
                        }
                        if ctx.depth_write {
                            *z = new_z;
                        }
                    }
                }
            });
        io
    }
}

//
// Trait impls
//

impl Default for Builder<'_> {
    fn default() -> Self {
        Self {
            dims: dims::SVGA_800_600,
            title: "// retrofire application //",
            vsync: true,
            fs: FullscreenType::Off,
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
