//! Frontend using the `minifb` crate for window creation and event handling.

use std::{
    ops::ControlFlow::{self, Break},
    time::Instant,
};

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::render::Texture;

use retrofire_core::{
    math::Vary,
    render::ctx::Context,
    render::raster::Frag,
    render::raster::Scanline,
    render::shader::FragmentShader,
    render::stats::Throughput,
    render::target::Target,
    util::buf::{Buf2, MutSlice2},
};

use crate::Frame;

/// A lightweight wrapper of a `minibuf` window.
pub struct Window {
    /// The SDL canvas.
    pub cvs: sdl2::render::WindowCanvas,
    /// The SDL event pump.
    pub ep: sdl2::EventPump,
    /// The width and height of the window.
    pub size: (u32, u32),
    /// Rendering context defaults.
    pub ctx: Context,
}

/// Builder for creating `Window`s.
pub struct Builder<'title> {
    pub size: (u32, u32),
    pub title: &'title str,
    pub max_fps: Option<f32>,
}

pub struct Framebuf<'a> {
    pub color_buf: MutSlice2<'a, u8>,
    pub depth_buf: MutSlice2<'a, f32>,
}

impl Default for Builder<'_> {
    fn default() -> Self {
        Self {
            size: (800, 600),
            title: "// retrofire application //",
            max_fps: Some(60.0),
        }
    }
}

impl<'t> Builder<'t> {
    /// Sets the width and height of the window.
    pub fn size(mut self, w: u32, h: u32) -> Self {
        self.size = (w, h);
        self
    }
    /// Sets the title of the window.
    pub fn title(mut self, title: &'t str) -> Self {
        self.title = title;
        self
    }
    /// Sets the frame rate cap of the window. `None` means unlimited
    /// frame rate (the main loop runs as fast as possible).
    pub fn max_fps(mut self, fps: Option<f32>) -> Self {
        self.max_fps = fps;
        self
    }

    /// Creates the window.
    pub fn build(self) -> Window {
        let Self { size, title, max_fps: _ } = self;

        let sdl = sdl2::init().expect("could not initialize SDL");

        let cvs = sdl
            .video()
            .expect("could not get SDL video subsystem")
            .window(title, size.0, size.1)
            .build()
            .expect("could not open SDL window")
            .into_canvas()
            .accelerated()
            .build()
            .expect("could not create SDL canvas");

        let ep = sdl
            .event_pump()
            .expect("could not get SDL event pump");

        let ctx = Context::default();

        Window { cvs, ep, size, ctx }
    }
}

impl Window {
    /// Returns a window builder.
    pub fn builder() -> Builder<'static> {
        Builder::default()
    }

    /// Updates the window content with pixel data from `fb`.
    ///
    /// The data is interpreted as colors in `0x00_RR_GG_BB` format.
    /// # Panics
    /// If `fb.len() < self.size.0 * self.size.1`.
    pub fn present(&mut self, tex: &Texture) {
        self.cvs.copy(&tex, None, None).expect("TODO");
        self.cvs.present();
    }

    /// Runs the main loop of the program, invoking the callback on each
    /// iteration to compute and draw the next frame.
    ///
    /// The main loop stops and this function returns if:
    /// * the user closes the window via the GUI (e.g. titlebar close button);
    /// * the Esc key is pressed; or
    /// * the callback returns `ControlFlow::Break`.
    pub fn run<F>(&mut self, mut frame_fn: F)
    where
        F: FnMut(&mut Frame<Self, Framebuf>) -> ControlFlow<()>,
    {
        let (w, h) = self.size;

        let tc = self.cvs.texture_creator();

        let mut tex = tc
            .create_texture_streaming(None, w, h)
            .expect("could not create output texture");

        let mut zbuf = Buf2::new(w, h);
        let mut ctx = self.ctx.clone();

        let start = Instant::now();
        let mut last = Instant::now();
        'main: loop {
            for e in self.ep.poll_iter() {
                match e {
                    Event::Quit { .. }
                    | Event::KeyDown {
                        keycode: Some(Keycode::Escape), ..
                    } => break 'main,

                    _ => (),
                }
            }

            if let Some(c) = ctx.depth_clear {
                zbuf.fill(c);
            }

            let cf = tex
                .with_lock(None, |cbuf, pitch| {
                    if let Some(c) = ctx.color_clear {
                        let [r, g, b, a] = c.0;

                        cbuf.chunks_exact_mut(4)
                            .for_each(|pix| pix.copy_from_slice(&[b, g, r, a]));
                    }

                    cbuf.as

                    let mut cbuf = MutSlice2::new(4 * w, h, pitch as u32, cbuf);

                    let frame = &mut Frame {
                        t: start.elapsed(),
                        dt: last.elapsed(),
                        buf: Framebuf {
                            color_buf: cbuf.as_mut_slice2(),
                            depth_buf: zbuf.as_mut_slice2(),
                        },
                        win: self,
                        ctx: &mut ctx,
                    };
                    last = Instant::now();
                    frame_fn(frame)
                })
                .expect("could not render to texture");

            if cf == Break(()) {
                break;
            }

            self.present(&tex);

            ctx.stats.borrow_mut().frames += 1.0;
        }
        println!("{}", ctx.stats.borrow());
    }
}

impl<'a> Target for Framebuf<'a> {
    fn rasterize<V, Fs>(
        &mut self,
        sl: Scanline<V>,
        fs: &Fs,
        ctx: &Context,
    ) -> Throughput
    where
        V: Vary,
        Fs: FragmentShader<Frag<V>>,
    {
        let Scanline { y, xs, frags } = sl;

        let x0 = xs.start;
        let x1 = xs.end.max(xs.start);
        let cbuf_span = &mut self.color_buf.as_mut_slice2()[y][4 * x0..4 * x1];
        let zbuf_span = &mut self.depth_buf.as_mut_slice2()[y][x0..x1];

        let mut io = Throughput { i: x1 - x0, o: 0 };
        frags
            .zip(cbuf_span.chunks_exact_mut(4))
            .zip(zbuf_span)
            .for_each(|(((pos, var), c), z)| {
                let new_z = pos.z();
                if ctx.depth_test(new_z, *z) {
                    let frag = Frag { pos, var };
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
