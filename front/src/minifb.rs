//! Frontend using the `minifb` crate for window creation and event handling.

use std::ops::ControlFlow::{self, Break};
use std::time::{Duration, Instant};

use minifb::{Key, WindowOptions};

use retrofire_core::render::ctx::Context;
use retrofire_core::render::target::Framebuf;
use retrofire_core::util::buf::{AsMutSlice2, Buf2};

use crate::Frame;

/// A lightweight wrapper of a `minibuf` window.
pub struct Window {
    /// The wrapped minifb window.
    pub imp: minifb::Window,
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
    pub opts: WindowOptions,
}

impl Default for Builder<'_> {
    fn default() -> Self {
        Self {
            size: (800, 600),
            title: "// retrofire application //",
            max_fps: Some(60.0),
            opts: WindowOptions::default(),
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
    /// Sets other `minifb` options.
    pub fn options(mut self, opts: WindowOptions) -> Self {
        self.opts = opts;
        self
    }

    /// Creates the window.
    pub fn build(self) -> Window {
        let Self { size, title, max_fps, opts } = self;
        let mut imp =
            minifb::Window::new(title, size.0 as usize, size.1 as usize, opts)
                .unwrap();
        imp.limit_update_rate(
            max_fps.map(|fps| Duration::from_secs_f32(1.0 / fps)),
        );
        let ctx = Context::default();
        Window { imp, size, ctx }
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
    pub fn present(&mut self, fb: &[u32]) {
        let (w, h) = self.size;
        self.imp
            .update_with_buffer(fb, w as usize, h as usize)
            .unwrap();
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
        F: FnMut(&mut Frame<Self>) -> ControlFlow<()>,
    {
        let (w, h) = self.size;
        let mut cbuf = Buf2::new_default(w, h);
        let mut zbuf = Buf2::new_default(w, h);
        let mut ctx = self.ctx.clone();

        let start = Instant::now();
        let mut last = Instant::now();
        loop {
            if self.should_quit() {
                break;
            }
            if let Some(c) = ctx.color_clear {
                cbuf.fill(c.to_argb_u32());
            }
            if let Some(c) = ctx.depth_clear {
                zbuf.fill(c);
            }

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
            if let Break(_) = frame_fn(frame) {
                break;
            }
            self.present(cbuf.data_mut());

            ctx.stats.borrow_mut().frames += 1.0;
        }
        println!("{}", ctx.stats.borrow());
    }

    fn should_quit(&self) -> bool {
        !self.imp.is_open() || self.imp.is_key_down(Key::Escape)
    }
}
