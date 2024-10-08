#![allow(unused)]

use core::cell::RefCell;
use core::mem::transmute;
use core::ops::{ControlFlow, ControlFlow::*, Deref};
use core::ptr::slice_from_raw_parts_mut;
use core::slice;
use core::time::Duration;

use alloc::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::Clamped;

use web_sys::js_sys::{Uint32Array, Uint8ClampedArray};
use web_sys::{
    CanvasRenderingContext2d as Context2d, Document,
    HtmlCanvasElement as Canvas, ImageData,
};

use retrofire_core::math::color::rgba;
use retrofire_core::render::{ctx::Context, stats::Stats, target::Framebuf};
use retrofire_core::util::buf::{AsMutSlice2, Buf2, MutSlice2};

use crate::Frame;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    pub fn log(msg: &str);

    #[wasm_bindgen(js_namespace = console)]
    pub fn error(msg: &str);

    #[wasm_bindgen]
    fn requestAnimationFrame(cb: &Closure<dyn FnMut(f32)>);
}

#[derive(Debug)]
pub struct Window {
    pub size: (u32, u32),
    pub ctx2d: Context2d,
    pub ctx: Context,
}

#[derive(Debug)]
pub struct Builder {
    size: (u32, u32),
}

impl Builder {
    pub fn size(self, w: u32, h: u32) -> Self {
        Self { size: (w, h), ..self }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self { size: (800, 600) }
    }
}

impl Window {
    pub fn builder() -> Builder {
        Builder::default()
    }

    pub fn new(w: u32, h: u32) -> Result<Self, &'static str> {
        log("Starting wasm app...");

        let doc = Self::document().ok_or("document object not found")?;
        let body = doc.body().ok_or("body element not found")?;

        let cvs = Self::create_canvas(w, h).ok_or("could not create canvas")?;
        body.append_child(&cvs);

        let ctx2d = Self::context2d(&cvs).ok_or("could not get context")?;
        let ctx = Context::default();

        log("done!");
        Ok(Self { size: (w, h), ctx2d, ctx })
    }

    pub fn run<F>(mut self, mut frame_fn: F)
    where
        F: FnMut(&mut Frame<Self>) -> ControlFlow<()> + 'static,
    {
        let (w, h) = self.size;
        let mut ctx = self.ctx.clone();

        let mut cbuf = Buf2::new(w, h);
        let mut zbuf = Buf2::new(w, h);

        let mut t_last = Duration::default();

        let mut outer: Rc<RefCell<Option<_>>> = Rc::default();
        let mut inner = outer.clone();
        outer
            .borrow_mut()
            .replace(Closure::new(move |ms| {
                // TODO add clear method to Framebuf?
                if let Some(c) = ctx.color_clear {
                    cbuf.fill(c.to_argb_u32());
                }
                if let Some(z) = ctx.depth_clear {
                    // Depth buffer contains reciprocal depth values
                    zbuf.fill(z.recip());
                }

                let t = Duration::from_secs_f32(ms / 1e3);
                let dt = t - t_last;
                let buf = Framebuf {
                    color_buf: cbuf.as_mut_slice2(),
                    depth_buf: zbuf.as_mut_slice2(),
                };
                let mut frame = Frame {
                    t,
                    dt,
                    buf,
                    ctx: &mut ctx,
                    win: &mut self,
                };

                if let Continue(_) = frame_fn(&mut frame) {
                    requestAnimationFrame(inner.borrow().as_ref().unwrap());
                } else {
                    let _ = inner.borrow_mut().take();
                }

                self.put_image_data(cbuf.data()).unwrap();
                t_last = t;
            }));
        requestAnimationFrame(outer.borrow().as_ref().unwrap());
    }

    pub fn document() -> Option<Document> {
        web_sys::window()?.document()
    }

    fn create_canvas(w: u32, h: u32) -> Option<Canvas> {
        Self::document()?
            .create_element("canvas")
            .ok()?
            .dyn_into()
            .map(|cvs: Canvas| {
                cvs.set_width(w);
                cvs.set_height(h);
                cvs
            })
            .ok()
    }

    fn context2d(cvs: &Canvas) -> Option<Context2d> {
        cvs.get_context("2d")
            .ok()
            .flatten()?
            .dyn_into()
            .ok()
    }

    fn put_image_data(&self, data: &[u32]) -> Result<(), &'static str> {
        // SAFETY: TODO
        let u8_data = unsafe {
            slice::from_raw_parts(
                data as *const [u32] as *const u8,
                data.len() * 4,
            )
        };
        let img =
            ImageData::new_with_u8_clamped_array(Clamped(u8_data), self.size.0)
                .map_err(|_| "could not create image data from color buf")?;

        self.ctx2d
            .put_image_data(&img, 0.0, 0.0)
            .map_err(|_| "failed blitting image data to canvas")
    }
}
