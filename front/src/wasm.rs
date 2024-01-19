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

use retrofire_core::render::stats::Stats;
use web_sys::{
    CanvasRenderingContext2d as Context, Document, HtmlCanvasElement as Canvas,
    ImageData,
};

use retrofire_core::render::target::Framebuf;
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
    size: (u32, u32),
    ctx: Context,
}

#[derive(Debug, Default)] // TODO needs custom default
pub struct Builder {
    size: (u32, u32),
    id: String,
}

impl Builder {
    pub fn size(self, w: u32, h: u32) -> Self {
        Self { size: (w, h), ..self }
    }
}

impl Window {
    pub fn builder() -> Builder {
        Builder::default()
    }

    pub fn new(w: u32, h: u32) -> Result<Self, &'static str> {
        log("Starting wasm app...");

        let doc = Self::document().ok_or("document object was not found")?;
        let body = doc.body().ok_or("body element was not found")?;

        let cvs = Self::create_canvas(w, h).ok_or("could not create canvas")?;
        body.append_child(&cvs);

        let ctx = Self::context(&cvs).ok_or("could not get context")?;

        log("done...");
        Ok(Self { size: (w, h), ctx })
    }

    pub fn run<F>(mut self, mut frame_fn: F)
    where
        F: FnMut(&mut Frame) -> ControlFlow<()> + 'static,
    {
        let w = self.size.0 as usize;
        let h = self.size.1 as usize;

        let mut cb = Buf2::new_default(w, h);
        let mut zb = Buf2::new_default(w, h);

        let mut stats = Stats::default();

        let mut t_last = Duration::default();

        let mut outer: Rc<RefCell<Option<_>>> = Default::default();
        let mut inner = outer.clone();
        outer.borrow_mut().replace(Closure::new(move |t| {
            cb.fill(0x7F_00_00_00);
            zb.fill(f32::INFINITY);

            let t = Duration::from_millis(t as u64);
            let dt = t - t_last;
            let mut frame = Frame {
                t,
                dt,
                buf: Framebuf {
                    color_buf: cb.as_mut_slice2(),
                    depth_buf: zb.as_mut_slice2(),
                },
                stats: &mut stats,
            };

            if let Continue(_) = frame_fn(&mut frame) {
                self.put_image_data(cb.data()).unwrap();

                self.ctx.set_font("12px sans-serif");
                self.ctx
                    .set_fill_style(&JsValue::from_str("#CCC"));
                self.ctx.fill_text(
                    &format!("fps: {:.1}", dt.as_secs_f32().recip()),
                    8.0,
                    16.0,
                );

                requestAnimationFrame(inner.borrow().as_ref().unwrap());
            } else {
                let _ = inner.borrow_mut().take();
            }
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

    fn context(cvs: &Canvas) -> Option<Context> {
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
        let img = ImageData::new_with_u8_clamped_array(
            Clamped(&u8_data),
            self.size.0,
        )
        .map_err(|_| "could not create image data from color buf")?;
        self.ctx
            .put_image_data(&img, 0.0, 0.0)
            .map_err(|_| "failed blitting image data to canvas")
    }
}
