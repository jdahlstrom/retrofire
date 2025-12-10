#![no_std]
#![no_main]

extern crate alloc;

use alloc::alloc::*;
use core::{ffi::c_void, panic::PanicInfo};

use libc::{abort, c_char, c_int, free, malloc, putchar, puts};

use re::prelude::*;

use re::math::mat::ProjMat3;
use re::render::{Model, render, shader};

#[global_allocator]
static ALLOC: Malloc = Malloc;

struct Malloc;

unsafe impl GlobalAlloc for Malloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { malloc(layout.size()) as *mut u8 }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _: Layout) {
        unsafe { free(ptr as *mut c_void) }
    }
}

#[panic_handler]
unsafe fn panic(_info: &PanicInfo) -> ! {
    unsafe { abort() }
}

#[unsafe(no_mangle)]
fn main() -> i32 {
    let verts = [
        vertex(pt3(-1.0, -1.0, 0.0), rgb(1.0, 0.0, 0.0)),
        vertex(pt3(0.0, 1.0, 0.0), rgb(0.4, 0.4, 1.0)),
        vertex(pt3(1.0, -1.0, 0.0), rgb(0.0, 0.8, 0.0)),
    ];

    let shader = shader::new(
        |v: Vertex3<Color3f>, mvp: &ProjMat3<Model>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f<_>>| frag.var.to_color4(),
    );

    let dims @ (w, h) = (640, 480);
    let modelview = translate3(0.0, 0.0, 2.0).to();
    let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(0, h)..pt2(w, 0));

    let mut framebuf = Buf2::<Color4>::new(dims);

    render(
        [tri(0, 1, 2)],
        verts,
        &shader,
        &modelview.then(&project),
        viewport,
        &mut framebuf,
        &Context::default(),
    );

    unsafe {
        puts("P6\n640 480 255\n\0".as_ptr() as *const c_char);
    }
    for &col in framebuf.data() {
        unsafe {
            putchar(col.r() as c_int);
            putchar(col.g() as c_int);
            putchar(col.b() as c_int);
        };
    }
    0
}
