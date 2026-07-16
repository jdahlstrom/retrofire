#![no_std]
#![no_main]

extern crate alloc;

use alloc::alloc::*;
use core::{ffi::CStr, ffi::c_void, panic::PanicInfo};

use libc::{abort, c_int, free, malloc, putchar, puts};

use re::prelude::*;

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
    unsafe {
        abort();
    }
}

#[unsafe(no_mangle)]
fn main() -> i32 {
    let verts = [
        vertex(pt3(-1.0, -1.0, 0.0), rgb(1.0, 0.0, 0.0)),
        vertex(pt3(0.0, 1.0, 0.0), rgb(1.0, 0.0, 0.0)),
        vertex(pt3(1.0, -1.0, 0.0), rgb(1.0, 0.0, 0.0)),
    ];

    let shader = shader::new(
        |v: Vertex3<Color3f>, mvp: &ProjMat3<Model>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f<_>>, _: &_| Some(frag.var.to_color3()),
    );

    let dims @ Dims(w, h) = Dims(640, 480);
    let modelview = translate((0.0, 0.0, 2.0)).to();
    let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(0, h)..pt2(w, 0));

    let mut framebuf = Buf2::<Color3>::new(dims);

    render(
        [tri(0, 1, 2)],
        verts,
        &shader,
        &modelview.then(&project),
        viewport,
        &mut framebuf,
        &Context::default(),
    );

    let header = CStr::from_bytes_with_nul(b"P6\n640 480 255\0").unwrap();
    unsafe {
        puts(header.as_ptr()); // Appends a newline
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
