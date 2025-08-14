//! ```text
//!                                                      ______
//!                         ___                       /´  ____/\
//!       __ ______ _____ /   /\_ _ ______ _____ __  /   /_/___/\ __ _____ ______
//!    ==/  ´ ____/ __   \   ____/ ´ ____/  __  ` __    ___,  /==/  ´  ___/ __   \
//!   ==/   /´=/   ______/  /==/   /´=/   /==/   /=/   /=/   /==/   /´=/   ______/\
//!  ==/   /==/   /____/   /__/   /==/   /__/   /=/   /=/   /__/   /==/   /______\/
//! ==/___/ ==\_______/\______/__/ ==\________,´_/   /==\______/__/ ==\________/\
//! ==\___\/ ==\______\/\_____\__\/ ==\______/_____,´ /==\_____\___\/==\_______\/
//!                                          \_____\,´
//! ```
//!
//! Core functionality of the `retrofire` project.
//!
//! Includes a math library with vectors, matrices, colors, and angles; basic
//! geometry primitives; a software 3D renderer with customizable shaders;
//! with more to come.
//!
//! # Crate features
//!
//! * `std`:
//!   Makes available items requiring I/O, timekeeping, or any floating-point
//!   functions not included in `core`. In particular this means trigonometric
//!   and transcendental functions.
//!
//!   If this feature is disabled, the crate only depends on `alloc`.
//!
//! * `libm`:
//!   Provides software implementations of floating-point functions via the
//!   [libm](https://crates.io/crates/libm) crate.
//!
//! * `mm`:
//!   Provides fast approximate implementations of floating-point functions
//!   via the [micromath](https://crates.io/crates/micromath) crate.
//!
//! All features are disabled by default.
//!
//! # Example
//!
//! ```
//! use retrofire_core::{prelude::*, util::*};
//!
//! let verts = [
//!     vertex(pt3(-0.8, 1.0, 0.0), rgb(1.0, 0.0, 0.0)),
//!     vertex(pt3(0.8, 1.0, 0.0), rgb(0.0, 0.8, 0.0)),
//!     vertex(pt3(0.0, -1.5, 0.0), rgb(0.4, 0.4, 1.0)),
//! ];
//!
//! let shader = shader::new(
//!     |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
//!         vertex(mvp.apply(&v.pos), v.attrib)
//!     },
//!     |frag: Frag<Color3f>| frag.var.to_color4(),
//! );
//!
//! let dims @ (w, h) = (640, 480);
//! let modelview = translate3(0.0, 0.0, 2.0).to();
//! let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
//! let viewport = viewport(pt2(0, 0)..pt2(w, h));
//!
//! let mut fb = Colorbuf {
//!     buf: Buf2::new(dims),
//!     fmt: pixfmt::Xrgb8888,
//! };
//!
//! render(
//!     [Tri([0, 1, 2])],
//!     verts,
//!     &shader,
//!     &modelview.then(&project),
//!     viewport,
//!     &mut fb,
//!     &Context::default(),
//! );
//!
//! assert_eq!(fb.buf[[w/2, h/2]], 0x00_74_66_65);
//! ```

#![no_std]

#[cfg(feature = "std")]
extern crate std;

// TODO make alloc optional
extern crate alloc;
extern crate core;

pub mod geom;
pub mod math;
pub mod render;
pub mod util;

/// Prelude module exporting many frequently used items.
pub mod prelude {
    pub use crate::{
        geom::{Mesh, Normal2, Normal3, Tri, Vertex, Vertex2, Vertex3, vertex},
        math::*,
        render::{raster::Frag, *},
        util::buf::{AsMutSlice2, AsSlice2, Buf2, MutSlice2, Slice2},
    };
}
