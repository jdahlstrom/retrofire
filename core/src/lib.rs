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
        render::*,
        util::buf::{AsMutSlice2, AsSlice2, Buf2, MutSlice2, Slice2},
    };
}
