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
//! * `micromath`:
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

pub mod prelude {
    pub use crate::geom::{vertex, Mesh, Tri, Vertex};

    pub use crate::math::angle::{degs, rads, turns, Angle};
    pub use crate::math::color::{
        hsl, hsla, rgb, rgba, Color3, Color3f, Color4, Color4f,
    };
    pub use crate::math::mat::{
        perspective, rotate_x, rotate_y, rotate_z, translate, viewport, Mat3x3,
        Mat4x4, Matrix,
    };
    pub use crate::math::vary::{lerp, Vary};
    pub use crate::math::vec::{
        splat, vec2, vec3, vec4, Vec2, Vec2i, Vec3, Vec3i, Vec4, Vec4i, Vector,
    };

    pub use crate::render::raster::Frag;
    pub use crate::render::shader::Shader;

    pub use crate::util::buf::{
        AsMutSlice2, AsSlice2, Buf2, MutSlice2, Slice2,
    };
}
