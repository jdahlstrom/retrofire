//! ```text
//!                                                     ______
//!                        ____                       /´  ____/\
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
//! # Cargo features
//!
//! * `std`:
//!   Makes available items requiring floating-point functions or timekeeping.
//!   If disabled, this crate only depends on `alloc`.
//!
//! * `micromath`:
//!   Provides an alternative, no-std implementation of floating-point
//!   functions via the [micromath](https://crates.io/crates/micromath) crate.
//!
//! All features are disabled by default.

#![cfg_attr(not(feature = "std"), no_std)]

// TODO make alloc optional
extern crate alloc;
extern crate core;

pub mod geom;
pub mod math;
pub mod render;
pub mod util;
