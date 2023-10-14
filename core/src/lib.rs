//! ```text
//!                                                     _______
//!                        ____                       /´  ____/_
//!       __ ______ ____ _/   /__ _ ______ _____ ____/   /_/___/  __ _____ _____
//!    ==/  ´ ____/ __   \   ____/ ´ ____/  __  ` __    ___.  /==/  ´  ___/ __   \
//!   ==/   /´=/   ______/  /==/   /´=/   /==/   /=/   /=/   /==/   /´=/   ______/
//!  ==/   /==/   /____/   /__/   /==/   /__/   /=/   /=/   /__/   /==/   /_____
//! ==/___/===\_______/\______,__/===\________/__/   /==\______/__/===\________/
//!                                          /_____,´
//! ```
//!
//! Core functionality of the `retrofire` project.
//!
//! Includes a math library with vectors, matrices, colors, and angles; basic
//! geometry primitives; a software 3D renderer with customizable shaders;
//! # Features
//! * `std`:
//!   Makes available items requiring floating-point functions or
//!   `std::time`. If disabled, this crate only depends on `alloc`.
//!
//! * `micromath`:
//!   Provides an alternative, no-std implementation of floating-point
//!   functions via [micromath](https://crates.io/crates/micromath).
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
