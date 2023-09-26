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
//! # Features
//! * `std`: Enables `std` support.
//!   This makes available items requiring `std::time` or floating-point
//!   functions. Disabled by default; only `alloc` is required.

#![cfg_attr(not(feature = "std"), no_std)]

// TODO make alloc optional
extern crate alloc;
extern crate core;

pub mod geom;
pub mod math;
pub mod render;
pub mod util;
