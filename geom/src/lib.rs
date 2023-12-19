#![no_std]

extern crate alloc;
extern crate core;
#[cfg(feature = "std")]
extern crate std;

pub mod io;
pub mod solids;
#[cfg(feature = "teapot")]
pub mod teapot;
