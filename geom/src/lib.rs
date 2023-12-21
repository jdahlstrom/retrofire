#![no_std]

extern crate alloc;
extern crate core;
#[cfg(feature = "std")]
extern crate std;

pub mod solids;

pub mod io;
#[cfg(feature = "teapot")]
pub mod teapot;
