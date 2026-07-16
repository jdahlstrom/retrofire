#![no_std]

extern crate alloc;
extern crate core;
#[cfg(feature = "std")]
extern crate std;

pub mod io;
pub mod isect;
pub mod solids;

mod tri;

pub use isect::Intersect;
pub use tri::triangulate;
