//! TODO doc
//!

#[cfg(feature = "alloc")]
pub use mesh::{Builder, Mesh};

pub use prim::*;

#[cfg(feature = "alloc")]
pub mod mesh;

mod prim;
