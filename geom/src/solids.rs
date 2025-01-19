//! Mesh approximations of various geometric shapes.

#[cfg(feature = "std")]
mod lathe;
mod platonic;
mod subdiv;

#[cfg(feature = "std")]
pub use lathe::*;

pub use platonic::*;
pub use subdiv::*;
