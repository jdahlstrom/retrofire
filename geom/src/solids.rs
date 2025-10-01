//! Mesh approximations of various geometric shapes.

#[cfg(feature = "std")]
mod lathe;
mod platonic;

use re::geom::{Mesh, mesh::Builder};

#[cfg(feature = "std")]
pub use lathe::*;

pub use platonic::*;

pub trait Build<A>: Sized {
    fn build(self) -> Mesh<A>;

    fn builder(self) -> Builder<A> {
        self.build().into_builder()
    }
}
