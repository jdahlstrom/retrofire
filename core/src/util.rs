//! Various utility types and functions.
//!
//! * Owned and borrowed 2D buffers
//! * PNM image file loading
//! * Pixel formats

#[cfg(feature = "std")]
use std::{fs::File, io, path::Path};

mod buf;
pub mod dims;
pub mod pixfmt;
pub mod pnm;
mod rect;

pub(super) mod re_exports {
    pub use super::{
        buf::{AsMutSlice2, AsSlice2, Buf2, MutSlice2, Slice2},
        dims::Dims,
        pixfmt::IntoPixel,
        rect::Rect,
    };
}

pub use re_exports::*;

pub trait Load: Sized {
    #[cfg(feature = "std")]
    type Error: From<io::Error>;

    #[cfg(not(feature = "std"))]
    type Error;

    #[cfg(feature = "std")]
    fn from_path(path: impl AsRef<Path>) -> Result<Self, Self::Error> {
        Self::from_reader(io::BufReader::new(File::open(path)?))
    }

    #[cfg(feature = "std")]
    fn from_reader(
        reader: impl io::Read + io::Seek,
    ) -> Result<Self, Self::Error>;

    fn from_slice(slice: &[u8]) -> Result<Self, Self::Error>;
}
