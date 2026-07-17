//! Various utility types and functions.

mod buf;
pub mod dims;
pub mod pixfmt;
#[cfg(feature = "alloc")]
pub mod pnm;
mod rect;

pub(super) mod re_exports {
    #[cfg(feature = "alloc")]
    pub use super::buf::Buf2;
    pub use super::{
        buf::{AsMutSlice2, AsSlice2, MutSlice2, Slice2},
        dims::Dims,
        pixfmt::IntoPixel,
        rect::Rect,
    };
}
pub use re_exports::*;
