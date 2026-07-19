//! Various utility types and functions.

mod buf;
pub mod dims;
pub mod pixfmt;
pub mod pnm;
mod rect;
pub mod seq;

pub(super) mod re_exports {
    pub use super::{
        buf::{AsMutSlice2, AsSlice2, Buf2, MutSlice2, Slice2},
        dims::Dims,
        pixfmt::IntoPixel,
        rect::Rect,
        seq::AsSlice,
    };
}
pub use re_exports::*;
