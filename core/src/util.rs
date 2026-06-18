//! Various utility types and functions.

mod buf;
pub mod pixfmt;
pub mod pnm;
mod rect;

pub(super) mod re_exports {
    pub use super::{
        Dims,
        buf::{AsMutSlice2, AsSlice2, Buf2, MutSlice2, Slice2},
        pixfmt::IntoPixel,
        rect::Rect,
    };
}
pub use re_exports::*;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Dims(pub u32, pub u32);

impl Dims {
    pub fn count(&self) -> usize {
        (self.0 as u64 * self.1 as u64)
            .try_into()
            .expect("count should fit in usize")
    }
}
