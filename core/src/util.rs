//! Various utility types and functions.

pub mod buf;
pub mod pixfmt;
pub mod pnm;
pub mod rect;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Dims(pub u32, pub u32);

impl Dims {
    pub fn count(&self) -> usize {
        (self.0 as u64 * self.1 as u64)
            .try_into()
            .expect("count should fit in usize")
    }
}
