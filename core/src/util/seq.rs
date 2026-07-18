use std::prelude::v1::Vec;

pub trait AsSlice {
    type Elem;

    fn as_slice(&self) -> &[Self::Elem];
}

pub struct Seq<D>(pub D);

impl<T> AsSlice for Vec<T> {
    type Elem = T;

    fn as_slice(&self) -> &[Self::Elem] {
        self.as_slice()
    }
}

impl<T> AsSlice for [T] {
    type Elem = T;

    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<T, const N: usize> AsSlice for [T; N] {
    type Elem = T;

    fn as_slice(&self) -> &[Self::Elem] {
        self.as_slice()
    }
}
