use std::ops::DerefMut;

pub mod io;
pub mod color;

#[derive(Clone)]
pub struct Buffer<T, B: DerefMut<Target=[T]> = Vec<T>> {
    pub width: usize,
    pub height: usize,
    pub data: B,
}
impl<T> Buffer<T, Vec<T>> {
    pub fn new(width: usize, height: usize, init: T) -> Self
        where T: Clone {
        Self {
            width, height,
            data: vec![init; width * height],
        }
    }
}
impl<'a, T> Buffer<T, &'a mut [T]> {
    pub fn borrow(width: usize, data: &'a mut [T]) -> Self {
        let height = data.len() / width;
        assert_eq!(data.len(), width * height);
        Self { width, height, data }
    }
}
