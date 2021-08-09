use std::ops::DerefMut;

pub mod io;
pub mod color;

#[derive(Clone)]
pub struct Buffer<T, B = Vec<T>>
where B: DerefMut<Target=[T]>
{
    width: usize,
    height: usize,
    data: B,
}

impl<T, B> Buffer<T, B>
where B: DerefMut<Target=[T]>
{
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn data(&self) -> &B {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut B {
        &mut self.data
    }

    #[inline(always)]
    pub fn get(&self, x: usize, y: usize) -> &T {
        &self.data[self.width * y + x]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[self.width * y + x]
    }

    #[inline(always)]
    pub fn put(&mut self, x: usize, y: usize, val: T) {
        (&mut self.data)[self.width * y + x] = val;
    }
}

impl<T> Buffer<T, Vec<T>> {
    pub fn new(width: usize, height: usize, init: T) -> Self
    where T: Clone {
        Self {
            width,
            height,
            data: vec![init; width * height],
        }
    }
    pub fn from_vec(width: usize, data: Vec<T>) -> Self {
        let height = data.len() / width;
        assert_eq!(data.len(), width * height);
        Self { width, height, data }
    }
}

impl<'a, T> Buffer<T, &'a mut [T]> {
    pub fn borrow(width: usize, data: &'a mut [T]) -> Self {
        let height = data.len() / width;
        assert_eq!(data.len(), width * height);
        Self { width, height, data }
    }
}
