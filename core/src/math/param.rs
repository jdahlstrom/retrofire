use core::ops::Range;

use crate::math::Lerp;

pub trait Parametric<T> {
    #[allow(unused)]
    fn eval(&self, t: f32) -> T;
}

impl<F, T> Parametric<T> for F
where
    F: Fn(f32) -> T,
{
    fn eval(&self, t: f32) -> T {
        self(t)
    }
}

impl<T: Lerp> Parametric<T> for Range<T> {
    fn eval(&self, t: f32) -> T {
        self.start.lerp(&self.end, t)
    }
}
