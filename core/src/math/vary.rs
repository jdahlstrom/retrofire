//! Linear interpolation

use core::mem;
use std::iter::{zip, Zip};

/// A trait for types that can be linearly interpolated and distributed
/// between two endpoints.
///
/// This trait is designed particularly for *varyings:* types that are
/// meant to be interpolated across the face of a polygon when rendering,
/// but the methods are of course useful for various purposes.
pub trait Vary: Sized {
    /// The iterator returned by the [vary][Self::vary] method.
    type Iter: Iterator<Item = Self>;
    /// The difference type of `Self`.
    type Diff;

    /// Returns an iterator that yields values such that the first value
    /// equals `self`, and each subsequent value is offset by `step` from its
    /// predecessor using the [step][Self::step] method. If `max` is `Some(n)`,
    /// stops after `n` steps, otherwise infinite.
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::math::vary::Vary;
    /// let mut  iter = 0.0f32.vary(0.2, Some(5));
    /// assert_eq!(iter.next(), Some(0.0));
    /// assert_eq!(iter.next(), Some(0.2));
    /// assert_eq!(iter.next(), Some(0.4));
    /// assert_eq!(iter.next(), Some(0.6));
    /// assert_eq!(iter.next(), Some(0.8));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn vary(self, step: Self::Diff, max: Option<u32>) -> Self::Iter;

    /// Returns the result of offsetting `self` by `delta`.
    /// For arithmetic types this is simply addition.
    fn step(&self, delta: &Self::Diff) -> Self;

    /// Returns the difference between `self` and `other`.
    /// For arithmetic types this is simply subtraction.
    fn diff(&self, other: &Self) -> Self::Diff;

    /// Scales `diff` by `s`.
    ///
    /// TODO it's a bit ugly to have this method here and not on Self::Diff.
    fn scale(diff: &Self::Diff, s: f32) -> Self::Diff;
}

#[derive(Copy, Clone, Debug)]
pub struct Iter<T: Vary> {
    pub val: T,
    pub step: T::Diff,
    pub n: Option<u32>,
}

impl<T: Vary, U: Vary> Vary for (T, U) {
    type Iter = Zip<T::Iter, U::Iter>;
    type Diff = (T::Diff, U::Diff);

    fn vary(self, step: Self::Diff, max: Option<u32>) -> Self::Iter {
        zip(self.0.vary(step.0, max), self.1.vary(step.1, max))
    }

    fn step(&self, delta: &Self::Diff) -> Self {
        (self.0.step(&delta.0), self.1.step(&delta.1))
    }

    fn diff(&self, other: &Self) -> Self::Diff {
        (self.0.diff(&other.0), self.1.diff(&other.1))
    }

    fn scale(diff: &Self::Diff, s: f32) -> Self::Diff {
        (T::scale(&diff.0, s), U::scale(&diff.1, s))
    }
}

impl<T: Vary> Iterator for Iter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match &mut self.n {
            Some(0) => return None,
            Some(n) => *n -= 1,
            None => (),
        }
        let new = self.val.step(&self.step);
        Some(mem::replace(&mut self.val, new))
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;

    use super::*;

    #[test]
    fn vary_f32() {
        use alloc::vec::Vec;
        let varying = (-6.0f32).vary(1.2, Some(10));
        assert_approx_eq!(
            varying.collect::<Vec<_>>()[..],
            [-6.0, -4.8, -3.6, -2.4, -1.2, 0.0, 1.2, 2.4, 3.6, 4.8]
        );
    }
}
