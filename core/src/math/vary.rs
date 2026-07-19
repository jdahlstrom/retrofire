//! Types that can be interpolated across a face when rendering.
//!
//! Common varying types include colors, texture coordinates,
//! and vertex normals.

use core::{
    mem,
    ops::{Range, RangeInclusive},
};

use super::Lerp;

pub trait ZDiv: Sized {
    #[must_use]
    fn z_div(self, _z: f32) -> Self {
        self
    }
}

/// A trait for types that can be linearly interpolated and distributed
/// between two endpoints.
///
/// This trait is designed particularly for *varyings:* values such as colors
/// and texture coordinates that are meant to be interpolated across the face
/// of a polygon when rendering, but the methods are useful for various purposes.
pub trait Vary: Lerp + ZDiv {
    /// The iterator returned by the [vary][Self::vary] method.
    type Iter: Iterator<Item = Self>;
    /// The difference type of `Self`.
    type Diff: Clone;

    /// Returns an iterator that yields values such that the first value
    /// equals `self`, and each subsequent value is offset by `step` from its
    /// predecessor using the [step][Self::step] method. If `max` is `Some(n)`,
    /// stops after `n` steps, otherwise infinite.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::Vary;
    ///
    /// let mut iter = 0.0f32.vary(0.2, Some(5));
    ///
    /// assert_eq!(iter.next(), Some(0.0));
    /// assert_eq!(iter.next(), Some(0.2));
    /// assert_eq!(iter.next(), Some(0.4));
    /// assert_eq!(iter.next(), Some(0.6));
    /// assert_eq!(iter.next(), Some(0.8));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn vary(self, step: Self::Diff, max: Option<u32>) -> Self::Iter;

    /// Linearly distributes `n` values between `self` and `other` *inclusive*.
    ///
    /// The first and last items emitted are `self` and `other` respectively.
    ///
    /// If `n` = 1, the only item emitted is `self`. If `n` = 0, emits nothing.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::Vary;
    ///
    /// let mut  v = 2.0.vary_to(8.0, 3);
    ///
    /// assert_eq!(v.next(), Some(2.0));
    /// assert_eq!(v.next(), Some(5.0));
    /// assert_eq!(v.next(), Some(8.0));
    /// assert_eq!(v.next(), None);
    #[inline]
    fn vary_to(self, other: Self, n: u32) -> Self::Iter {
        let recip_dt = if n <= 1 {
            // Dummy value, no actual steps taken if n is 0 or 1
            1.0
        } else {
            // Fencepost problem: n - 1 steps to yield n values
            1.0 / (n - 1) as f32
        };
        let step = self.dv_dt(&other, recip_dt); // Borrowck...
        self.vary(step, Some(n))
    }

    /// Returns, conceptually, `(other - self) / dt`.
    fn dv_dt(&self, other: &Self, recip_dt: f32) -> Self::Diff;

    /// Returns the result of offsetting `self` by `delta`, conceptually
    /// `self + delta`.
    #[must_use]
    fn step(&self, delta: &Self::Diff) -> Self;
}

pub trait RangeExt<T> {
    /// Returns an iterator of values linearly distributed over a range.
    fn vary(self, steps: u32) -> impl Iterator<Item = T>;
}

impl<T: Vary> RangeExt<T> for Range<T> {
    /// Returns an iterator of linearly distributed values over a half-open
    /// range.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::vary::RangeExt;
    ///
    /// // If steps = 0 or if the range is empty, returns an empty iterator.
    ///
    /// let mut iter = (1.0..5.0).vary(0);
    /// assert_eq!(iter.next(), None);
    ///
    /// let mut iter = (1.0..1.0).vary(123);
    /// assert_eq!(iter.next(), None);
    ///
    /// // If steps = 1, returns an iterator that yields one value: self.start.
    ///
    /// let mut iter = (1.0..5.0).vary(1);
    /// assert_eq!(iter.next(), Some(1.0));
    ///
    /// // Otherwise, returns an iterator over n uniformly spaced values such
    /// // that self.end would be the (n+1)th value.
    ///
    /// let mut iter = (1.0..5.0).vary(4);
    /// assert_eq!(iter.next(), Some(1.0));
    /// assert_eq!(iter.next(), Some(2.0));
    /// assert_eq!(iter.next(), Some(3.0));
    /// assert_eq!(iter.next(), Some(4.0));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn vary(self, steps: u32) -> impl Iterator<Item = T> {
        let recip_dt = if steps == 0 {
            // Dummy value, no actual steps taken if steps = 0
            1.0
        } else {
            1.0 / steps as f32
        };
        let step = self.start.dv_dt(&self.end, recip_dt); // Borrowck...
        self.start.clone().vary(step, Some(steps))
    }
}
impl<T: Vary> RangeExt<T> for RangeInclusive<T> {
    /// Returns an iterator of linearly distributed values over a closed range.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::vary::RangeExt;
    ///
    /// // If steps = 0 or if the range is empty, returns an empty iterator.
    ///
    /// let mut iter = (1.0..5.0).vary(0);
    /// assert_eq!(iter.next(), None);
    ///
    /// let mut iter = (1.0..1.0).vary(123);
    /// assert_eq!(iter.next(), None);
    ///
    /// // If steps = 1, returns an iterator that yields one value: self.start.
    ///
    /// let mut iter = (1.0..5.0).vary(1);
    /// assert_eq!(iter.next(), Some(1.0));
    ///
    /// // Otherwise, returns an iterator over n uniformly spaced values such
    /// // that self.end is the last value yielded.
    ///
    /// let mut iter = (1.0..5.0).vary(3);
    /// assert_eq!(iter.next(), Some(1.0));
    /// assert_eq!(iter.next(), Some(3.0));
    /// assert_eq!(iter.next(), Some(5.0));
    /// ```
    fn vary(self, steps: u32) -> impl Iterator<Item = T> {
        let recip_dt = if steps <= 1 {
            // Dummy value, no actual steps taken if n is 0 or 1
            1.0
        } else {
            // Fencepost problem: n - 1 steps to yield n values
            1.0 / (steps - 1) as f32
        };
        let step = self.start().dv_dt(&self.end(), recip_dt); // Borrowck...
        self.start().clone().vary(step, Some(steps))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Iter<T: Vary> {
    pub val: T,
    pub step: T::Diff,
    pub n: Option<u32>,
}

//
// Trait impls
//

impl Vary for () {
    type Iter = Iter<()>;
    type Diff = ();

    fn vary(self, (): (), n: Option<u32>) -> Iter<()> {
        Iter { val: (), step: (), n }
    }
    fn dv_dt(&self, (): &(), _: f32) {}
    fn step(&self, (): &()) {}
}

impl<T: Vary, U: Vary> Vary for (T, U) {
    type Iter = Iter<Self>;
    type Diff = (T::Diff, U::Diff);

    #[inline]
    fn vary(self, step: Self::Diff, n: Option<u32>) -> Self::Iter {
        Iter { val: self, step, n }
    }

    #[inline]
    fn dv_dt(&self, other: &Self, recip_dt: f32) -> Self::Diff {
        (
            self.0.dv_dt(&other.0, recip_dt),
            self.1.dv_dt(&other.1, recip_dt),
        )
    }

    #[inline]
    fn step(&self, (d0, d1): &Self::Diff) -> Self {
        (self.0.step(d0), self.1.step(d1))
    }
}

impl ZDiv for () {}

impl<T: ZDiv, U: ZDiv> ZDiv for (T, U) {
    #[inline]
    fn z_div(self, z: f32) -> Self {
        (self.0.z_div(z), self.1.z_div(z))
    }
}

impl ZDiv for f32 {
    #[inline]
    fn z_div(self, z: f32) -> Self {
        self / z
    }
}

impl<T: Vary> Iterator for Iter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.n == Some(0) {
            return None;
        }
        if let Some(n) = &mut self.n {
            *n -= 1;
        }
        let new = self.val.step(&self.step);
        Some(mem::replace(&mut self.val, new))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(n) = self.n {
            (n as usize, Some(n as usize))
        } else {
            (0, None)
        }
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

    #[test]
    fn vary_to_zero() {
        assert_eq!(1.0.vary_to(2.0, 0).next(), None);
    }

    #[test]
    fn vary_to_one() {
        let mut v = 1.0.vary_to(2.0, 1);
        assert_eq!(v.next(), Some(1.0));
        assert_eq!(v.next(), None);
    }

    #[test]
    fn vary_to_two() {
        let mut v = 1.0.vary_to(2.0, 2);
        assert_eq!(v.next(), Some(1.0));
        assert_eq!(v.next(), Some(2.0));
        assert_eq!(v.next(), None);
    }

    #[test]
    fn vary_range() {
        let mut v = (1.0..2.0).vary(2);
        assert_eq!(v.next(), Some(1.0));
        assert_eq!(v.next(), Some(1.5));
        assert_eq!(v.next(), None);
    }

    #[test]
    fn vary_range_inclusive() {
        let mut v = (1.0..=2.0).vary(3);
        assert_eq!(v.next(), Some(1.0));
        assert_eq!(v.next(), Some(1.5));
        assert_eq!(v.next(), Some(2.0));
        assert_eq!(v.next(), None);
    }
}
