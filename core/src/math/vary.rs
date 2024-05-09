//! Support for varyings: types that can be linearly interpolated across
//! a face when rendering, such as colors, normals, or texture coordinates.

use core::mem;

/// A trait for types that can be linearly interpolated and distributed
/// between two endpoints.
///
/// This trait is designed particularly for *varyings:* types that are
/// meant to be interpolated across the face of a polygon when rendering,
/// but the methods are useful for various purposes.
pub trait Vary: Sized + Clone {
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
    /// # use retrofire_core::math::vary::Vary;
    /// let mut iter = 0.0f32.vary(0.2, Some(5));
    /// assert_eq!(iter.next(), Some(0.0));
    /// assert_eq!(iter.next(), Some(0.2));
    /// assert_eq!(iter.next(), Some(0.4));
    /// assert_eq!(iter.next(), Some(0.6));
    /// assert_eq!(iter.next(), Some(0.8));
    /// assert_eq!(iter.next(), None);
    /// ```
    fn vary(self, step: Self::Diff, max: Option<u32>) -> Self::Iter;

    /// Linearly distributes values between `self` and `other` *inclusive*.
    #[inline]
    fn vary_to(self, other: Self, n: u32) -> Self::Iter {
        let step = self.dv_dt(&other, 1.0 / n as f32);
        self.vary(step, Some(n + 1))
    }

    /// Returns, conceptually, `(other - self) / dt`.
    fn dv_dt(&self, other: &Self, recip_dt: f32) -> Self::Diff;

    /// Returns the result of offsetting `self` by `delta`, conceptually
    /// `self + delta`.
    #[must_use]
    fn step(&self, delta: &Self::Diff) -> Self;

    /// Performs perspective division.
    #[must_use]
    fn z_div(&self, z: f32) -> Self;

    /// Linearly interpolates between `self` and `other`.
    ///
    /// This method does not panic if `t < 0.0` or `t > 1.0`, or if `t`
    /// is a `NaN`, but the return value in those cases is unspecified.
    /// Individual implementations may offer stronger guarantees.
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::math::vary::Vary;
    /// assert_eq!(2.0.lerp(&5.0, 0.0), 2.0);
    /// assert_eq!(2.0.lerp(&5.0, 0.5), 3.5);
    /// assert_eq!(2.0.lerp(&5.0, 1.0), 5.0);
    /// ```
    #[inline]
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self.step(&self.dv_dt(other, t))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Iter<T: Vary> {
    pub val: T,
    pub step: T::Diff,
    pub n: Option<u32>,
}

#[inline]
pub fn lerp<V: Vary>(t: f32, from: V, to: V) -> V {
    from.lerp(&to, t)
}

impl Vary for () {
    type Iter = Iter<()>;
    type Diff = ();

    fn vary(self, _: Self::Diff, n: Option<u32>) -> Self::Iter {
        Iter { val: (), step: (), n }
    }
    fn dv_dt(&self, _: &Self, _: f32) {}
    fn step(&self, _: &Self::Diff) {}

    fn z_div(&self, _: f32) {}
}

impl<T: Vary, U: Vary> Vary for (T, U) {
    type Iter = Iter<Self>;
    type Diff = (T::Diff, U::Diff);

    fn vary(self, step: Self::Diff, n: Option<u32>) -> Self::Iter {
        Iter { val: self, step, n }
    }
    fn dv_dt(&self, other: &Self, recip_dt: f32) -> Self::Diff {
        (
            self.0.dv_dt(&other.0, recip_dt),
            self.1.dv_dt(&other.1, recip_dt),
        )
    }
    fn step(&self, (d0, d1): &Self::Diff) -> Self {
        (self.0.step(d0), self.1.step(d1))
    }

    fn z_div(&self, z: f32) -> Self {
        (self.0.z_div(z), self.1.z_div(z))
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
