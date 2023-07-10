//! Linear interpolation

/// A trait for types that can be linearly interpolated and distributed
/// between two endpoints.
///
/// This trait is especially designed for *varyings:* types that are
/// meant to be interpolated across the face of a polygon when rendering,
/// but the methods are of course useful for a multitude of purposes.
pub trait Vary: Sized {
    /// The iterator returned by the [vary][Self::vary] method.
    type Iter: Iterator<Item = Self>;

    /// Returns an iterator that yields values such that the first value
    /// equals `self`, and each subsequent value is offset by `step` from its
    /// predecessor using the [step][Self::step] method. If `max` is `Some(n)`,
    /// stops after `n` steps, otherwise infinite.
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::math::vary::Vary;
    /// let mut  iter = 0.0f32.vary(0.2, Some(5));
    /// assert_eq(iter.next(), 0.0);
    /// assert_eq(iter.next(), 0.2);
    /// assert_eq(iter.next(), 0.4);
    /// assert_eq(iter.next(), 0.6);
    /// assert_eq(iter.next(), 0.8);
    /// assert_eq(iter.next(), None);
    /// ```
    fn vary(self, step: Self, max: Option<u32>) -> Self::Iter;

    /// Linearly interpolates between `self` and `other`.
    ///
    /// This method does not panic if `t < 0.0` or `t > 1.0`,
    /// or if `t` is `NaN`, but the return value is unspecified.
    /// Individual implementations may offer stronger guarantees.
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::math::vary::Vary;
    /// assert_eq!(2.0.lerp(5.0, 0.0), 2.0);
    ///
    /// assert_eq!(2.0.lerp(5.0, 1.0), 5.0);
    ///
    /// assert_eq!(2.0.lerp(5.0, 0.5), 3.5);
    /// ```
    fn lerp(self, other: Self, t: f32) -> Self;

    /// Returns the result of offsetting `self` by `delta`.
    /// For normal arithmetic types this is simply addition.
    fn step(self, delta: Self) -> Self;
}

#[derive(Copy, Clone, Debug)]
pub struct Iter<T> {
    pub val: T,
    pub step: T,
    pub n: Option<u32>,
}

impl Vary for f32 {
    type Iter = Iter<Self>;

    fn vary(self, step: Self, max: Option<u32>) -> Self::Iter {
        Iter::new(self, step, max)
    }

    fn lerp(self, other: Self, t: f32) -> Self {
        self + t * (other - self)
    }

    /// Returns `self + delta`.
    fn step(self, delta: Self) -> Self {
        self + delta
    }
}

impl<T: Copy + Vary> Iter<T> {
    pub fn new(val: T, step: T, max: Option<u32>) -> Self {
        Self { val, step, n: max }
    }
}

impl<T: Copy + Vary> Iterator for Iter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if let Some(0) = self.n {
            return None;
        }
        if let Some(n) = &mut self.n {
            *n -= 1;
        }
        let res = self.val;
        self.val = self.val.step(self.step);
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;

    use super::*;

    #[test]
    fn vary_f32() {
        let iter = (-6.0f32).vary(1.2, Some(10));
        assert_approx_eq!(
            *iter.collect::<alloc::vec::Vec<_>>().as_slice(),
            [-6.0, -4.8, -3.6, -2.4, -1.2, 0.0, 1.2, 2.4, 3.6, 4.8]
        );
    }
}
