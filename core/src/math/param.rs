use core::marker::PhantomData;
use core::ops::Range;

use super::Lerp;

/// Represents a single-variable parametric curve.
// TODO More documentation
// TODO Associated type instead of parameter?
pub trait Parametric<T> {
    /// Returns the value of `self` at `t`.
    ///
    /// The "canonical" domain of this function is `t` ∈ [0.0, 1.0],
    /// but implementations should return "reasonable" values outside
    /// the unit interval as well.
    #[allow(unused)]
    fn eval(&self, t: f32) -> T;

    fn iter(&self, step: f32) -> Iter<'_, T, Self> {
        Iter {
            param: self,
            t: 0.0,
            step,
            end: 1.0,
            _pd: PhantomData,
        }
    }

    fn iter_n(&self, n: u32) -> Iter<'_, T, Self> {
        self.iter(1.0 / n as f32)
    }
}

pub struct Iter<'a, T, P: ?Sized> {
    pub(crate) param: &'a P,
    pub(crate) t: f32,
    pub(crate) step: f32,
    pub(crate) end: f32,
    pub(crate) _pd: PhantomData<T>,
}

impl<F: Fn(f32) -> T, T> Parametric<T> for F {
    /// Returns `self(t)`.
    fn eval(&self, t: f32) -> T {
        self(t)
    }
}

impl<T: Lerp> Parametric<T> for Range<T> {
    /// Linearly interpolates between `self.start` and `self.end`.
    ///
    /// Equivalent to `<Self as Lerp>::lerp(&self.start, &self.end, t)`.
    ///
    /// See also [`Lerp::lerp`].
    ///
    /// # Examples
    /// ```
    /// use core::ops::Range;
    /// use retrofire_core::math::{Parametric, pt2, Point2};
    ///
    /// let range: Range<Point2> = pt2(-2.0, 1.0)..pt2(3.0, 2.0);
    ///
    /// assert_eq!(range.eval(0.0), range.start);
    /// assert_eq!(range.eval(0.5), pt2(0.5, 1.5));
    /// assert_eq!(range.eval(1.0), range.end);
    /// ```
    fn eval(&self, t: f32) -> T {
        self.start.lerp(&self.end, t)
    }
}

impl<T, P: Parametric<T>> Iterator for Iter<'_, T, P> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        let Self { param, t, step, end, .. } = self;
        (*t <= *end + 0.5 * *step).then(|| {
            let res = param.eval(*t);
            *t += *step;
            res
        })
    }
}
