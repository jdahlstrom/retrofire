pub mod mat;
pub mod vec;
pub mod transform;
pub mod rand;

pub trait ApproxEq: Sized + Copy {
    type Scalar: PartialOrd + Copy;
    const EPSILON: Self::Scalar;

    fn approx_eq(self, rhs: Self) -> bool {
        self.abs_diff(rhs) < Self::EPSILON
    }

    fn abs_diff(self, rhs: Self) -> Self::Scalar;
}

impl ApproxEq for f32 {
    type Scalar = Self;
    const EPSILON: Self = 1e-6;

    fn abs_diff(self, rhs: Self) -> Self {
        (self - rhs).abs()
    }
}

impl ApproxEq for f64 {
    type Scalar = Self;
    const EPSILON: Self = 1e-13;

    fn abs_diff(self, rhs: Self) -> Self {
        (self - rhs).abs()
    }
}

pub trait Linear<Scalar>
where
    Self: Sized,
    Scalar: Sized,
{
    fn add(self, other: Self) -> Self;
    fn mul(self, s: Scalar) -> Self;

    /// # Linear combination of `self` and `other`.
    fn lincomb(self, s: Scalar, other: Self, r: Scalar) -> Self {
        self.mul(s).add(other.mul(r))
    }
}

impl Linear<f32> for f32 {
    fn add(self, other: Self) -> Self {
        self + other
    }

    fn mul(self, s: f32) -> Self {
        self * s
    }
}

impl<T> Linear<T> for () {
    fn add(self, _: Self) -> Self {
        self
    }
    fn mul(self, _: T) -> Self {
        self
    }
}

impl<S, T, U> Linear<S> for (T, U)
where S: Copy,
      T: Linear<S>,
      U: Linear<S> {

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        (self.0.add(other.0), self.1.add(other.1))
    }

    #[inline(always)]
    fn mul(self, s: S) -> Self {
        (self.0.mul(s), self.1.mul(s))
    }
}

/// Linear interpolation between `a` and `b`.
///
/// Example:
/// ```
/// # use math::lerp;
/// assert_eq!(1.6, lerp(0.7, 3.0, 1.0))
/// ```
pub fn lerp<T: Linear<f32>>(t: f32, a: T, b: T) -> T {
    a.lincomb(1.0 - t, b, t)
}

#[cfg(test)]
mod tests {
    pub mod util {
        use core::fmt::Debug;

        use crate::ApproxEq;

        #[derive(Debug, Copy, Clone)]
        struct Approx<T>(T);

        impl<T> PartialEq<T> for Approx<T>
        where
            T: ApproxEq + Copy,
        {
            fn eq(&self, other: &T) -> bool {
                self.0.approx_eq(*other)
            }
        }

        pub fn assert_approx_eq<T>(actual: T, expected: T)
        where
            T: ApproxEq + Debug + Copy,
            T::Scalar: Debug,
        {
            assert_eq!(
                Approx(actual),
                expected,
                "\n(difference={:?},epsilon={:?})",
                actual.abs_diff(expected),
                T::EPSILON
            );
        }

        pub fn assert_approx_ne<T>(actual: T, expected: T)
        where
            T: ApproxEq + Debug + Copy,
            T::Scalar: Debug,
        {
            assert_ne!(
                Approx(actual),
                expected,
                "\n(difference={:?},epsilon={:?})",
                actual.abs_diff(expected),
                T::EPSILON
            );
        }
    }

    #[test]
    fn lerp_float() {
        use crate::lerp;
        assert_eq!(0.0, lerp(0.0, 0.0, 1.0));
        assert_eq!(1.0, lerp(1.0, 0.0, 1.0));

        assert_eq!(1.0, lerp(0.0, 1.0, -1.0));
        assert_eq!(-1.0, lerp(1.0, 1.0, -1.0));

        assert_eq!(1.0, lerp(0.5, -1.0, 3.0));
        assert_eq!(-0.5, lerp(0.5, 2.0, -3.0));
    }
}
