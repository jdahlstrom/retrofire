//! Testing and asserting approximate equality.

use core::iter::zip;

/// Trait for testing approximate equality.
///
/// Floating-point types are only an approximation of real numbers due to their
/// finite precision. The presence of rounding errors means that two floats may
/// not compare equal even if their counterparts in ℝ would. Even such a simple
/// expression as `0.1 + 0.2 == 0.3` will evaluate to false due to precision
/// issues.
///
/// Approximate equality is a more robust way to compare floating-point values
/// than strict equality. Two values are considered approximately equal if their
/// absolute difference is less than some small value, "epsilon". The choice of
/// the epsilon value is not an exact science, and depends on how much error
/// has accrued in the computation of the values.
///
/// Moreover, due to the nature of floating point, a naive comparison against
/// a fixed value does not work well. Rather, the epsilon should be *relative*
/// to the magnitude of the values being compared.
pub trait ApproxEq<Other: ?Sized = Self, Epsilon = Self> {
    /// Returns whether `self` and `other` are approximately equal.
    /// Uses the epsilon returned by [`Self::relative_epsilon`].
    fn approx_eq(&self, other: &Other) -> bool {
        self.approx_eq_eps(other, &Self::relative_epsilon())
    }

    /// Returns whether `self` and `other` are approximately equal,
    /// using the relative epsilon `rel_eps`.
    fn approx_eq_eps(&self, other: &Other, rel_eps: &Epsilon) -> bool;

    /// Returns the default relative epsilon of type `E`.
    fn relative_epsilon() -> Epsilon;
}

impl ApproxEq for f32 {
    fn approx_eq_eps(&self, other: &Self, rel_eps: &Self) -> bool {
        use super::float::f32;
        let diff = f32::abs(self - other);
        diff <= *rel_eps * f32::abs(*self).max(1.0)
    }

    fn relative_epsilon() -> Self {
        if cfg!(any(feature = "std", feature = "libm")) {
            1e-6
        } else {
            5e-3
        }
    }
}

impl<E, T: Sized + ApproxEq<T, E>> ApproxEq<Self, E> for [T] {
    fn approx_eq_eps(&self, other: &Self, rel_eps: &E) -> bool {
        self.len() == other.len()
            && zip(self, other).all(|(s, o)| s.approx_eq_eps(o, rel_eps))
    }
    fn relative_epsilon() -> E {
        T::relative_epsilon()
    }
}

impl<E, T: Sized + ApproxEq<T, E>, const N: usize> ApproxEq<Self, E>
    for [T; N]
{
    fn approx_eq_eps(&self, other: &Self, rel_eps: &E) -> bool {
        self.as_slice().approx_eq_eps(other, rel_eps)
    }
    fn relative_epsilon() -> E {
        T::relative_epsilon()
    }
}

impl<E, T: ApproxEq<T, E>> ApproxEq<Self, E> for Option<T> {
    fn approx_eq_eps(&self, other: &Self, rel_eps: &E) -> bool {
        match (self, other) {
            (Some(s), Some(o)) => s.approx_eq_eps(o, rel_eps),
            (Some(_), None) | (None, Some(_)) => false,
            (None, None) => true,
        }
    }

    fn relative_epsilon() -> E {
        T::relative_epsilon()
    }
}

/// Asserts that two values are approximately equal.
/// Requires that the left operand has an applicable [`ApproxEq`] impl
/// and that both operands impl `Debug` unless a custom message is given.
///
/// # Panics
///
/// If the given values are not approximately equal.
///
/// # Examples
/// `assert_eq` would fail, but `assert_approx_eq` passes:
/// ```
/// # use retrofire_core::assert_approx_eq;
/// assert_ne!(0.1 + 0.2, 0.3);
/// assert_approx_eq!(0.1 + 0.2, 0.3);
/// ```
/// A relative epsilon is used:
/// ```
/// # use retrofire_core::assert_approx_eq;
/// assert_ne!(1e7, 1e7 + 1.0);
/// assert_approx_eq!(1e7, 1e7 + 1.0);
/// ```
/// A custom epsilon can be given:
/// ```
/// # use retrofire_core::assert_approx_eq;
/// assert_approx_eq!(100.0, 101.0, eps = 0.01);
/// ```
/// Like `assert_eq`, this macro supports custom panic messages.
/// The epsilon, if present, must come before the format string.
/// ```should_panic
/// # use std::f32;
/// # use retrofire_core::assert_approx_eq;
/// assert_approx_eq!(f32::sin(3.14), 0.0, eps = 0.001,
///     "3.14 is not a good approximation of {}!", f32::consts::PI);
/// ```
#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
        match (&$a, &$b) {
            (a, b) => $crate::assert_approx_eq!(
                *a, *b,
                "assertion failed: `{a:?} ≅ {b:?}`"
            )
        }
    };
    ($a:expr, $b:expr, eps = $eps:literal) => {
        match (&$a, &$b) {
            (a, b) => $crate::assert_approx_eq!(
                *a, *b, eps = $eps,
                "assertion failed: `{a:?} ≅ {b:?}`"
            )
        }
    };
    ($a:expr, $b:expr, $fmt:literal $(, $args:expr)*) => {{
        use $crate::math::approx::ApproxEq;
        match (&$a, &$b) {
            (a, b) => assert!(ApproxEq::approx_eq(a, b), $fmt $(, $args)*)
        }
    }};
    ($a:expr, $b:expr, eps = $eps:literal, $fmt:literal $(, $args:expr)*) => {{
        use $crate::math::approx::ApproxEq;
        match (&$a, &$b) {
            (a, b) => assert!(
                ApproxEq::approx_eq_eps(a, b, &$eps),
                $fmt $(, $args)*
            )
        }
    }};
}

#[cfg(test)]
mod tests {

    mod f32 {
        #[test]
        fn approx_eq_zero() {
            assert_approx_eq!(0.0, 0.0);
            assert_approx_eq!(-0.0, 0.0);
            assert_approx_eq!(0.0, -0.0);
        }

        #[test]
        fn approx_eq_positive() {
            assert_approx_eq!(0.0, 0.0000001);
            assert_approx_eq!(0.0000001, 0.0);
            assert_approx_eq!(0.9999999, 1.0);
            assert_approx_eq!(1.0, 1.0000001);
            assert_approx_eq!(1.0e10, 1.0000001e10);
        }

        #[test]
        fn approx_eq_negative() {
            assert_approx_eq!(0.0, -0.0000001);
            assert_approx_eq!(-0.0000001, 0.0);
            assert_approx_eq!(-1.0, -1.0000001);
            assert_approx_eq!(-0.9999999, -1.0);
            assert_approx_eq!(-1.0e10, -1.0000001e10);
        }

        #[test]
        fn approx_eq_custom_epsilon() {
            assert_approx_eq!(0.0, 0.001, eps = 0.01);
            assert_approx_eq!(0.0, -0.001, eps = 0.01);
            assert_approx_eq!(1.0, 0.999, eps = 0.01);
            assert_approx_eq!(100.0, 99.9, eps = 0.01);
        }

        #[test]
        #[should_panic]
        fn zero_not_approx_eq_to_one() {
            assert_approx_eq!(0.0, 1.0);
        }
        #[test]
        #[should_panic]
        fn one_not_approx_eq_to_1_00001() {
            assert_approx_eq!(1.0, 1.00001);
        }
        #[test]
        #[should_panic]
        fn inf_not_approx_eq_to_inf() {
            assert_approx_eq!(f32::INFINITY, f32::INFINITY);
        }
        #[test]
        #[should_panic]
        fn nan_not_approx_eq_to_nan() {
            assert_approx_eq!(f32::NAN, f32::NAN);
        }
    }
}
