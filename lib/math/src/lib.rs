use std::f32::consts::{PI, TAU};
use std::fmt;
use std::ops::{Add, Mul, Neg};

use Angle::*;

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

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Angle {
    /// Radians
    Rad(f32),
    /// Degrees
    Deg(f32),
    /// Multiples of full angle, ie. 2π radians
    Tau(f32),
}

impl Angle {
    pub fn as_deg(self) -> f32 {
        match self {
            Rad(r) => r * 180.0 / PI,
            Deg(d) => d,
            Tau(t) => t * 360.0,
        }
    }
    pub fn as_rad(self) -> f32 {
        match self {
            Rad(r) => r,
            Deg(d) => d * PI / 180.0,
            Tau(t) => t * TAU,
        }
    }
    pub fn as_tau(self) -> f32 {
        match self {
            Rad(r) => r / TAU,
            Deg(d) => d / 360.0,
            Tau(t) => t,
        }
    }

    fn map(self, mut f: impl FnMut(f32) -> f32) -> Angle {
        match self {
            Rad(r) => Rad(f(r)),
            Deg(d) => Deg(f(d)),
            Tau(t) => Tau(f(t)),
        }
    }

    pub fn sin(self) -> f32 { self.as_rad().sin() }
    pub fn cos(self) -> f32 { self.as_rad().cos() }
    pub fn sin_cos(self) -> (f32, f32) { self.as_rad().sin_cos() }
    pub fn tan(self) -> f32 { self.as_rad().tan() }

    pub fn asin(x: f32) -> Angle { Rad(x.asin()) }
    pub fn acos(x: f32) -> Angle { Rad(x.acos()) }
    pub fn atan(x: f32) -> Angle { Rad(x.atan()) }
    pub fn atan2(y: f32, x: f32) -> Angle { Rad(y.atan2(x)) }
}

impl fmt::Display for Angle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision();
        match self {
            Rad(r) => write!(f, "∡ {:.p$} rad", r, p=p.unwrap_or(2)),
            Deg(d) => write!(f, "∡ {:.p$}°", d, p=p.unwrap_or(0)),
            Tau(t) => write!(f, "∡ {:.p$}𝜏", t, p=p.unwrap_or(1)),
        }
    }
}

impl Mul<Angle> for f32 {
    type Output = Angle;
    fn mul(self, rhs: Angle) -> Angle { rhs.map(|a| self * a) }
}

impl Neg for Angle {
    type Output = Angle;
    fn neg(self) -> Self::Output { self.map(Neg::neg) }
}

impl Add for Angle {
    type Output = Angle;

    fn add(self, rhs: Self) -> Self::Output {
        match self {
            Rad(r) => Rad(r + rhs.as_rad()),
            Deg(d) => Deg(d + rhs.as_deg()),
            Tau(t) => Tau(t + rhs.as_tau()),
        }
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
