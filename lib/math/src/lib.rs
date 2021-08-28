use std::cmp::Ordering;
use std::f32::consts::{PI, TAU};
use std::fmt;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use Angle::*;

pub mod mat;
pub mod vec;
pub mod transform;
pub mod rand;
pub mod spline;

pub trait ApproxEq: Sized + Copy {
    type Scalar: PartialOrd + Copy;

    fn approx_eq(self, rhs: Self) -> bool {
        self.approx_eq_eps(rhs, self.epsilon())
    }

    fn approx_eq_eps(self, rhs: Self, epsilon: Self::Scalar) -> bool {
        self.abs_diff(rhs) <= epsilon
    }

    fn epsilon(self) -> Self::Scalar;

    fn abs_diff(self, rhs: Self) -> Self::Scalar;
}

impl ApproxEq for f32 {
    type Scalar = Self;
    fn epsilon(self) -> f32 {
        2.0 * self.abs().max(1.0) * f32::EPSILON
    }
    fn abs_diff(self, rhs: Self) -> Self {
        (self - rhs).abs()
    }
}

impl ApproxEq for f64 {
    type Scalar = Self;
    fn epsilon(self) -> f64 {
        2.0 * self.abs().max(1.0) * f64::EPSILON
    }
    fn abs_diff(self, rhs: Self) -> Self {
        (self - rhs).abs()
    }
}

impl<T> ApproxEq for &[T]
where
    T: ApproxEq,
    <T as ApproxEq>::Scalar: PartialOrd,
{
    type Scalar = <T as ApproxEq>::Scalar;

    fn epsilon(self) -> Self::Scalar {
        // TODO
        self[0].epsilon()
    }

    fn abs_diff(self, rhs: Self) -> Self::Scalar {
        assert!(!self.is_empty());
        assert_eq!(self.len(), rhs.len());

        self.iter().zip(rhs.iter())
            .map(|(a, b)| a.abs_diff(*b))
            .max_by(|c, d| c.partial_cmp(d).unwrap())
            .unwrap()
    }
}


pub trait Linear<Scalar> where Self: Sized {
    fn add(self, other: Self) -> Self;
    fn mul(self, s: Scalar) -> Self;
    fn neg(self) -> Self;
    fn sub(self, other: Self) -> Self { self.add(other.neg()) }

    /// # Linear combination of `self` and `other`.
    fn lincomb(self, s: Scalar, other: Self, r: Scalar) -> Self {
        self.mul(s).add(other.mul(r))
    }
}

impl Linear<f32> for f32 {
    #[inline]
    fn add(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn mul(self, s: f32) -> Self {
        self * s
    }
    #[inline]
    fn neg(self) -> Self {
        -self
    }
}

impl<T> Linear<T> for () {
    fn add(self, _: Self) -> Self { self }
    fn mul(self, _: T) -> Self { self }
    fn neg(self) -> Self { self }
}

impl<S, T, U> Linear<S> for (T, U)
where S: Copy,
      T: Linear<S>,
      U: Linear<S>
{
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        (self.0.add(other.0), self.1.add(other.1))
    }

    #[inline(always)]
    fn mul(self, s: S) -> Self {
        (self.0.mul(s), self.1.mul(s))
    }

    #[inline(always)]
    fn neg(self) -> Self {
        (self.0.neg(), self.1.neg())
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Angle {
    /// Radians
    Rad(f32),
    /// Degrees
    Deg(f32),
    /// Multiples of full angle, ie. 2π radians
    Tau(f32),
}

impl Angle {

    pub const ZERO: Angle = Deg(0.0);
    pub const RIGHT: Angle = Deg(90.0);
    pub const STRAIGHT: Angle = Deg(180.0);
    pub const FULL: Angle = Deg(360.0);

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

    pub fn sin(self) -> f32 { self.as_rad().sin() }
    pub fn cos(self) -> f32 { self.as_rad().cos() }
    pub fn sin_cos(self) -> (f32, f32) { self.as_rad().sin_cos() }
    pub fn tan(self) -> f32 { self.as_rad().tan() }

    pub fn asin(x: f32) -> Angle { Rad(x.asin()) }
    pub fn acos(x: f32) -> Angle { Rad(x.acos()) }
    pub fn atan(x: f32) -> Angle { Rad(x.atan()) }
    pub fn atan2(y: f32, x: f32) -> Angle { Rad(y.atan2(x)) }

    pub fn clamp(self, min: Angle, max: Angle) -> Angle {
        let min = min.as_rad();
        let max = max.as_rad();
        let clamped = self.as_rad().clamp(min, max);

        self.from_rad(clamped)
    }

    pub fn wrap(self, min: Angle, max: Angle) -> Angle {
        assert!(min <= max);

        let min = min.as_rad();
        let max = max.as_rad();
        let wrapped = min + (self.as_rad() - min).rem_euclid(max - min);

        self.from_rad(wrapped)
    }


    fn from_rad(self, rad: f32) -> Angle {
        match self {
            Rad(_) => Rad(rad),
            Deg(_) => Deg(rad * 180.0 / PI),
            Tau(_) => Tau(rad / TAU),
        }
    }
}

impl Default for Angle {
    fn default() -> Self { Self::ZERO }
}

impl PartialEq for Angle {
    fn eq(&self, other: &Self) -> bool {
        self.as_rad().eq(&other.as_rad())
    }
}

impl PartialOrd for Angle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_rad().partial_cmp(&other.as_rad())
    }
}

impl fmt::Display for Angle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let p = f.precision();
        match self {
            Rad(r) => write!(f, "/_ {:.p$} rad", r, p=p.unwrap_or(2)),
            Deg(d) => write!(f, "/_ {:.p$}°", d, p=p.unwrap_or(1)),
            Tau(t) => write!(f, "/_ {:.p$} tau", t, p=p.unwrap_or(2)),
        }
    }
}

impl Mul<Angle> for f32 {
    type Output = Angle;
    fn mul(self, rhs: Angle) -> Angle {
        rhs.from_rad(self * rhs.as_rad())
    }
}

impl Neg for Angle {
    type Output = Angle;
    fn neg(self) -> Angle { -1.0 * self }
}

impl Add for Angle {
    type Output = Angle;

    fn add(self, rhs: Self) -> Angle {
        self.from_rad(self.as_rad() + rhs.as_rad())
    }
}

impl AddAssign for Angle {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Angle {
    type Output = Angle;

    fn sub(self, rhs: Self) -> Angle {
        self + -rhs
    }
}


impl SubAssign for Angle {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

pub fn total_ord_f32(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(b).unwrap_or_else(|| {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => unreachable!()
        }
    })
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
    use crate::Angle::*;
    use crate::tests::util::*;

    pub mod util {
        use core::fmt::Debug;

        use crate::ApproxEq;

        #[derive(Debug, Copy, Clone)]
        pub(crate) struct Approx<T>(pub(crate) T);

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
                "\n(delta={:?},eps={:?})",
                actual.abs_diff(expected),
                actual.epsilon()
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
                actual.epsilon()
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

    #[test]
    fn angle_add_sub() {
        assert!(matches!(Deg(45.0) + Tau(0.125), Deg(a) if a == 90.0));
        assert!(matches!(Tau(0.5) - Deg(90.0), Tau(a) if a == 0.25));
    }


    #[test]
    fn angle_mul() {
        assert!(matches!(3.0 * Deg(60.0), Deg(a) if a == 180.0));
        assert!(matches!(3.0 * Rad(1.0), Rad(a) if a == 3.0));
        assert!(matches!(4.0 * Tau(0.25), Tau(a) if a == 1.0));
    }

    #[test]
    fn angle_wrap() {
        let a = Deg(360.0);
        assert_eq!(0.0, a.wrap(Deg(0.0), Deg(90.0)).as_deg());

        let b = Deg(56.0);
        assert_approx_eq(56.0, b.wrap(Deg(-180.0), Deg(180.0)).as_deg());

        let c = Deg(181.0);
        assert_approx_eq(-179.0, c.wrap(Deg(-180.0), Deg(180.0)).as_deg());

        let d = Deg(-78.0);
        assert_approx_eq(282.0, d.wrap(Deg(0.0), Deg(360.0)).as_deg());

        let e = Deg(-45.0);
        assert_approx_eq(135.0, e.wrap(Tau(0.0), Tau(0.5)).as_deg());
    }

    #[test]
    fn angle_clamp() {
        let a = Deg(123.0);
        assert_approx_eq(90.0, a.clamp(Deg(0.0), Deg(90.0)).as_deg());

        let b = Deg(-42.0);
        assert_approx_eq(-15.0, b.clamp(Deg(-15.0), Deg(15.0)).as_deg());

        let c = Deg(-270.0);
        assert_approx_eq(-0.5, c.clamp(Tau(-0.5), Tau(0.5)).as_tau());
    }
}
