//! Angular quantities.

use core::f32::consts::{PI, TAU};
use core::fmt::{self, Debug, Display};
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::math::approx::ApproxEq;
use crate::math::vec::{Affine, Linear};

const RADS_PER_DEG: f32 = PI / 180.0;
const RADS_PER_TURN: f32 = TAU;

/// Represents an angular quantity.
///
/// Prevents the confusion between degrees and radians by requiring the use of
/// one of the named constructors to create an `Angle`, as well as one of the
/// named getter methods to obtain the angle as a raw `f32` value.
#[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Angle(f32);

impl Angle {
    /// A zero degree angle.
    pub const ZERO: Self = Self(0.0);
    /// A 90 degree angle.
    pub const RIGHT: Self = Self(RADS_PER_TURN / 4.0);
    /// A 180 degree angle.
    pub const STRAIGHT: Self = Self(RADS_PER_TURN / 2.0);
    /// A 360 degree angle.
    pub const FULL: Self = Self(RADS_PER_TURN);

    /// Returns the value of `self` in radians.
    pub const fn to_rads(self) -> f32 {
        self.0
    }
    /// Returns the value of `self` in degrees.
    pub fn to_degs(self) -> f32 {
        self.0 / RADS_PER_DEG
    }
    /// Returns the value of `self` in turns.
    pub fn to_turns(self) -> f32 {
        self.0 / RADS_PER_TURN
    }

    /// Returns the minimum of `self` and `other`.
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }
    /// Returns the maximum of `self` and `other`.
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }
    /// Returns `self` clamped to the range `min..=max`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use retrofire_core::math::angle::degs;
    /// assert_eq!(degs(100.0).clamp(degs(0.0), degs(90.0)), degs(90.0));
    ///
    /// assert_eq!(degs(45.0).clamp(degs(0.0), degs(90.0)), degs(45.0));
    ///
    /// assert_eq!(degs(-10.0).clamp(degs(0.0), degs(90.0)), degs(0.0));
    /// ```
    #[must_use]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }
}

#[cfg(feature = "std")]
impl Angle {
    /// Returns the sine of `self`.
    pub fn sin(self) -> f32 {
        self.0.sin()
    }
    /// Returns the cosine of `self`.
    pub fn cos(self) -> f32 {
        self.0.sin()
    }
    /// Simultaneously computes the sine and cosine of `self`.
    pub fn sin_cos(self) -> (f32, f32) {
        self.0.sin_cos()
    }
    /// Returns the tangent of `self`.
    pub fn tan(self) -> f32 {
        self.0.sin()
    }
    /// Returns `self` "wrapped around" to the range `min..max`.
    ///
    /// # Examples
    /// ```
    /// # use retrofire_core::assert_approx_eq;
    /// # use retrofire_core::math::angle::*;
    /// assert_approx_eq!(degs(400.0).wrap(Angle::ZERO, Angle::FULL), degs(40.0))
    /// ```
    #[must_use]
    pub fn wrap(self, Self(min): Self, Self(max): Self) -> Self {
        Self(min + (self.0 - min).rem_euclid(max - min))
    }
}

/// Returns an `Angle` of `a` radians.
pub const fn rads(a: f32) -> Angle {
    Angle(a)
}

/// Returns an `Angle` of `a` degrees.
pub fn degs(a: f32) -> Angle {
    Angle(a * RADS_PER_DEG)
}

/// Returns an `Angle` of `a` turns.
pub fn turns(a: f32) -> Angle {
    Angle(a * RADS_PER_TURN)
}

/// Returns the arcsine of `x` as an `Angle`.
#[cfg(feature = "std")]
pub fn asin(x: f32) -> Angle {
    Angle(x.asin())
}

/// Returns the arccosine of `x` as an `Angle`.
#[cfg(feature = "std")]
pub fn acos(x: f32) -> Angle {
    Angle(x.acos())
}

/// Returns the arctangent of `y` and `x` as an `Angle`.
#[cfg(feature = "std")]
pub fn atan2(y: f32, x: f32) -> Angle {
    Angle(y.atan2(x))
}

impl ApproxEq for Angle {
    fn approx_eq_eps(&self, other: &Self, eps: &Self) -> bool {
        self.0.approx_eq_eps(&other.0, &eps.0)
    }

    fn relative_epsilon() -> Self {
        Self(f32::relative_epsilon())
    }
}

impl Affine for Angle {
    type Space = ();
    type Diff = Self;
    const DIM: usize = 1;

    #[inline]
    fn add(&self, other: &Self) -> Self {
        *self + *other
    }
    #[inline]
    fn sub(&self, other: &Self) -> Self {
        *self - *other
    }
}

impl Linear for Angle {
    type Scalar = f32;

    #[inline]
    fn zero() -> Self {
        Angle::ZERO
    }
    #[inline]
    fn neg(&self) -> Self {
        -*self
    }
    #[inline]
    fn mul(&self, scalar: f32) -> Self {
        *self * scalar
    }
}

impl Display for Angle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            Display::fmt(&self.to_rads(), f)?;
            f.write_str(" rad")
        } else {
            Display::fmt(&self.to_degs(), f)?;
            f.write_str("°")
        }
    }
}

impl Debug for Angle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Angle(")?;
        Display::fmt(self, f)?;
        f.write_str(")")
    }
}

impl Add for Angle {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}
impl Sub for Angle {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}
impl Neg for Angle {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Mul<f32> for Angle {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self(self.0 * rhs)
    }
}
impl Div<f32> for Angle {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Self(self.0 / rhs)
    }
}
impl Rem for Angle {
    type Output = Self;
    fn rem(self, rhs: Angle) -> Self {
        Self(self.0 % rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::assert_approx_eq;
    use crate::math::vary::Vary;

    use super::*;

    #[test]
    fn rads_to_degs() {
        assert_eq!(rads(PI).to_degs(), 180.0);
    }

    #[test]
    fn rads_to_turns() {
        assert_eq!(rads(PI).to_turns(), 0.5);
    }

    #[test]
    fn degs_to_rads() {
        assert_eq!(degs(180.0).to_rads(), PI);
    }

    #[test]
    fn degs_to_turns() {
        assert_eq!(degs(360.0).to_turns(), 1.0);
    }

    #[test]
    fn turns_to_rads() {
        assert_eq!(turns(1.0).to_rads(), TAU);
    }

    #[test]
    fn turns_to_degs() {
        assert_eq!(turns(1.0).to_degs(), 360.0);
    }

    #[test]
    fn clamping() {
        let min = degs(-45.0);
        let max = degs(45.0);
        assert_eq!(degs(60.0).clamp(min, max), max);
        assert_eq!(degs(10.0).clamp(min, max), degs(10.0));
        assert_eq!(degs(-50.0).clamp(min, max), min);
    }

    #[test]
    #[cfg(feature = "std")]
    fn wrapping() {
        use crate::assert_approx_eq;

        let a = degs(540.0).wrap(Angle::ZERO, Angle::FULL);
        assert_approx_eq!(a, degs(180.0));

        let a = degs(225.0).wrap(-Angle::STRAIGHT, Angle::STRAIGHT);
        assert_approx_eq!(a, degs(-135.0));
    }

    #[test]
    fn lerping() {
        let a = degs(30.0).lerp(&degs(60.0), 0.2);
        assert_eq!(a, degs(36.0));
    }

    #[test]
    fn varying() {
        let mut i = degs(45.0).vary(degs(15.0), Some(4));

        assert_approx_eq!(i.next(), Some(degs(45.0)));
        assert_approx_eq!(i.next(), Some(degs(60.0)));
        assert_approx_eq!(i.next(), Some(degs(75.0)));
        assert_approx_eq!(i.next(), Some(degs(90.0)));
        assert_approx_eq!(i.next(), None);
    }
}
