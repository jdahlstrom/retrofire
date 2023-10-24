//! Angular quantities.

use core::f32::consts::{PI, TAU};
use core::fmt::{self, Debug, Display};
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::math::approx::ApproxEq;
use crate::math::vec::{Affine, Linear};

//
// Types
//

/// A scalar angular quantity.
///
/// Prevents confusion between degrees and radians by requiring the use of
/// one of the named constructors to create an `Angle`, as well as one of
/// the named getter methods to obtain the angle as a raw `f32` value.
#[derive(Copy, Clone, Default, PartialEq)]
#[repr(transparent)]
pub struct Angle(f32);

const RADS_PER_DEG: f32 = PI / 180.0;
const RADS_PER_TURN: f32 = TAU;

//
// Free fns and consts
//

/// Returns an angle of `a` radians.
pub fn rads(a: f32) -> Angle {
    Angle(a)
}

/// Returns an angle of `a` degrees.
pub fn degs(a: f32) -> Angle {
    Angle(a * RADS_PER_DEG)
}

/// Returns an angle of `a` turns.
pub fn turns(a: f32) -> Angle {
    Angle(a * RADS_PER_TURN)
}

/// Returns the arcsine of `x` as an `Angle`.
///
/// The return value is in the range [-90°, 90°].
///
/// # Examples
/// ```
/// # use retrofire_core::assert_approx_eq;
/// # use retrofire_core::math::angle::*;
/// assert_approx_eq!(asin(1.0), degs(90.0));
/// assert_approx_eq!(asin(-1.0), degs(-90.0));
/// ```
/// # Panics
/// If `x` is outside the range [-1.0, 1.0].
#[cfg(feature = "std")]
pub fn asin(x: f32) -> Angle {
    assert!(-1.0 <= x && x <= 1.0);
    Angle(x.asin())
}

/// Returns the arccosine of `x` as an `Angle`.
///
/// The return value is in the range [-90°, 90°].
///
/// # Examples
/// ```
/// # use retrofire_core::math::angle::*;
/// assert_eq!(acos(1.0), degs(0.0));
/// ```
/// # Panics
/// If `x` is outside the range [-1.0, 1.0].
#[cfg(feature = "std")]
pub fn acos(x: f32) -> Angle {
    Angle(x.acos())
}

/// Returns the four-quadrant arctangent of `y` and `x` as an `Angle`.
///
/// The returned angle is equal to [`y.atan2(x)`][f32::atan2].
///
/// # Examples
/// ```
/// # use retrofire_core::math::angle::*;
/// assert_eq!(atan2(0.0, 1.0), degs(0.0));
/// assert_eq!(atan2(2.0, 2.0), degs(45.0));
/// assert_eq!(atan2(3.0, 0.0), degs(90.0));
/// ```
#[cfg(feature = "std")]
pub fn atan2(y: f32, x: f32) -> Angle {
    Angle(y.atan2(x))
}

//
// Inherent impls
//

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
    /// # Examples
    /// ```
    /// # use std::f32;
    /// # use retrofire_core::math::degs;
    /// assert_eq!(degs(90.0).to_rads(), f32::consts::FRAC_PI_2);
    /// ```
    pub const fn to_rads(self) -> f32 {
        self.0
    }
    /// Returns the value of `self` in degrees.
    /// # Examples
    /// ```
    /// # use retrofire_core::math::turns;
    /// assert_eq!(turns(2.0).to_degs(), 720.0);
    pub fn to_degs(self) -> f32 {
        self.0 / RADS_PER_DEG
    }
    /// Returns the value of `self` in turns.
    /// # Examples
    /// ```
    /// # use retrofire_core::math::degs;
    /// assert_eq!(degs(180.0).to_turns(), 0.5);
    /// ```
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
    /// let (min, max) = (degs(0.0), degs(45.0));
    ///
    /// assert_eq!(degs(100.0).clamp(min, max), max);
    ///
    /// assert_eq!(degs(30.0).clamp(min, max), degs(30.0));
    ///
    /// assert_eq!(degs(-10.0).clamp(min, max), min);
    /// ```
    #[must_use]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }
}

#[cfg(feature = "std")]
impl Angle {
    /// Returns the sine of `self`.
    /// # Examples
    /// ```
    /// # use retrofire_core::math::angle::*;
    /// assert_eq!(degs(30.0).sin(), 0.5)
    /// ```
    pub fn sin(self) -> f32 {
        self.0.sin()
    }
    /// Returns the cosine of `self`.
    /// # Examples
    /// ```
    /// # use retrofire_core::assert_approx_eq;
    /// # use retrofire_core::math::angle::*;
    /// assert_approx_eq!(degs(60.0).cos(), 0.5)
    /// ```
    pub fn cos(self) -> f32 {
        self.0.cos()
    }
    /// Simultaneously computes the sine and cosine of `self`.
    /// # Examples
    /// ```
    /// # use retrofire_core::assert_approx_eq;
    /// # use retrofire_core::math::angle::*;
    /// let (sin, cos) = degs(90.0).sin_cos();
    /// assert_approx_eq!(sin, 1.0);
    /// assert_approx_eq!(cos, 0.0);
    /// ```
    pub fn sin_cos(self) -> (f32, f32) {
        self.0.sin_cos()
    }
    /// Returns the tangent of `self`.
    /// # Examples
    /// ```
    /// # use retrofire_core::math::angle::*;
    /// assert_eq!(degs(45.0).tan(), 1.0)
    /// ```
    pub fn tan(self) -> f32 {
        self.0.tan()
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
    pub fn wrap(self, min: Self, max: Self) -> Self {
        Self(min.0 + (self.0 - min.0).rem_euclid(max.0 - min.0))
    }
}

//
// Local trait impls
//

impl ApproxEq for Angle {
    // TODO Should this account for wraparound?
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

//
// Foreign trait impls
//

impl Display for Angle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (val, unit) = if f.alternate() {
            (self.to_rads() / PI, "𝜋 rad")
        } else {
            (self.to_degs(), "°")
        };
        Display::fmt(&val, f)?;
        f.write_str(unit)
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

    #[cfg(feature = "std")]
    #[test]
    fn trig_functions() {
        assert_eq!(degs(0.0).sin(), 0.0);
        assert_eq!(degs(0.0).cos(), 1.0);

        assert_approx_eq!(degs(30.0).sin(), 0.5);
        assert_approx_eq!(degs(60.0).cos(), 0.5);

        let (sin, cos) = degs(90.0).sin_cos();
        assert_approx_eq!(sin, 1.0);
        assert_approx_eq!(cos, 0.0);

        assert_approx_eq!(degs(-45.0).tan(), -1.0);
        assert_approx_eq!(degs(0.0).tan(), 0.0);
        assert_approx_eq!(degs(45.0).tan(), 1.0);
        assert_approx_eq!(degs(135.0).tan(), -1.0);
        assert_approx_eq!(degs(225.0).tan(), 1.0);
        assert_approx_eq!(degs(315.0).tan(), -1.0, eps = 1e-6);
    }

    #[cfg(feature = "std")]
    #[test]
    fn inverse_trig_functions() {
        assert_approx_eq!(asin(-1.0), degs(-90.0));
        assert_approx_eq!(asin(0.0), degs(0.0));
        assert_approx_eq!(asin(0.5), degs(30.0));
        assert_approx_eq!(asin(1.0), degs(90.0));

        assert_approx_eq!(acos(-1.0), degs(180.0));
        assert_approx_eq!(acos(0.0), degs(90.0));
        assert_approx_eq!(acos(0.5), degs(60.0));
        assert_approx_eq!(acos(1.0), degs(0.0));

        assert_approx_eq!(atan2(0.0, 1.0), degs(0.0));
        assert_approx_eq!(atan2(1.0, 1.0), degs(45.0));
        assert_approx_eq!(atan2(1.0, -1.0), degs(135.0));
        assert_approx_eq!(atan2(-1.0, -1.0), degs(-135.0));
        assert_approx_eq!(atan2(-1.0, 1.0), degs(-45.0));
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
