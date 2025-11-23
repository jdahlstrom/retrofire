//! Angular quantities, including scalar angles and angular vectors.

use core::{
    f32::consts::{PI, TAU},
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use crate::math::{Affine, ApproxEq, Linear, Vector, vary::ZDiv};
#[cfg(feature = "fp")]
use crate::math::{Vec2, Vec3, float::f32, vec2, vec3};

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

/// Tag type for a polar coordinate space
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Polar<B>(PhantomData<B>);

/// Tag type for a spherical coordinate space.
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Spherical<B>(PhantomData<B>);

/// A polar coordinate vector, with radius and azimuth components.
pub type PolarVec<B = ()> = Vector<[f32; 2], Polar<B>>;

/// A spherical coordinate vector, with radius, azimuth, and altitude
/// (elevation) components.
pub type SphericalVec<B = ()> = Vector<[f32; 3], Spherical<B>>;

//
// Free fns and consts
//

/// Returns an angle of `a` radians.
pub const fn rads(a: f32) -> Angle {
    Angle(a)
}

/// Returns an angle of `a` degrees.
pub const fn degs(a: f32) -> Angle {
    Angle(a * RADS_PER_DEG)
}

/// Returns an angle of `a` turns.
pub const fn turns(a: f32) -> Angle {
    Angle(a * RADS_PER_TURN)
}

/// Returns the arcsine of `x` as an `Angle`.
///
/// The return value is in the range [-90°, 90°].
///
/// # Examples
/// ```
/// use retrofire_core::assert_approx_eq;
/// use retrofire_core::math::{degs, asin};
///
/// assert_approx_eq!(asin(1.0), degs(90.0));
/// assert_approx_eq!(asin(-1.0), degs(-90.0));
/// ```
/// # Panics
/// If `x` is outside the range [-1.0, 1.0].
#[cfg(feature = "fp")]
pub fn asin(x: f32) -> Angle {
    assert!(-1.0 <= x && x <= 1.0);
    Angle(f32::asin(x))
}

/// Returns the arccosine of `x` as an `Angle`.
///
/// The return value is in the range [-90°, 90°].
///
/// # Examples
/// ```
/// use retrofire_core::assert_approx_eq;
/// use retrofire_core::math::{acos, degs};
///
/// assert_approx_eq!(acos(1.0), degs(0.0));
/// ```
/// # Panics
/// If `x` is outside the range [-1.0, 1.0].
#[cfg(feature = "fp")]
pub fn acos(x: f32) -> Angle {
    Angle(f32::acos(x))
}

/// Returns the four-quadrant arctangent of `y` and `x` as an `Angle`.
///
/// The returned angle is equal to [`y.atan2(x)`][f32::atan2].
///
/// # Examples
/// ```
/// use retrofire_core::math::{atan2, degs};
///
/// assert_eq!(atan2(0.0, 1.0), degs(0.0));
/// assert_eq!(atan2(2.0, 2.0), degs(45.0));
/// assert_eq!(atan2(-3.0, 0.0), degs(-90.0));
/// ```
#[cfg(feature = "fp")]
pub fn atan2(y: f32, x: f32) -> Angle {
    Angle(f32::atan2(y, x))
}

/// Returns a polar coordinate vector with azimuth `az` and radius `r`.
pub const fn polar<B>(r: f32, az: Angle) -> PolarVec<B> {
    Vector::new([r, az.to_rads()])
}

/// Returns a spherical coordinate vector with azimuth `az`,
/// altitude `alt`, and radius `r`.
///
/// An altitude of +90° corresponds to straight up and -90° to straight down.
pub const fn spherical<B>(r: f32, az: Angle, alt: Angle) -> SphericalVec<B> {
    Vector::new([r, az.to_rads(), alt.to_rads()])
}

const RADS_PER_DEG: f32 = PI / 180.0;
const RADS_PER_TURN: f32 = TAU;

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
    /// use std::f32;
    /// use retrofire_core::math::degs;
    ///
    /// assert_eq!(degs(90.0).to_rads(), f32::consts::FRAC_PI_2);
    /// ```
    pub const fn to_rads(self) -> f32 {
        self.0
    }
    /// Returns the value of `self` in degrees.
    /// # Examples
    /// ```
    /// use retrofire_core::math::turns;
    ///
    /// assert_eq!(turns(2.0).to_degs(), 720.0);
    pub fn to_degs(self) -> f32 {
        self.0 / RADS_PER_DEG
    }
    /// Returns the value of `self` in turns.
    /// # Examples
    /// ```
    /// use retrofire_core::math::degs;
    ///
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
    /// ```
    /// use retrofire_core::math::degs;
    ///
    /// let (min, max) = (degs(0.0), degs(45.0));
    ///
    /// assert_eq!(degs(100.0).clamp(min, max), max);
    /// assert_eq!(degs(30.0).clamp(min, max), degs(30.0));
    /// assert_eq!(degs(-10.0).clamp(min, max), min);
    /// ```
    #[must_use]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }

    /// Returns `self` "wrapped around" to the range `min..max`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::math::{degs, turns};
    ///
    /// // 400 (mod 360) = 40
    /// assert_approx_eq!(degs(400.0).wrap(turns(0.0), turns(1.0)), degs(40.0))
    /// ```
    #[must_use]
    pub fn wrap(self, min: Self, max: Self) -> Self {
        use super::float::f32;
        Self(min.0 + f32::rem_euclid(self.0 - min.0, max.0 - min.0))
    }
}

#[cfg(feature = "fp")]
impl Angle {
    /// Returns the sine of `self`.
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::math::degs;
    ///
    /// assert_approx_eq!(degs(30.0).sin(), 0.5)
    /// ```
    pub fn sin(self) -> f32 {
        f32::sin(self.0)
    }
    /// Returns the cosine of `self`.
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::math::degs;
    ///
    /// assert_approx_eq!(degs(60.0).cos(), 0.5)
    /// ```
    pub fn cos(self) -> f32 {
        f32::cos(self.0)
    }
    /// Simultaneously computes the sine and cosine of `self`.
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::math::degs;
    ///
    /// let (sin, cos) = degs(90.0).sin_cos();
    /// assert_approx_eq!(sin, 1.0);
    /// assert_approx_eq!(cos, 0.0);
    /// ```
    pub fn sin_cos(self) -> (f32, f32) {
        (self.sin(), self.cos())
    }
    /// Returns the tangent of `self`.
    /// # Examples
    /// ```
    /// use retrofire_core::math::degs;
    /// assert_eq!(degs(45.0).tan(), 1.0)
    /// ```
    pub fn tan(self) -> f32 {
        f32::tan(self.0)
    }
}

impl<B> PolarVec<B> {
    /// Returns the radial component of `self`.
    #[inline]
    pub fn r(&self) -> f32 {
        self.0[0]
    }
    /// Returns the azimuthal component of `self`.
    #[inline]
    pub fn az(&self) -> Angle {
        rads(self.0[1])
    }

    /// Returns `self` converted to the equivalent Cartesian 2-vector.
    ///
    /// Let the components of `self` be `(r, az)`. Then the result `(x, y)`
    /// equals `(r * cos(az), r * sin(az))`.
    ///
    /// ```text
    /// +y
    /// ^     ^
    /// | +r /
    /// |   /
    /// |  /_ +az
    /// | /  \
    /// +----------> +x
    /// ```
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::math::{vec2, polar, degs};
    ///
    /// let vec2 = vec2::<f32, ()>;
    ///
    /// assert_approx_eq!(polar(2.0, degs(0.0)).to_cart(), vec2(2.0, 0.0));
    /// assert_approx_eq!(polar(3.0, degs(90.0)).to_cart(), vec2(0.0, 3.0));
    /// assert_approx_eq!(polar(4.0, degs(-180.0)).to_cart(), vec2(-4.0, 0.0));
    ///
    /// ```
    #[cfg(feature = "fp")]
    pub fn to_cart(&self) -> Vec2<B> {
        let (y, x) = self.az().sin_cos();
        vec2(x, y) * self.r()
    }
}

impl<B> SphericalVec<B> {
    /// Returns the radial component of `self`.
    #[inline]
    pub fn r(&self) -> f32 {
        self.0[0]
    }
    /// Returns the azimuthal component of `self`.
    #[inline]
    pub fn az(&self) -> Angle {
        rads(self.0[1])
    }
    /// Returns the altitude (elevation) component of `self`.
    #[inline]
    pub fn alt(&self) -> Angle {
        rads(self.0[2])
    }

    /// Returns `self` converted to the equivalent Cartesian 3-vector.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::math::{degs, spherical, vec3, SphericalVec};
    ///
    /// let mut v = spherical::<()>(1.0, degs(0.0), degs(0.0));
    /// assert_approx_eq!(v.to_cart(), vec3(1.0, 0.0, 0.0));
    ///
    /// v = spherical(2.0, degs(90.0), degs(0.0));
    /// assert_approx_eq!(v.to_cart(), vec3(0.0, 0.0, -2.0));
    ///
    /// v = spherical(3.0, degs(0.0), degs(90.0));
    /// assert_approx_eq!(v.to_cart(), vec3(0.0, 3.0, 0.0));
    /// ```
    #[cfg(feature = "fp")]
    pub fn to_cart(&self) -> Vec3<B> {
        // First about z by alt, then about y by az:
        //
        // ( caz  0 saz )   ( calt -salt  0 )   ( r )
        // (   0  1   0 ) · ( salt  calt  0 ) · ( 0 )
        // (-saz  0 caz )   (   0      0  1 )   ( 0 )
        //
        //   ( caz  0 saz )   ( calt·r )   ( caz·calt·r )
        // = (   0  1   0 ) · ( salt·r ) = (     salt·r )
        //   (-saz  0 caz )   (      0 )   (-saz·calt·r )

        let (sin_alt, cos_alt) = self.alt().sin_cos();
        let (sin_az, cos_az) = self.az().sin_cos();
        self.r() * vec3(cos_az * cos_alt, sin_alt, -sin_az * cos_alt)
    }
}

#[cfg(feature = "fp")]
impl<B> Vec2<B> {
    /// Returns `self` converted into the equivalent polar coordinate vector.
    ///
    /// The `r` component of the result equals `self.len()`.
    ///
    /// The `az` component equals the angle between the vector and the x-axis
    /// in the range (-180°, 180°] such that positive `y` maps to positive `az`.
    /// ```text
    /// +y
    /// ^    ^
    /// |   /
    /// |  /_ +az
    /// | /  \
    /// +----------> +x
    /// ```
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::assert_approx_eq;
    /// use retrofire_core::math::{vec2, degs};
    ///
    /// let vec2 = vec2::<f32, ()>;
    ///
    /// // A non-negative x and zero y maps to zero azimuth
    /// assert_eq!(vec2(0.0, 0.0).to_polar().az(), degs(0.0));
    /// assert_eq!(vec2(1.0, 0.0).to_polar().az(), degs(0.0));
    ///
    /// // A zero x and positive y maps to right angle azimuth
    /// assert_eq!(vec2(0.0, 1.0).to_polar().az(), degs(90.0));
    ///
    /// // A zero x and negative y maps to negative right angle azimuth
    /// assert_eq!(vec2(0.0, -1.0).to_polar().az(), degs(-90.0));
    ///
    /// // A negative x and zero y maps to straight angle azimuth
    /// assert_approx_eq!(vec2(-1.0, 0.0).to_polar().az(), degs(180.0));
    /// ```
    pub fn to_polar(&self) -> PolarVec<B> {
        let r = self.len();
        let az = atan2(self.y(), self.x());
        polar(r, az)
    }
}

#[cfg(feature = "fp")]
impl<B> Vec3<B> {
    /// Converts `self` into the equivalent spherical coordinate vector.
    ///
    /// Returns a vector (r, az, alt) such that:
    /// * `r` equals `self.len()`
    /// * `az`is the angle between `self` and the xy-plane in the range
    ///   (-180°, 180°] such that positive `z` maps to *negative* `az`, and
    /// * `alt` is the angle between `self` and the xz-plane in the
    /// range [-90°, 90°] such that positive `y` maps to positive `alt`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{vec3, spherical, degs};
    ///
    /// // The positive x-axis lies at zero azimuth and altitude
    /// assert_eq!(
    ///     vec3(1.0, 0.0, 0.0).to_spherical(),
    ///     spherical::<()>(1.0, degs(0.0), degs(0.0))
    /// );
    /// // The positive y-axis lies at 90° altitude
    /// assert_eq!(
    ///     vec3(0.0, 2.0, 0.0).to_spherical(),
    ///     spherical::<()>(2.0, degs(0.0), degs(90.0))
    /// );
    /// // The positive z-axis lies at *-90°* azimuth
    /// assert_eq!(
    ///     vec3(0.0, 0.0, 3.0).to_spherical(),
    ///     spherical::<()>(3.0, degs(-90.0), degs(0.0))
    /// );
    /// ```
    pub fn to_spherical(&self) -> SphericalVec<B> {
        let [x, y, z] = self.0;
        let az = atan2(-z, x);
        let alt = atan2(y, f32::sqrt(x * x + z * z));
        let r = self.len();
        spherical(r, az, alt)
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
        Self::ZERO
    }
    #[inline]
    fn mul(&self, scalar: f32) -> Self {
        *self * scalar
    }
}

impl ZDiv for Angle {}

//
// Foreign trait impls
//

impl<B> Debug for Polar<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("Pol")
    }
}
impl<B> Debug for Spherical<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("Sph")
    }
}

impl<B> Default for SphericalVec<B> {
    fn default() -> Self {
        Self::new([1.0, 0.0, 0.0])
    }
}
impl<B> Default for PolarVec<B> {
    fn default() -> Self {
        Self::new([1.0, 0.0])
    }
}

impl Display for Angle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (val, unit) = if f.alternate() {
            (self.to_rads() / PI, "·𝜋 rad")
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
        Debug::fmt(&(self.to_rads() / PI), f)?;
        f.write_str("·𝜋 rad)")
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
    fn rem(self, rhs: Self) -> Self {
        Self(self.0 % rhs.0)
    }
}

impl Mul<Angle> for f32 {
    type Output = Angle;

    fn mul(self, rhs: Angle) -> Self::Output {
        rhs * self
    }
}

impl AddAssign for Angle {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl SubAssign for Angle {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
impl MulAssign<f32> for Angle {
    fn mul_assign(&mut self, rhs: f32) {
        self.0 *= rhs;
    }
}
impl DivAssign<f32> for Angle {
    fn div_assign(&mut self, rhs: f32) {
        self.0 /= rhs;
    }
}

#[cfg(feature = "fp")]
impl<B> From<PolarVec<B>> for Vec2<B> {
    /// Converts a polar vector into the equivalent Cartesian vector.
    ///
    /// See [PolarVec::to_cart] for more information.
    fn from(p: PolarVec<B>) -> Self {
        p.to_cart()
    }
}

#[cfg(feature = "fp")]
impl<B> From<Vec2<B>> for PolarVec<B> {
    /// Converts a Cartesian 2-vector into the equivalent polar vector.
    ///
    /// See [Vec2::to_polar] for more information.
    fn from(v: Vec2<B>) -> Self {
        v.to_polar()
    }
}

#[cfg(feature = "fp")]
impl<B> From<SphericalVec<B>> for Vec3<B> {
    /// Converts a spherical coordinate vector to a Euclidean 3-vector.
    ///
    /// See [SphericalVec::to_cart] for more information.
    fn from(v: SphericalVec<B>) -> Self {
        v.to_cart()
    }
}

#[cfg(feature = "fp")]
impl<B> From<Vec3<B>> for SphericalVec<B> {
    /// Converts a Cartesian 3-vector into the equivalent spherical vector.
    ///
    /// See [Vec3::to_spherical] for more information.
    fn from(v: Vec3<B>) -> Self {
        v.to_spherical()
    }
}

#[cfg(test)]
#[allow(unused, nonstandard_style)]
mod tests {
    use core::f32::consts::{PI, TAU};
    use std::eprintln;

    use crate::{
        assert_approx_eq,
        math::{self, Lerp, SQRT_3, Vary, Vec2, Vec3},
    };

    use super::*;

    const fn vec2(x: f32, y: f32) -> Vec2 {
        math::vec2(x, y)
    }
    const fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
        math::vec3(x, y, z)
    }

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

    #[cfg(feature = "fp")]
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
        assert_approx_eq!(degs(315.0).tan(), -1.0);
    }

    // TODO Micromath requires large epsilon here
    #[cfg(all(feature = "fp", not(feature = "mm")))]
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
        assert_approx_eq!(atan2(1.0, SQRT_3), degs(30.0));
        assert_approx_eq!(atan2(1.0, -1.0), degs(135.0));
        assert_approx_eq!(atan2(-SQRT_3, -1.0), degs(-120.0));
        assert_approx_eq!(atan2(-1.0, 1.0), degs(-45.0));
    }

    #[test]
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

    #[cfg(feature = "fp")]
    #[test]
    fn polar_to_cartesian_zero_r() {
        assert_eq!(polar(0.0, degs(0.0)).to_cart(), vec2(0.0, 0.0));
        assert_eq!(polar(0.0, degs(30.0)).to_cart(), vec2(0.0, 0.0));
        assert_eq!(polar(0.0, degs(-120.0)).to_cart(), vec2(0.0, 0.0));
    }

    #[cfg(feature = "fp")]
    #[test]
    fn polar_to_cartesian_zero_az() {
        assert_eq!(polar(2.0, degs(0.0)).to_cart(), vec2(2.0, 0.0));
        assert_eq!(polar(-3.0, degs(0.0)).to_cart(), vec2(-3.0, 0.0));
    }

    #[cfg(feature = "fp")]
    #[test]
    fn polar_to_cartesian() {
        assert_approx_eq!(polar(2.0, degs(60.0)).to_cart(), vec2(1.0, SQRT_3));

        assert_approx_eq!(
            polar(3.0, degs(-90.0)).to_cart(),
            vec2(0.0, -3.0),
            eps = 1e-6
        );
        assert_approx_eq!(polar(4.0, degs(270.0)).to_cart(), vec2(0.0, -4.0));

        assert_approx_eq!(
            polar(5.0, turns(1.25)).to_cart(),
            vec2(0.0, 5.0),
            eps = 2e-6
        );
    }

    #[cfg(feature = "fp")]
    #[test]
    fn cartesian_to_polar_zero_y() {
        assert_approx_eq!(vec2(0.0, 0.0).to_polar(), polar(0.0, degs(0.0)));
        assert_eq!(vec2(1.0, 0.0).to_polar(), polar(1.0, degs(0.0)));
    }
    #[cfg(feature = "fp")]
    #[test]
    fn cartesian_to_polar() {
        assert_approx_eq!(vec2(SQRT_3, 1.0).to_polar(), polar(2.0, degs(30.0)));
        assert_eq!(vec2(0.0, 2.0).to_polar(), polar(2.0, degs(90.0)));
        assert_approx_eq!(vec2(-3.0, 0.0).to_polar(), polar(3.0, degs(180.0)));
        assert_eq!(vec2(0.0, -4.0).to_polar(), polar(4.0, degs(-90.0)));
    }

    const fn sph(r: f32, az: f32, alt: f32) -> SphericalVec {
        spherical(r, degs(az), degs(alt))
    }
    #[rustfmt::skip]
    const CART_SPH: [(Vec3, SphericalVec); 10] = [
        (vec3(   0.0,    0.0,     0.0), sph(0.0,   0.0,  0.0)),

        (vec3(    1.0,   0.0,     0.0), sph(1.0,   0.0,  0.0)),
        (vec3( SQRT_3,   0.0,    -1.0), sph(2.0,  30.0,  0.0)),
        (vec3(    1.0,   0.0, -SQRT_3), sph(2.0,  60.0,  0.0)),
        (vec3(    0.0,   0.0,    -2.0), sph(2.0,  90.0,  0.0)),
        (vec3(-SQRT_3,   0.0,    -1.0), sph(2.0, 150.0,  0.0)),

        // Doesn't roundtrip due to imprecision and
        // the discontinuity from 180° to -180° :(
        (vec3(  -3.0,    0.0,     0.0), sph(3.0, 180.0,  0.0)),

        (vec3(SQRT_3,    1.0,     0.0), sph(2.0,  0.0,  30.0)),
        (vec3(   1.0, SQRT_3,     0.0), sph(2.0,  0.0,  60.0)),
        (vec3(   0.0,    2.0,     0.0), sph(2.0,  0.0,  90.0)),
        (vec3(   0.0,   -3.0,     0.0), sph(3.0,  0.0, -90.0)),
    ];

    #[cfg(feature = "fp")]
    #[test]
    fn spherical_to_cartesian() {
        for (cart, sp) in CART_SPH {
            let actual = sp.to_cart();
            eprintln!("Testing {sp:?} -> {cart:?}");
            assert_approx_eq!(actual, cart);
        }
    }

    #[cfg(feature = "fp")]
    #[test]
    fn cartesian_to_spherical() {
        for (cart, sp) in CART_SPH {
            let actual = cart.to_spherical();
            eprintln!("Testing {cart:?} -> {sp:?}");
            assert_approx_eq!(actual, sp);
        }
    }
}
