//! Linear algebra and other useful mathematics.
//!
//! Includes [vectors][self::vec], [matrices][mat], [colors][color],
//! [angles][angle], [Bezier splines][spline] and [pseudo-random numbers][rand],
//! as well as support for custom [varying][vary] types and utilities such as
//! approximate equality comparisons.
//!
//! This library is more strongly typed than many other similar math libraries.
//! It aims  to diagnose at compile time many errors that might otherwise only
//! manifest as graphical glitches, runtime panics, or even – particularly in
//! languages that are unsafe-by-default – undefined behavior.
//!
//! In particular, vectors and colors are tagged with a type that represents
//! the *space* they're embedded in, and values in different spaces cannot be
//! mixed without explicit conversion (transformation). Matrices, similarly,
//! are tagged by both source and destination space, and can only be applied
//! to matching vectors. Angles are strongly typed as well, to allow working
//! with different angular units without confusion.

pub use {
    angle::{
        Angle, PolarVec, SphericalVec, degs, polar, rads, spherical, turns,
    },
    approx::ApproxEq,
    color::{Color, Color3, Color3f, Color4, Color4f, rgb, rgba},
    mat::{
        Mat3x3, Mat4x4, Matrix, orthographic, perspective, scale, scale3,
        translate, translate3, viewport,
    },
    param::Parametric,
    point::{Point, Point2, Point2u, Point3, pt2, pt3},
    space::{Affine, Linear},
    spline::{BezierSpline, CubicBezier, smootherstep, smoothstep},
    vary::Vary,
    vec::{Vec2, Vec2i, Vec3, Vec3i, Vector, splat, vec2, vec3},
};
#[cfg(feature = "fp")]
pub use {
    angle::{acos, asin, atan2},
    mat::{orient_y, orient_z, rotate, rotate_x, rotate_y, rotate_z, rotate2},
};

pub mod angle;
pub mod approx;
pub mod color;
pub mod float;
pub mod mat;
pub mod param;
pub mod point;
pub mod rand;
pub mod space;
pub mod spline;
pub mod vary;
pub mod vec;

/// Trait for linear interpolation between two values.
pub trait Lerp: Sized {
    /// Linearly interpolates between `self` and `other`.
    ///
    /// if `t` = 0, returns `self`; if `t` = 1, returns `other`.
    /// For 0 < `t` < 1, returns the weighted average of `self` and `other`
    /// ```text
    /// (1 - t) * self + t * other
    /// ```
    ///
    /// This method does not panic if `t < 0.0` or `t > 1.0`, or if `t`
    /// is a `NaN`, but the return value in those cases is unspecified.
    /// Individual implementations may offer stronger guarantees.
    fn lerp(&self, other: &Self, t: f32) -> Self;

    /// Returns the (unweighted) average of `self` and `other`.
    fn midpoint(&self, other: &Self) -> Self {
        self.lerp(other, 0.5)
    }
}

/// Linearly interpolates between two values.
///
/// For more information, see [`Lerp::lerp`].
#[inline]
pub fn lerp<T: Lerp>(t: f32, from: T, to: T) -> T {
    from.lerp(&to, t)
}

impl<T> Lerp for T
where
    T: Affine<Diff: Linear<Scalar = f32>>,
{
    /// Linearly interpolates between `self` and `other`.
    ///
    /// if `t` = 0, returns `self`; if `t` = 1, returns `other`.
    /// For 0 < `t` < 1, returns the affine combination
    /// ```text
    /// (1 - t) * self + t * other
    /// ```
    /// or rearranged:
    /// ```text
    /// self + t * (other - self)
    /// ```
    ///
    /// If `t < 0.0` or `t > 1.0`, returns the appropriate extrapolated value.
    /// If `t` is a NaN, the result is unspecified.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::*;
    ///
    /// assert_eq!(2.0.lerp(&5.0, 0.0), 2.0);
    /// assert_eq!(2.0.lerp(&5.0, 0.25), 2.75);
    /// assert_eq!(2.0.lerp(&5.0, 0.75), 4.25);
    /// assert_eq!(2.0.lerp(&5.0, 1.0), 5.0);
    ///
    /// let v0: Vec2 = vec2(-2.0, 1.0);
    /// let v1 = vec2(3.0, -1.0);
    /// assert_eq!(v0.lerp(&v1, 0.8), vec2(2.0, -0.6));
    ///
    /// let p0: Point2 = pt2(-10.0, 5.0);
    /// let p1 = pt2(-5.0, 0.0);
    /// assert_eq!(p0.lerp(&p1, 0.4),pt2(-8.0, 3.0));
    /// ```
    fn lerp(&self, other: &Self, t: f32) -> Self {
        self.add(&other.sub(self).mul(t))
    }
}

impl Lerp for () {
    fn lerp(&self, _: &Self, _: f32) {}
}

impl<U: Lerp, V: Lerp> Lerp for (U, V) {
    fn lerp(&self, (u, v): &Self, t: f32) -> Self {
        (self.0.lerp(&u, t), self.1.lerp(&v, t))
    }
}
