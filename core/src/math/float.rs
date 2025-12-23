//! Floating-point compatibility API.
//!
//! Most floating-point functions are currently unavailable in `no_std`.
//! This module provides the missing functions using either the `libm` or
//! `micromath` crate, depending on which feature is enabled. As a fallback,
//! it also implements a critical subset of the functions even if none of
//! the features is enabled.

#[cfg(feature = "libm")]
pub mod libm {
    pub use libm::floorf as floor;

    pub use libm::powf;
    pub use libm::sqrtf as sqrt;

    pub use libm::cosf as cos;
    pub use libm::sinf as sin;
    pub use libm::tanf as tan;

    pub use libm::acosf as acos;
    pub use libm::asinf as asin;
    pub use libm::atan2f as atan2;

    pub use libm::expf as exp;
    pub use libm::log2;

    pub use super::fallback::rem_euclid;

    #[inline]
    pub fn recip_sqrt(x: f32) -> f32 {
        1.0 / sqrt(x)
    }
}

#[cfg(feature = "mm")]
pub mod mm {
    use micromath::F32Ext as mm;

    #[inline]
    pub fn floor(x: f32) -> f32 {
        mm::floor(x)
    }
    #[inline]
    pub fn rem_euclid(x: f32, m: f32) -> f32 {
        mm::rem_euclid(x, m)
    }
    /// Returns the approximate square root of `x`.
    #[inline]
    pub fn sqrt(x: f32) -> f32 {
        let y = mm::sqrt(x);
        // Two rounds of Newton's method
        let y = 0.5 * (y + (x / y));
        0.5 * (y + (x / y))
    }
    /// Returns the approximate reciprocal of the square root of `x`.
    #[inline]
    pub fn recip_sqrt(x: f32) -> f32 {
        let y = mm::invsqrt(x);
        // A round of Newton's method
        y * (1.5 - 0.5 * x * y * y)
    }
    #[inline]
    pub fn powf(x: f32, y: f32) -> f32 {
        mm::powf(x, y)
    }

    #[inline]
    pub fn sin(x: f32) -> f32 {
        mm::sin(x)
    }
    #[inline]
    pub fn cos(x: f32) -> f32 {
        mm::cos(x)
    }
    #[inline]
    pub fn tan(x: f32) -> f32 {
        mm::tan(x)
    }
    #[inline]
    pub fn asin(x: f32) -> f32 {
        mm::asin(x)
    }
    #[inline]
    pub fn acos(x: f32) -> f32 {
        mm::acos(x)
    }
    #[inline]
    pub fn atan2(y: f32, x: f32) -> f32 {
        #[cfg(debug_assertions)]
        if y == 0.0 && x == 0.0 {
            // Micromath yields a NaN but others return zero
            return 0.0;
        }
        mm::atan2(y, x)
    }

    #[inline]
    pub fn exp(x: f32) -> f32 {
        mm::exp(x)
    }
    #[inline]
    pub fn log2(x: f32) -> f32 {
        mm::log2(x)
    }
}

pub mod fallback {
    use crate::math::float::fast_recip_sqrt;

    /// Returns the largest integer less than or equal to `x`.
    #[inline]
    pub fn floor(x: f32) -> f32 {
        (x as i64 - (x < 0.0) as i64) as f32
    }
    /// Returns the least non-negative remainder of `x` (mod `m`).
    #[inline]
    pub fn rem_euclid(x: f32, m: f32) -> f32 {
        let r = x % m;
        r + if r < 0.0 { m.abs() } else { 0.0 }
    }
    /// Returns the approximate reciprocal of the square root of `x`.
    #[inline]
    pub fn recip_sqrt(x: f32) -> f32 {
        fast_recip_sqrt(x)
    }
    #[inline]
    pub fn sqrt(x: f32) -> f32 {
        1.0 / recip_sqrt(x)
    }
}

/// Returns a fast approximation of the reciprocal square root of a number.
#[inline]
pub fn fast_recip_sqrt(x: f32) -> f32 {
    // https://en.wikipedia.org/wiki/Fast_inverse_square_root
    const MAGIC: u32 = 0x5f37_5a86;
    let mut y = f32::from_bits(MAGIC.saturating_sub(x.to_bits() >> 1));
    // A round of Newton's method
    y = y * (1.5 - 0.5 * x * y * y);
    //y = y * (1.5 - 0.5 * x * y * y);
    y
}

#[cfg(feature = "std")]
#[allow(non_camel_case_types)]
pub type f32 = core::primitive::f32;

#[allow(unused)]
pub(crate) trait RecipSqrt {
    fn recip_sqrt(x: Self) -> Self;
}

#[cfg(feature = "std")]
impl RecipSqrt for f32 {
    #[inline]
    fn recip_sqrt(x: f32) -> f32 {
        x.powf(-0.5)
    }
}

#[cfg(all(feature = "libm", not(feature = "std")))]
pub use libm as f32;

#[cfg(all(feature = "mm", not(feature = "std"), not(feature = "libm")))]
pub use mm as f32;

#[cfg(not(feature = "fp"))]
pub use fallback as f32;

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use core::f32::consts::*;

    use super::{RecipSqrt, f32, *};
    use crate::assert_approx_eq;

    #[cfg(feature = "libm")]
    #[test]
    fn libm_functions() {
        assert_eq!(libm::floor(1.5), 1.0);
        assert_eq!(libm::floor(0.99), 0.0);
        assert_eq!(libm::floor(-0.0), 0.0);
        assert_eq!(libm::floor(-1.1), -2.0);

        assert_approx_eq!(libm::rem_euclid(1.6, 0.5), 0.1);
        assert_approx_eq!(libm::rem_euclid(-1.6, 0.5), 0.4);
        assert_approx_eq!(libm::rem_euclid(1.6, -0.5), 0.1);
        assert_approx_eq!(libm::rem_euclid(-1.6, -0.5), 0.4);

        assert_eq!(libm::sqrt(9.0), 3.0);
        assert_eq!(libm::sqrt(16.0), 4.0);
        assert!(libm::sqrt(-1.0).is_nan());
        assert_eq!(libm::recip_sqrt(9.0), 1.0 / 3.0);
        assert_eq!(libm::recip_sqrt(0.0), f32::INFINITY);
        assert!(libm::recip_sqrt(-1.0).is_nan());

        assert_eq!(libm::powf(3.0, 2.0), 9.0);
        assert_eq!(libm::powf(-3.0, 2.0), 9.0);
        assert_eq!(libm::powf(3.0, -2.0), 1.0 / 9.0);
        assert_eq!(libm::powf(-3.0, 3.0), -27.0);

        assert_approx_eq!(libm::sin(FRAC_PI_6), 0.5);
        assert_eq!(libm::cos(PI), -1.0);

        assert_eq!(libm::exp(1.0), E);
        assert_approx_eq!(libm::exp(2.0), E * E);
        assert_eq!(libm::log2(8.0), 3.0);
        assert!(libm::log2(-1.0).is_nan());
    }

    #[cfg(feature = "mm")]
    #[test]
    fn mm_functions() {
        assert_eq!(mm::floor(1.5), 1.0);
        assert_eq!(mm::floor(0.99), 0.0);
        assert_eq!(mm::floor(-0.0), 0.0);
        assert_eq!(mm::floor(-1.1), -2.0);

        assert_approx_eq!(mm::rem_euclid(1.6, 0.5), 0.1);
        assert_approx_eq!(mm::rem_euclid(-1.6, 0.5), 0.4);
        assert_approx_eq!(mm::rem_euclid(1.6, -0.5), 0.1);
        assert_approx_eq!(mm::rem_euclid(-1.6, -0.5), 0.4);

        assert_approx_eq!(mm::sqrt(9.0), 3.0);
        assert_eq!(mm::sqrt(16.0), 4.0);
        assert!(mm::sqrt(-1.0).is_nan());
        assert_approx_eq!(mm::recip_sqrt(9.0), 1.0 / 3.0);
        // mm doesn't check for zero, just gives a big number
        assert_approx_eq!(mm::recip_sqrt(0.0), 1.9818e19);
        // mm doesn't check for negative, panics due to sub overflow
        //assert!(mm::recip_sqrt(-1.0).is_nan());

        assert_approx_eq!(mm::powf(3.0, 2.0), 9.0);
        assert_approx_eq!(mm::powf(-3.0, 2.0), 9.0);
        assert_approx_eq!(mm::powf(3.0, -2.0), 1.0 / 9.0);
        assert_approx_eq!(mm::powf(-3.0, 3.0), -27.0);

        assert_approx_eq!(mm::sin(FRAC_PI_6), 0.5);
        assert_eq!(mm::cos(PI), -1.0);

        assert_eq!(mm::exp(1.0), E);
        assert_approx_eq!(mm::exp(2.0), E * E);
        assert_approx_eq!(mm::log2(8.0), 3.0);
        // mm doesn't check for negative, panics due to sub overflow
        //assert!(mm::log2(-1.0).is_nan());
    }

    #[cfg(feature = "std")]
    #[test]
    fn std_functions() {
        assert_eq!(f32::floor(-0.0), 0.0);

        assert_approx_eq!(f32::rem_euclid(1.6, 0.5), 0.1);
        assert_approx_eq!(f32::rem_euclid(-1.6, 0.5), 0.4);
        assert_approx_eq!(f32::rem_euclid(1.6, -0.5), 0.1);
        assert_approx_eq!(f32::rem_euclid(-1.6, -0.5), 0.4);

        assert_eq!(f32::sqrt(9.0), 3.0);
        assert!(f32::sqrt(-1.0).is_nan());
        assert_eq!(f32::recip_sqrt(9.0), 1.0 / 3.0);
        assert_eq!(f32::recip_sqrt(0.0), f32::INFINITY);
        assert!(f32::recip_sqrt(-1.0).is_nan());

        assert_eq!(f32::cos(PI), -1.0);
    }

    #[cfg(not(feature = "fp"))]
    #[test]
    fn fallback_functions() {
        use fallback as fb;
        assert_eq!(fb::floor(1.5), 1.0);
        assert_eq!(fb::floor(0.99), 0.0);
        assert_eq!(fb::floor(-0.0), 0.0);
        assert_eq!(fb::floor(-1.1), -2.0);

        assert_approx_eq!(fb::rem_euclid(1.6, 0.5), 0.1);
        assert_approx_eq!(fb::rem_euclid(-1.6, 0.5), 0.4);
        assert_approx_eq!(fb::rem_euclid(1.6, -0.5), 0.1);
        assert_approx_eq!(fb::rem_euclid(-1.6, -0.5), 0.4);

        assert_approx_eq!(fb::sqrt(9.0), 3.0);
        assert_approx_eq!(fb::sqrt(16.0), 4.0);
        assert!(fb::sqrt(-1.0).is_nan());
        assert_approx_eq!(fb::recip_sqrt(9.0), 1.0 / 3.0);
        // doesn't check for infinity, just returns a big number
        assert_approx_eq!(fb::recip_sqrt(0.0), 2.9727e19);
        assert!(fb::recip_sqrt(-1.0).is_nan());

        // assert_eq!(fb::powf(3.0, 2.0), 9.0);
        // assert_eq!(fb::powf(-3.0, 2.0), 9.0);
        // assert_eq!(fb::powf(3.0, -2.0), 1.0 / 9.0);
        // assert_eq!(fb::powf(-3.0, 3.0), -27.0);
        //
        // assert_approx_eq!(libm::sin(FRAC_PI_6), 0.5);
        // assert_eq!(libm::cos(PI), -1.0);
        //
        // assert_eq!(libm::exp(1.0), E);
        // assert_approx_eq!(libm::exp(2.0), E * E);
        // assert_eq!(libm::log2(8.0), 3.0);
        // assert!(libm::log2(-1.0).is_nan());
    }
}
