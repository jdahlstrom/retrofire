#[cfg(feature = "libm")]
pub mod libm {
    pub use libm::fabsf as abs;
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

    pub fn recip_sqrt(x: f32) -> f32 {
        powf(x, -0.5)
    }
}

#[cfg(feature = "mm")]
pub mod mm {
    use micromath::F32Ext as mm;

    #[inline]
    pub fn abs(x: f32) -> f32 {
        mm::abs(x)
    }
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
        // One round of Newton's method
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
}

pub mod fallback {
    /// Returns the absolute value of `x`.
    #[inline]
    pub fn abs(x: f32) -> f32 {
        f32::from_bits(x.to_bits() & !0x8000_0000)
    }
    /// Returns the largest integer less than or equal to `x`.
    #[inline]
    pub fn floor(x: f32) -> f32 {
        (x as i64 - x.is_sign_negative() as i64) as f32
    }
    // Returns the least non-negative remainder of `x` (mod `m`).
    #[inline]
    pub fn rem_euclid(x: f32, m: f32) -> f32 {
        x % m + (x.is_sign_negative() as u32 as f32) * m
    }
    /// Returns the approximate reciprocal of the square root of `x`.
    #[inline]
    pub fn recip_sqrt(x: f32) -> f32 {
        // https://en.wikipedia.org/wiki/Fast_inverse_square_root
        let y = f32::from_bits(0x5f37_5a86 - (x.to_bits() >> 1));
        // A round of Newton's method
        y * (1.5 - 0.5 * x * y * y)
    }
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

    use crate::assert_approx_eq;

    #[cfg(feature = "libm")]
    #[test]
    fn libm_functions() {
        use super::libm;
        use core::f32::consts::PI;
        assert_eq!(libm::cos(PI), -1.0);
        assert_eq!(libm::sqrt(9.0), 3.0);
    }

    #[cfg(feature = "mm")]
    #[test]
    fn mm_functions() {
        use core::f32::consts::*;

        use super::f32;

        assert_approx_eq!(f32::sin(FRAC_PI_6), 0.5);
        assert_eq!(f32::cos(PI), -1.0);
        assert_eq!(f32::sqrt(16.0), 4.0);
        assert_approx_eq!(f32::sqrt(9.0), 3.0, eps = 1e-3);
    }

    #[cfg(feature = "std")]
    #[test]
    fn std_functions() {
        use super::f32;
        use core::f32::consts::PI;
        assert_eq!(f32::cos(PI), -1.0);
        assert_eq!(f32::sqrt(9.0), 3.0);
    }

    #[cfg(not(feature = "fp"))]
    #[test]
    fn fallback_functions() {
        use super::{f32, RecipSqrt};

        assert_eq!(f32::floor(1.23), 1.0);
        assert_eq!(f32::floor(0.0), 0.0);
        assert_eq!(f32::floor(-1.23), -2.0);

        assert_eq!(f32::abs(1.23), 1.23);
        assert_eq!(f32::abs(0.0), 0.0);
        assert_eq!(f32::abs(-1.23), 1.23);

        assert_approx_eq!(f32::rem_euclid(1.23, 4.0), 1.23);
        assert_approx_eq!(f32::rem_euclid(4.0, 4.0), 0.0);
        assert_approx_eq!(f32::rem_euclid(5.67, 4.0), 1.67);
        assert_approx_eq!(f32::rem_euclid(-1.23, 4.0), 2.77);
    }

    #[test]
    fn recip_sqrt() {
        use super::{f32, RecipSqrt};
        assert_approx_eq!(f32::recip_sqrt(2.0), f32::sqrt(0.5), eps = 1e-2);
        assert_approx_eq!(f32::recip_sqrt(9.0), 1.0 / 3.0, eps = 1e-3);
    }
}
