//! Colors and color spaces.

use core::{
    array,
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    ops::Index,
};

use crate::math::{float::f32, vary::ZDiv, Affine, Linear, Vector};

//
// Types
//

// TODO Document color spaces better

/// A generic color type, similar to [`Vector`].
///
/// # Type parameters
/// * `Repr`: the representation of the components of `Self`.
/// Color components are also called *channels*.
/// * `Space`: the color space that `Self` is an element of.
#[repr(transparent)]
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Color<Repr, Space>(pub Repr, PhantomData<Space>);

/// The (S)RGB (red, green, blue) color space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Rgb;

/// The (S)RGB (red, green, blue) color space with alpha (opacity).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Rgba;

/// Linear RGB (red, green, blue) color space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct LinRgb;

/// The HSL color space (hue, saturation, luminance).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Hsl;

/// The HSL color space (hue, saturation, luminance) with alpha (opacity).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Hsla;

/// A color with three `u8` channels, by default RGB.
pub type Color3<Space = Rgb> = Color<[u8; 3], Space>;

/// A color with four `u8` channels, by default RGBA.
pub type Color4<Space = Rgba> = Color<[u8; 4], Space>;

/// A color with three `f32` channels, by default RGB.
///
/// The nominal range of values for each channel is [0.0, 1.0], but values
/// outside the range can be useful as intermediate results in calculations.
pub type Color3f<Space = Rgb> = Color<[f32; 3], Space>;

/// A color with four `f32` channels, by default RGBA.
///
/// The nominal range of values for each channel is [0.0, 1.0], but values
/// outside the range can be useful as intermediate results in calculations.
pub type Color4f<Space = Rgba> = Color<[f32; 4], Space>;

/// Returns a new RGB color with the given color channels.
pub const fn rgb<Ch>(r: Ch, g: Ch, b: Ch) -> Color<[Ch; 3], Rgb> {
    Color([r, g, b], PhantomData)
}
/// Returns a new RGBA color with the given color channels.
pub const fn rgba<Ch>(r: Ch, g: Ch, b: Ch, a: Ch) -> Color<[Ch; 4], Rgba> {
    Color([r, g, b, a], PhantomData)
}
/// Returns a new RGB color with all channels set to the same value.
pub const fn gray<Ch: Copy>(lum: Ch) -> Color<[Ch; 3], Rgb> {
    rgb(lum, lum, lum)
}

/// Returns a new HSL color with the given color channels.
pub const fn hsl<Ch>(h: Ch, s: Ch, l: Ch) -> Color<[Ch; 3], Hsl> {
    Color([h, s, l], PhantomData)
}
/// Returns a new HSLA color with the given color channels.
pub const fn hsla<Ch>(h: Ch, s: Ch, l: Ch, a: Ch) -> Color<[Ch; 4], Hsla> {
    Color([h, s, l, a], PhantomData)
}

/// Exponent for gamma conversion [from sRGB to linear sRGB][1].
///
/// [1]: Color3f<Rgb>::to_linear
pub const GAMMA: f32 = 2.2;
/// Exponent for gamma conversion [from linear sRGB to sRGB][1].
///
/// [1]: Color3f<LinRgb>::to_srgb
pub const INV_GAMMA: f32 = 1.0 / GAMMA;

//
// Inherent impls
//

impl Color3<Rgb> {
    #[inline]
    /// Returns a `u32` containing the component bytes of `self`
    /// in format `0x00_RR_GG_BB`.
    pub const fn to_rgb_u32(self) -> u32 {
        let [r, g, b] = self.0;
        u32::from_be_bytes([0x00, r, g, b])
    }

    /// Returns `self` as RGBA, with alpha set to 0xFF (fully opaque).
    #[inline]
    pub const fn to_rgba(self) -> Color4 {
        let [r, g, b] = self.0;
        rgba(r, g, b, 0xFF)
    }

    /// Returns the HSL color equivalent to `self`.
    pub fn to_hsl(self) -> Color3<Hsl> {
        // Fixed point multiplier
        const M: i32 = 256;

        let [r, g, b] = self.0.map(i32::from);

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min; // Always non-negative

        let h = if d == 0 {
            0
        } else if max == r {
            (((g - b) * M) / d).rem_euclid(6 * M)
        } else if max == g {
            ((b - r) * M) / d + (2 * M)
        } else {
            ((r - g) * M) / d + (4 * M)
        };
        let h = h / 6;
        let l = (max + min + 1) / 2;
        let s = if l == 0 || l == 255 {
            0
        } else {
            (d * M) / (M - (2 * l - M).abs())
        };

        [h, s, l].map(|c| c.clamp(0, 255) as u8).into()
    }
}

impl Color4<Rgba> {
    /// Returns `self` as RGB, discarding the alpha channel.
    #[inline]
    pub fn to_rgb(self) -> Color3<Rgb> {
        let [r, g, b, _] = self.0;
        rgb(r, g, b)
    }

    /// Returns a `u32` containing the component bytes of `self`
    /// in format `0xRR_GG_BB_AA`.
    pub const fn to_rgba_u32(self) -> u32 {
        u32::from_be_bytes(self.0)
    }

    /// Returns a `u32` containing the component bytes of `self`
    /// in format `0xAA_RR_GG_BB`.
    #[inline]
    pub const fn to_argb_u32(self) -> u32 {
        self.to_rgba_u32().rotate_right(8)
    }

    /// Returns the HSLA color equivalent to `self`.
    pub fn to_hsla(self) -> Color4<Hsla> {
        let [r, g, b, _] = self.0;
        let [h, s, l] = rgb(r, g, b).to_hsl().0;
        [h, s, l, self.a()].into()
    }
}

impl Color3f<Rgb> {
    /// Returns `self` as RGBA, with alpha set to 1.0 (fully opaque).
    #[inline]
    pub const fn to_rgba(self) -> Color4f {
        let [r, g, b] = self.0;
        rgba(r, g, b, 1.0)
    }

    /// Returns a `Color3` with the components of `self` mapped to `u8`
    /// with `(c.clamp(0.0, 1.0) * 255.0) as u8`.
    #[inline]
    pub fn to_color3(self) -> Color3 {
        self.to_u8().into()
    }

    /// Returns a `Color4` with alpha 0xFF and the components of `self`
    /// mapped to `u8` with `(c.clamp(0.0, 1.0) * 255.0) as u8`.
    #[inline]
    pub fn to_color4(self) -> Color4 {
        let [r, g, b] = self.to_u8();
        rgba(r, g, b, 0xFF)
    }

    #[inline]
    fn to_u8(self) -> [u8; 3] {
        self.0.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8)
    }

    /// Returns `self` gamma-converted to linear RGB space.
    ///
    /// Linear interpolation, used to compute eg. gradients and blending,
    /// is just an approximation if carried out in a nonlinear, gamma-corrected
    /// color space such as the standard sRGB space. For the visually optimal
    /// results, colors should be interpolated in a linear space and only
    /// [converted to sRGB][1] right before writing to the output. Conversion,
    /// however, incurs a small performance penalty.
    ///
    /// [1]: Color3f<LinRgb>::to_srgb()
    #[cfg(feature = "fp")]
    #[inline]
    pub fn to_linear(self) -> Color3f<LinRgb> {
        self.0.map(|c| f32::powf(c, GAMMA)).into()
    }

    /// Returns the HSL color equivalent to `self`.
    pub fn to_hsl(self) -> Color3f<Hsl> {
        let [r, g, b] = self.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min;

        let h = if d == 0.0 {
            0.0
        } else if max == r {
            f32::rem_euclid((g - b) / d, 6.0)
        } else if max == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
        let h = h / 6.0;
        let l = (max + min) / 2.0;
        let s = if l == 0.0 || l == 1.0 {
            0.0
        } else {
            d / (1.0 - f32::abs(2.0 * l - 1.0))
        };

        for ch in [h, s, l] {
            debug_assert!(0.0 <= ch && ch <= 1.0, "channel oob: {ch:?}");
        }

        hsl(h, s, l)
    }
}

impl Color4f<Rgba> {
    /// Returns `self` as RGB, discarding the alpha channel.
    pub fn to_rgb(self) -> Color3f<Rgb> {
        let [r, g, b, _] = self.0;
        rgb(r, g, b)
    }
    /// Returns a `Color3` with the components of `self` mapped to `u8`
    /// with `(c.clamp(0.0, 1.0) * 255.0) as u8`, discarding alpha.
    #[inline]
    pub fn to_color3(self) -> Color3 {
        let [r, g, b, _] = self.to_u8();
        [r, g, b].into()
    }
    /// Returns a `Color4` with the components of `self` mapped
    /// to `u8` with `(c.clamp(0.0, 1.0) * 255.0) as u8`.
    #[inline]
    pub fn to_color4(self) -> Color4 {
        self.to_u8().into()
    }
    #[inline]
    fn to_u8(self) -> [u8; 4] {
        self.0.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8)
    }

    /// Returns the HSLA color equivalent to `self`.
    pub fn to_hsla(self) -> Color4f<Hsla> {
        let [r, g, b, _] = self.0;
        let [h, s, l] = rgb(r, g, b).to_hsl().0;
        [h, s, l, self.a()].into()
    }
}

impl Color3f<LinRgb> {
    /// Returns `self` gamma-converted to sRGB space.
    ///
    /// Linear interpolation, used to compute eg. gradients and blending,
    /// is just an approximation if carried out in a nonlinear, gamma-corrected
    /// color space such as the standard sRGB space. For visually optimal
    /// results, sRGB input colors should be [converted to linear space][1]
    /// before interpolation, and right before writing to the output.
    /// Conversion, however, incurs a small performance penalty.
    ///
    /// [1]: Color3f<Rgb>::to_linear()
    #[cfg(feature = "fp")]
    #[inline]
    pub fn to_srgb(self) -> Color3f<Rgb> {
        self.0.map(|c| f32::powf(c, INV_GAMMA)).into()
    }
}

impl Color3<Hsl> {
    /// Returns the RGB color equivalent to `self`.
    pub fn to_rgb(self) -> Color3<Rgb> {
        // Fixed-point multiplier
        const M: i32 = 256;

        let [h, s, l] = self.0.map(i32::from);
        let h = h * 6;

        let c = (M - (2 * l - M).abs()) * s;
        let x = c * (M - (h % (2 * M) - M).abs());
        let m = M * l - c / 2;

        let c = c / M;
        let x = x / M / M;
        let m = m / M;

        let rgb = match h / M {
            0 => [c, x, 0],
            1 => [x, c, 0],
            2 => [0, c, x],
            3 => [0, x, c],
            4 => [x, 0, c],
            5 => [c, 0, x],
            _ => unreachable!(),
        };
        rgb.map(|ch| {
            let ch = ch + m;
            debug_assert!(0 <= ch && ch < 256, "channel oob: {:?}", ch);
            ch as u8
        })
        .into()
    }
}

impl Color3f<Hsl> {
    /// Returns the RGB color equivalent to `self`.
    pub fn to_rgb(self) -> Color3f<Rgb> {
        let [h, s, l] = self.0;
        let h = h * 6.0;

        let c = (1.0 - f32::abs(2.0 * l - 1.0)) * s;
        let x = c * (1.0 - f32::abs(h % 2.0 - 1.0));
        let m = 1.0 * l - c / 2.0;

        let rgb = match (h - 0.5) as i32 {
            0 => [c, x, 0.0],
            1 => [x, c, 0.0],
            2 => [0.0, c, x],
            3 => [0.0, x, c],
            4 => [x, 0.0, c],
            5 => [c, 0.0, x],
            _ => unreachable!("h={h}"),
        };

        rgb.map(|ch| {
            let ch = ch + m;
            debug_assert!(0.0 <= ch && ch <= 1.0, "channel oob: {ch:?}");
            ch
        })
        .into()
    }
}

impl Color4<Hsla> {
    /// Returns `self` as HSL, discarding the alpha channel.
    pub fn to_hsl(self) -> Color3<Hsl> {
        let [h, s, l, _] = self.0;
        hsl(h, s, l)
    }
    /// Returns the RGBA color equivalent to `self`.
    pub fn to_rgba(self) -> Color4<Rgba> {
        let [r, g, b] = self.to_hsl().to_rgb().0;
        rgba(r, g, b, self.a())
    }
}
impl Color4f<Hsla> {
    /// Returns `self` as HSL, discarding the alpha channel.
    pub fn to_hsl(self) -> Color3f<Hsl> {
        let [h, s, l, _] = self.0;
        hsl(h, s, l)
    }
    /// Returns the RGBA color equivalent to `self`.
    pub fn to_rgba(self) -> Color4f<Rgba> {
        let [r, g, b] = self.to_hsl().to_rgb().0;
        rgba(r, g, b, self.a())
    }
}

impl<R, Sc> Color<R, Rgb>
where
    R: Index<usize, Output = Sc>,
    Sc: Copy,
{
    /// Returns the red component of `self`.
    pub fn r(&self) -> Sc {
        self.0[0]
    }
    /// Returns the green component of `self`.
    pub fn g(&self) -> Sc {
        self.0[1]
    }
    /// Returns the blue component of `self`.
    pub fn b(&self) -> Sc {
        self.0[2]
    }
}

impl<R, Sc> Color<R, Rgba>
where
    R: Index<usize, Output = Sc>,
    Sc: Copy,
{
    /// Returns the red component of `self`.
    pub fn r(&self) -> Sc {
        self.0[0]
    }
    /// Returns the green component of `self`.
    pub fn g(&self) -> Sc {
        self.0[1]
    }
    /// Returns the blue component of `self`.
    pub fn b(&self) -> Sc {
        self.0[2]
    }
    /// Returns the alpha component of `self`.
    pub fn a(&self) -> Sc {
        self.0[3]
    }
}

impl<R, Sc> Color<R, Hsl>
where
    R: Index<usize, Output = Sc>,
    Sc: Copy,
{
    /// Returns the hue component of `self`.
    pub fn h(&self) -> Sc {
        self.0[0]
    }
    /// Returns the saturation component of `self`.
    pub fn s(&self) -> Sc {
        self.0[1]
    }
    /// Returns the luminance component of `self`.
    pub fn l(&self) -> Sc {
        self.0[2]
    }
}

impl<R, Sc> Color<R, Hsla>
where
    R: Index<usize, Output = Sc>,
    Sc: Copy,
{
    /// Returns the hue component of `self`.
    pub fn h(&self) -> Sc {
        self.0[0]
    }
    /// Returns the saturation component of `self`.
    pub fn s(&self) -> Sc {
        self.0[1]
    }
    /// Returns the luminance component of `self`.
    pub fn l(&self) -> Sc {
        self.0[2]
    }
    /// Returns the alpha component of `self`.
    pub fn a(&self) -> Sc {
        self.0[3]
    }
}

//
// Local trait impls
//

impl<Sp, const DIM: usize> Affine for Color<[u8; DIM], Sp> {
    type Space = Sp;
    // Color<i32> is currently not Linear, so use Vector for now
    type Diff = Vector<[i32; DIM], Sp>;

    const DIM: usize = DIM;

    fn add(&self, other: &Self::Diff) -> Self {
        array::from_fn(|i| {
            let sum = i32::from(self.0[i]) + other.0[i];
            sum.clamp(0, u8::MAX as i32) as u8
        })
        .into()
    }
    fn sub(&self, other: &Self) -> Self::Diff {
        array::from_fn(|i| i32::from(self.0[i]) - i32::from(other.0[i])).into()
    }
}

impl<Sp, const DIM: usize> Affine for Color<[f32; DIM], Sp> {
    type Space = Sp;
    type Diff = Self;

    const DIM: usize = DIM;

    #[inline]
    fn add(&self, other: &Self::Diff) -> Self {
        array::from_fn(|i| self.0[i] + other.0[i]).into()
    }
    #[inline]
    fn sub(&self, other: &Self) -> Self::Diff {
        array::from_fn(|i| self.0[i] - other.0[i]).into()
    }
}

impl<Sp, const DIM: usize> Linear for Color<[f32; DIM], Sp> {
    type Scalar = f32;

    /// Returns the all-zeroes color (black).
    fn zero() -> Self {
        [0.0; DIM].into()
    }
    #[inline]
    fn mul(&self, scalar: Self::Scalar) -> Self {
        array::from_fn(|i| self.0[i] * scalar).into()
    }
}

impl<Sc, Sp, const N: usize> ZDiv for Color<[Sc; N], Sp> where Sc: ZDiv + Copy {}

//
// Foreign trait impls
//

impl<R: Debug, Space: Debug + Default> Debug for Color<R, Space> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Color<{:?}>{:?}", Space::default(), self.0)
    }
}

impl<R, Sp> From<R> for Color<R, Sp> {
    #[inline]
    fn from(els: R) -> Self {
        Self(els, PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_components() {
        assert_eq!(rgb(0xFF, 0, 0).r(), 0xFF);
        assert_eq!(rgb(0, 0xFF, 0).g(), 0xFF);
        assert_eq!(rgb(0, 0, 0xFF).b(), 0xFF);

        assert_eq!(rgba(0xFF, 0, 0, 0).r(), 0xFF);
        assert_eq!(rgba(0, 0xFF, 0, 0).g(), 0xFF);
        assert_eq!(rgba(0, 0, 0xFF, 0).b(), 0xFF);
        assert_eq!(rgba(0, 0, 0, 0xFF).a(), 0xFF);
    }
    #[test]
    fn rgb_to_u32() {
        assert_eq!(rgb(0x11, 0x22, 0x33).to_rgb_u32(), 0x00_11_22_33);
    }
    #[test]
    fn rgba_to_u32() {
        assert_eq!(rgba(0x11, 0x22, 0x33, 0x44).to_rgba_u32(), 0x11_22_33_44);
        assert_eq!(rgba(0x11, 0x22, 0x33, 0x44).to_argb_u32(), 0x44_11_22_33);
    }

    #[test]
    fn rgb_to_hsl() {
        let cases = [
            // Grays
            (gray(0), hsl(0, 0, 0)),
            (gray(64), hsl(0, 0, 64)),
            (gray(160), hsl(0, 0, 160)),
            (gray(255), hsl(0, 0, 255)),
            // 100% RGB
            (rgb(255, 0, 0), hsl(0, 255, 128)),
            (rgb(0, 255, 0), hsl(85, 255, 128)),
            (rgb(0, 0, 255), hsl(170, 255, 128)),
            // 100% CMY
            (rgb(255, 255, 0), hsl(42, 255, 128)),
            (rgb(255, 0, 255), hsl(213, 255, 128)),
            (rgb(0, 255, 255), hsl(128, 255, 128)),
            // 50% RGB
            (rgb(128, 0, 0), hsl(0, 255, 64)),
            (rgb(0, 128, 0), hsl(85, 255, 64)),
            (rgb(0, 0, 128), hsl(170, 255, 64)),
            // 50% CMY
            (rgb(128, 128, 0), hsl(42, 255, 64)),
            (rgb(128, 0, 128), hsl(213, 255, 64)),
            (rgb(0, 128, 128), hsl(128, 255, 64)),
        ];

        for (rgb, hsl) in cases {
            assert_eq!(rgb.to_hsl(), hsl, "{rgb:?} vs {hsl:?}");
        }
    }

    #[test]
    fn hsl_to_rgb() {
        // Not exactly the same as in `rgb_to_hsl` due to rounding errors
        let cases = [
            // Grays
            (gray(0), hsl(0, 0, 0)),
            (gray(64), hsl(0, 0, 64)),
            (gray(160), hsl(0, 0, 160)),
            (gray(255), hsl(0, 0, 255)),
            // 100% RGB
            (rgb(255, 0, 0), hsl(0, 255, 128)),
            (rgb(1, 255, 0), hsl(85, 255, 128)),
            (rgb(0, 3, 255), hsl(170, 255, 128)), // !
            // 100% CMY
            (rgb(255, 251, 0), hsl(42, 255, 128)), // !
            (rgb(253, 0, 255), hsl(213, 255, 128)), // !
            (rgb(0, 255, 255), hsl(128, 255, 128)),
            // 50% RGB
            (rgb(127, 0, 0), hsl(0, 255, 64)), // !
            (rgb(0, 127, 0), hsl(85, 255, 64)), // !
            (rgb(0, 1, 127), hsl(170, 255, 64)), // !
            // 50% CMY
            (rgb(127, 125, 0), hsl(42, 255, 64)), // !
            (rgb(126, 0, 127), hsl(213, 255, 64)), // !
            (rgb(0, 127, 127), hsl(128, 255, 64)), // !
        ];

        for (rgb, hsl) in cases {
            assert_eq!(hsl.to_rgb(), rgb, "{hsl:?} vs {rgb:?}");
        }
    }

    const RGB_HSL_FLOAT_CASES: [(Color3f, Color3f<Hsl>); 16] = [
        // Grays
        (gray(0.0), hsl(0.0, 0.0, 0.0)),
        (gray(0.25), hsl(0.0, 0.0, 0.25)),
        (gray(0.625), hsl(0.0, 0.0, 0.625)),
        (gray(1.0), hsl(0.0, 0.0, 1.0)),
        // 100% RGB
        (rgb(1.0, 0.0, 0.0), hsl(0.0, 1.0, 0.5)),
        (rgb(0.0, 1.0, 0.0), hsl(1.0 / 3.0, 1.0, 0.5)),
        (rgb(0.0, 0.0, 1.0), hsl(2.0 / 3.0, 1.0, 0.5)),
        // 100% CMY
        (rgb(1.0, 1.0, 0.0), hsl(1.0 / 6.0, 1.0, 0.5)),
        (rgb(1.0, 0.0, 1.0), hsl(5.0 / 6.0, 1.0, 0.5)),
        (rgb(0.0, 1.0, 1.0), hsl(0.5, 1.0, 0.5)),
        // 50% RGB
        (rgb(0.5, 0.0, 0.0), hsl(0.0, 1.0, 0.25)),
        (rgb(0.0, 0.5, 0.0), hsl(1.0 / 3.0, 1.0, 0.25)),
        (rgb(0.0, 0.0, 0.5), hsl(2.0 / 3.0, 1.0, 0.25)),
        // 50% CMY
        (rgb(0.5, 0.5, 0.0), hsl(1.0 / 6.0, 1.0, 0.25)),
        (rgb(0.5, 0.0, 0.5), hsl(5.0 / 6.0, 1.0, 0.25)),
        (rgb(0.0, 0.5, 0.5), hsl(0.5, 1.0, 0.25)),
    ];

    #[test]
    fn hsl_to_rgb_float() {
        for (rgb, hsl) in RGB_HSL_FLOAT_CASES {
            assert_eq!(hsl.to_rgb(), rgb, "{hsl:?} to {rgb:?}");
        }
    }
    #[test]
    fn rgb_to_hsl_float() {
        for (rgb, hsl) in RGB_HSL_FLOAT_CASES {
            assert_eq!(rgb.to_hsl(), hsl, "{rgb:?} to {hsl:?}");
        }
    }
}
