//! Colors and color spaces.

use core::ops::{
    Add, AddAssign, Div, DivAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign,
};
use core::{
    array,
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

use super::{Affine, Linear, Vector, vary::ZDiv};

//
// Types
//

// TODO Document color spaces better

/// A generic color type, similar to [`Vector`].
///
/// # Type parameters
/// * `Repr`: the representation of the components of `Self`.
///   Color components are also called *channels*.
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
    /// Returns `self` as RGBA, with alpha set to 0xFF (fully opaque).
    #[inline]
    pub const fn to_rgba(self) -> Color4 {
        let [r, g, b] = self.0;
        rgba(r, g, b, 0xFF)
    }

    pub fn to_color3f(self) -> Color3f {
        self.0.map(|c| c as f32 / 255.0).into()
    }

    /// Returns the HSL color equivalent to `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::color::{hsl, rgb};
    ///
    /// let red = rgb(1.0, 0.0, 0.0);
    /// assert_eq!(red.to_hsl(), hsl(0.0, 1.0, 0.5));
    ///
    /// let light_blue = rgb(0.5, 0.5, 1.0);
    /// assert_eq!(light_blue.to_hsl(), hsl(2.0/3.0, 1.0, 0.75));
    /// ```
    pub fn to_hsl(self) -> Color3<Hsl> {
        // Fixed point multiplier
        const _1: i32 = 256;

        let [r, g, b] = self.0.map(i32::from);

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min; // Always non-negative

        let h = if d == 0 {
            0
        } else if max == r {
            (((g - b) * _1) / d).rem_euclid(6 * _1)
        } else if max == g {
            ((b - r) * _1) / d + (2 * _1)
        } else {
            ((r - g) * _1) / d + (4 * _1)
        };
        let h = h / 6;
        let l = (max + min + 1) / 2;
        let s = if l == 0 || l == 255 {
            0
        } else {
            (d * _1) / (_1 - (2 * l - _1).abs())
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

    pub fn to_color4f(self) -> Color4f {
        self.0.map(|c| c as f32 / 255.0).into()
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
        use super::float::f32;
        self.0.map(|c| f32::powf(c, GAMMA)).into()
    }

    /// Returns the HSL color equivalent to `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::color::{hsl, rgb};
    ///
    /// let red = rgb(0xFF, 0, 0);
    /// assert_eq!(red.to_hsl(), hsl(0, 0xFF, 0x80));
    ///
    /// let light_blue = rgb(0x80, 0x80, 0xFF);
    /// assert_eq!(light_blue.to_hsl(), hsl(0xAA, 0xFE, 0xC0));
    /// ```
    pub fn to_hsl(self) -> Color3f<Hsl> {
        let [r, g, b] = self.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let d = max - min;

        let h = if d == 0.0 {
            0.0
        } else if max == r {
            super::float::f32::rem_euclid((g - b) / d, 6.0)
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
            d / (1.0 - (2.0 * l - 1.0).abs())
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
    /// Linear interpolation, used to compute e.g. gradients and blending,
    /// is just an approximation if carried out in a nonlinear, gamma-corrected
    /// color space such as the standard sRGB space. For visually correct
    /// results, sRGB input colors should be [converted to linear space][1]
    /// before interpolation, and right before writing to the output.
    /// Conversion, however, incurs a small performance penalty.
    ///
    /// [1]: Color3f<Rgb>::to_linear()
    #[cfg(feature = "fp")]
    #[inline]
    pub fn to_srgb(self) -> Color3f<Rgb> {
        use super::float::f32;
        self.0.map(|c| f32::powf(c, INV_GAMMA)).into()
    }
}

impl Color3<Hsl> {
    /// Returns the RGB color equivalent to `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::color::{hsl, rgb};
    ///
    /// let red = hsl(0, 0xFF, 0x80);
    /// assert_eq!(red.to_rgb(), rgb(0xFF, 0, 0));
    ///
    /// let light_blue = hsl(0xAB, 0xFF, 0xC0);
    /// assert_eq!(light_blue.to_rgb(), rgb(0x80, 0x80, 0xFF));
    /// ```
    pub fn to_rgb(self) -> Color3<Rgb> {
        // Fixed-point multiplier
        const _1: i32 = 256;

        let [h, s, l] = self.0.map(i32::from);
        let h = 6 * h;

        let c = (_1 - (2 * l - _1).abs()) * s;
        let x = c * (_1 - (h % (2 * _1) - _1).abs());
        let m = _1 * l - c / 2;

        // Normalize
        let [c, x, m] = [c / _1, x / _1 / _1, m / _1];

        let rgb = hcx_to_rgb(h / _1, c, x, 0);
        rgb.map(|ch| {
            let ch = ch + m;
            debug_assert!(0 <= ch && ch < _1, "channel oob: {:?}", ch);
            ch as u8
        })
        .into()
    }
}

impl Color3f<Hsl> {
    /// Returns the RGB color equivalent to `self`.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::color::{hsl, rgb};
    ///
    /// let red = hsl(0.0f32, 1.0, 0.5);
    /// assert_eq!(red.to_rgb(), rgb(1.0, 0.0, 0.0));
    ///
    /// let light_blue = hsl(2.0 / 3.0, 1.0, 0.75);
    /// assert_eq!(light_blue.to_rgb(), rgb(0.5, 0.5, 1.0));
    /// ```
    pub fn to_rgb(self) -> Color3f<Rgb> {
        let [h, s, l] = self.0;
        let h = 6.0 * h;

        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let x = c * (1.0 - (h % 2.0 - 1.0).abs());
        let m = l - c / 2.0;

        let rgb = hcx_to_rgb(h as i32, c, x, 0.0);
        rgb.map(|ch| {
            let ch = ch + m;
            debug_assert!(-1e6 <= ch && ch <= 1.0 + 1e6, "channel oob: {ch:?}");
            ch
        })
        .into()
    }
}

fn hcx_to_rgb<T: Display>(h: i32, c: T, x: T, z: T) -> [T; 3] {
    match h {
        0 => [c, x, z],
        1 => [x, c, z],
        2 => [z, c, x],
        3 => [z, x, c],
        4 => [x, z, c],
        5 | 6 => [c, z, x],
        _ => unreachable!("h = {h}, c = {c}, x = {z}"),
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

//
// Arithmetic trait impls
//

//
// Arithmetic traits
//

/// The color += color operator.
impl<R, D, Sp> AddAssign<D> for Color<R, Sp>
where
    Self: Affine<Diff = D>,
{
    #[inline]
    fn add_assign(&mut self, rhs: D) {
        *self = Affine::add(&*self, &rhs);
    }
}

/// The color -= color operator.
impl<R, D, Sp> SubAssign<D> for Color<R, Sp>
where
    D: Linear,
    Self: Affine<Diff = D>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: D) {
        *self += rhs.neg();
    }
}

// The color *= scalar operator.
impl<R, Sc, Sp> MulAssign<Sc> for Color<R, Sp>
where
    Self: Linear<Scalar = Sc>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Sc) {
        *self = Linear::mul(&*self, rhs);
    }
}

// The color /= scalar operator.
impl<R, Sp> DivAssign<f32> for Color<R, Sp>
where
    Self: Linear<Scalar = f32>,
{
    #[inline]
    fn div_assign(&mut self, rhs: f32) {
        use crate::math::ApproxEq;
        debug_assert!(!rhs.approx_eq(&0.0));
        *self = Linear::mul(&*self, rhs.recip());
    }
}

/// The color negation operator.
impl<R, Sp> Neg for Color<R, Sp>
where
    Self: Linear,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        <Self as Linear>::neg(&self)
    }
}

/// The scalar * color operator.
impl<R, Sp> Mul<Color<R, Sp>> for <Color<R, Sp> as Linear>::Scalar
where
    Color<R, Sp>: Linear,
{
    type Output = Color<R, Sp>;

    #[inline]
    fn mul(self, rhs: Color<R, Sp>) -> Color<R, Sp> {
        rhs * self
    }
}

// The color + color operator.
impl_op!(Add::add, Color, <Self as Affine>::Diff, +=, bound=Affine);
// The color - color operator.
impl_op!(Sub::sub, Color, <Self as Affine>::Diff, -=, bound=Affine);
// The color * scalar operator.
impl_op!(Mul::mul, Color, <Self as Linear>::Scalar, *=);
// The color / scalar operator.
impl_op!(Div::div, Color, f32, /=, bound=Linear<Scalar = f32>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_components() {
        assert_eq!(rgb(0xFF, 0, 0).r(), 0xFF);
        assert_eq!(rgb(0, 0xFF, 0).g(), 0xFF);
        assert_eq!(rgb(0, 0, 0xFF).b(), 0xFF);

        assert_eq!(rgba(0xFF, 0, 0, 0).r(), 0xFF);
        assert_eq!(rgba(0, 0xFF, 0, 0).g(), 0xFF);
        assert_eq!(rgba(0, 0, 0xFF, 0).b(), 0xFF);
        assert_eq!(rgba(0, 0, 0, 0xFF).a(), 0xFF);
    }

    #[test]
    fn hsl_components() {
        assert_eq!(hsl(0xFF, 0, 0).h(), 0xFF);
        assert_eq!(hsl(0, 0xFF, 0).s(), 0xFF);
        assert_eq!(hsl(0, 0, 0xFF).l(), 0xFF);

        assert_eq!(hsla(0xFF, 0, 0, 0).h(), 0xFF);
        assert_eq!(hsla(0, 0xFF, 0, 0).s(), 0xFF);
        assert_eq!(hsla(0, 0, 0xFF, 0).l(), 0xFF);
        assert_eq!(hsla(0, 0, 0, 0xFF).a(), 0xFF);
    }

    #[test]
    fn rgb_f32_ops() {
        let lhs = rgb(0.5, 0.625, 0.75);
        let rhs = rgb(0.125, 0.25, 0.375);

        assert_eq!(lhs + rhs, rgb(0.625, 0.875, 1.125));
        assert_eq!(lhs - rhs, rgb(0.375, 0.375, 0.375));
        assert_eq!(lhs * 0.5, rgb(0.25, 0.3125, 0.375));
        assert_eq!(0.5 * lhs, rgb(0.25, 0.3125, 0.375));
        assert_eq!(lhs / 2.0, rgb(0.25, 0.3125, 0.375));
        assert_eq!(-lhs, rgb(-0.5, -0.625, -0.75));
    }

    #[test]
    fn rgb_u8_ops() {
        let lhs = rgb(0x77, 0x88, 0x99);
        let rhs = [0x11_i32, 0x33, 0x55].into();

        assert_eq!(lhs + rhs, rgb(0x88, 0xBB, 0xEE));
        assert_eq!(lhs - rhs, rgb(0x66, 0x55, 0x44));
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
