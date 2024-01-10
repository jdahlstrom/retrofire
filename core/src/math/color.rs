use core::array;
use core::fmt::{self, Debug, Formatter};
use core::marker::PhantomData;
use core::ops::Index;

use crate::math::space::{Affine, Linear};
use crate::math::vec::Vector;

//
// Types
//

/// A generic color type, similar to [`Vector`].
///
/// # Type parameters
/// * `Repr`: the representation of the components of `Self`.
/// Color components are also called *channels*.
/// * `Space`: the color space that `Self` is an element of.
#[repr(transparent)]
#[derive(Copy, Clone, Default, Eq, PartialEq)]
pub struct Color<Repr, Space>(pub Repr, PhantomData<Space>);

/// The (S)RGB color space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Rgb;

/// The (S)RGBA color space (RGB plus alpha, or opacity).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Rgba;

/// The HSL color space (hue, saturation, luminance) .
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Hsl;

/// The HSL space with alpha (opacity).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Hsla;

/// An RGB color with `u8` components.
pub type Color3<Space = Rgb> = Color<[u8; 3], Space>;

/// An RGBA color with `u8` components.
pub type Color4<Space = Rgba> = Color<[u8; 4], Space>;

/// An RGB color with `f32` components.
pub type Color3f<Space = Rgb> = Color<[f32; 3], Space>;

/// An RGBA color with `f32` components.
pub type Color4f<Space = Rgba> = Color<[f32; 4], Space>;

/// Returns a new RGB color with `r`, `g`, and `b` components.
pub const fn rgb<Ch>(r: Ch, g: Ch, b: Ch) -> Color<[Ch; 3], Rgb> {
    Color([r, g, b], PhantomData)
}
/// Returns a new RGBA color with `r`, `g`, `b`, and `c` components.
pub const fn rgba<Ch>(r: Ch, g: Ch, b: Ch, a: Ch) -> Color<[Ch; 4], Rgba> {
    Color([r, g, b, a], PhantomData)
}

//
// Inherent impls
//

impl Color3 {
    #[inline]
    /// Returns a `u32` containing the component bytes of `self`
    /// in format `0x00_RR_GG_BB`.
    pub const fn to_rgb_u32(self) -> u32 {
        let [r, g, b] = self.0;
        u32::from_be_bytes([0x00, r, g, b])
    }
}

impl Color4 {
    #[inline]
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
}

impl Color3f {
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
        [r, g, b, 0xFF].into()
    }
    #[inline]
    fn to_u8(self) -> [u8; 3] {
        self.0.map(|c| (c.clamp(0.0, 1.0) * 255.0) as u8)
    }
}

impl Color4f {
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

    /// TODO
    pub fn to_hsl(&self) -> Color<R, Hsl> {
        todo!()
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

    /// TODO
    pub fn to_hsla(&self) -> Color<R, Hsla> {
        todo!()
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
    /// TODO
    pub fn to_rgb(&self) -> Color<R, Rgb> {
        todo!()
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
    /// TODO
    pub fn to_rgba(&self) -> Color<R, Rgba> {
        todo!()
    }
}

//
// Local trait impls
//

impl<Sp, const DIM: usize> Affine for Color<[u8; DIM], Sp>
where
    [i16; DIM]: Default,
{
    type Space = Sp;
    type Diff = Vector<[i16; DIM], Sp>;

    const DIM: usize = DIM;

    fn add(&self, other: &Self::Diff) -> Self {
        array::from_fn(|i| {
            (i16::from(self.0[i]) + other.0[i]).clamp(0, u8::MAX as i16) as u8
        })
        .into()
    }
    fn sub(&self, other: &Self) -> Self::Diff {
        array::from_fn(|i| i16::from(self.0[i]) - i16::from(other.0[i])).into()
    }
}

impl<Sp, const DIM: usize> Affine for Color<[f32; DIM], Sp>
where
    [f32; DIM]: Default,
{
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

impl<Sp, const DIM: usize> Linear for Color<[f32; DIM], Sp>
where
    [f32; DIM]: Default,
{
    type Scalar = f32;

    /// Returns the all-zeroes color (black).
    fn zero() -> Self {
        <[f32; DIM]>::default().into()
    }
    #[inline]
    fn neg(&self) -> Self {
        array::from_fn(|i| -self.0[i]).into()
    }
    #[inline]
    fn mul(&self, scalar: Self::Scalar) -> Self {
        array::from_fn(|i| self.0[i] * scalar).into()
    }
}

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
}
