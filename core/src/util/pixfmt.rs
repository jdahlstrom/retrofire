use crate::math::{Color3, Color4};

// TODO Do we need the trait at all?
// pub trait PixelFmt {}

pub trait IntoPixel<T, F>: Sized {
    /// Converts `self` to `T` in format `F`.
    fn into_pixel(self) -> T;

    /// Converts `self` to `T`, taking an `F` to help type inference.
    ///
    /// This can be used to avoid the awkward fully-qualified syntax
    /// `IntoPixel::<_, F>::into_pixel(self)`.
    fn into_pixel_fmt(self, _: F) -> T {
        self.into_pixel()
    }
}

/// Eight-bit channels in R,G,B order.
#[derive(Copy, Clone, Default)]
pub struct Rgb888;
/// 5,6,5-bit channels in R,G,B order.
#[derive(Copy, Clone, Default)]
pub struct Rgb565;

/// Eight-bit channels in X,R,G,B order, where X is unused.
#[derive(Copy, Clone, Default)]
pub struct Xrgb8888;
/// Eight-bit channels in R,G,B,A order.
#[derive(Copy, Clone, Default)]
pub struct Rgba8888;
/// Eight-bit channels in A,R,G,B order.
#[derive(Copy, Clone, Default)]
pub struct Argb8888;
/// Eight-bit channels in B,G,R,A order.
#[derive(Copy, Clone, Default)]
pub struct Bgra8888;

/// Four-bit channels in R,G,B,A order.
#[derive(Copy, Clone, Default)]
pub struct Rgba4444;

// Impls for Color3

impl IntoPixel<u32, Xrgb8888> for Color3 {
    /// Converts `self` to a `u32` in 0x00_RR_GG_BB format.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{rgb, Color3};
    /// use retrofire_core::util::pixfmt::{IntoPixel, Xrgb8888};
    ///
    /// let color: Color3 = rgb(0x33, 0x66, 0x99);
    ///
    /// assert_eq!(color.into_pixel_fmt(Xrgb8888), 0x00_33_66_99);
    /// ```
    #[inline]
    fn into_pixel(self) -> u32 {
        let [r, g, b] = self.0;
        // [0x00, 0xRR, 0xGG, 0xBB] -> 0x00_RR_GG_BB
        u32::from_be_bytes([0, r, g, b])
    }
}
impl IntoPixel<[u8; 3], Rgb888> for Color3 {
    /// Converts `self` to (0xRR, 0xGG, 0xBB) bytes.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{rgb, Color3};
    /// use retrofire_core::util::pixfmt::{IntoPixel, Rgb888};
    ///
    /// let color: Color3 = rgb(0x33, 0x66, 0x99);
    ///
    /// assert_eq!(color.into_pixel_fmt(Rgb888), [0x33, 0x66, 0x99]);
    /// ```
    #[inline]
    fn into_pixel(self) -> [u8; 3] {
        self.0
    }
}

impl IntoPixel<u16, Rgb565> for Color3 {
    /// Converts `self` to a `u16` in 0bRRRRR_GGGGGG_BBBBB format.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{rgb, Color3};
    /// use retrofire_core::util::pixfmt::{IntoPixel, Rgb565};
    ///
    /// let color: Color3 = rgb(0x80, 0x40, 0x20);
    ///
    /// let word_565: u16 = color.into_pixel_fmt(Rgb565);
    /// assert_eq!(word_565, 0x8204); // 0b10000_010000_00100
    /// ```
    #[inline]
    fn into_pixel(self) -> u16 {
        let [r, g, b] = self.0;
        (r as u16 >> 3 & 0x1F) << 11
            | (g as u16 >> 2 & 0x3F) << 5
            | (b as u16 >> 3 & 0x1F)
    }
}

impl IntoPixel<[u8; 2], Rgb565> for Color3 {
    #[inline]
    fn into_pixel(self) -> [u8; 2] {
        let c: u16 = self.into_pixel();
        c.to_ne_bytes()
    }
}

// Impls for Color4

impl<F> IntoPixel<u32, F> for Color4
where
    Self: IntoPixel<[u8; 4], F>,
{
    #[inline]
    fn into_pixel(self) -> u32 {
        // From [0xAA, 0xBB, 0xCC, 0xDD] to 0xAA_BB_CC_DD -> big-endian!
        u32::from_be_bytes(self.into_pixel())
    }
}

impl IntoPixel<u32, Xrgb8888> for Color4 {
    /// Converts `self` to a `u32` in 0x0RGB format, discarding the value of
    /// the alpha channel.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{rgba, Color4};
    /// use retrofire_core::util::pixfmt::{IntoPixel, Xrgb8888};
    ///
    /// let color: Color4 = rgba(0x11, 0x22, 0x33, 0x44);
    ///
    /// assert_eq!(color.into_pixel_fmt(Xrgb8888), 0x00_11_22_33);
    /// ```
    #[inline]
    fn into_pixel(self) -> u32 {
        let [r, g, b, _] = self.0;
        // From [0x00, 0xRR, 0xGG, 0xBB] to 0x00_RR_GG_BB -> big-endian!
        u32::from_be_bytes([0, r, g, b])
    }
}
impl IntoPixel<[u8; 4], Rgba8888> for Color4 {
    /// Converts `self` to (0xRR, 0xGG, 0xBB, 0xAA) bytes.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{rgba, Color4};
    /// use retrofire_core::util::pixfmt::{IntoPixel, Rgba8888};
    ///
    /// let color: Color4 = rgba(0x11, 0x22, 0x33, 0x44);
    /// let rgba: [u8; 4] = color.into_pixel_fmt(Rgba8888);
    ///
    /// assert_eq!(rgba, [0x11, 0x22, 0x33, 0x44]);
    /// ```
    #[inline]
    fn into_pixel(self) -> [u8; 4] {
        self.0
    }
}
impl IntoPixel<[u8; 4], Argb8888> for Color4 {
    /// Converts `self` to (0xAA, 0xRR, 0xGG, 0xBB) bytes.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{rgba, Color4};
    /// use retrofire_core::util::pixfmt::{Argb8888, IntoPixel};
    ///
    /// let color: Color4 = rgba(0x11, 0x22, 0x33, 0x44);
    /// let argb: [u8; 4] = color.into_pixel_fmt(Argb8888);
    ///
    /// assert_eq!(argb, [0x44, 0x11, 0x22, 0x33]);
    /// ```
    #[inline]
    fn into_pixel(self) -> [u8; 4] {
        let [r, g, b, a] = self.0;
        [a, r, g, b]
    }
}
impl IntoPixel<[u8; 4], Bgra8888> for Color4 {
    /// Converts `self` to (0xBB, 0xGG, 0xRR, 0xAA) bytes.
    ///
    /// # Examples
    /// ```
    /// use retrofire_core::math::{rgba, Color4};
    /// use retrofire_core::util::pixfmt::{Bgra8888, IntoPixel};
    ///
    /// let color: Color4 = rgba(0x11, 0x22, 0x33, 0x44);
    /// let bgra: [u8; 4] = color.into_pixel_fmt(Bgra8888);
    ///
    /// assert_eq!(bgra, [0x33, 0x22, 0x11, 0x44]);
    /// ```
    #[inline]
    fn into_pixel(self) -> [u8; 4] {
        let [r, g, b, a] = self.0;
        [b, g, r, a]
    }
}
impl IntoPixel<[u8; 3], Rgb888> for Color4 {
    /// Converts `self` to bytes in RGB order, discarding alpha.
    #[inline]
    fn into_pixel(self) -> [u8; 3] {
        [self.r(), self.g(), self.b()]
    }
}
impl IntoPixel<[u8; 2], Rgba4444> for Color4 {
    /// Converts `self` to two bytes, one nibble (four bits) per channel in RGBA order.
    #[inline]
    fn into_pixel(self) -> [u8; 2] {
        let c: u16 = self.into_pixel_fmt(Rgba4444);
        c.to_ne_bytes()
    }
}
impl IntoPixel<u16, Rgba4444> for Color4 {
    /// Converts `self` to a `u16`, one nibble (four bits) per channel in RGBA order.
    #[inline]
    fn into_pixel(self) -> u16 {
        let [r, g, b, a] = self.0;

        // [0xBA, 0xRG] in little-endian
        (r as u16 >> 4) << 12
            | (g as u16 >> 4) << 8
            | (b as u16 >> 4) << 4
            | (a as u16 >> 4)
    }
}
impl IntoPixel<u16, Rgb565> for Color4 {
    #[inline]
    fn into_pixel(self) -> u16 {
        self.to_rgb().into_pixel()
    }
}
impl IntoPixel<[u8; 2], Rgb565> for Color4 {
    #[inline]
    fn into_pixel(self) -> [u8; 2] {
        let c: u16 = self.into_pixel_fmt(Rgb565);
        c.to_ne_bytes()
    }
}

#[cfg(test)]
#[allow(clippy::unusual_byte_groupings)]
mod tests {
    use super::*;
    use crate::math::{color::hex, rgb};

    const COL3: Color3 = hex("#112233");

    #[test]
    fn color3_to_rgb888() {
        let pix: u32 = COL3.into_pixel_fmt(Xrgb8888);
        assert_eq!(pix, 0x00_11_22_33);
    }

    #[test]
    fn color3_to_rgb565() {
        let pix: u16 = rgb(0x40, 0x20, 0x10).into_pixel();
        assert_eq!(pix, 0b01000_001000_00010_u16);

        let pix: [u8; 2] = rgb(0x40u8, 0x20, 0x10).into_pixel_fmt(Rgb565);
        assert_eq!(pix, [0b000_00010, 0b01000_001]);
    }

    const COL4: Color4 = hex("#11223344");

    #[test]
    fn color4_to_rgba8888() {
        let pix: u32 = COL4.into_pixel_fmt(Rgba8888);
        assert_eq!(pix, 0x11_22_33_44);

        let pix: [u8; 4] = COL4.into_pixel_fmt(Rgba8888);
        assert_eq!(pix, [0x11, 0x22, 0x33, 0x44]);
    }

    #[test]
    fn color4_to_argb8888() {
        let pix: u32 = COL4.into_pixel_fmt(Argb8888);
        assert_eq!(pix, 0x44_11_22_33);

        let pix: [u8; 4] = COL4.into_pixel_fmt(Argb8888);
        assert_eq!(pix, [0x44, 0x11, 0x22, 0x33]);
    }

    #[test]
    fn color4_to_bgra8888() {
        let pix: u32 = COL4.into_pixel_fmt(Bgra8888);
        assert_eq!(pix, 0x33_22_11_44);

        let pix: [u8; 4] = COL4.into_pixel_fmt(Bgra8888);
        assert_eq!(pix, [0x33, 0x22, 0x11, 0x44]);
    }

    #[test]
    fn color4_to_rgba4444() {
        let pix: [u8; 2] = COL4.into_pixel_fmt(Rgba4444);
        assert_eq!(pix, [0x34, 0x12]);

        let pix: u16 = COL4.into_pixel_fmt(Rgba4444);
        assert_eq!(pix, 0x1234);
    }
}
