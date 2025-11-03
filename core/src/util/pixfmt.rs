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

#[derive(Copy, Clone, Default)]
pub struct Rgba5551;

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

#[derive(Copy, Clone, Default)]
pub struct Rgb332;

#[derive(Copy, Clone, Default)]
pub struct Lum8;

// Impls for Color3

impl IntoPixel<u32, Rgb888> for Color3 {
    fn into_pixel(self) -> u32 {
        let [r, g, b] = self.0;
        // [0x00, 0xRR, 0xGG, 0xBB] -> 0x00_RR_GG_BB
        u32::from_be_bytes([0, r, g, b])
    }
}
impl IntoPixel<[u8; 3], Rgb888> for Color3 {
    fn into_pixel(self) -> [u8; 3] {
        self.0
    }
}

impl IntoPixel<u16, Rgb565> for Color3 {
    /// Packs `self` into a `u16` in `0bRRRRR_GGGGGG_BBBBB` format.
    fn into_pixel(self) -> u16 {
        let [r, g, b] = self.0;
        (r as u16 >> 3 & 0x1F) << 11
            | (g as u16 >> 2 & 0x3F) << 5
            | (b as u16 >> 3 & 0x1F)
    }
}

impl IntoPixel<[u8; 2], Rgb565> for Color3 {
    /// Packs `self` into two bytes in `[0bGGG_BBBBB, 0bRRRRR_GGG]` format.
    // TODO is this correct?
    fn into_pixel(self) -> [u8; 2] {
        let c: u16 = self.into_pixel();
        c.to_ne_bytes()
    }
}

impl IntoPixel<u8, Rgb332> for Color3 {
    /// Packs `self` into a single byte in `0bRRR_GGG_BB` format.
    fn into_pixel(self) -> u8 {
        let [r, g, b] = self.0;
        (r & 0b111_000_00) | (g >> 3 & 0b000_111_00) | (b >> 6)
    }
}

// Impls for Color4
/*
impl<F> IntoPixel<u32, F> for Color4
where
    Self: IntoPixel<[u8; 4], F>,
{
    fn into_pixel(self) -> u32 {
        // From [0xAA, 0xBB, 0xCC, 0xDD] to 0xAA_BB_CC_DD -> big-endian!
        u32::from_be_bytes(self.into_pixel())
    }
}*/

impl<T, F> IntoPixel<T, F> for Color4
where
    Color3: IntoPixel<T, F>,
{
    fn into_pixel(self) -> T {
        self.to_rgb().into_pixel()
    }
}

impl IntoPixel<u32, Xrgb8888> for Color4 {
    fn into_pixel(self) -> u32 {
        let [r, g, b, _] = self.0;
        // From [0x00, 0xRR, 0xGG, 0xBB] to 0x00_RR_GG_BB -> big-endian!
        u32::from_be_bytes([0, r, g, b])
    }
}
impl IntoPixel<[u8; 4], Rgba8888> for Color4 {
    fn into_pixel(self) -> [u8; 4] {
        self.0
    }
}
impl IntoPixel<[u8; 4], Argb8888> for Color4 {
    fn into_pixel(self) -> [u8; 4] {
        let [r, g, b, a] = self.0;
        [a, r, g, b]
    }
}
impl IntoPixel<[u8; 4], Bgra8888> for Color4 {
    fn into_pixel(self) -> [u8; 4] {
        let [r, g, b, a] = self.0;
        [b, g, r, a]
    }
}
/*impl IntoPixel<[u8; 3], Rgb888> for Color4 {
    fn into_pixel(self) -> [u8; 3] {
        [self.r(), self.g(), self.b()]
    }
}*/
impl IntoPixel<[u8; 2], Rgba4444> for Color4 {
    fn into_pixel(self) -> [u8; 2] {
        let c: u16 = self.into_pixel_fmt(Rgba4444);
        c.to_ne_bytes()
    }
}
impl IntoPixel<u16, Rgba4444> for Color4 {
    fn into_pixel(self) -> u16 {
        let [r, g, b, a] = self.0.map(|c| c as u16 >> 4);
        // [0xBA, 0xRG] in little-endian
        r << 12 | g << 8 | b << 4 | a
    }
}
/*impl IntoPixel<u16, Rgb565> for Color4 {
    fn into_pixel(self) -> u16 {
        self.to_rgb().into_pixel()
    }
}
impl IntoPixel<[u8; 2], Rgb565> for Color4 {
    fn into_pixel(self) -> [u8; 2] {
        let c: u16 = self.into_pixel_fmt(Rgb565);
        c.to_ne_bytes()
    }
}*/
impl IntoPixel<u16, Rgba5551> for Color4 {
    /// Packs `self` into a `u16` in a `0bRRRRR_GGGGG_BBBBB_A` format.
    /// The alpha value `0xFF` is considered opaque, any other value
    /// fully transparent.
    fn into_pixel(self) -> u16 {
        let [r, g, b, a] = self.0;
        (r as u16 >> 3 & 0x1F) << 11
            | (g as u16 >> 3 & 0x1F) << 6
            | (b as u16 >> 3 & 0x1F) << 1
            | (a == 0xFF) as u16
    }
}
impl IntoPixel<[u8; 2], Rgba5551> for Color4 {
    fn into_pixel(self) -> [u8; 2] {
        let c: u16 = self.into_pixel_fmt(Rgba5551);
        c.to_ne_bytes()
    }
}

impl IntoPixel<[u8; 1], Rgb332> for Color4 {
    /// Packs `self` into a single byte in `0bRRR_GGG_BB` format.
    fn into_pixel(self) -> [u8; 1] {
        [self.to_rgb().into_pixel()]
    }
}

#[cfg(test)]
mod tests {
    use crate::math::{rgb, rgba};

    use super::*;

    const COL3: Color3 = rgb(0x11u8, 0x22, 0x33);

    #[test]
    fn color3_to_rgb888() {
        let pix: u32 = COL3.into_pixel_fmt(Rgb888);
        assert_eq!(pix, 0x00_11_22_33);
    }

    #[test]
    fn color3_to_rgb565() {
        let pix: u16 = rgb(0x40, 0x20, 0x10).into_pixel();
        assert_eq!(pix, 0b01000_001000_00010_u16);

        let pix: [u8; 2] = rgb(0x40u8, 0x20, 0x10).into_pixel_fmt(Rgb565);
        assert_eq!(pix, [0b000_00010, 0b01000_001]);
    }

    #[test]
    fn color3_to_rgb332() {
        let pix: u8 = rgb(0xFF, 0x00, 0xFF).into_pixel();
        assert_eq!(pix, 0b111_000_11, "bits: {pix:b}");

        let pix: u8 = rgb(0x00, 0x0FF, 0x00).into_pixel();
        assert_eq!(pix, 0b000_111_00, "bits: {pix:b}");
    }

    #[test]
    fn color4_to_rgba8888() {
        let col = rgba(0x11u8, 0x22, 0x33, 0x44);

        let pix: u32 = col.into_pixel_fmt(Rgba8888);
        assert_eq!(pix, 0x11_22_33_44);

        let pix: [u8; 4] = col.into_pixel_fmt(Rgba8888);
        assert_eq!(pix, [0x11, 0x22, 0x33, 0x44]);
    }

    const COL4: Color4 = rgba(0x11u8, 0x22, 0x33, 0x44);

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

    #[test]
    fn color4_to_rgba5551_u16() {
        let pix: u16 = rgba(0x40u8, 0x20, 0x10, 0).into_pixel_fmt(Rgba5551);
        assert_eq!(pix, 0b01000_00100_00010_0_u16, "bits: {pix:b}");
        let pix: u16 = rgba(0x40u8, 0x20, 0x10, 0x80).into_pixel_fmt(Rgba5551);
        assert_eq!(pix, 0b01000_00100_00010_0_u16, "bits: {pix:b}");
        let pix: u16 = rgba(0x40u8, 0x20, 0x10, 0xFF).into_pixel_fmt(Rgba5551);
        assert_eq!(pix, 0b01000_00100_00010_1_u16, "bits: {pix:b}");
    }

    #[test]
    fn color4_to_rgba5551_2u8() {
        let pix: [u8; 2] = rgba(0x40u8, 0x20, 0x10, 0).into_pixel_fmt(Rgba5551);
        assert_eq!(pix, [0b00_00010_0, 0b01000_001]);
        let pix: [u8; 2] =
            rgba(0x40u8, 0x20, 0x10, 0xFF).into_pixel_fmt(Rgba5551);
        assert_eq!(pix, [0b00_00010_1, 0b01000_001]);
    }
}
