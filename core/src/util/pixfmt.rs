use crate::math::color::{Color, Color3, Color4};
pub trait Fmt<T> {}

pub trait ToFmt<T, F: Fmt<T>> {
    fn write(&self, t: &mut T);

    fn to_fmt(&self, _: F) -> T
    where
        T: Sized + Default,
    {
        let mut to = Default::default();
        ToFmt::write(self, &mut to);
        to
    }
}

pub struct Rgb888;
pub struct Rgba8888;
pub struct Argb8888;

/// 0x00_RR_GG_BB
impl Fmt<u32> for Rgb888 {}
/// [0xRR, 0xGG, 0xBB]
impl Fmt<[u8; 3]> for Rgb888 {}
/// 0xRR_GG_BB_AA
impl Fmt<u32> for Rgba8888 {}
/// 0xAA_RR_GG_BB
impl Fmt<u32> for Argb8888 {}

impl ToFmt<u32, Rgb888> for Color3 {
    fn write(&self, t: &mut u32) {
        let [r, g, b] = self.0;
        *t = u32::from_be_bytes([0, r, g, b]);
    }
}
impl ToFmt<[u8; 3], Rgb888> for Color3 {
    fn write(&self, t: &mut [u8; 3]) {
        *t = self.0;
    }
}

impl ToFmt<u32, Rgb888> for Color4 {
    fn write(&self, t: &mut u32) {
        self.to_rgb().write(Rgb888, t)
    }
}
impl ToFmt<u32, Rgba8888> for Color4 {
    fn write(&self, t: &mut u32) {
        *t = u32::from_be_bytes(self.0);
    }
}
impl ToFmt<u32, Argb8888> for Color4 {
    fn write(&self, t: &mut u32) {
        let [r, g, b, a] = self.0;
        *t = u32::from_be_bytes([a, r, g, b]);
    }
}

impl<R, Sp> Color<R, Sp> {
    pub fn write<T, F>(&self, _: F, to: &mut T)
    where
        F: Fmt<T>,
        Self: ToFmt<T, F>,
    {
        ToFmt::write(self, to);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::color::{rgb, rgba};

    const RGB: Color3 = rgb(0x44, 0x88, 0xCC);
    const RGBA: Color4 = rgba(0x33, 0x66, 0x99, 0xCC);

    #[test]
    fn rgb_to_u32() {
        let actual: u32 = RGB.to_fmt(Rgb888);
        assert_eq!(actual, 0x00_44_88_CC);
    }
    #[test]
    fn rgba_to_u32() {
        assert_eq!(RGBA.to_fmt(Rgba8888), 0x33_66_99_CC);
        assert_eq!(RGBA.to_fmt(Argb8888), 0xCC_33_66_99);
    }
}
