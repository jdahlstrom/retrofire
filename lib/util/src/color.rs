use math::Linear;
use math::vec::Vec4;
use std::ops::Mul;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct Color(pub u32);

pub const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color(u32::from_be_bytes([0, r, g, b]))
}

pub const fn gray(val: u8) -> Color {
    rgb(val, val, val)
}

pub const BLACK: Color = rgb(0, 0, 0);
pub const RED: Color = rgb(255, 0, 0);
pub const GREEN: Color = rgb(0, 255, 0);
pub const BLUE: Color = rgb(0, 0, 255);
pub const WHITE: Color = rgb(255, 255, 255);

impl Color {
    pub fn r(&self) -> u8 { self.to_argb()[1] }
    pub fn g(&self) -> u8 { self.to_argb()[2] }
    pub fn b(&self) -> u8 { self.to_argb()[3] }

    pub fn to_argb(&self) -> [u8; 4] {
        self.0.to_be_bytes()
    }
}

impl Mul<Color> for Color {
    type Output = Color;

    fn mul(self, rhs: Color) -> Color {
        let r = (self.r() as u16 * rhs.r() as u16) >> 8;
        let g = (self.g() as u16 * rhs.g() as u16) >> 8;
        let b = (self.b() as u16 * rhs.b() as u16) >> 8;
        rgb(r as u8, g as u8, b as u8)
    }
}

impl Mul<u8> for Color {
    type Output = Color;

    fn mul(self, rhs: u8) -> Color {
        self * gray(rhs)
    }
}

// TODO This impl is fundamentally broken
impl Linear<f32> for Color {
    fn add(self, other: Self) -> Self {
        rgb(self.r().saturating_add(other.r()),
            self.g().saturating_add(other.g()),
            self.b().saturating_add(other.b()))
    }

    fn mul(self, s: f32) -> Self {
        rgb((self.r() as f32 * s) as u8,
            (self.g() as f32 * s) as u8,
            (self.b() as f32 * s) as u8)
    }

    fn neg(self) -> Self {
        rgb(self.r().wrapping_neg(),
            self.g().wrapping_neg(),
            self.b().wrapping_neg())
    }
}

impl From<Vec4> for Color {
    fn from(v: Vec4) -> Color {
        let c = 255. * v;
        rgb(c.x as u8, c.y as u8, c.z as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_components() {
        assert_eq!(0xFF, RED.r());
        assert_eq!(0xFF, GREEN.g());
        assert_eq!(0xFF, BLUE.b());
    }

    #[test]
    fn color_from_u32() {
        assert_eq!(0x23, Color(0x01234567).r());
        assert_eq!(0x45, Color(0x01234567).g());
        assert_eq!(0x67, Color(0x01234567).b());
    }

    #[test]
    fn color_from_rgb() {
        assert_eq!(Color(0x012345), rgb(0x01, 0x23, 0x45));
    }

    #[test]
    fn color_to_argb() {
        assert_eq!([0x01, 0x23, 0x45, 0x67], Color(0x01234567).to_argb());
        assert_eq!([0x00, 0x01, 0x23, 0x45], rgb(0x01, 0x23, 0x45).to_argb());
    }
}
