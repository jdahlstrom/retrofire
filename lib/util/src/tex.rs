use std::ops::Deref;
use math::Linear;

use crate::buf::Buffer;
use crate::color::Color;

#[derive(Copy, Clone, Debug, Default)]
pub struct TexCoord {
    pub u: f32, pub v: f32, pub w: f32
}

pub const fn uv(u: f32, v: f32) -> TexCoord {
    TexCoord { u, v, w: 1.0 }
}

pub const U: TexCoord = uv(1.0, 0.0);
pub const V: TexCoord = uv(0.0, 1.0);

impl Linear<f32> for TexCoord {
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            u: self.u + other.u,
            v: self.v + other.v,
            w: self.w + other.w,
        }
    }
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self {
            u: s * self.u,
            v: s * self.v,
            w: s * self.w,
        }
    }
    #[inline]
    fn neg(self) -> Self {
        Self {
            u: -self.u,
            v: -self.v,
            w: -self.w,
        }
    }

    fn perspective_div(self) -> Self {
        let w = 1.0 / self.w;
        uv(w * self.u, w * self.v)
    }
}

#[derive(Clone)]
pub struct Texture<D = Vec<Color>> {
    w: f32,
    h: f32,
    w_mask: usize,
    h_mask: usize,
    data: D,
}

impl<D: Deref<Target=[Color]>> Texture<D> {
    pub fn width(&self) -> f32 {
        self.w
    }
    pub fn height(&self) -> f32 {
        self.h
    }

    pub fn sample(&self, TexCoord { u, v, w }: TexCoord) -> Color {
        let w = 1.0 / w;
        let u = (self.w * u * w).floor() as isize as usize & self.w_mask;
        let v = (self.h * v * w).floor() as isize as usize & self.h_mask;

        // TODO enforce invariants and use get_unchecked
        self.data[self.w as usize * v + u]
    }
}

impl<'a> Texture<&'a [Color]> {
    pub fn borrow(width: usize, height: usize, data: &'a [Color]) -> Self {
        assert!(width.is_power_of_two());
        assert!(height.is_power_of_two());
        assert!(height * width <= data.len());

        Self {
            w: width as f32,
            h: height as f32,
            w_mask: width - 1,
            h_mask: height - 1,
            data,
        }
    }
}

impl Texture<Vec<Color>> {
    pub fn owned(width: usize, height: usize, data: &[Color]) -> Self {
        assert!(width.is_power_of_two());
        assert!(height.is_power_of_two());
        assert!(height * width <= data.len());

        Self {
            w: width as f32,
            h: height as f32,
            w_mask: width - 1,
            h_mask: height - 1,
            data: data.to_vec(),
        }
    }

    pub fn solid(width: usize, height: usize, color: Color) -> Self {
        assert!(width.is_power_of_two());
        assert!(height.is_power_of_two());

        Texture {
            w: width as f32,
            h: height as f32,
            w_mask: width - 1,
            h_mask: height - 1,
            data: vec![color; width * height],
        }
    }
}

impl From<Buffer<Color>> for Texture<Vec<Color>> {
    fn from(buf: Buffer<Color>) -> Self {
        assert!(buf.width().is_power_of_two());
        assert!(buf.height().is_power_of_two());

        let w = buf.width();
        let h = buf.height();

        Self {
            w: w as f32,
            h: h as f32,
            w_mask: w - 1,
            h_mask: h - 1,
            data: buf.into_data(),
        }
    }
}
