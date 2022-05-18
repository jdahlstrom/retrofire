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
    data: D,
}

impl<D: Deref<Target=[Color]>> Texture<D> {
    #[inline]
    pub fn width(&self) -> f32 {
        self.w
    }
    #[inline]
    pub fn height(&self) -> f32 {
        self.h
    }
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Color {
        self.data[self.w as usize * y + x]
    }
}

impl<'a> Texture<&'a [Color]> {
    pub fn borrow(width: usize, height: usize, data: &'a [Color]) -> Self {
        debug_assert!(height * width <= data.len());

        Self {
            w: width as f32,
            h: height as f32,
            data: data,
        }
    }
}

impl Texture<Vec<Color>> {
    pub fn owned(width: usize, height: usize, data: &[Color]) -> Self {
        debug_assert!(height * width <= data.len());

        Self {
            w: width as f32,
            h: height as f32,
            data: data.to_vec(),
        }
    }

    pub fn solid(width: usize, height: usize, color: Color) -> Self {
        Self {
            w: width as f32,
            h: height as f32,
            data: vec![color; width * height],
        }
    }
}

impl From<Buffer<Color>> for Texture<Vec<Color>> {
    fn from(buf: Buffer<Color>) -> Self {
        Self {
            w: buf.width() as f32,
            h: buf.height() as f32,
            data: buf.into_data(),
        }
    }
}

pub struct SamplerRepeatPot { w_mask: usize, h_mask: usize }

impl SamplerRepeatPot {
    pub fn new(tex: &Texture<impl Deref<Target=[Color]>>) -> Self {
        let w = tex.width() as usize;
        let h = tex.height() as usize;
        assert!(w.is_power_of_two(), "width must be a power of two, was {w}");
        assert!(h.is_power_of_two(), "height must be a power of two, was {h}");
        Self {
            w_mask: w - 1,
            h_mask: h - 1,
        }
    }

    pub fn sample(
        &self,
        tex: &Texture<impl Deref<Target=[Color]>>,
        uv: TexCoord
    ) -> Color {
        let u = (tex.width() * uv.u).floor() as isize as usize & self.w_mask;
        let v = (tex.height() * uv.v).floor() as isize as usize & self.h_mask;

        // TODO enforce invariants and use get_unchecked
        tex.data[(self.w_mask + 1) * v + u]
    }

    pub fn sample_abs(
        &self,
        tex: &Texture<impl Deref<Target=[Color]>>,
        uv: TexCoord
    ) -> Color {
        let u = (uv.u).floor() as isize as usize & self.w_mask;
        let v = (uv.v).floor() as isize as usize & self.h_mask;

        // TODO enforce invariants and use get_unchecked
        tex.data[(self.w_mask + 1) * v + u]
    }
}

pub struct SamplerClamp;

impl SamplerClamp {
    pub fn sample(
        &self,
        tex: &Texture<impl Deref<Target=[Color]>>,
        uv: TexCoord
    ) -> Color {
        let u = ((tex.width() - 1.0) * uv.u.clamp(0.0, 1.0)).floor() as isize as usize;
        let v = ((tex.height() - 1.0) * uv.v.clamp(0.0, 1.0)).floor() as isize as usize;

        // TODO enforce invariants and use get_unchecked
        tex.data[tex.width() as usize * v + u]
    }
}

pub struct SamplerOnce;

impl SamplerOnce {

    pub fn sample(
        &self,
        tex: &Texture<impl Deref<Target=[Color]>>,
        uv: TexCoord
    ) -> Option<Color> {
        let u = (tex.width() * uv.u).floor() as isize as usize;
        let v = (tex.height() * uv.v).floor() as isize as usize;

        debug_assert!(u < tex.width() as usize);
        debug_assert!(v < tex.height() as usize);

        // TODO enforce invariants and use get_unchecked
        tex.data.get(tex.width() as usize * v + u).copied()
    }
}
