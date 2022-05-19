use std::ops::Deref;

use math::{Angle, ApproxEq, Linear};
use math::vec::Vec4;

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

impl ApproxEq for TexCoord {
    type Scalar = f32;

    fn epsilon(self) -> f32 {
        1e-6
    }

    fn abs_diff(self, rhs: Self) -> f32 {
        let ud = rhs.u - self.u;
        let vd = rhs.v - self.v;
        let wd = rhs.w - self.w;
        (ud * ud + vd * vd + wd * wd).sqrt()
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

pub fn map_planar(pt: Vec4, basis_u: Vec4, basis_v: Vec4) -> TexCoord {
    let u = basis_u.dot(pt);
    let v = basis_v.dot(pt);
    uv(u, v)
}

pub fn map_spherical(pt: Vec4, ref_dir: Vec4) -> TexCoord {
    let (az, alt) = if pt.x.approx_eq(0.0) && pt.z.approx_eq(0.0) {
        let az = ref_dir.to_spherical().1;
        let alt = Angle::Tau(-0.25 * pt.y.signum());
        (az, alt)
    } else {
        let (_, az, alt) = pt.to_spherical();
        (az, alt)
    };

    let mut u = az.as_tau().rem_euclid(1.0);
    let fourth_quad = ref_dir.x.is_sign_negative() && ref_dir.z > 0.0;
    if u < 0.25 && fourth_quad { u += 1.0; }
    let v = 2.0 * alt.as_tau() + 0.5;

    uv(u, v)
}

pub fn map_cylindrical(mut pt: Vec4, ref_dir: Vec4) -> TexCoord {
    if pt.x.approx_eq(0.0) && pt.z.approx_eq(0.0) {
        pt.x = ref_dir.x;
        pt.z = ref_dir.z;
    }
    let (_, az) = pt.to_polar();
    let mut u = az.as_tau().rem_euclid(1.0);
    let fourth_quad = ref_dir.x.is_sign_negative() && ref_dir.z > 0.0;
    if u < 0.25 && fourth_quad { u += 1.0 }
    let v = pt.y * 0.5 + 0.5;

    uv(u, v)
}

pub fn map_cube(Vec4 { x, y, z, .. }: Vec4, n: Vec4) -> TexCoord {
    /*
        u=0   1/3  2/3   1
      v=0 +----+----+----+
          | +X | +Y | +Z |
          |    |    |    |
      1/2 +----+----+----+
          | -X | -Y | -Z |
          |    |    |    |
        1 +----+----+----+
    */

    let abs = n.map(f32::abs);
    let (u, v);
    if abs.x >= abs.y && abs.x >= abs.z {
        u = z / x;
        v = y / x + if n.x < 0.0 { 2.0 } else { 0.0 }
    } else if abs.y >= abs.x && abs.y >= abs.z {
        u = x / y + 2.0;
        v = z / y + if n.y < 0.0 { 2.0 } else { 0.0 }
    } else {
        u = x / z + 4.0;
        v = y / z + if n.z < 0.0 { 2.0 } else { 0.0 }
    };
    uv((u + 1.0) / 6.0, (v + 1.0) / 4.0)
}

pub fn map_cube_env(n: Vec4) -> TexCoord {
    uv(n.x * 0.5 + 0.5, n.y * 0.5 + 0.5)
}

#[cfg(test)]
mod tests {
    use math::test_util::*;
    use math::vec::*;

    use super::*;

    #[test]
    fn planar_map_zero() {
        assert!(uv(0.0, 0.0).approx_eq(map_planar(ZERO, X, Y)));
        assert!(uv(0.0, 0.0).approx_eq(map_planar(ORIGIN, X, Y)));
    }
    #[test]
    fn planar_map_zero_with_offset() {
        assert_approx_eq(uv(0.0, 0.0), map_planar(ZERO, X+2.0*W, Y-3.0*W));
        assert_approx_eq(uv(2.0, -3.0), map_planar(ORIGIN, X+2.0*W, Y-3.0*W));
    }

    #[test]
    fn spherical_map_x() {
        assert_approx_eq(uv(0.25, 0.5), map_spherical(X, X));
        assert_approx_eq(uv(0.75, 0.5), map_spherical(-X, X));
    }
    #[test]
    fn spherical_map_y() {
        assert_approx_eq(uv(0.0, 0.0), map_spherical(Y, Z));
        assert_approx_eq(uv(0.25, 0.0), map_spherical(Y, X));
        assert_approx_eq(uv(0.0, 1.0), map_spherical(-Y, Z));
        assert_approx_eq(uv(0.25, 1.0), map_spherical(-Y, X));
    }
    #[test]
    fn spherical_map_z() {
        assert_approx_eq(uv(0.0, 0.5), map_spherical(Z, Z));
        assert_approx_eq(uv(1.0, 0.5), map_spherical(Z, dir(-0.0, 0.0, 1.0)));
        assert_approx_eq(uv(0.5, 0.5), map_spherical(-Z, -Z));
        assert_approx_eq(uv(0.5, 0.5), map_spherical(-Z, -Z));
    }

    #[test]
    fn cube_map_x() {
        // TODO
        dbg!(map_cube(X+Y-Z, X));
        dbg!(map_cube(X+Y+Z, X));
        dbg!(map_cube(X-Y-Z, X));
        dbg!(map_cube(X-Y+Z, X));
    }
}
