use math::{Angle, ApproxEq, Linear, vec::Vec4};
use math::Angle::Tau;

use crate::buf::Buffer;
use crate::color::Color;

#[derive(Copy, Clone, Debug, Default)]
pub struct TexCoord {
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

pub const fn uv(u: f32, v: f32) -> TexCoord {
    TexCoord { u, v, w: 1.0 }
}

pub const U: TexCoord = uv(1.0, 0.0);
pub const V: TexCoord = uv(0.0, 1.0);

impl ApproxEq for TexCoord {
    type Scalar = f32;

    fn epsilon(self) -> f32 {
        1e-6
    }

    fn abs_diff(self, rhs: Self) -> f32 {
        self.u.abs_diff(rhs.u)
            + self.v.abs_diff(rhs.v)
    }
}

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
}

#[derive(Clone)]
pub struct Texture {
    w: f32,
    h: f32,
    buf: Buffer<Color>,
}

impl Texture {
    pub fn new(width: usize, data: &[Color]) -> Texture {
        assert!(width.is_power_of_two());
        let height = data.len() / width;
        assert!(height.is_power_of_two());
        assert_eq!(height * width, data.len());

        Texture {
            w: width as f32,
            h: height as f32,
            buf: Buffer::from_vec(width, data.to_vec()),
        }
    }

    pub fn solid(width: usize, height: usize, color: Color) -> Texture {
        assert!(width.is_power_of_two());
        assert!(height.is_power_of_two());

        Texture {
            w: width as f32,
            h: height as f32,
            buf: Buffer::new(width, height, color),
        }
    }

    pub fn width(&self) -> f32 {
        self.w
    }
    pub fn height(&self) -> f32 {
        self.h
    }

    pub fn sample(&self, TexCoord { u, v, w }: TexCoord) -> Color {
        let buf = &self.buf;
        let w = 1.0 / w;
        let u = (self.w * u * w).floor() as isize as usize & (buf.width() - 1);
        let v = (self.h * v * w).floor() as isize as usize & (buf.height() - 1);

        // TODO enforce invariants and use get_unchecked
        *buf.get(u, v)
    }
}

impl From<Buffer<Color>> for Texture {
    fn from(buf: Buffer<Color>) -> Self {
        Self {
            w: buf.width() as f32,
            h: buf.height() as f32,
            buf,
        }
    }
}

pub fn map_planar(basis_u: Vec4, basis_v: Vec4, pt: Vec4) -> TexCoord {
    let u = basis_u.dot(pt);
    let v = basis_v.dot(pt);
    uv(u, v)
}

pub fn map_spherical(Vec4 { x, y, z, .. }: Vec4) -> TexCoord {
    let u = if x == 0.0 && x.is_sign_negative() {
        1.0
    } else {
        Angle::atan2(x, z).wrap(Tau(0.0), Tau(1.0)).as_tau()
    };
    let v = 2.0 * Angle::atan2((x * x + z * z).sqrt(), y)
        .wrap(Tau(0.0), Tau(1.0)).as_tau();

    uv(u, v)
}

pub fn map_hemispheric(Vec4 { x, y, z, .. }: Vec4) -> TexCoord {
    let u = 2.0 * Angle::atan2(x.abs(), z).wrap(Tau(0.0), Tau(1.0)).as_tau();
    let v = 2.0 * Angle::atan2((x * x + z * z).sqrt(), y).as_tau();

    uv(u, v)
}

pub fn map_cylindrical(v: Vec4) -> TexCoord {
    let u = Angle::atan2(v.x, v.z).as_tau().rem_euclid(1.0);
    let v = v.y;
    uv(u, v)
}

pub fn map_cube(v: Vec4) -> TexCoord {
    let Vec4 { x, y, z, .. } = v.map(f32::abs);

    if x > y && x > z {
        uv(v.y, v.z)
    } else if y > x && y > z {
        uv(v.x, v.z)
    } else {
        uv(v.x, v.y)
    }
}

#[cfg(test)]
mod tests {
    use math::test_util::*;
    use math::vec::*;

    use super::*;

    #[test]
    fn planar_map_zero() {
        assert!(uv(0.0, 0.0).approx_eq(map_planar(X, Y, ZERO)));
        assert!(uv(0.0, 0.0).approx_eq(map_planar(X, Y, ORIGIN)));
    }

    #[test]
    fn planar_map_zero_with_offset() {
        assert_approx_eq(uv(0.0, 0.0), map_planar(X + 2.0 * W, Y - 3.0 * W, ZERO));
        assert_approx_eq(uv(2.0, -3.0), map_planar(X + 2.0 * W, Y - 3.0 * W, ORIGIN));
    }

    #[test]
    fn spherical_map_x() {
        assert_approx_eq(uv(0.25, 0.5), map_spherical(pt(1.0, 0.0, 0.0)));
        assert_approx_eq(uv(0.75, 0.5), map_spherical(pt(-1.0, 0.0, 0.0)));
    }

    #[test]
    fn spherical_map_y() {
        assert_approx_eq(uv(0.0, 0.0), map_spherical(pt(0.0, 1.0, 0.0)));
        assert_approx_eq(uv(0.5, 0.0), map_spherical(pt(0.0, 1.0, -0.0)));
        assert_approx_eq(uv(0.0, 1.0), map_spherical(pt(0.0, -1.0, 0.0)));
        assert_approx_eq(uv(0.5, 1.0), map_spherical(pt(0.0, -1.0, -0.0)));
    }

    #[test]
    fn spherical_map_z() {
        assert_approx_eq(uv(0.0, 0.5), map_spherical(pt(0.0, 0.0, 1.0)));
        assert_approx_eq(uv(0.5, 0.5), map_spherical(pt(0.0, 0.0, -1.0)));
    }
}
