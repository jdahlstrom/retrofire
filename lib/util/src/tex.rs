use math::{Angle, ApproxEq, lerp, Linear, vec::Vec4};
use math::Angle::Tau;

use crate::buf::Buffer;
use crate::color::{BLACK, BLUE, Color, GREEN, RED, rgb};

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
    mips: [usize; 4],
    buf: Buffer<Color>,
}

impl Texture {
    pub fn new(width: usize, data: &[Color]) -> Texture {
        assert!(width.is_power_of_two(), "width not power of two: {}", width);

        let len = data.len();
        let height = len / width;

        assert!(height.is_power_of_two(), "height not power of two: {}", height);
        assert_eq!(height * width, len, "{} x {} != {}", height, width, len);


        let mips = [0, len, len + len / 4, len + len / 4 + len / 16];
        let mut data = data.to_vec();
        data.resize(mips[2], RED);
        data.resize(mips[3], GREEN);
        data.resize(mips[3] + len / 64, BLUE);

        let mut gen_mip = |lvl: usize| {
            let from_w = width >> (lvl - 1);
            let from_h = height >> (lvl - 1);
            let to_w = from_w >> 1;
            let to_h = from_h >> 1;

            eprintln!("Generating mip {:?}=>{:?}", (from_w, from_h), (to_w, to_h));

            let (from_slice, to_slice) = data.split_at_mut(mips[lvl]);
            let from_slice = &from_slice[mips[lvl-1]..];

            for j in 0..to_h {
                for i in 0..to_w {
                    let from_idx = from_w * (2 * j) + (2 * i);
                    let to_idx = to_w * j + i;

                    let widen = |c: u32|
                        (c as u64 & 0xFF000000) << 24 |
                            (c as u64 & 0xFF0000) << 16 |
                            (c as u64 & 0xFF00) << 8 |
                            (c as u64 & 0xFF);

                    let from00 = widen(from_slice[from_idx.saturating_sub(from_w+1)].0);
                    let from01 = widen(from_slice[from_idx.saturating_sub(from_w)].0);
                    let from02 = widen(from_slice[from_idx.saturating_sub(from_w)+1].0);
                    let from10 = widen(from_slice[from_idx.saturating_sub(1)].0);
                    let from11 = widen(from_slice[from_idx].0);
                    let from12 = widen(from_slice[from_idx + 1].0);
                    let from20 = widen(from_slice[from_idx + from_w - 1].0);
                    let from21 = widen(from_slice[from_idx + from_w].0);
                    let from22 = widen(from_slice[from_idx + from_w + 1].0);

                    let from = from00 + from01 + from02
                        + from10 + from11 + from12
                        + from20 + from21 + from22;

                    let from = 3 * from11 + 2 * from12 + 2 * from21 + from22;

                    let to = (from / 8) & 0xFF |
                        (((from >> 16) / 8 & 0xFF) << 8) |
                        (((from >> 32) / 8 & 0xFF) << 16) |
                        (((from >> 48) / 8 & 0xFF) << 24);

                    /*let from = from00 + 2 * from01 + from02
                        + 2 * from10 + 4 * from11 + 2 * from12
                        + from20 + 2 * from21 + from22;

                    let to =
                        ((from >> 4) & 0xFF) |
                        ((from >> 20 & 0xFF) << 8) |
                        ((from >> 36 & 0xFF) << 16) |
                        ((from >> 52 & 0xFF) << 24);*/

                    to_slice[to_idx].0 = to as u32;
                }
            }
        };
        /*gen_mip(1);
        gen_mip(2);
        gen_mip(3);*/

        eprintln!("Created mipped texture");
        eprintln!("  width={} height={}", width, height);
        eprintln!("  mip sizes={:?}", [len, len / 4, len / 16, len / 64]);
        eprintln!("  offsets={:?} orig len={} new len={}", mips, len, data.len());

        Texture {
            w: width as f32,
            h: height as f32,
            mips,
            buf: Buffer::from_vec(width, data),
        }
    }

    pub fn new_npot(width: usize, data: &[Color]) -> Texture {
        assert!(width % 16 == 0);
        let height = data.len() / width;
        assert!(height % 16 == 0);
        assert_eq!(height * width, data.len());

        Texture {
            w: width as f32,
            h: height as f32,
            mips: [0, 0, 0, 0],
            buf: Buffer::from_vec(width, data.to_vec()),
        }
    }

    pub fn solid(width: usize, height: usize, color: Color) -> Texture {
        assert!(width.is_power_of_two());
        assert!(height.is_power_of_two());

        Texture {
            w: width as f32,
            h: height as f32,
            mips: [0, 0, 0, 0],
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

    pub fn sample_mip(&self, TexCoord { u, v, w }: TexCoord, lvl: usize) -> Color {
        let width = (self.buf.width() >> lvl).max(1);
        let height = (self.height() as usize >> lvl).max(1);
        let data = &self.buf.data()[self.mips[lvl]..self.mips[lvl]+width*height];
        let w = 1.0 / w;
        let u = (width as f32 * u * w).floor() as isize as usize & (width - 1);
        let v = (height as f32 * v * w).floor() as isize as usize & (height - 1);

        // TODO enforce invariants and use get_unchecked
        *data.get(width * v + u)
            .unwrap_or_else(|| panic!(
                "data={:?} lvl={} offset={} len={}/{} w={}/{} h={}/{} u={} v={}",
                data, lvl, self.mips[lvl], self.buf.data().len(), data.len(),
                self.buf.width(), width, self.height(), height, u, v
            ))
    }

    pub fn sample_abs_npot(&self, TexCoord { u, v, w }: TexCoord) -> Color {
        let buf = &self.buf;
        let w = 1.0 / w;
        let u = ((u * w).floor() as isize as usize).rem_euclid(buf.width());
        let v = ((v * w).floor() as isize as usize).rem_euclid(buf.height());

        // TODO enforce invariants and use get_unchecked
        *buf.get(u, v)
    }

    pub fn sample_abs_clamped(&self, TexCoord { u, v, w }: TexCoord) -> Color {
        let buf = &self.buf;
        let w = 1.0 / w;
        let u = ((u * w).floor() as usize).clamp(0, buf.width() - 1);
        let v = ((v * w).floor() as usize).clamp(0, buf.height() - 1);

        // TODO enforce invariants and use get_unchecked
        *buf.get(u, v)
    }

    pub fn sample_abs_once(&self, TexCoord { u, v, w }: TexCoord) -> Color {
        let buf = &self.buf;
        let w = 1.0 / w;
        let u = (u * w).round() as isize;
        let v = (v * w).round() as isize;

        if u < 0 {
            BLUE
        } else if u >= buf.width() as isize {
            RED
        } else if v < 0 {
            rgb(0, 255, 255)
        } else if v >= buf.height() as isize {
            rgb(255, 255, 0)
        } else {
            // TODO enforce invariants and use get_unchecked
            *buf.get(u as usize, v as usize)
        }
    }

    pub fn sample_abs_npot_bilinear(&self, TexCoord { u, v, w }: TexCoord) -> Color {
        let buf = &self.buf;
        let w = 1.0 / w;
        let u = u * w;
        let v = v * w;
        let u_fl = u.floor();
        let v_fl = v.floor();
        let u0 = (u_fl as isize as usize).rem_euclid(buf.width());
        let v0 = (v_fl as isize as usize).rem_euclid(buf.height());
        let u1 = (u0 + 1).rem_euclid(buf.width());
        let v1 = (v0 + 1).rem_euclid(buf.height());
        let uf = u - u_fl;
        let vf = v - v_fl;

        // TODO enforce invariants and use get_unchecked
        let c00 = *buf.get(u0, v0);
        let c01 = *buf.get(u0, v1);
        let c10 = *buf.get(u1, v0);
        let c11 = *buf.get(u1, v1);

        let c1 = lerp(uf, c00, c10);
        let c2 = lerp(uf, c01, c11);

        lerp(vf, c1, c2)
    }
}

impl From<Buffer<Color>> for Texture {
    fn from(buf: Buffer<Color>) -> Self {
        Self {
            w: buf.width() as f32,
            h: buf.height() as f32,
            mips: [0, 0, 0, 0],
            buf,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Sampler<'a> {
    tex: &'a Texture,
}

impl<'a> Sampler<'a> {
    pub fn sample(uv: TexCoord) -> Color {
        todo!()
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
