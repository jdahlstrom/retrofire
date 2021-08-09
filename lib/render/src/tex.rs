use util::Buffer;
use math::Linear;
use util::color::Color;

#[derive(Copy, Clone, Debug)]
pub struct TexCoord {
    pub u: f32, pub v: f32, pub w: f32
}

pub const fn uv(u: f32, v: f32) -> TexCoord {
    TexCoord { u, v, w: 1.0 }
}

pub const U: TexCoord = uv(1.0, 0.0);
pub const V: TexCoord = uv(0.0, 1.0);

impl Linear<f32> for TexCoord {
    fn add(self, other: Self) -> Self {
        Self {
            u: self.u + other.u,
            v: self.v + other.v,
            w: self.w + other.w,
        }
    }

    fn mul(self, s: f32) -> Self {
        Self {
            u: s * self.u,
            v: s * self.v,
            w: s * self.w,
        }
    }

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
        let u = (self.w * u * w) as isize as usize & (buf.width() - 1);
        let v = (self.h * v * w) as isize as usize & (buf.height() - 1);

        // TODO enforce invariants and use get_unchecked
        *buf.get(u, v)
    }
}

impl From<Buffer<Color>> for Texture {
    fn from(buf: Buffer<Color>) -> Self {
        Self {
            w: buf.width() as f32,
            h: buf.height() as f32,
            buf
        }
    }
}
