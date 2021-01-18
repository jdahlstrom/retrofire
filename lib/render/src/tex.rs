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
        TexCoord {
            u: self.u + other.u,
            v: self.v + other.v,
            w: self.w + other.w,
        }
    }

    fn mul(self, s: f32) -> Self {
        TexCoord {
            u: s * self.u,
            v: s * self.v,
            w: s * self.w,
        }
    }
}

#[derive(Clone)]
pub struct Texture {
    width: f32,
    height: f32,
    buf: Buffer<Color>,
}

impl Texture {
    pub fn new(width: usize, data: &[Color]) -> Texture {
        assert!(width.is_power_of_two());
        let height = data.len() / width;
        assert!(height.is_power_of_two());
        assert_eq!(height * width, data.len());

        Texture {
            width: width as f32,
            height: height as f32,
            buf: Buffer {
                width, height,
                data: data.to_vec(),
            }
        }
    }

    pub fn solid(width: usize, height: usize, color: Color) -> Texture {
        assert!(width.is_power_of_two());
        assert!(height.is_power_of_two());

        Texture {
            width: width as f32,
            height: height as f32,
            buf: Buffer::new(width, height, color),
        }
    }

    pub fn sample(&self, TexCoord { u, v, w }: TexCoord) -> Color {
        let buf = &self.buf;
        let w = 1.0 / w;
        let u = (self.width * u * w) as usize & (buf.width - 1);
        let v = (self.height * v * w) as usize & (buf.height - 1);

        // TODO enforce invariants and use get_unchecked
        buf.data[buf.width * v + u]
    }
}

impl From<Buffer<Color>> for Texture {
    fn from(buf: Buffer<Color>) -> Self {
        Self {
            width: buf.width as f32,
            height: buf.height as f32,
            buf
        }
    }
}
