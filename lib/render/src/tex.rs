use math::Linear;
use crate::color::Color;

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

#[derive(Debug, Clone)]
pub struct Texture {
    width: usize,
    height: usize,
    fwidth: f32,
    fheight: f32,
    data: Vec<Color>,
}

impl Texture {

    pub fn new(width: usize, data: &[Color]) -> Texture {
        assert!(width.is_power_of_two());
        let height = data.len() / width;
        assert!(height.is_power_of_two());
        assert_eq!(height * width, data.len());

        Texture {
            width, height,
            fwidth: width as f32,
            fheight: height as f32,
            data: data.to_vec(),
        }
    }

    pub fn solid(width: usize, height: usize, color: Color) -> Texture {
        assert!(width.is_power_of_two());
        assert!(height.is_power_of_two());

        Texture {
            width, height,
            fwidth: width as f32,
            fheight: height as f32,
            data: vec![color; width * height],
        }
    }

    pub fn sample(&self, TexCoord { u, v, w }: TexCoord) -> Color {
        let w = 1.0 / w;
        let u = (self.fwidth * u * w) as usize & (self.width - 1);
        let v = (self.fheight * v * w) as usize & (self.height - 1);

        self.data[self.width * v + u]
    }
}
