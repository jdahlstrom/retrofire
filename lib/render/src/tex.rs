use math::Linear;

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