use math::Linear;
use util::Buffer;
use util::color::Color;

#[derive(Copy, Clone, Debug, Default)]
pub struct TexCoord {
    pub u: f32, pub v: f32, pub w: f32
}

pub const fn uv(u: f32, v: f32) -> TexCoord {
    TexCoord { u, v, w: 1.0 }
}

pub const U: TexCoord = uv(1.0, 0.0);
pub const V: TexCoord = uv(0.0, 1.0);

impl TexCoord {
    #[inline]
    pub fn w_div(self) -> Self {
        let iw = self.w.recip();
        uv(iw * self.u, iw * self.v)
    }
}

impl Linear<f32> for TexCoord {
    #[inline]
    fn add(self, Self { u, v, w }: Self) -> Self {
        Self { u: self.u + u, v: self.v + v, w: self.w + w }
    }
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self { u: s * self.u, v: s * self.v, w: s * self.w }
    }
    #[inline]
    fn neg(self) -> Self {
        Self { u: -self.u, v: -self.v, w: -self.w }
    }
    #[inline]
    fn w_div(self) -> Self {
        debug_assert_ne!(self.w, 0.0);
        let w = self.w.recip();
        Self { u: self.u * w, v: self.v * w, w: 1.0 }
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
        let u = (self.w * u) as isize as usize & (buf.width() - 1);
        let v = (self.h * v) as isize as usize & (buf.height() - 1);

        // TODO enforce invariants and use get_unchecked
        *buf.get(u, v)
    }

    pub fn sample_abs_norepeat_nopc(&self, uv: TexCoord) -> Color {
        let buf = &self.buf;
        let u = uv.u as isize as usize;
        let v = uv.v as isize as usize;

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


#[derive(Copy, Clone)]
pub struct Sampler<'a> {
    tex: &'a Texture,
    uv: TexCoord,
}

impl<'a> Sampler<'a> {

    pub fn new(tex: &'a Texture, uv: TexCoord) -> Self {
        Self {
            tex,
            uv: TexCoord {
                u: uv.u * tex.w,
                v: uv.v * tex.h,
                w: uv.w,
            }
        }
    }

    pub fn sample(&self) -> Color {
        let Self { tex, uv } = self;

        let width = tex.buf.width();
        let height = tex.buf.height();
        let u = uv.u as isize as usize & (width - 1);
        let v = uv.v as isize as usize & (height - 1);

        unsafe {
            // SAFETY: Ensured by bitmasks to be within bounds
            *tex.buf.data().get_unchecked(width * v + u)
        }
    }
}

impl<'a> Linear<f32> for Sampler<'a> {
    #[inline]
    fn add(self, other: Self) -> Self {
        Self { uv: self.uv.add(other.uv), ..self }
    }
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self { uv: self.uv.mul(s), ..self }
    }
    #[inline]
    fn neg(self) -> Self {
        Self { uv: self.uv.neg(), ..self }
    }
    #[inline]
    fn w_div(self) -> Self {
        Self { uv: self.uv.w_div(), ..self }
    }
}
