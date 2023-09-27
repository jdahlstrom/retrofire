use crate::math::vec::{Real, Vec2, Vector};
use crate::util::buf::{AsSlice2, Buf2, Slice2};

// Basis of the texture space
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Tex;

pub type TexCoord = Vec2<Real<2, Tex>>;

impl TexCoord {
    pub fn u(&self) -> f32 {
        self.0[0]
    }
    pub fn v(&self) -> f32 {
        self.0[1]
    }
}

#[inline]
pub const fn uv(u: f32, v: f32) -> TexCoord {
    Vector::new([u, v])
}

#[derive(Copy, Clone)]
pub struct Texture<D> {
    w: f32,
    h: f32,
    data: D,
}

impl<D> Texture<D> {
    #[inline]
    pub fn width(&self) -> f32 {
        self.w
    }
    #[inline]
    pub fn height(&self) -> f32 {
        self.h
    }
}

impl<C> From<Buf2<C>> for Texture<Buf2<C>> {
    fn from(data: Buf2<C>) -> Self {
        Self {
            w: data.width() as f32,
            h: data.height() as f32,
            data,
        }
    }
}

impl<'a, C> From<Slice2<'a, C>> for Texture<Slice2<'a, C>> {
    fn from(data: Slice2<'a, C>) -> Self {
        Self {
            w: data.width() as f32,
            h: data.height() as f32,
            data,
        }
    }
}

/// A texture sampler that repeats the texture infinitely modulo the texture
/// dimensions. For performance reasons, `SamplerRepeatPot` only accepts
/// textures with dimensions that are powers of two.
#[derive(Copy, Clone, Debug)]
pub struct SamplerRepeatPot {
    w_mask: i32,
    h_mask: i32,
}

#[cfg(feature = "std")]
impl SamplerRepeatPot {
    pub fn new<C>(tex: &Texture<impl AsSlice2<C>>) -> Self {
        let w = tex.width() as u32;
        let h = tex.height() as u32;
        assert!(w.is_power_of_two(), "width must be 2^n, was {w}");
        assert!(h.is_power_of_two(), "height must be 2^n, was {h}");
        Self {
            w_mask: w as i32 - 1,
            h_mask: h as i32 - 1,
        }
    }

    pub fn sample<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        let scaled_uv = uv(tex.width() * tc.u(), tex.height() * tc.v());
        self.sample_abs(tex, scaled_uv)
    }

    pub fn sample_abs<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        let u = (tc.u()).floor() as i32 & self.w_mask;
        let v = (tc.v()).floor() as i32 & self.h_mask;

        // TODO enforce invariants and use get_unchecked
        tex.data.as_slice2()[[u, v]]
    }
}

/// A texture sampler that clamps out-of-bounds coordinates
/// to the nearest valid coordinate in both dimensions.
#[derive(Copy, Clone, Debug)]
pub struct SamplerClamp;

#[cfg(feature = "std")]
impl SamplerClamp {
    pub fn sample<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        self.sample_abs(tex, uv(tc.u() * tex.w, tc.v() * tex.h))
    }

    pub fn sample_abs<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        let u = (tc.u().clamp(0.0, tex.w - 1.0)).floor() as i32;
        let v = (tc.v().clamp(0.0, tex.h - 1.0)).floor() as i32;
        tex.data.as_slice2()[[u, v]]
    }
}

/// A texture sampler that assumes all texture coordinates are within bounds.
///
/// Out-of-bounds coordinates may cause graphical glitches or runtime panics
/// but no undefined behavior. In particular, if the texture data is a slice
/// of a larger buffer, `SamplerOnce` may read out of bounds of the slice but
/// not the backing buffer.
#[derive(Copy, Clone, Debug)]
pub struct SamplerOnce;

impl SamplerOnce {
    pub fn sample<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        let scaled_uv = uv(tex.width() * tc.u(), tex.height() * tc.v());
        self.sample_abs(tex, scaled_uv)
    }
    pub fn sample_abs<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        let u = tc.u() as i32;
        let v = tc.v() as i32;

        let d = tex.data.as_slice2();
        debug_assert!(u < d.width() as i32, "u={u}");
        debug_assert!(v < d.height() as i32, "v={v}");

        d[[u, v]]
    }
}

#[cfg(test)]
mod tests {
    use crate::math::color::{rgb, Color3};
    use crate::render::tex::{
        uv, SamplerClamp, SamplerOnce, SamplerRepeatPot, Texture,
    };
    use crate::util::buf::Buf2;

    #[rustfmt::skip]
    fn tex() -> Texture<Buf2<Color3>> {
        Texture::from(Buf2::from_vec(
            2, 2, vec![
                rgb(0xFF, 0, 0),
                rgb(0, 0xFF, 0),
                rgb(0, 0, 0xFF),
                rgb(0xFF, 0xFF, 0),
           ]
        ))
    }

    #[test]
    #[cfg(feature = "std")]
    fn sampler_repeat_pot() {
        let tex = tex();
        let s = SamplerRepeatPot::new(&tex);

        assert_eq!(s.sample(&tex, uv(-0.1, 0.0)), rgb(0, 0xFF, 0));
        assert_eq!(s.sample(&tex, uv(0.0, -0.1)), rgb(0, 0, 0xFF));

        assert_eq!(s.sample(&tex, uv(1.0, 0.0)), rgb(0xFF, 0, 0));
        assert_eq!(s.sample(&tex, uv(0.0, 1.0)), rgb(0xFF, 0, 0));

        assert_eq!(s.sample(&tex, uv(4.8, 0.2)), rgb(0, 0xFF, 0));
        assert_eq!(s.sample(&tex, uv(0.2, 4.8)), rgb(0, 0, 0xFF));
    }

    #[test]
    #[cfg(feature = "std")]
    fn sampler_clamp() {
        let tex = tex();
        let s = SamplerClamp;

        assert_eq!(s.sample(&tex, uv(-1.0, 0.0)), rgb(0xFF, 0, 0));
        assert_eq!(s.sample(&tex, uv(0.0, -1.0)), rgb(0xFF, 0, 0));

        assert_eq!(s.sample(&tex, uv(1.5, 0.0)), rgb(0, 0xFF, 0));
        assert_eq!(s.sample(&tex, uv(0.0, 1.5)), rgb(0, 0, 0xFF));

        assert_eq!(s.sample(&tex, uv(1.5, 1.5)), rgb(0xFF, 0xFF, 0));
    }

    #[test]
    fn sampler_once() {
        let tex = tex();
        let s = SamplerOnce;

        assert_eq!(s.sample(&tex, uv(0.0, 0.0)), rgb(0xFF, 0, 0));
        assert_eq!(s.sample(&tex, uv(0.5, 0.0)), rgb(0, 0xFF, 0));
        assert_eq!(s.sample(&tex, uv(0.0, 0.5)), rgb(0, 0, 0xFF));
        assert_eq!(s.sample(&tex, uv(0.5, 0.5)), rgb(0xFF, 0xFF, 0));
    }
}
