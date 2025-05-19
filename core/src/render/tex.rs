//! Textures and texture samplers.

use crate::math::{pt2, vec2, Point2u, Vec2, Vector};
use crate::util::{
    buf::{AsSlice2, Buf2, Slice2},
    Dims,
};

/// Basis of the texture space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Tex;

/// Texture coordinate vector. Texture coordinates can be either absolute,
/// in range (0, 0)..(w, h) for some texture with dimensions w and h, or
/// relative, in range (0, 0)..(1, 1), in which case they are independent
/// of the actual dimensions of the texture.
pub type TexCoord = Vec2<Tex>;

impl TexCoord {
    /// Returns the u (horizontal) component of `self`.
    pub const fn u(&self) -> f32 {
        self.0[0]
    }
    /// Returns the v (vertical) component of `self`.
    pub const fn v(&self) -> f32 {
        self.0[1]
    }
}

/// Returns a new texture coordinate with components `u` and `v`.
#[inline]
pub const fn uv(u: f32, v: f32) -> TexCoord {
    Vector::new([u, v])
}

/// A texture type. Can contain either owned or borrowed pixel data.
///
/// Textures are used to render *texture mapped* geometry, by interpolating
/// texture coordinates across polygon faces. To read, or *sample*, from a
/// `Texture`, use one of the `Sampler*` types defined in this module.
///
/// Multiple textures can be packed into a single larger memory buffer, often
/// called a "texture atlas" or "sprite sheet". Each texture borrows a region
/// of the larger buffer.
///
/// * TODO Mipmapping
/// * TODO Bilinear filtering sampler
#[derive(Copy, Clone)]
pub struct Texture<D> {
    w: f32,
    h: f32,
    data: D,
}

#[derive(Clone)]
pub struct Atlas<C> {
    pub layout: Layout,
    pub texture: Texture<Buf2<C>>,
}

/// Method of arranging sub-textures in an [atlas][Atlas].
#[derive(Copy, Clone, Debug)]
pub enum Layout {
    /// A regular grid of equal-sized cells.
    Grid { sub_dims: Dims },
}

impl<D> Texture<D> {
    /// Returns the width of `self` as `f32`.
    #[inline]
    pub fn width(&self) -> f32 {
        self.w
    }
    /// Returns the height of `self` as `f32`.
    #[inline]
    pub fn height(&self) -> f32 {
        self.h
    }
    /// Returns the pixel data of `self`.
    pub fn data(&self) -> &D {
        &self.data
    }
}

impl<C> Atlas<C> {
    /// Creates a new texture atlas from a texture.
    pub fn new(layout: Layout, texture: Texture<Buf2<C>>) -> Self {
        Self { layout, texture }
    }

    /// Returns the top-left and bottom-right pixel coordinates
    /// of the sub-texture with index `i`.
    fn rect(&self, i: u32) -> [Point2u; 2] {
        match self.layout {
            Layout::Grid { sub_dims: (sub_w, sub_h) } => {
                let subs_per_row = self.texture.data.width() / sub_w;
                let top_left =
                    pt2(i % subs_per_row * sub_w, i / subs_per_row * sub_h);
                [top_left, top_left + vec2(sub_w as i32, sub_h as i32)]
            }
        }
    }

    /// Returns the sub-texture with index `i`.
    pub fn get(&self, i: u32) -> Texture<Slice2<C>> {
        let [p0, p1] = self.rect(i);
        self.texture.data.slice(p0..p1).into()
    }

    /// Returns the texture coordinates of the sub-texture with index `i`.
    pub fn coords(&self, i: u32) -> [TexCoord; 4] {
        let tex_w = self.texture.width();
        let tex_h = self.texture.height();
        let [(x0, y0), (x1, y1)] = self
            .rect(i)
            .map(|p| (p.x() as f32 / tex_w, p.y() as f32 / tex_h));
        [uv(x0, y0), uv(x1, y0), uv(x0, y1), uv(x1, y1)]
    }
}

//
// Trait impls
//

impl<C> From<Buf2<C>> for Texture<Buf2<C>> {
    /// Creates a new texture from owned pixel data.
    fn from(data: Buf2<C>) -> Self {
        Self {
            w: data.width() as f32,
            h: data.height() as f32,
            data,
        }
    }
}

impl<'a, C> From<Slice2<'a, C>> for Texture<Slice2<'a, C>> {
    /// Creates a new texture from borrowed pixel data.
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
    w_mask: u32,
    h_mask: u32,
}

impl SamplerRepeatPot {
    /// Creates a new `SamplerRepeatPot` based on the dimensions of `tex`.
    /// # Panics
    /// If the width or height of `tex` is not a power of two.
    pub fn new<C>(tex: &Texture<impl AsSlice2<C>>) -> Self {
        let w = tex.width() as u32;
        let h = tex.height() as u32;
        assert!(w.is_power_of_two(), "width must be 2^n, was {w}");
        assert!(h.is_power_of_two(), "height must be 2^n, was {h}");
        Self { w_mask: w - 1, h_mask: h - 1 }
    }

    /// Returns the color in `tex` at `tc` in relative coordinates, such that
    /// coordinates outside `0.0..1.0` are wrapped to the valid range.
    ///
    /// Uses nearest neighbor sampling.
    pub fn sample<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        let scaled_uv = uv(tex.width() * tc.u(), tex.height() * tc.v());
        self.sample_abs(tex, scaled_uv)
    }

    /// Returns the color in `tex` at `tc` in absolute coordinates, such that
    /// coordinates outside `0.0..tex.width()` and `0.0..tex.height()` are
    /// wrapped to the valid range.
    ///
    /// Uses nearest neighbor sampling.
    pub fn sample_abs<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        use crate::math::float::f32;
        // Convert first to signed int to avoid clamping to zero
        let u = f32::floor(tc.u()) as i32 as u32 & self.w_mask;
        let v = f32::floor(tc.v()) as i32 as u32 & self.h_mask;

        tex.data.as_slice2()[[u, v]]
    }
}

/// A texture sampler that clamps out-of-bounds coordinates
/// to the nearest valid coordinate in both dimensions.
#[derive(Copy, Clone, Debug)]
pub struct SamplerClamp;

#[cfg(feature = "fp")]
impl SamplerClamp {
    /// Returns the color in `tex` at `tc` such that coordinates outside
    /// the range `0.0..1.0` are clamped to the range endpoints.
    ///
    /// Uses nearest neighbor sampling.
    pub fn sample<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        self.sample_abs(tex, uv(tc.u() * tex.w, tc.v() * tex.h))
    }

    /// Returns the color in `tex` at `tc` in absolute coordinates, such that
    /// coordinates outside `0.0..tex.width()` and `0.0..tex.height()` are
    /// clamped to the range endpoints.
    ///
    /// Uses nearest neighbor sampling.
    pub fn sample_abs<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        use crate::math::float::f32;
        let u = f32::floor(tc.u().clamp(0.0, tex.w - 1.0)) as u32;
        let v = f32::floor(tc.v().clamp(0.0, tex.h - 1.0)) as u32;
        tex.data.as_slice2()[[u, v]]
    }
}

/// A texture sampler that assumes all texture coordinates are within bounds.
///
/// Out-of-bounds coordinates may cause graphical glitches or runtime panics
/// but not undefined behavior. In particular, if the texture data is a slice
/// of a larger buffer, `SamplerOnce` may read out of bounds of the slice but
/// not the backing buffer.
#[derive(Copy, Clone, Debug)]
pub struct SamplerOnce;

impl SamplerOnce {
    /// Returns the color in `tex` at `tc` such that both coordinates are
    /// assumed to be in the range `0.0..1.0`.
    ///
    /// Uses nearest neighbor sampling. Passing out-of-range coordinates
    /// to this function is sound (not UB) but is not otherwise specified.
    ///
    /// # Panics
    /// May panic if `tc` is not in the valid range.
    pub fn sample<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        let scaled_uv = uv(tex.width() * tc.u(), tex.height() * tc.v());
        self.sample_abs(tex, scaled_uv)
    }
    /// Returns the color in `tex` at `tc` such that the coordinates are
    /// assumed to be in the ranges `0.0..tex.width()` and `0.0..tex.height()`
    /// respectively.
    ///
    /// Uses nearest neighbor sampling. Passing out-of-range coordinates
    /// to this function is sound (not UB) but is not otherwise specified.
    ///
    /// # Panics
    /// May panic if `tc` is not in the valid range.
    pub fn sample_abs<C: Copy>(
        &self,
        tex: &Texture<impl AsSlice2<C>>,
        tc: TexCoord,
    ) -> C {
        let u = tc.u() as u32;
        let v = tc.v() as u32;

        let d = tex.data.as_slice2();
        debug_assert!(u < d.width(), "u={u}");
        debug_assert!(v < d.height(), "v={v}");

        d[[u, v]]
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::math::{rgb, Color3};
    use crate::util::buf::Buf2;

    use super::*;

    #[rustfmt::skip]
    fn tex() -> Texture<Buf2<Color3>> {
        Texture::from(Buf2::new_from(
            (2, 2), vec![
                rgb(0xFF, 0, 0),
                rgb(0, 0xFF, 0),
                rgb(0, 0, 0xFF),
                rgb(0xFF, 0xFF, 0),
           ]
        ))
    }

    #[test]
    #[cfg(feature = "fp")]
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
    #[cfg(feature = "fp")]
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
