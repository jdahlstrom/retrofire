//! Textures and texture samplers.

use crate::{
    geom::Normal3,
    math::{Color3, Point2u, Point3, Vec2, Vector, inv_lerp, lerp, pt2, vec2},
    util::{AsSlice2, Buf2, Dims, Slice2},
};

pub trait Sample<D: AsSlice2>: Sized {
    /// Returns the color in `tex` at `tc` in relative coordinates.
    #[inline]
    fn sample(&self, tex: &Texture<D>, tc: TexCoord) -> D::Elem {
        self.sample_abs(tex, tc * uv(tex.w, tex.h))
    }

    /// Returns the color in `tex` at `tc` in absolute coordinates.
    fn sample_abs(&self, tex: &Texture<D>, tc: TexCoord) -> D::Elem;
}

/// Basis of the texture space.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Tex;

/// Texture coordinate vector. Texture coordinates can be either absolute,
/// in range (0, 0)..(w, h) for some texture with dimensions w and h, or
/// relative, in range (0, 0)..(1, 1), in which case they are independent
/// of the actual dimensions of the texture.
pub type TexCoord = Vec2<Tex>;

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
#[derive(Copy, Clone, Debug)]
pub struct Texture<D> {
    w: f32,
    h: f32,
    data: D,
}

#[derive(Clone, Debug)]
pub struct Atlas<D> {
    pub layout: Layout,
    pub texture: Texture<D>,
}

/// Method of arranging sub-textures in an [atlas][Atlas].
#[derive(Copy, Clone, Debug)]
pub enum Layout {
    /// A regular grid of equal-sized cells.
    Grid { sub_dims: Dims },
}

/// Returns a new texture coordinate with components `u` and `v`.
#[inline]
pub const fn uv(u: f32, v: f32) -> TexCoord {
    Vector::new([u, v])
}

/// Returns a texture coordinate in a cube map.
///
/// A cube map texture is a composite of six subtextures in a 3x2 grid.
/// Each subtexture corresponds to one of the six cardinal directions:
/// right (+x), left (-x), top (+y), bottom (-y), front (+z), back (-z).
///
/// The subtexture is chosen based on which component of `dir` has the greatest
/// absolute value. The texture coordinates within the subtexture are based on
/// the zy, xz, or xy components of `pos` such that the range  [-1.0, 1.0] is
/// transformed to the range of uv values in the appropriate subtexture.
///
/// ```text
///     u
///     0       1/3      2/3       1
/// v 0 +--------+--------+--------+
///     | y   +x | z   +y | +z   y |
///     | ^      | ^      |      ^ |
///   1 | + -> z | + -> x | x <- + |
///   / +---zy---+---xz---+---xy---+
///   2 | -x   y | + -> x | y   -z |
///     |      ^ | v      | ^      |
///     | z <- + | z   -y | + -> x |
///   1 +--------+--------+--------+
///
///
///
///  +-- u       +--------+
///  |           | z   +y |
///  v           | ^      |
///              | + -> x |
///     +--------+--------+--------+--------+
///     | -x   y | y   -z | y   +x | +z   y |
///     |      ^ | ^      | ^      |      ^ |
///     | z <- + | + -> x | + -> z | x <- + |
///     +--------+--------+--------+--------+
///              | + -> x |
///              | v      |
///              | z   -y |
///              +--------+
///
///
///
/// ```
pub fn cube_map(pos: Point3, dir: Normal3) -> TexCoord {
    // ..-1.0..1.0.. -> 0.0..1.0
    let [x, y, z] = pos
        .map(|x| inv_lerp(x, -1.0, 1.0).clamp(0.0, 1.0))
        .0;
    let [ax, ay, az] = dir.map(f32::abs).0;

    let axis; // 0=x, 1=y, 2=z
    let sign; // 0=+, 1=-

    let tc = if ax > ay && ax > az {
        axis = 0.0;
        if x > 0.5 {
            sign = 0.0;
            (z, 1.0 - y)
        } else {
            sign = 1.0;
            (1.0 - z, 1.0 - y)
        }
    } else if ay > az {
        axis = 1.0;
        if y > 0.5 {
            sign = 0.0;
            (x, 1.0 - z)
        } else {
            sign = 1.0;
            (x, z)
        }
    } else {
        axis = 2.0;
        if z > 0.5 {
            sign = 1.0;
            (1.0 - x, 1.0 - y)
        } else {
            sign = 0.0;
            (x, 1.0 - y)
        }
    };

    (uv(axis, sign) + tc.into()) / uv(3.0, 2.0)
}

//
// Inherent impls
//

impl TexCoord {
    /// Returns the u (horizontal) component of `self`.
    #[inline]
    pub const fn u(&self) -> f32 {
        self.0[0]
    }
    /// Returns the v (vertical) component of `self`.
    #[inline]
    pub const fn v(&self) -> f32 {
        self.0[1]
    }
}

impl<D> Texture<D> {
    /// Returns the width of `self` as `f32`.
    #[inline]
    pub const fn width(&self) -> f32 {
        self.w
    }
    /// Returns the height of `self` as `f32`.
    #[inline]
    pub const fn height(&self) -> f32 {
        self.h
    }
    /// Returns the pixel data of `self`.
    pub const fn data(&self) -> &D {
        &self.data
    }
}

impl<D: AsSlice2> Atlas<D> {
    /// Creates a new texture atlas from a texture.
    pub fn new(layout: Layout, texture: Texture<D>) -> Self {
        Self { layout, texture }
    }

    /// Creates a texture atlas with a grid layout.
    pub fn grid(sub_dims: Dims, texture: Texture<D>) -> Self {
        Self::new(Layout::Grid { sub_dims }, texture)
    }

    /// Returns the top-left and bottom-right pixel coordinates
    /// of the sub-texture with index `i`.
    fn rect(&self, i: u32) -> [Point2u; 2] {
        match self.layout {
            Layout::Grid { sub_dims: Dims(sub_w, sub_h) } => {
                let subs_per_row =
                    self.texture.data.as_slice2().width() / sub_w;
                let top_left =
                    pt2(i % subs_per_row * sub_w, i / subs_per_row * sub_h);
                [top_left, top_left + vec2(sub_w as i32, sub_h as i32)]
            }
        }
    }

    /// Returns the texture coordinates of the sub-texture with index `i`.
    ///
    /// The coordinates are the top-left, top-right, bottom-left, and
    /// bottom-right corners of the texture, in that order.
    ///
    /// Note that currently this method does not check `i` is actually a valid
    /// index and may return coordinates with values greater than one.
    // TODO Error handling, more readable result type
    pub fn coords(&self, i: u32) -> [TexCoord; 4] {
        let tex_w = self.texture.width();
        let tex_h = self.texture.height();
        let [(x0, y0), (x1, y1)] = self
            .rect(i)
            .map(|p| (p.x() as f32 / tex_w, p.y() as f32 / tex_h));
        [uv(x0, y0), uv(x1, y0), uv(x0, y1), uv(x1, y1)]
    }
}

impl<T> Atlas<Buf2<T>> {
    /// Returns the sub-texture with index `i`.
    ///
    /// # Panics
    /// If `i` is out of bounds.
    // TODO Improve error reporting
    #[inline]
    pub fn get(&self, i: u32) -> Texture<Slice2<'_, T>> {
        let [p0, p1] = self.rect(i);
        self.texture.data.slice(p0..p1).into()
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
/// dimensions.
///
/// For performance reasons, `SamplerRepeatPot` only accepts textures with
/// dimensions that are powers of two.
#[derive(Copy, Clone, Debug)]
pub struct SamplerRepeatPot {
    w_mask: u32,
    h_mask: u32,
}

impl SamplerRepeatPot {
    /// Creates a new `SamplerRepeatPot` based on the dimensions of `tex`.
    /// # Panics
    /// If the width or height of `tex` is not a power of two.
    pub fn new(tex: &Texture<impl AsSlice2>) -> Self {
        let w = tex.width() as u32;
        let h = tex.height() as u32;
        assert!(w.is_power_of_two(), "width must be 2^n, was {w}");
        assert!(h.is_power_of_two(), "height must be 2^n, was {h}");
        Self { w_mask: w - 1, h_mask: h - 1 }
    }
}
impl<D: AsSlice2<Elem: Copy>> Sample<D> for SamplerRepeatPot {
    /// Returns the color in `tex` at `tc` in absolute coordinates.
    ///
    /// Coordinates outside `0.0..tex.width()` and `0.0..tex.height()` are
    /// wrapped to the valid range. Uses nearest neighbor sampling.
    #[inline]
    fn sample_abs(&self, tex: &Texture<D>, tc: TexCoord) -> D::Elem {
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

impl<D: AsSlice2<Elem: Copy>> Sample<D> for SamplerClamp {
    /// Returns the color in `tex` at `tc` in absolute coordinates.
    ///
    /// Coordinates outside `0.0..tex.width()` and `0.0..tex.height()` are
    /// clamped to the range endpoints. Uses nearest neighbor sampling.
    #[inline]
    fn sample_abs(&self, tex: &Texture<D>, tc: TexCoord) -> D::Elem {
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
/// not of the backing buffer.
#[derive(Copy, Clone, Debug)]
pub struct SamplerOnce;

impl<D: AsSlice2<Elem: Copy>> Sample<D> for SamplerOnce {
    /// Returns the color in `tex` at `tc` such that the coordinates are
    /// assumed to be in the ranges `0.0..tex.width()` and `0.0..tex.height()`
    /// respectively.
    ///
    /// Uses nearest neighbor sampling. Passing out-of-range coordinates
    /// to this function is sound (not UB) but is not otherwise specified.
    ///
    /// # Panics
    /// May panic if `tc` is not in the valid range.
    #[inline]
    fn sample_abs(&self, tex: &Texture<D>, tc: TexCoord) -> D::Elem {
        let u = tc.u() as u32;
        let v = tc.v() as u32;

        let d = tex.data.as_slice2();
        debug_assert!(u < d.width(), "u={u}");
        debug_assert!(v < d.height(), "v={v}");

        d[[u, v]]
    }
}

/// Texture sampler with bilinear filtering.
pub struct SamplerBilinear<S>(pub S);

impl<D: AsSlice2<Elem = Color3>, S: Sample<D>> Sample<D>
    for SamplerBilinear<S>
{
    /// Uses the inner sampler to sample the texture in the four integer
    /// positions surrounding the given texture coordinate and linearly
    /// interpolates the returned color value.
    #[inline]
    fn sample_abs(&self, tex: &Texture<D>, tc: TexCoord) -> D::Elem {
        use crate::math::float::f32;

        let u = tc.u().clamp(0.0, tex.w - 1.0);
        let v = tc.v().clamp(0.0, tex.h - 1.0);

        let u0 = f32::floor(u);
        let v0 = f32::floor(v);
        let u1 = u0 + 1.0;
        let v1 = v0 + 1.0;

        // TODO costly conversion to float and back
        let c00 = self.0.sample_abs(tex, uv(u0, v0)).to_color3f();
        let c01 = self.0.sample_abs(tex, uv(u0, v1)).to_color3f();
        let c10 = self.0.sample_abs(tex, uv(u1, v0)).to_color3f();
        let c11 = self.0.sample_abs(tex, uv(u1, v1)).to_color3f();

        let c0 = lerp(u - u0, c00, c10);
        let c1 = lerp(u - u0, c01, c11);

        lerp(v - v0, c0, c1).to_color3()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::math::{Color3, Vec3, rgb};
    use crate::util::Buf2;

    use super::*;

    #[rustfmt::skip]
    fn tex() -> Texture<Buf2<Color3>> {
        Texture::from(Buf2::new_from(
            Dims(2, 2), vec![
                rgb(0xFF, 0, 0),
                rgb(0, 0xFF, 0),
                rgb(0, 0, 0xFF),
                rgb(0xFF, 0xFF, 0),
           ]
        ))
    }

    #[test]
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

    #[test]
    fn cube_mapping() {
        let zero = Point3::origin();

        let tc = cube_map(zero, Vec3::X);
        assert_eq!(tc, uv(1.0 / 6.0, 0.25));
        let tc = cube_map(zero, -Vec3::X);
        assert_eq!(tc, uv(1.0 / 6.0, 0.75));

        let tc = cube_map(zero, Vec3::Y);
        assert_eq!(tc, uv(3.0 / 6.0, 0.25));
        let tc = cube_map(zero, -Vec3::Y);
        assert_eq!(tc, uv(3.0 / 6.0, 0.75));

        let tc = cube_map(zero, Vec3::Z);
        assert_eq!(tc, uv(5.0 / 6.0, 0.25));
        let tc = cube_map(zero, -Vec3::Z);
        assert_eq!(tc, uv(5.0 / 6.0, 0.75));
    }
}
