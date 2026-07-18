use core::fmt;
use std::dbg;
#[cfg(feature = "std")]
use std::io;

use crate::geom::{Mesh, Tri, Vertex, Vertex3, tri, vertex};
use crate::math::{
    Color3, Color4, Mat4, Point2, Point3, ProjMat3, ProjVec3, Vec2,
    color::gray, orthographic, pt2, pt3, vec2, vec3, viewport,
};
use crate::util::{Buf2, Dims};

use super::{
    BBox, Context, Frag, FragmentShader, Model, Target, VertexShader, View,
    tex::*,
};

/// Text represented as texture-mapped geometry, one quad per glyph.
#[derive(Clone)]
pub struct Text {
    pub font: Atlas<Color3>,
    pub geom: Mesh<TexCoord>,
    pub color: Color3,
    pub anchor: Point3,
    pub align: Align,
    cursor: Point2<Model>,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum Align {
    #[default]
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

pub struct Shader<'a>(&'a Text);

pub type Batch<'a> = super::Batch<
    &'a [Tri<usize>],
    &'a [Vertex3<TexCoord>],
    (),
    Shader<'a>,
    (),
    Context,
>;

//
// Inherent impls
//

impl Text {
    /// Creates a new empty text object with the given font.
    pub fn new(font: Atlas<Color3>) -> Self {
        Self {
            font,
            geom: Mesh::default(),
            color: gray(0xFF),
            anchor: Point3::default(),
            align: Align::default(),
            cursor: Point2::default(),
        }
    }

    /// Sets the anchor point of the text.
    #[must_use]
    pub fn anchor(mut self, pt: impl Into<Point3>) -> Self {
        self.anchor = pt.into();
        self
    }

    /// Erases all text from `self`.
    pub fn clear(&mut self) {
        self.cursor = Point2::origin();
        self.geom.faces.clear();
        self.geom.verts.clear();
    }

    /// Returns a shader for rendering text.
    pub fn shader(&self) -> Shader<'_> {
        Shader(self)
    }

    /// Renders this text to a render target in 2D.
    ///
    /// For more customizable rendering, see the [`batch`][Self::batch] function.
    pub fn render(&self, target: &mut impl Target) {
        // TODO keep track of this state
        let [right, bot, _] = BBox::of(&self.geom).1.0;

        let proj: ProjMat3<View> =
            orthographic(pt3(0.0, 0.0, -1.0), pt3(self.cursor.x(), bot, 1.0))
                .to();

        let left_top = self.left_top();
        let left_top = pt2(left_top.x() as u32, left_top.y() as u32);
        let right_bot = left_top + vec2(right as i32, bot as i32);
        let viewport = viewport(left_top..right_bot);

        self.batch()
            .uniform((&Mat4::identity(), &proj))
            .viewport(viewport)
            .target(target)
            .render();
    }

    pub fn left_top(&self) -> Point3<Model> {
        use Align::*;
        let off = match self.align {
            TopLeft => (0.0, 0.0),
            TopCenter => (0.5, 0.0),
            TopRight => (1.0, 0.0),
            CenterLeft => (0.0, 0.5),
            Center => (0.5, 0.5),
            CenterRight => (1.0, 0.5),
            BottomLeft => (0.0, 1.0),
            BottomCenter => (0.5, 1.0),
            BottomRight => (1.0, 1.0),
        };
        let [right, bot, _] = BBox::of(&self.geom).1.0;
        self.anchor.to() - (Vec2::from(off) * vec2(right, bot)).to_vec3()
    }

    /// Returns a `Batch` with the geometry and shader set to render this text.
    ///
    /// Useful for customized text rendering.
    pub fn batch(&self) -> Batch<'_> {
        super::Batch::from(&self.geom).shader(self.shader())
    }

    /// Samples the font at a texture coordinate.
    #[inline]
    fn sample(&self, uv: TexCoord) -> Option<Color4> {
        let col = SamplerClamp.sample(&self.font.texture, uv);
        (col != gray(0)).then_some(self.color.to_rgba())
    }

    fn write_char(&mut self, idx: u32) {
        let Self { font, geom, cursor, .. } = self;

        let Layout::Grid { sub_dims } = font.layout;
        let (glyph_w, glyph_h) = (sub_dims.0 as f32, sub_dims.1 as f32);

        let [tl, tr, bl, br] = font.coords(idx);
        // TODO doesn't work when the text is written in several pieces,
        //      such as when writing a formatted string. Total row and col
        //      counts are only known when everything is written.
        /*let offset = vec2(
            anchor.x() * glyph_w * cols as f32,
            anchor.y() * glyph_h * rows as f32,
        );*/
        let offset = vec2(0.0, 0.0);
        let pos = (*cursor - offset).to_pt3().to();
        let l = geom.verts.len();

        geom.verts.extend([
            vertex(pos, tl),
            vertex(pos + vec3(glyph_w, 0.0, 0.0), tr),
            vertex(pos + vec3(0.0, glyph_h, 0.0), bl),
            vertex(pos + vec3(glyph_w, glyph_h, 0.0), br),
        ]);
        geom.faces.push(tri(l, l + 1, l + 3));
        geom.faces.push(tri(l, l + 3, l + 2));

        *cursor += vec2(glyph_w, 0.0);
    }
}

/// Renders ("bakes") the byte string into a buffer.
pub fn bake<T>(s: &[u8], font: &Atlas<T>) -> Buf2<T>
where
    T: Copy + Default,
{
    let rows = s.split(|&c| c == b'\n');
    let (mut num_rows, mut num_cols) = (0, 0);

    for row in rows.clone() {
        num_rows += 1;
        num_cols = num_cols.max(row.len() as u32);
    }
    if num_rows == 0 || num_cols == 0 {
        return Buf2::new(Dims(0, 0));
    }

    let Layout::Grid { sub_dims: Dims(gw, gh) } = font.layout;
    let mut buf = Buf2::new(Dims(num_cols * gw, num_rows * gh));

    let (mut x, mut y) = (0, 0);
    for row in rows {
        for ch in row {
            let dest = (x..x + gw, y..y + gh);
            buf.slice_mut(dest)
                .copy_from(*font.get((*ch).into()).data());
            x += gw;
        }
        (x, y) = (0, y + gh);
    }
    buf
}

//
// Trait impls
//

pub type Uniform<'a> = (&'a Mat4<Model, View>, &'a ProjMat3<View>);

impl VertexShader<Vertex3<TexCoord>, Uniform<'_>> for Shader<'_> {
    type Output = Vertex<ProjVec3, TexCoord>;

    fn shade_vertex(
        &self,
        v: Vertex3<TexCoord>,
        (modelview, project): Uniform,
    ) -> Self::Output {
        let origin = self.0.anchor.to();
        let mut view_origin = modelview.apply(&origin);
        view_origin[2] = 3.0;

        let view_vp = (v.pos.to_vec() + self.0.left_top().to_vec())
            * vec3(1.0, -1.0, 1.0)
            * 0.01;

        let view_pos = view_origin + view_vp.to();

        let clip_pos = project.apply(&view_pos);

        vertex(clip_pos, v.attrib)
    }
}
impl FragmentShader<TexCoord, Uniform<'_>> for Shader<'_> {
    fn shade_fragment(
        &self,
        frag: Frag<TexCoord>,
        _: Uniform,
    ) -> Option<Color4> {
        self.0.sample(frag.var)
    }
}

#[cfg(feature = "std")]
impl io::Write for Text {
    /// Creates geometry to represent the bytes in `buf`.
    ///
    /// This method uses each byte in `buf` as an index to the font. Only up to
    /// 256 glyphs are thus supported. The font should have enough glyphs to
    /// cover each byte value in `buf`.
    ///
    /// Because a one-to-one mapping from bytes to glyphs is used, the result
    /// will be [mojibake][1] if the buffer contains UTF-8 encoded data beyond
    /// the ASCII range.
    ///
    /// [1]: https://en.wikipedia.org/wiki/Mojibake
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        /*let (rows, cols) = buf
            .split(|&b| b == b'\n')
            .fold((0, 0), |(rs, cs), row| (rs + 1, cs.max(row.len() as u32)));
        if rows == 0 || cols == 0 {
            return Ok(0);
        }*/
        let Layout::Grid { sub_dims } = self.font.layout;
        let glyph_h = sub_dims.1 as f32;

        for &b in buf {
            match b {
                b'\n' => self.cursor = pt2(0.0, self.cursor.y() + glyph_h),
                _ => self.write_char(b.into()),
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl fmt::Write for Text {
    /// Creates geometry to represent the characters in `s`.
    ///
    /// This method iterates over the `char`s of the string, and uses the value
    /// of each `char` as an index into the font. As such, the font should have
    /// enough glyphs to cover all the characters used.
    fn write_str(&mut self, s: &str) -> fmt::Result {
        /*let (rows, cols) = s
            .split(|c| c == '\n')
            .fold((0, 0), |(rs, cs), row| {
                (rs + 1, cs.max(row.chars().count() as u32))
            });
        if rows == 0 || cols == 0 {
            return Ok(());
        }*/
        let Layout::Grid { sub_dims } = self.font.layout;
        let glyph_h = sub_dims.1 as f32;

        for c in s.chars() {
            match c {
                '\n' => self.cursor = pt2(0.0, self.cursor.y() + glyph_h),
                _ => self.write_char(c.into()),
            }
        }
        Ok(())
    }
}
