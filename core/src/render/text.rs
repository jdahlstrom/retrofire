use core::fmt;
#[cfg(feature = "std")]
use std::io;

use crate::geom::{Tri, Vertex3, mesh::VecMesh, tri, vertex};
use crate::math::{
    Color3, Color4, Point2, ProjMat3, Vec2, color::gray, orthographic, pt2,
    pt3, vec2, vec3, viewport,
};
use crate::util::{Buf2, Dims};

use super::{BBox, Context, Frag, Model, Shader, Target, shader, tex::*};

/// Text represented as texture-mapped geometry, one quad per glyph.
#[derive(Clone)]
pub struct Text {
    pub font: Atlas<Color3>,
    pub geom: VecMesh<TexCoord>,
    pub color: Color3,
    pub anchor: Point2,
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

pub type Batch<'a, Shd> = super::Batch<
    &'a [Tri<usize>],
    &'a [Vertex3<TexCoord>],
    (),
    Shd,
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
            geom: VecMesh::default(),
            color: gray(0xFF),
            anchor: Point2::default(),
            align: Align::default(),
            cursor: Point2::default(),
        }
    }

    /// Sets the anchor point of the text.
    #[must_use]
    pub fn anchor(mut self, pt: impl Into<Point2>) -> Self {
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
    pub fn shader(
        &self,
    ) -> impl Shader<Vertex3<TexCoord>, TexCoord, &ProjMat3<Model>> {
        shader::new(
            |v: Vertex3<_>, tf: &ProjMat3<_>| {
                vertex(tf.apply(&v.pos.to()), v.attrib)
            },
            |frag: Frag<TexCoord>, _| self.sample(frag.var),
        )
    }

    /// Renders this text to a render target in 2D.
    ///
    /// For more customizable rendering, see the [`batch`][Self::batch] function.
    pub fn render(&self, target: &mut impl Target) {
        let BBox(_lt, rb) = BBox::of(&self.geom);
        let [r, b, _] = rb.0;

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
        let pos = self.anchor - Vec2::from(off) * vec2(r, b);

        let proj: ProjMat3<Model> =
            orthographic(pt3(0.0, 0.0, -1.0), pt3(self.cursor.x(), b, 1.0))
                .to();
        let pos = pt2(pos.x() as _, pos.y() as _);
        let wh = vec2(r as _, b as _);
        let viewport = viewport(pos..pos + wh);

        self.batch()
            .uniform(&proj)
            .viewport(viewport)
            .target(target)
            .render();
    }

    /// Returns a `Batch` with the geometry and shader set to render this text.
    ///
    /// Useful for customized text rendering.
    pub fn batch(
        &self,
    ) -> Batch<'_, impl Shader<Vertex3<TexCoord>, TexCoord, &ProjMat3<Model>>>
    {
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

//
// Trait impls
//

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
