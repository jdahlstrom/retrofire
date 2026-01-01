use core::fmt::{self, Write};
use core::ops::Range;
#[cfg(feature = "std")]
use std::io;

use crate::geom::{Mesh, Tri, Vertex, Vertex3, tri, vertex};
use crate::math::{
    Color3, Color4, Point2, Point2u, ProjMat3, ProjVec3, Vec2, color::gray,
    orthographic, pt2, pt3, vec2, vec3, viewport,
};
use crate::util::buf::Buf2;

use super::{
    Batch, Context, Frag, FragmentShader, Model, Target, VertexShader,
    tex::{Atlas, Layout, SamplerClamp, TexCoord},
};

/// Text represented as texture-mapped geometry, one quad per glyph.
#[derive(Clone)]
pub struct Text {
    pub font: Atlas<Color3>,
    pub geom: Mesh<TexCoord>,
    // TODO Private until fixed
    _anchor: Vec2,
    cursor: Point2,
}

pub struct Console {
    text: Text,
    left_top: Point2u,
    right_bot: Point2u,
}

pub struct TextShader<'a>(&'a Text);

impl VertexShader<Vertex3<TexCoord>, ProjMat3<Model>> for TextShader<'_> {
    type Output = Vertex<ProjVec3, TexCoord>;

    fn shade_vertex(
        &self,
        v: Vertex3<TexCoord>,
        tf: ProjMat3<Model>,
    ) -> Self::Output {
        vertex(tf.apply(&v.pos), v.attrib)
    }
}

impl FragmentShader<TexCoord> for TextShader<'_> {
    fn shade_fragment(&self, f: Frag<TexCoord>) -> Option<Color4> {
        let c = self.0.sample(f.var);
        (c != gray(0)).then_some(c.to_rgba())
    }
}

pub type TextBatch<'a> = Batch<
    Tri<usize>,
    Vertex3<TexCoord>,
    ProjMat3<Model>,
    TextShader<'a>,
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
            _anchor: Vec2::default(),
            cursor: Point2::default(),
        }
    }

    /// Sets the anchor point of the text.
    ///
    /// The anchor is a vector that determines how the text is aligned relative
    /// to the (local) origin. The default is (0, 0) which places the origin to
    /// the top left corner. Use (0.5, 0.5) to center the text vertically and
    /// horizontally relative to the origin.
    ///
    /// Note that this value does not affect how individual lines of text
    /// are aligned relative to each other.
    // TODO private until fixed
    fn _anchor(mut self, x: f32, y: f32) -> Self {
        self._anchor = vec2(x, y);
        self
    }

    /// Erases all text from `self`.
    pub fn clear(&mut self) {
        self.cursor = Point2::origin();
        self.geom.faces.clear();
        self.geom.verts.clear();
    }

    /// Samples the font at `uv`.
    pub fn sample(&self, uv: TexCoord) -> Color3 {
        // TODO Figure out why coords go out of bounds -> SamplerOnce panics
        SamplerClamp.sample(&self.font.texture, uv)
    }

    fn write_char(&mut self, idx: u32) {
        let Self { font, geom, cursor, .. } = self;

        let (gw, gh) = font.dims(idx);
        let (glyph_w, glyph_h) = (gw as f32, gh as f32);

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

    fn newline(&mut self) {
        let Layout::Grid { sub_dims } = self.font.layout;
        // TODO variable line height support
        self.cursor = pt2(0.0, self.cursor.y() + sub_dims.1 as f32)
    }
}

impl Console {
    pub fn new(font: Atlas<Color3>, bounds: Range<Point2u>) -> Self {
        Self {
            text: Text::new(font),
            left_top: bounds.start,
            right_bot: bounds.end,
        }
    }

    pub fn print(&mut self, s: &str) {
        _ = self.text.write_str(s);
    }
    pub fn println(&mut self, s: &str) {
        self.print(s);
        self.print("\n");
    }

    pub fn write_fmt(&mut self, args: fmt::Arguments) {
        _ = self.text.write_fmt(args)
    }

    pub fn clear(&mut self) {
        self.text.clear();
    }

    pub fn batch(&self) -> TextBatch<'_> {
        let Self {
            left_top: lt, right_bot: rb, ..
        } = self;

        let [w, h] = (*rb - *lt).0;
        let projection =
            orthographic(pt3(0.0, 0.0, 0.0), pt3(w as f32, h as f32, 0.0));

        let viewport = viewport(*lt..*rb);

        Batch::new()
            .mesh(&self.text.geom)
            .uniform(projection.to())
            .shader(TextShader(&self.text))
            .viewport(viewport)
    }

    pub fn render(&self, target: impl Target, ctx: &Context) {
        self.batch().target(target).context(ctx).render();
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
        for &b in buf {
            match b {
                b'\n' => self.newline(),
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
        for c in s.chars() {
            match c {
                '\n' => self.newline(),
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
        return Buf2::new((0, 0));
    }

    let Layout::Grid { sub_dims: (gw, gh) } = font.layout;
    let mut buf = Buf2::new((num_cols * gw, num_rows * gh));

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
