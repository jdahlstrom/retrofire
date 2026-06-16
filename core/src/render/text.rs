use alloc::vec::Vec;
use core::fmt::{self, Write};
use core::ops::Range;

#[cfg(feature = "std")]
use std::io;

use crate::geom::{Mesh, Tri, Vertex, Vertex3, tri, vertex};
use crate::math::{
    Color3, Color4, Point2, Point2u, ProjMat3, ProjVec3, Vec2, Vec3,
    color::gray, orthographic, pt2, pt3, translate, vec2, vec3, viewport,
};
use crate::util::buf::Buf2;

use super::{
    BBox, Batch, Context, Frag, FragmentShader, Model, Screen, Target,
    VertexShader, tex::*,
};

/// Text represented as texture-mapped geometry, one quad per glyph.
#[derive(Clone)]
pub struct Text {
    pub font: Atlas<Color3>,
    pub geom: Mesh<TexCoord>,
    pub color: Color3,
    pub anchor: Point2<Model>,
    pub align: Align,
    cursor: Point2<Model>,
    /// The index of the first primitive of each line in geom.faces.
    lines: Vec<usize>,
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

pub struct Console {
    text: Text,
    left_top: Point2u<Screen>,
    right_bot: Point2u<Screen>,
    // Store here to allow passing by ref to shader
    transform: ProjMat3<Model>,
    pub open: bool,
}

pub struct TextShader<'a>(&'a Text);

pub type TextBatch<'a, Uni> =
    Batch<Tri<usize>, Vertex3<TexCoord>, Uni, TextShader<'a>, (), Context>;

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
            anchor: Point2::origin(),
            align: Align::default(),
            cursor: Point2::origin(),
            lines: Vec::default(),
        }
    }

    /// Sets the anchor point of the text.
    pub fn anchor(mut self, pt: impl Into<Point2<Model>>) -> Self {
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
    pub fn shader(&self) -> TextShader<'_> {
        TextShader(self)
    }

    /// Renders this text to a render target in 2D.
    ///
    /// For more customizable rendering, see the [`batch`][Self::batch] function.
    pub fn render(&self, target: &mut impl Target) {
        if self.geom.faces.is_empty() {
            return;
        }
        let right_bot = BBox::of(&self.geom).1.xy().to_vec();

        // Transform to screen
        let offset = self.align.offset().to();
        let pos: Point2<Screen> = (self.anchor - offset * right_bot).to();

        let proj: ProjMat3<Model> = orthographic(
            pt3(0.0, 0.0, -1.0),
            pt3(right_bot.x(), right_bot.y(), 1.0),
        )
        .to();
        let pos = pos.to_u();
        let viewport = viewport(pos..pos + right_bot.to_i().to());

        self.batch()
            .uniform(&proj)
            .viewport(viewport)
            .target(target)
            .render();
    }

    /// Returns a `Batch` with the geometry and shader set to render this text.
    ///
    /// Useful for customized text rendering.
    pub fn batch(&self) -> TextBatch<'_, ()> {
        Batch::new()
            .mesh(&self.geom)
            .shader(self.shader())
    }

    /// Samples the font at a texture coordinate.
    #[inline]
    fn sample(&self, uv: TexCoord) -> Option<Color4> {
        let col = SamplerClamp.sample(&self.font.texture, uv);
        (col != gray(0)).then_some(self.color.to_rgba())
    }

    fn write_char(&mut self, idx: u32) {
        let Self { font, geom, cursor, .. } = self;

        let (gw, gh) = font.dims(idx);
        let (glyph_w, glyph_h) = (gw as f32, gh as f32);

        let [tl, tr, bl, br] = font.coords(idx);
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
        let Layout::Grid { sub_dims: (_, h) } = self.font.layout;
        // TODO variable line height support
        self.cursor = pt2(0.0, self.cursor.y() + h as f32);
        self.lines.push(self.geom.faces.len());
    }
}

impl Console {
    pub fn new(font: Atlas<Color3>, bounds: Range<Point2u<Screen>>) -> Self {
        let dims = bounds.end - bounds.start;
        Self {
            text: Text::new(font),
            left_top: bounds.start,
            right_bot: bounds.end,
            transform: orthographic(
                pt3(0.0, 0.0, -1.0),
                pt3(dims.x() as f32, dims.y() as f32, 1.0),
            )
            .to(),
            open: true,
        }
    }

    pub fn print(&mut self, s: &str) {
        _ = write!(self, "{}", s);
    }
    pub fn println(&mut self, s: &str) {
        _ = writeln!(self, "{}", s);
    }

    pub fn write_fmt(&mut self, args: fmt::Arguments) {
        #[rustfmt::skip]
        let Self { left_top, right_bot, text, transform, .. } = self;

        let lines_old = text.lines.len() as u32;
        text.write_fmt(args).expect("cannot fail");
        let lines_new = text.lines.len() as u32;

        let glyph_h = text.font.dims(0).1;
        let vis_lines = (right_bot.y() - left_top.y()) / glyph_h;
        if lines_new > vis_lines {
            let scroll = lines_new - lines_old;
            *transform = transform
                .compose(&translate((glyph_h * scroll) as f32 * -Vec3::Y).to())
        }
    }

    pub fn clear(&mut self) {
        self.text.clear();
    }

    pub fn batch(&self) -> TextBatch<'_, &ProjMat3<Model>> {
        Batch::new()
            .mesh(&self.text.geom)
            .uniform(&self.transform)
            .shader(self.text.shader())
            .viewport(viewport(self.left_top..self.right_bot))
    }

    pub fn render(&self, target: impl Target, ctx: &Context) {
        if self.open {
            self.batch().target(target).context(ctx).render();
        }
    }
}

impl Align {
    pub fn offset(&self) -> Vec2 {
        use Align::*;
        match self {
            TopLeft => (0.0, 0.0),
            TopCenter => (0.5, 0.0),
            TopRight => (1.0, 0.0),
            CenterLeft => (0.0, 0.5),
            Center => (0.5, 0.5),
            CenterRight => (1.0, 0.5),
            BottomLeft => (0.0, 1.0),
            BottomCenter => (0.5, 1.0),
            BottomRight => (1.0, 1.0),
        }
        .into()
    }
}

//
// Trait impls
//

impl VertexShader<Vertex3<TexCoord>, &ProjMat3<Model>> for TextShader<'_> {
    type Output = Vertex<ProjVec3, TexCoord>;

    fn shade_vertex(
        &self,
        v: Vertex3<TexCoord>,
        tf: &ProjMat3<Model>,
    ) -> Self::Output {
        vertex(tf.apply(&v.pos), v.attrib)
    }
}

impl FragmentShader<TexCoord, &ProjMat3<Model>> for TextShader<'_> {
    fn shade_fragment(
        &self,
        f: Frag<TexCoord>,
        _: &ProjMat3<Model>,
    ) -> Option<Color4> {
        self.0.sample(f.var)
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

#[cfg(test)]
mod tests {
    //use super::*;

    #[test]
    fn test_text() {
        todo!()
    }
}
