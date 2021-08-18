use crate::tex::{uv, TexCoord, Texture};
use crate::{RasterOps, Render, Renderer};
use geom::Sprite;
use math::mat::Mat4;
use math::transform::Transform;
use math::vec::pt;

pub struct Font {
    pub glyph_w: u16,
    pub glyph_h: u16,
    pub glyphs: Texture,
}

impl Font {
    pub fn glyph_bounds(&self, c: char) -> Option<[TexCoord; 4]> {
        if c.is_ascii_control() || !c.is_ascii() {
            return None;
        }
        let Font {
            glyph_w: gw,
            glyph_h: gh,
            glyphs,
        } = self;

        let (tw, th) = (glyphs.width(), glyphs.height());

        let c = c as u16 - 0x20;
        let (x0, y0) = ((c & 0xF) * gw, c / 0x10 * gh);
        let (x1, y1) = (((c & 0xF) + 1) * gw, (c / 0x10 + 1) * gh);
        Some([
            uv(x0 as f32 / tw, y0 as f32 / th),
            uv(x1 as f32 / tw, y0 as f32 / th),
            uv(x1 as f32 / tw, y1 as f32 / th),
            uv(x0 as f32 / tw, y1 as f32 / th),
        ])
    }
}

pub struct Text<'a> {
    pub font: &'a Font,
    x: Vec<Sprite<TexCoord>>,
}

impl<'a> Text<'a> {
    pub fn new(font: &'a Font, text: &str) -> Text<'a> {
        let x = text
            .chars()
            .map(|c| font.glyph_bounds(c))
            .flatten()
            .enumerate()
            .map(|(i, bounds)| Sprite {
                center: pt((i as u16 * font.glyph_w) as f32, 0.0, 0.0),
                width: font.glyph_w as f32,
                height: font.glyph_h as f32,
                vertex_attrs: bounds,
                face_attr: (),
            })
            .collect();

        Text { font, x }
    }
}

impl<'a> Transform for Text<'a> {
    fn transform(&mut self, tf: &Mat4) {
        for c in &mut self.x {
            c.transform(tf);
        }
    }
}

impl<'a> Render<TexCoord, ()> for Text<'a> {
    fn render<R>(&self, rdr: &mut Renderer, raster: &mut R)
    where
        R: RasterOps<TexCoord, ()>,
    {
        for s in &self.x {
            s.render(rdr, raster);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::Fragment;
    use crate::Raster;
    use util::color::BLACK;
    use util::io::load_ppm;
    use math::transform::{orthogonal, viewport, translate};
    use math::vec::dir;

    #[test]
    fn render_text() {
        let fnt = &Font {
            glyph_w: 6,
            glyph_h: 10,
            glyphs: Texture::from(
                load_ppm("../../resources/font_6x10.ppm").unwrap(),
            ),
        };
        let txt = Text::new(fnt, "Hello, World!");
        let rdr = &mut Renderer::default();
        rdr.modelview = translate(dir(3.0, 5.0, 0.0));
        rdr.projection = orthogonal(pt(0.0, 0.0, -1.0), pt(80.0, 10.0, 1.0));
        rdr.viewport = viewport(0.0, 10.0, 80.0, 0.0);
        let mut out = [[BLACK; 80]; 10];
        txt.render(
            rdr,
            &mut Raster {
                shade: |frag: Fragment<TexCoord>, _| {
                    txt.font.glyphs.sample(frag.varying)
                },
                test: |_| true,
                output: |(x, y): (usize, usize), c| {
                    out[y][x] = c;
                },
            },
        );
        let mut hash: u32 = 0;
        for row in &out {
            for &c in row {
                hash = hash.wrapping_mul(31)
                    .wrapping_add(c.0).wrapping_add(1);
                eprint!("{}", if c == BLACK { ' ' } else { '#' })
            }
            eprintln!();
        }
        assert_eq!(2712754115, hash);
    }
}
