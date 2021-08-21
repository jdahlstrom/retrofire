use std::mem::replace;

use geom::Sprite;
use math::mat::Mat4;
use math::transform::Transform;
use math::vec::pt;
use util::color::Color;

use crate::{RasterOps, Render, Renderer};
use crate::raster::Fragment;
use crate::shade::{Shader, ShaderImpl};
use crate::tex::{TexCoord, Texture, uv};

pub struct Font {
    pub glyph_w: u16,
    pub glyph_h: u16,
    pub glyphs: Texture,
}

impl Font {
    pub fn glyph_bounds(&self, c: char) -> Option<[TexCoord; 4]> {
        if !c.is_ascii() {
            return None;
        }
        let Font { glyph_w: gw, glyph_h: gh, glyphs, } = self;
        let (tw, th) = (glyphs.width(), glyphs.height());

        let c = c as u16;
        let (x0, y0) = ((c % 0x10) * gw, c / 0x10 * gh);
        let (x1, y1) = (((c % 0x10) + 1) * gw, (c / 0x10 + 1) * gh);
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
    geom: Vec<Sprite<TexCoord>>,
}

impl<'a> Text<'a> {
    pub fn new(font: &'a Font, text: &str) -> Text<'a> {
        let pos = (0, 0);
        let geom = text.chars()
            .scan(pos, |(x, y), c| {
                let ch = if c == '\n' {
                    *x = 0;
                    *y += font.glyph_h;
                    None
                } else if let Some(bounds) = font.glyph_bounds(c) {
                    let oldx = replace(x, *x + font.glyph_w);
                    Some(Sprite {
                        center: pt(oldx as f32, *y as f32, 0.0),
                        width: font.glyph_w as f32,
                        height: font.glyph_h as f32,
                        vertex_attrs: bounds,
                        face_attr: (),
                    })
                } else {
                    None
                };
                Some(ch)
            })
            .flatten()
            .collect();

        Text { font, geom }
    }
}

impl<'a> Transform for Text<'a> {
    fn transform(&mut self, tf: &Mat4) {
        for s in &mut self.geom {
            s.transform(tf);
        }
    }
}

impl<'a> Render<(), TexCoord, Color> for Text<'a> {
    fn render<S, R>(&self, rdr: &mut Renderer, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), TexCoord, TexCoord, Color>,
        R: RasterOps,
    {
        for s in &self.geom {
            s.render(rdr, &mut ShaderImpl {
                vs: |v| shader.shade_vertex(v),
                fs: |f: Fragment<TexCoord>| {
                    let col = self.font.glyphs.sample(f.varying);
                    shader.shade_fragment(f.varying(col))
                },
            }, raster);
        }
    }
}

#[cfg(test)]
mod tests {
    use math::transform::{orthogonal, translate, viewport};
    use math::vec::dir;
    use util::color::{BLACK, Color};
    use util::io::load_pnm;

    use crate::Raster;
    use crate::raster::Fragment;

    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    #[test]
    fn render_text() {
        const STR: &str = "Hello, World!";
        const GW: usize = 6;
        const GH: usize = 10;
        const BW: usize = GW * STR.len();
        const BH: usize = GH;

        let fnt = &Font {
            glyph_w: GW as u16,
            glyph_h: GH as u16,
            glyphs: Texture::from(
                load_pnm(&format!("../../resources/font_{}x{}.ppm", GW, GH))
                    .unwrap(),
            ),
        };
        let txt = Text::new(fnt, "Hello, World!");
        let rdr = &mut Renderer::default();
        rdr.modelview = translate(dir(GW as f32 / 2.0, GH as f32 / 2.0, 0.0));
        rdr.projection = orthogonal(pt(0.0, 0.0, -1.0), pt(BW as f32, BH as f32, 1.0));
        rdr.viewport = viewport(0.0, BH as f32, BW as f32, 0.0);
        let mut out = [[BLACK; BW]; BH];
        txt.render(
            rdr,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |frag: Fragment<_>| {
                    Some(frag.varying)
                },
            },
            &mut Raster {
                test: |_| true,
                output: |frag: Fragment<(f32, Color)>| {
                    out[frag.coord.1][frag.coord.0] = frag.varying.1;
                },
            },
        );


        for row in &out {
            let row = row.iter()
                .map(|&c| ['█', ' '][(c == BLACK) as usize])
                .collect::<String>();
            eprintln!("{}", row);
        }
        let hasher = &mut DefaultHasher::new();
        out.hash(hasher);
        let hash = hasher.finish();

        assert_eq!(10889892750907716637, hash);
    }
}
