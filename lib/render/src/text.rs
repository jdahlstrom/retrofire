use std::mem::replace;

use geom::{Align, Sprite};
use geom::mesh::Vertex;
use math::vec::pt;
use util::color::Color;
use util::tex::{SamplerOnce, TexCoord, Texture, uv};

use crate::{Batch, Rasterize, render::Render, State};
use crate::raster::Fragment;
use crate::shade::{Shader, ShaderImpl};

pub struct Font {
    pub glyph_w: u16,
    pub glyph_h: u16,
    pub glyphs: Texture,
}

impl Font {
    fn glyph_bounds(&self, c: char) -> Option<[TexCoord; 4]> {
        if !c.is_ascii() {
            return None;
        }
        let (col, row) = (c as u16 % 0x10, c as u16 / 0x10);
        let (x0, y0) = (col * self.glyph_w, row * self.glyph_h);
        let (x1, y1) = (x0 + self.glyph_w, y0 + self.glyph_h);
        Some([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            .map(|(u, v)| uv(u as f32, v as f32)))
    }
}

pub struct Text<'a> {
    pub font: &'a Font,
    geom: Vec<Sprite<Vertex<TexCoord>>>,
}

impl<'a> Text<'a> {
    pub fn new(font: &'a Font, text: &str) -> Text<'a> {
        let geom = text.chars()
            .scan((0, 0), |(x, y), c| {
                let ch = if c == '\n' {
                    *x = 0;
                    *y += font.glyph_h;
                    None
                } else if let Some(bounds) = font.glyph_bounds(c) {
                    let oldx = replace(x, *x + font.glyph_w);
                    Some(Sprite::new(
                        pt(oldx as f32, *y as f32, 0.0),
                        Align::TopRight,
                        font.glyph_w as f32,
                        font.glyph_h as f32,
                        bounds,
                        (),
                    ))
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

impl<'a> Render<(), TexCoord, Color> for Text<'a> {
    fn render<S, R>(&self, st: &mut State, shader: &mut S, raster: &mut R)
    where
        S: Shader<(), TexCoord, Color, VtxOut = TexCoord>,
        R: Rasterize,
    {
        let mut prims = vec![];
        let mut verts = vec![];

        for s in &self.geom {
            let l = verts.len();
            prims.push(Sprite {
                verts: [l, l+1, l+2, l+3],
                face_attr: s.face_attr
            });
            verts.extend(s.verts);
        }

        let batch = Batch { prims, verts };
        batch.render(st, &mut ShaderImpl {
            vs: |v| shader.shade_vertex(v),
            fs: |f: Fragment<TexCoord>| {
                SamplerOnce.sample_abs(&self.font.glyphs, f.varying)
                    .and_then(|col| shader.shade_fragment(f.varying(col)))
            },
        }, raster);

    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use math::transform::{orthogonal, viewport};
    use util::color::{BLACK, Color};
    use util::io::load_pnm;

    use crate::Raster;
    use crate::raster::Fragment;

    use super::*;

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
                load_pnm(&format!("../../resources/font_{}x{}.pbm", GW, GH))
                    .unwrap(),
            ),
        };
        let txt = Text::new(fnt, "Hello, World!");
        let st = &mut State::default();
        st.projection = orthogonal(pt(0.0, 0.0, -1.0), pt(BW as f32, BH as f32, 1.0));
        st.viewport = viewport(0.0, BH as f32, BW as f32, 0.0);
        let mut out = [[BLACK; BW]; BH];
        txt.render(
            st,
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
