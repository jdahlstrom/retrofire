use std::io;
use std::time::{Duration, Instant};

use ascii_forge::prelude::{self as af, Render};

use re::prelude::*;

use re::math::spline::smootherstep;
use re::render::raster::Scanline;
use re::render::shader::FragmentShader;
use re::render::stats::Throughput;
use re::render::target::{Config, Target};
use re::render::{render, ModelToProj};
use re_geom::solids::*;

struct Framebuf<'a>(&'a mut af::Buffer, Buf2<f32>);

impl Target for Framebuf<'_> {
    fn rasterize<V, Fs>(
        &mut self,
        scanline: Scanline<V>,
        frag_shader: &Fs,
        _: Config,
    ) -> Throughput
    where
        V: Vary,
        Fs: FragmentShader<Frag<V>>,
    {
        for (pos, var) in scanline.frags {
            let [x, y, z] = pos.0;
            let f = Frag { pos: vec3(x, y, 0.0).to(), var };

            let curr_z = &mut self.1[vec2(x as i32, y as i32)];
            if z > *curr_z {
                continue;
            }
            let Some(c) = frag_shader.shade_fragment(f) else {
                continue;
            };
            *curr_z = z;

            //let col =
            //    af::Cell::chr(b" .-,:;co%#@W"[c.r() as usize / 24] as char);
            let col = af::Color::Rgb { r: c.r(), g: c.g(), b: c.b() };
            //let col = af::Color::AnsiValue(
            //    16 + (c.r() / 43) * 36 + (c.g() / 43) * 6 + (c.b() / 43),
            //);
            let cs = af::ContentStyle {
                background_color: Some(col),
                ..Default::default()
            };
            let c = af::Cell::style(af::StyledContent::new(cs, ' '));
            c.render((x as u16, y as u16).into(), self.0);
        }
        let w = scanline.xs.len();
        Throughput { i: w, o: w }
    }
}

fn main() -> io::Result<()> {
    let shader = Shader::new(
        |v: Vertex<_, _>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Normal3>| {
            let [r, g, b] = (128.0 * (frag.var + splat(1.0))).0;
            rgba(r as u8, g as u8, b as u8, 0)
        },
    );

    let objs = [
        Box {
            left_bot_near: vec3(-1.0, -0.4, -0.6),
            right_top_far: vec3(1.0, 0.4, 0.6),
        }
        .build(),
        Octahedron.build(),
        Cylinder {
            sectors: 9,
            capped: true,
            radius: 0.8,
        }
        .build(),
        Cone {
            sectors: 7,
            capped: true,
            base_radius: 1.2,
            apex_radius: 0.2,
        }
        .build(),
        Lathe {
            pts: vec![
                vec2(1.0, -1.0),
                vec2(1.0, -0.75),
                vec2(0.3, -0.5),
                vec2(0.3, 0.5),
                vec2(1.0, 0.75),
                vec2(1.0, 1.0),
            ],
            sectors: 13,
            capped: true,
        }
        .build(),
        Torus {
            major_radius: 0.8,
            minor_radius: 0.3,
            major_sectors: 17,
            minor_sectors: 9,
        }
        .build(),
    ];

    let camera = translate(vec3(0.0, 0.0, 5.0));

    let mut win = af::Window::init()?;
    af::handle_panics();

    let mut idx = 0;
    let mut new_idx = idx;
    let mut anim = None;
    let start = Instant::now();
    let mut prev = start;

    loop {
        let secs = start.elapsed().as_secs_f32();
        let d_secs = prev.elapsed().as_secs_f32();
        prev = Instant::now();
        let af::Vec2 { x: w, y: h } = win.size();

        let spin = rotate_x(rads(secs * 0.7)).then(&rotate_z(rads(secs)));
        let carousel = rotate_y(turns(smootherstep(anim.unwrap_or(0.0))));
        let project = perspective(1.5, 0.5 * w as f32 / h as f32, 0.1..1000.0);
        let viewport = viewport(vec2(0, 0)..vec2(w as i32, h as i32));
        let mvp: Mat4x4<ModelToProj> = spin
            .then(&camera)
            .then(&carousel)
            .to()
            .then(&project);

        win.update(Duration::from_millis(10))?;
        if win.code(af::KeyCode::Esc) {
            break;
        } else if win.code(af::KeyCode::Char(' ')) {
            anim = anim.or(Some(0.0));
            new_idx = idx + 1;
        }
        if let Some(a) = &mut anim {
            *a += d_secs;
            if *a >= 0.5 {
                idx = new_idx;
            }
            if *a >= 1.0 {
                anim = None
            }
        }

        let Mesh { faces, verts } = &objs[idx % objs.len()];

        let mut buf = win.buffer_mut();
        buf.clear();

        "Press Space to cycle between objects...".render(af::vec2(0, 0), buf);
        format!("FPS {:.1}", d_secs.recip()).render(af::vec2(0, 1), buf);

        let mut framebuf = Framebuf(
            &mut buf,
            Buf2::new_with(w as usize, h as usize, |_, _| f32::INFINITY),
        );
        render(&faces, &verts, &shader, &mvp, viewport, &mut framebuf);
    }
    win.restore()
}
