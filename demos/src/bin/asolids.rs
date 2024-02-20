use std::process::exit;
use std::time::{Duration, Instant};
use std::{env, io};

use ascii_forge::prelude::{self as af, Render, Stylize};

use re::prelude::*;

use re::math::spline::smootherstep;
use re::render::raster::Scanline;
use re::render::shader::FragmentShader;
use re::render::stats::Throughput;
use re::render::target::{Config, Target};
use re::render::{render, ModelToProj};
use re_geom::solids::*;

struct Framebuf<'a>(&'a mut af::Buffer, Buf2<f32>, Shading);

#[derive(Copy, Clone, Debug)]
enum Shading {
    Ascii,
    EightBit,
    TrueColor,
}

fn main() -> io::Result<()> {
    let shading = match env::args().nth(1).as_deref() {
        Some("ascii") | None => Shading::Ascii,
        Some("8bit") => Shading::EightBit,
        Some("24bit") => Shading::TrueColor,
        Some(_) => exit(1),
    };

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
            radius: 0.8,
            capped: true,
        }
        .build(),
        Cone {
            sectors: 11,
            base_radius: 1.2,
            apex_radius: 0.2,
            capped: true,
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
            sectors: 6,
            capped: true,
        }
        .build(),
        Torus {
            major_radius: 0.8,
            minor_radius: 0.3,
            major_sectors: 15,
            minor_sectors: 7,
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
        if let Some(t) = &mut anim {
            *t += d_secs;
            if *t >= 0.5 {
                idx = new_idx;
            }
            if *t >= 1.0 {
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
            shading,
        );
        render(&faces, &verts, &shader, &mvp, viewport, &mut framebuf);
    }
    win.restore()
}

impl Target for Framebuf<'_> {
    fn rasterize<V, Fs>(
        &mut self,
        scanline: Scanline<V>,
        shader: &Fs,
        _: Config,
    ) -> Throughput
    where
        V: Vary,
        Fs: FragmentShader<Frag<V>>,
    {
        for (pos, var) in scanline.frags {
            let curr_z = &mut self.1[vec2(pos.x() as i32, pos.y() as i32)];
            if pos.z() > *curr_z {
                continue;
            }
            let Some(c) = shader.shade_fragment(Frag { pos, var }) else {
                continue;
            };
            *curr_z = pos.z();
            let [r, g, b, _] = c.0;
            let cell = match self.2 {
                Shading::Ascii => {
                    // quick and dirty monochrome conversion
                    let shade = (r / 3 + g / 2 + b / 10) / 24;
                    (b" .,:;coO%#@W"[shade as usize] as char).on_black()
                }
                Shading::EightBit => {
                    ' '.on(af::Color::AnsiValue(
                        // 6x6x6 color cube in range 16..232
                        16 + (r / 43) * 36 + (g / 43) * 6 + (b / 43),
                    ))
                }
                Shading::TrueColor => ' '.on(af::Color::Rgb { r, g, b }),
            };
            af::Cell::from(cell)
                .render((pos.x() as u16, pos.y() as u16).into(), self.0);
        }
        let w = scanline.xs.len();
        Throughput { i: w, o: w }
    }
}
