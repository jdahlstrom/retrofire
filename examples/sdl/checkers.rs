use std::convert::identity;

use sdl2::event::Event;
use sdl2::keyboard::Scancode;

use geom::mesh::Mesh;
use geom::solids::unit_cube;
use math::Angle::{self, Deg};
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::*;
use render::raster::Fragment;
use render::scene::*;
use render::shade::ShaderImpl;
use render::tex::*;
use render::text::{Font, Text};
use util::io::load_pnm;

use crate::runner::*;
use util::color::{WHITE, BLACK};
use util::Buffer;
use core::iter;

mod runner;

fn checkers() -> Mesh<TexCoord, usize> {
    let size: usize = 40;
    let isize = size as i32;

    let mut vs = vec![];
    let mut texcoords = vec![];
    for j in -isize..=isize {
        for i in -isize..=isize {
            vs.push(pt(i as f32, 0.0, j as f32));
            texcoords.push(uv(i as f32, j as f32));
        }
    }
    let mut fs = vec![];
    for j in 0..2 * size {
        for i in 0..2 * size {
            let w = 2 * size + 1;
            fs.push([w * j + i, w * (j + 1) + i + 1, w * j + i + 1]);
            fs.push([w * j + i, w * (j + 1) + i, w * (j + 1) + i + 1]);
        }
    }
    Mesh::builder().verts(vs.clone()).faces(fs)
        .vertex_attrs(texcoords)
        .face_attrs(iter::repeat(1))
        .build()
        .validate().unwrap()
}


fn crates() -> Vec<Obj<Mesh<TexCoord, usize>>> {
    let mut objects = vec![];
    objects.push(Obj { tf: translate(-Y), geom: checkers() });

    for j in -10..=10 {
        for i in -10..=10 {
            let geom = unit_cube()
                // Texcoords
                .vertex_attrs([
                    uv(1.0, 1.0), uv(0.0, 1.0), uv(1.0, 0.0), uv(0.0, 0.0),
                    uv(0.0, 1.0), uv(1.0, 1.0), uv(0.0, 0.0), uv(1.0, 0.0),
                ])
                .face_attrs(iter::repeat(0))
                .build();
            let tf = translate(dir(4. * i as f32, 0., 4. * j as f32));
            objects.push(Obj { tf, geom });
        }
    }
    objects
}


fn main() {
    let margin = 50.0;
    let w = 800.0;
    let h = 600.0;

    let camera = Mat4::identity();

    let mut objects = crates();
    objects.push(Obj { tf: Mat4::identity(), geom: checkers() });

    let mut scene = Scene { objects, camera };

    let mut rdr = Renderer::new();
    rdr.options.perspective_correct = true;
    rdr.projection = perspective(0.1, 50., w / h, Deg(90.0));
    rdr.viewport = viewport(margin, h - margin, w - margin, margin);

    let tex = [
        Texture::from(load_pnm("examples/sdl/crate.ppm").unwrap()),
        Texture::from(Buffer::from_vec(2, vec![BLACK, WHITE, WHITE, BLACK])),
    ];

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    let mut cam = FpsCamera::new(pt(0.0, 0.0, -2.5), Angle::ZERO);

    let shader = &mut ShaderImpl {
        vs: identity,
        fs: |frag: Fragment<_, usize>| Some(tex[frag.uniform].sample(frag.varying)),
    };

    let font = &Font {
        glyph_w: 6,
        glyph_h: 10,
        glyphs: Texture::from(
            load_pnm("resources/font_6x10.pbm").unwrap(),
        ),
    };

    let mut t = 0.0;
    let mut sample = Stats::default();
    let sample_interval = 0.1;
    let mut text = Text::new(font, "");

    runner.run(|mut frame| {

        rdr.render_scene(&scene, shader, &mut frame.buf);

        let mut hud = Renderer::new();
        hud.projection = orthogonal(pt(0.0, 0.0, -1.0), pt(w, h, 1.0));
        hud.viewport = viewport(0.0, h, w, 0.0);
        hud.modelview = translate(dir(margin + 2.0, margin - 12.0, 0.0));

        t += frame.delta_t;
        if t > sample_interval {
            let stats = rdr.stats.diff(&sample);
            text = Text::new(font, &stats.to_string().to_ascii_uppercase());
            sample = rdr.stats;
            t = 0.0;
        }

        text.render(
            &mut hud,
            &mut ShaderImpl {
                vs: identity,
                fs: |frag: Fragment<_>| (frag.varying == WHITE).then(|| WHITE),
            },
            &mut frame.buf
        );

        let mut cam_move = ZERO;
        {
            use Scancode::*;
            for scancode in &frame.pressed_keys {
                match scancode {
                    W => cam_move.z += 4.0,
                    S => cam_move.z -= 2.0,
                    D => cam_move.x += 3.0,
                    A => cam_move.x -= 3.0,

                    P => frame.screenshot("screenshot.ppm")?,

                    _ => {}
                }
            }
            for e in frame.events {
                match e {
                    Event::MouseMotion { xrel, yrel, .. } => {
                        cam.rotate(
                            0.6 * Deg(xrel as f32),
                            0.6 * Deg(yrel as f32)
                        );
                    }
                    Event::KeyDown { scancode: Some(M), .. } => {
                        println!(
                            "pos={} az={} alt={}",
                            cam.pos, cam.azimuth, cam.altitude
                        );
                    }
                    _ => (),
                }
            }
        }

        cam.translate(frame.delta_t * cam_move);
        scene.camera = cam.world_to_view();

        rdr.stats.frames += 1;

        Ok(Run::Continue)
    }).unwrap();

    runner.print_stats(rdr.stats);
}
