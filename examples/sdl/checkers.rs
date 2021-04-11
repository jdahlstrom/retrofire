use sdl2::event::Event;
use sdl2::keyboard::Scancode;

use geom::mesh::Mesh;
use geom::solids::unit_cube;
use math::Angle;
use math::Angle::Deg;
use math::mat::Mat4;
use math::transform::*;
use math::vec::*;
use render::*;
use render::raster::Fragment;
use render::scene::*;
use render::tex::*;
use util::io::load_ppm;

use crate::runner::*;

mod runner;

fn checkers() -> Mesh<TexCoord, ()> {
    let size: usize = 40;
    let isize = size as i32;

    let mut vs = vec![];
    let mut texcoords = vec![];
    for j in -isize..=isize {
        for i in -isize..=isize {
            vs.push(pt(i as f32, 0.0, j as f32));
            texcoords.push(uv(0.0, 0.0));
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
        .build()
        .validate().unwrap()
}


fn crates() -> Vec<Obj<TexCoord, ()>> {
    let mut objects = vec![];
    objects.push(Obj { tf: translate(-Y), mesh: checkers() });

    for j in -10..=10 {
        for i in -10..=10 {
            let mesh = unit_cube()
                // Texcoords
                .vertex_attrs([
                    uv(1.0, 1.0), uv(0.0, 1.0), uv(1.0, 0.0), uv(0.0, 0.0),
                    uv(0.0, 1.0), uv(1.0, 1.0), uv(0.0, 0.0), uv(1.0, 0.0),
                ].iter().copied())
                .build();
            let tf = translate(dir(4. * i as f32, 0., 4. * j as f32));
            objects.push(Obj { tf, mesh });
        }
    }
    objects
}


fn main() {
    let margin = 50;
    let w = 800;
    let h = 600;

    let camera = Mat4::identity();

    let mut objects = crates();
    objects.push(Obj { tf: Mat4::identity(), mesh: checkers() });

    let mut scene = Scene { objects, camera };

    let mut rdr = Renderer::new();
    rdr.options.perspective_correct = true;
    rdr.projection = perspective(0.1, 50., w as f32 / h as f32, Deg(90.0));
    rdr.viewport = viewport(margin as f32, (h - margin) as f32,
                            (w - margin) as f32, margin as f32);

    let tex = Texture::from(load_ppm("examples/sdl/crate.ppm").unwrap());

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    let mut cam = FpsCamera::new(pt(0.0, 0.0, -2.5), Angle::ZERO);

    runner.run(|mut frame| {
        {
            let Frame { buf, zbuf, .. } = &mut frame;
            rdr.render_scene(&scene, &mut Raster {
                shade: |frag: Fragment<_>, _| tex.sample(frag.varying),
                test: |frag| zbuf.test(frag),
                output: |(x, y), c| buf.plot(x, y, c),
            });
        }


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
