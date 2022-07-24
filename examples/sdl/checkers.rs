use std::convert::identity;
use std::ops::ControlFlow::*;

use sdl2::event::Event;
use sdl2::keyboard::Scancode;

use front::sdl::*;
use geom::bbox::BoundingBox;
use geom::mesh::{Face, Mesh, vertex_indices};
use geom::solids::UnitCube;
use math::Angle::{self, Deg};
use math::transform::*;
use math::vec::*;
use render::{Render, State};
use render::raster::Fragment;
use render::scene::*;
use render::scene::Obj;
use render::shade::ShaderImpl;
use render::Stats;
use render::text::{Font, Text};
use util::buf::Buffer;
use util::color::{BLACK, WHITE};
use util::io::load_pnm;
use util::tex::*;

fn coords<T: Copy>(r: impl Iterator<Item=T> + Clone)
    -> impl Iterator<Item=(T, T)>
{
    r.clone().flat_map(move |j| r.clone().map(move |i| (i, j)))
}

type NormAndTc = (Vec4, TexCoord);
type TexIdx = usize;

fn checkers() -> Mesh<NormAndTc, TexIdx> {
    let size = 40.0;
    let vcs = [-X-Z, -X+Z, X-Z, X+Z].map(|c| c * size + W);

    let tcs = [uv(0.0, 0.0), uv(0.0, size), uv(size, 0.0), uv(size, size)];

    let verts = vec![
        vertex_indices(0, [0, 0]),
        vertex_indices(1, [0, 1]),
        vertex_indices(2, [0, 2]),
        vertex_indices(3, [0, 3]),
    ];

    let faces = vec![
        Face { verts: [0, 1, 3], attr: 0 },
        Face { verts: [0, 3, 2], attr: 0 }
    ];

    let bbox = BoundingBox::of(&vcs);

    Mesh {
        verts,
        faces,
        bbox,
        vertex_coords: vcs.into(),
        vertex_attrs: (vec![Y], tcs.into()),
        face_attrs: vec![1],
    }
}

fn objects() -> Vec<Obj<Mesh<NormAndTc, TexIdx>>> {
    let mut objects = vec![];
    objects.push(Obj { tf: translate(-Y), geom: checkers() });

    let crates = coords(-10..=10)
        .map(|(i, j)| {
            let geom = UnitCube.with_normals_and_texcoords();

            let geom = Mesh {
                verts: geom.verts,
                vertex_coords: geom.vertex_coords,
                vertex_attrs: geom.vertex_attrs,
                faces: geom.faces,
                face_attrs: vec![0],
                bbox: geom.bbox,
            };

            let tf = translate(dir(4. * i as f32, 0., 4. * j as f32));
            Obj { tf, geom }
        });
    objects.extend(crates);

    objects
}

fn main() {
    let margin = 50.0;
    let w = 800.0;
    let h = 600.0;

    let mut runner = SdlRunner::new(w as u32, h as u32).unwrap();

    let mut st = State::new();
    st.options.perspective_correct = true;
    st.projection = perspective(0.1, 50., w / h, Deg(90.0));
    st.viewport = viewport(margin, h - margin, w - margin, margin);

    let mut cam = FpsCamera::new(pt(0.0, 0.0, -2.5), Angle::ZERO);

    let mut scene = Scene {
        objects: objects(),
        camera: cam.world_to_view(),
    };

    let tex = [
        Texture::from(load_pnm("examples/sdl/crate.ppm").unwrap()),
        Texture::from(Buffer::from_vec(2, vec![BLACK, WHITE, WHITE, BLACK])),
    ];

    let samplers = [
        SamplerRepeatPot::new(&tex[0]),
        SamplerRepeatPot::new(&tex[1]),
    ];


    let shader = &mut ShaderImpl {
        vs: identity,
        fs: |frag: Fragment<NormAndTc, TexIdx>| {
            let texi = frag.uniform;
            Some(samplers[texi].sample(&tex[texi], frag.varying.1))
        },
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
        scene.render(&mut st, shader, &mut frame.buf);

        let mut hud = State::new();
        hud.projection = orthogonal(pt(0.0, 0.0, -1.0), pt(w, h, 1.0));
        hud.viewport = viewport(0.0, h, w, 0.0);
        hud.modelview = translate(dir(margin + 2.0, margin - 12.0, 0.0));

        t += frame.delta_t;
        if t > sample_interval {
            let stats = st.stats.diff(&sample);
            text = Text::new(font, &stats.avg_per_sec().to_string());
            sample = st.stats;
            t = 0.0;
        }

        text.render(
            &mut hud,
            &mut ShaderImpl {
                vs: identity,
                fs: |frag: Fragment<_>| Some(frag.varying),
            },
            &mut frame.buf,
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

                    P => if let e @ Err(_) = frame.screenshot("screenshot.ppm") {
                        return Break(e);
                    }

                    _ => {}
                }
            }
            for e in frame.events {
                match e {
                    Event::MouseMotion { xrel, yrel, .. } => {
                        cam.rotate(
                            0.6 * Deg(xrel as f32),
                            0.6 * Deg(yrel as f32),
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

        st.stats.frames += 1;

        Continue(())
    }).unwrap();

    runner.print_stats(st.stats);
}
