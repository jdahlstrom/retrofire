use core::ops::ControlFlow::*;

use re::prelude::*;

use re::math::color::gray;
use re::render::tex::{Atlas, Layout, SamplerOnce};
use re::render::{
    ModelToProj, cam::FirstPerson, cam::Fov, clip::Status::*, scene::Obj,
    tex::SamplerClamp,
};
// Try also Rgb565 or Rgba4444
use re::util::pnm::parse_pbm_or_pgm;
use re::util::{pixfmt::Rgba8888, pnm::read_pnm};
use re_front::sdl2::Window;
use re_geom::solids::{Build, Cube};

fn main() {
    let mut win = Window::builder()
        .title("retrofire//crates")
        .pixel_fmt(Rgba8888)
        .vsync(false)
        .build()
        .expect("should create window");

    static FONT: &[u8] = include_bytes!("../../assets/font_8x12.pbm");
    let font = read_pnm(FONT).expect("valid pnm");
    let font = Atlas::new(Layout::Grid { sub_dims: (8, 12) }, font.into());

    static CRATE: &[u8] = include_bytes!("../../assets/crate.ppm");
    let crate_tex = Texture::from(read_pnm(CRATE).expect("data exists"));

    let light_dir = vec3(-2.0, 1.0, -4.0).normalize();

    let floor_shader = shader::new(
        |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Vec2>| {
            let even_odd = (frag.var.x() > 0.5) ^ (frag.var.y() > 0.5);
            gray(if even_odd { 0.8 } else { 0.1 }).to_color4()
        },
    );
    let crate_shader = shader::new(
        |v: Vertex3<(Normal3, TexCoord)>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<(Normal3, TexCoord)>| {
            let (n, uv) = frag.var;
            let kd = lerp(n.dot(&light_dir).max(0.0), 0.4, 1.0);
            let col = SamplerClamp.sample(&crate_tex, uv);
            (col.to_color3f() * kd).to_color4()
        },
    );

    let (w, h) = win.dims;
    let mut cam = Camera::new(win.dims)
        .transform(FirstPerson::default())
        .viewport((10..w - 10, h - 10..10))
        .perspective(Fov::Diagonal(degs(90.0)), 0.1..1000.0);

    let floor = floor();
    let crates = crates();

    let mut last_dt = 0.0;
    win.run(|frame| {
        let t_secs = frame.t.as_secs_f32();
        let dt_secs = frame.dt.as_secs_f32();
        //
        // Camera
        //

        let mut cam_vel = Vec3::zero();

        let ep = &frame.win.ev_pump;

        for key in ep.keyboard_state().pressed_scancodes() {
            use sdl2::keyboard::Scancode as Sc;
            match key {
                Sc::W => cam_vel[2] += 4.0,
                Sc::S => cam_vel[2] -= 2.0,
                Sc::D => cam_vel[0] += 3.0,
                Sc::A => cam_vel[0] -= 3.0,
                _ => {}
            }
        }

        let ms = ep.relative_mouse_state();
        cam.transform.rotate(
            turns(ms.x() as f32) * -0.001,
            turns(ms.y() as f32) * -0.001,
        );
        cam.transform.translate(cam_vel.mul(dt_secs));

        //
        // Render
        //

        let world_to_project = &cam.world_to_project();

        let batch = Batch::new()
            .viewport(cam.viewport)
            .context(frame.ctx);

        // Floor
        {
            let Obj { geom, bbox, tf } = &floor;
            let model_to_project = tf.then(&world_to_project);
            if bbox.visibility(&model_to_project) != Hidden {
                batch
                    .clone()
                    .mesh(geom)
                    .uniform(&model_to_project)
                    .shader(floor_shader)
                    .target(&mut frame.buf)
                    .render();
            }
        }

        // Crates

        for Obj { geom, bbox, tf } in &crates {
            frame.ctx.stats.borrow_mut().objs.i += 1;

            let model_to_project = tf.then(&world_to_project);

            // TODO Also if `Visible`, no further clipping or culling needed
            if bbox.visibility(&model_to_project) == Hidden {
                continue;
            }

            batch
                // TODO Try to get rid of clone
                .clone()
                .mesh(geom)
                .uniform(&model_to_project)
                // TODO Allow setting shader before uniform
                .shader(crate_shader)
                // TODO storing &mut target makes Batch not Clone, maybe
                //      pass to render() instead. OTOH then a Frame::batch
                //      helper wouldn't be as useful. Maybe just wrap the
                //      target in a RefCell?
                .target(&mut frame.buf)
                .render();

            frame.ctx.stats.borrow_mut().objs.o += 1;
        }

        // UI
        let (w, h) = frame.win.dims;

        let mean_dt = last_dt.lerp(&dt_secs, 0.1);
        let mut fps_counter = Text::new(&font);
        _ = write!(fps_counter, "{:.1}", mean_dt.recip());
        last_dt = mean_dt;

        let ui = Camera::new(frame.win.dims).transform(Mat4x4::identity());

        ui.render(
            &fps_counter.geom.faces,
            &fps_counter.geom.verts,
            &Mat4x4::identity(),
            &shader::new(
                |v: Vertex3<_>, (m2v, ()): (&Mat4x4<ModelToProj>, ())| {
                    vertex(m2v.apply(&v.pos), v.attrib)
                },
                |f: Frag<_>| {
                    (fps_counter.sample(f.var).r() > 0)
                        .then_some(gray(0xff).to_rgba())
                },
            ),
            (),
            &mut frame.buf,
            &Default::default(),
        );

        Continue(())
    })
    .expect("should run")
}

fn crates() -> Vec<Obj<(Normal3, TexCoord)>> {
    let obj = Obj::new(Cube { side_len: 2.0 }.build());

    let mut res = vec![];
    let n = 30;
    for i in (-n..=n).step_by(5) {
        for j in (-n..=n).step_by(5) {
            res.push(Obj {
                tf: translate3(i as f32, 0.0, j as f32).to(),
                ..obj.clone()
            });
        }
    }
    res
}
fn floor() -> Obj<Vec2> {
    let mut bld = Mesh::builder();

    let size = 50;
    for j in -size..=size {
        for i in -size..=size {
            let i_odd = i & 1;
            let j_odd = j & 1;

            let pos = pt3(i as f32, -1.0, j as f32);
            let attrib = vec2(i_odd as f32, j_odd as f32);
            bld.push_vert(pos, attrib);

            if j > -size && i > -size {
                let w = size * 2 + 1;
                let j = size + j;
                let i = size + i;
                let [a, b, c, d] = [
                    w * (j - 1) + (i - 1),
                    w * (j - 1) + i,
                    w * j + (i - 1),
                    w * j + i,
                ]
                .map(|i| i as usize);

                if i_odd ^ j_odd != 0 {
                    bld.push_face(a, c, d);
                    bld.push_face(a, d, b);
                } else {
                    bld.push_face(b, c, d);
                    bld.push_face(b, a, c)
                }
            }
        }
    }
    Obj::new(bld.build())
}
