use core::ops::ControlFlow::*;

use re::prelude::*;

use re::core::math::color::gray;
use re::core::render::{
    cam::{FirstPerson, Fov, PitchYawRoll},
    clip::Status::*,
    scene::Obj,
    tex::SamplerClamp,
};
// Try also Rgb565 or Rgba4444
use re::core::util::{pixfmt::Rgba8888, pnm::read_pnm};

use re::front::sdl2::Window;
use re::geom::solids::{Build, Cube};

fn main() {
    let mut win = Window::builder()
        .title("retrofire//crates")
        .pixel_fmt(Rgba8888)
        .build()
        .expect("should create window");

    let tex_data = *include_bytes!("../../assets/crate.ppm");
    let tex = Texture::from(read_pnm(&tex_data[..]).expect("data exists"));

    let light_dir = vec3(-2.0, 1.0, -4.0).normalize();

    let floor_shader = shader::new(
        |v: Vertex3<_>, mvp: &ProjMat3<_>| vertex(mvp.apply(&v.pos), v.attrib),
        |frag: Frag<Vec2>| {
            let even_odd = (frag.var.x() > 0.5) ^ (frag.var.y() > 0.5);
            gray(if even_odd { 0.8 } else { 0.1 }).to_color4()
        },
    );
    let crate_shader = shader::new(
        |v: Vertex3<(Normal3, TexCoord)>, mvp: &ProjMat3<_>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<(Normal3, TexCoord)>| {
            let (n, uv) = frag.var;
            let kd = lerp(n.dot(&light_dir).max(0.0), 0.4, 1.0);
            let col = SamplerClamp.sample(&tex, uv);
            (col.to_color3f() * kd).to_color4()
        },
    );

    let (w, h) = win.dims;
    let mut cam = Camera::new(win.dims)
        //.transform(FirstPerson::default())
        .viewport((10..w - 10, h - 10..10))
        .transform(PitchYawRoll::default())
        .perspective(Fov::Diagonal(degs(90.0)), 0.1..1000.0);

    let floor = floor();
    let crates = crates();

    let mut target_vel_z = 0.0;
    let mut cam_vel = Vec3::zero();
    let (mut p, mut y, mut r) = (degs(0.0), degs(0.0), degs(0.0));
    win.run(|frame| {
        let dt = frame.dt.as_secs_f32();

        //
        // Camera
        //

        let ep = &frame.win.ev_pump;

        // target angular vels
        let (mut tp, mut ty, mut tr) = (degs(0.0), degs(0.0), degs(0.0));

        let rv = degs(90.0) * dt;
        for key in ep.keyboard_state().pressed_scancodes() {
            use sdl2::keyboard::Scancode as Sc;
            match key {
                Sc::W => target_vel_z += 0.1,
                Sc::S => target_vel_z -= 0.05,
                Sc::A => tr = rv,
                Sc::D => tr = -rv,

                Sc::Down => tp = -rv,
                Sc::Up => tp = rv,

                Sc::Left => ty = -rv,
                Sc::Right => ty = rv,
                _ => {}
            }
        }

        let ra = 4.0;
        p = p.lerp(&tp, ra * dt);
        y = y.lerp(&ty, ra * dt);
        r = r.lerp(&tr, ra * dt);

        let target_vel = cam
            .transform
            .view_to_world()
            .apply(&(target_vel_z * Vec3::Z));

        let acc = 8.0;
        cam_vel = cam_vel.lerp(&target_vel, acc * dt);

        cam.transform.rotate(p, y, r);

        cam.transform
            .translate_to(cam.transform.pos + cam_vel * dt);

        //
        // Render
        //

        let world_to_project = &cam.world_to_project();

        let batch = Batch::new()
            .viewport(cam.viewport)
            .target(frame.buf)
            .context(frame.ctx);

        // Floor
        {
            let Obj { bbox, tf, geom } = &floor;
            let model_to_project = tf.then(&world_to_project);
            if bbox.visibility(&model_to_project) != Hidden {
                let mut b = batch
                    .clone()
                    .mesh(geom)
                    .shader(floor_shader)
                    .uniform(&model_to_project);
                b.render();
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
                .shader(crate_shader)
                .uniform(&model_to_project)
                .render();

            frame.ctx.stats.borrow_mut().objs.o += 1;
        }

        Continue(())
    })
    .expect("should run");
}

fn crates() -> Vec<Obj<(Normal3, TexCoord)>> {
    let obj = Obj::new(Cube { side_len: 2.0 }.build());

    let mut res = vec![];
    let n = 30;
    for i in (-n..=n).step_by(5) {
        for j in (-n..=n).step_by(5) {
            res.push(Obj {
                tf: translate3(i as f32, 0.0, j as f32).to(),
                // TODO Same geometry cloned many times
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
