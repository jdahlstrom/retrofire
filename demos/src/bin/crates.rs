use core::ops::ControlFlow::*;

use re::prelude::*;

use re::math::color::gray;
use re::render::{
    ModelToProj, cam::FirstPerson, cam::Fov, clip::Status::Hidden, scene::Obj,
};
// Try also Rgb565 or Rgba4444
use re::util::pixfmt::Rgba8888;

use re_front::sdl2::Window;
use re_geom::solids::Cube;

fn main() {
    let mut win = Window::builder()
        .title("retrofire//crates")
        .pixel_fmt(Rgba8888)
        .build()
        .expect("should create window");

    let floor_shader = shader::new(
        |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f>| frag.var.to_color4(),
    );
    let crate_shader = shader::new(
        |v: Vertex3<Normal3>, mvp: &Mat4x4<ModelToProj>| {
            let [x, y, z] = ((v.attrib + splat(1.0)) / 2.0).0;
            vertex(mvp.apply(&v.pos), rgb(x, y, z))
        },
        |frag: Frag<Color3f>| frag.var.to_color4(),
    );

    let (w, h) = win.dims;
    let mut cam = Camera::new(win.dims)
        .transform(FirstPerson::default())
        .viewport((10..w - 10, h - 10..10))
        .perspective(Fov::Diagonal(degs(90.0)), 0.1..1000.0);

    let floor = floor();
    let crates = crates();

    win.run(|frame| {
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
        cam.transform
            .translate(cam_vel.mul(frame.dt.as_secs_f32()));

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

        Continue(())
    })
    .expect("should run")
}

fn crates() -> Vec<Obj<Normal3>> {
    let krate = Cube { side_len: 2.0 }.build();
    let obj = Obj::new(krate);

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
fn floor() -> Obj<Color3f> {
    let mut bld = Mesh::builder();

    let size = 50;
    for j in -size..=size {
        for i in -size..=size {
            let even_odd = ((i & 1) ^ (j & 1)) == 1;

            let pos = pt3(i as f32, -1.0, j as f32);
            let col = if even_odd { gray(0.2) } else { gray(0.9) };
            bld.push_vert(pos, col);

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

                if even_odd {
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
