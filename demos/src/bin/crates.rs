use core::ops::{ControlFlow::*, Range};
use re::prelude::*;
use sdl2::event::Event;
use std::ops::Deref;

use re::core::math::{
    color::gray,
    rand::{DefaultRng, Distrib},
    spline::BSpline,
};
use re::core::render::{
    Model, World,
    cam::{FirstPerson, Fov},
    clip::Status::*,
    debug,
    scene::Obj,
    shader,
    tex::SamplerClamp,
};
// Try also Rgb565 or Rgba4444
use re::core::util::{pixfmt::Rgba8888, pnm::read_pnm};

use re::front::sdl2::Window;
use re::geom::solids::{Build, Cube, extrude};

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
            gray(if even_odd { 0.3 } else { 0.1 }).to_color4()
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
        .transform(FirstPerson::default())
        //.transform(Mat4::identity())
        .viewport((10..w - 10, h - 10..10))
        .perspective(Fov::Diagonal(degs(90.0)), 0.05..100.0);

    let floor = floor();
    let crates = crates();

    let rng = &mut DefaultRng::from_time();
    let pts: Range<Point3<World>> =
        pt3(-10.0, 0.0, -10.0)..pt3(10.0, 20.0, 10.0);
    //let dirs = vec3(-1.0, -0.1, -1.0)..vec3(1.0, 0.1, 1.0);

    // let rays = (pts.clone(), dirs);
    // let _rays = rays
    //     .samples(rng)
    //     .map(|(p, d)| Ray(p, 40.0 * d))
    //     .take(10);

    let pts = pts.samples(rng).take(50);
    let b = BSpline::new(pts);
    let mut bf = b.frame_iter(0.0003);

    let poly = 0.0
        .vary_to(1.0, 5)
        .take(4)
        .map(|a| polar(0.05, turns(a)).to_cart().to_pt());
    let tube = extrude(poly, b.frame_iter(0.002))
        .with_vertex_normals()
        .build();

    //win.ctx.face_cull = None;

    let mut paused = true;
    let mut mat: Mat4<World> = bf.next().unwrap();
    win.run(|frame| {
        //
        // Camera
        //

        let mut cam_vel: Vec3 = Vec3::zero();

        let ep = &mut frame.win.ev_pump;

        use sdl2::keyboard::Scancode as Sc;
        for key in ep.keyboard_state().pressed_scancodes() {
            match key {
                Sc::W => cam_vel[2] += 4.0,
                Sc::S => cam_vel[2] -= 2.0,
                Sc::D => cam_vel[0] += 3.0,
                Sc::A => cam_vel[0] -= 3.0,

                _ => {}
            }
        }
        for e in &frame.win.events {
            if matches!(e, Event::KeyDown { scancode: Some(Sc::P), .. }) {
                paused = !paused;
            }
        }

        if !paused {
            let ms = ep.relative_mouse_state();
            cam.transform.rotate(
                turns(ms.x() as f32) * 0.001,
                turns(ms.y() as f32) * -0.001,
            );

            let Some(m) = bf.next() else {
                return Break(());
            };
            mat = m;
        }
        let sdl = frame.win.canvas.window_mut().subsystem().sdl();

        sdl.mouse().capture(!paused);
        sdl.mouse().show_cursor(paused);
        sdl.mouse().set_relative_mouse_mode(!paused);

        //cam.transform.pos = mat.origin() + 0.2 * Vec3::Y;
        // cam.transform
        //     .look_at(cam.transform.pos + mat.linear().col_vec(2));

        //    .translate(cam_vel.mul(frame.dt.as_secs_f32()));

        //
        // Render
        //

        //let world_to_project = &cam.world_to_project();

        let mat: Mat4<World> = translate3(0.0, 0.25, 0.0).to().then(&mat);

        let world_to_project = mat
            .inverse()
            .then(&cam.world_to_view())
            .then(&cam.project);

        let batch = Batch::new()
            .viewport(cam.viewport)
            .target(frame.buf)
            .context(frame.ctx);

        batch
            .clone()
            .mesh(&tube)
            .shader(shader::new(
                |v: Vertex3<Normal3, _>, tf: &ProjMat3<_>| {
                    vertex(tf.apply(&v.pos.to()), v.attrib)
                },
                |f: Frag<Normal3>| debug::dir_to_rgb(f.var).to_color4(),
            ))
            .uniform(&world_to_project)
            .render();

        // Spline
        let circle = debug::circle::<Model>(Point3::origin(), 1.0)
            //debug::sphere(Point3::origin(), 0.1)
            //debug::basis(m2w)
            //debug::ray(Point3::origin(), Vec3::Z)
            .viewport(cam.viewport)
            .target(frame.buf)
            .context(frame.ctx);

        for m2w in b.frame_iter(0.01) {
            let m2p: ProjMat3<World> = m2w.to().then(&world_to_project);

            circle
                .clone()
                .uniform(&scale(0.18.into()).to().then(&m2p))
                .render();
        }

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
        /*
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
        }*/

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
