use core::ops::ControlFlow::*;

use re::prelude::*;

use re::core::math::{
    color::gray,
    rand::{DefaultRng, Distrib, Uniform},
};

use re::core::render::{
    cam::{Fov, PitchYawRoll},
    clip::Status::*,
    scene::Obj,
};
// Try also Rgb565 or Rgba4444
use re::core::util::{pixfmt::Rgba8888, pnm::read_pnm};

use re::front::sdl2::Window;
use re::geom::solids::Build;

fn main() {
    let mut win = Window::builder()
        .title("retrofire//crates")
        .pixel_fmt(Rgba8888)
        .build()
        .expect("should create window");

    let tex_data = *include_bytes!("../../assets/crate.ppm");
    let tex = Texture::from(read_pnm(&tex_data[..]).expect("data exists"));

    let light_dir = vec3(-2.0, 3.0, -4.0).normalize();

    let terrain_shader = shader::new(
        |v: Vertex3<Normal3>, mvp: &ProjMat3<_>| {
            vertex(mvp.apply(&v.pos), (v.pos.y(), v.attrib))
        },
        |frag: Frag<(f32, Normal3)>| {
            let l = frag.var.1.dot(&light_dir).max(0.0);
            let h = inv_lerp(frag.var.0, -32.0, 32.0);

            let c = if h < 0.05 {
                rgb(0.2, 0.8, 0.2)
            } else if h < 0.4 {
                lerp(h * 2.5, rgb(0.2, 0.8, 0.2), rgb(0.0, 0.3, 0.05))
            } else if h < 0.7 {
                lerp((h - 0.4) * 3.3, rgb(0.0, 0.3, 0.05), rgb(0.4, 0.4, 0.35))
            } else {
                lerp((h - 0.7) * 2.0, rgb(0.4, 0.4, 0.35), gray(1.1))
            };
            (l * rgb(1.0, 1.0, 0.8) * c + rgb(0.15, 0.15, 0.3)).to_color4()
        },
    );

    let (w, h) = win.dims;
    let mut cam = Camera::new(win.dims)
        //.transform(FirstPerson::default())
        .viewport((10..w - 10, h - 10..10))
        .transform(PitchYawRoll::default())
        .perspective(Fov::Diagonal(degs(90.0)), 0.1..1000.0);

    let floor = floor();

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
                Sc::W => target_vel_z += 10.0,
                Sc::S => target_vel_z -= 10.0,
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

        let acc = 20.0;
        cam_vel = cam_vel.lerp(&target_vel, (acc * dt).min(1.0));

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
                    .shader(terrain_shader)
                    .uniform(&model_to_project);
                b.render();
            }
        }

        // Crates

        /*for Obj { geom, bbox, tf } in &crates {
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

#[inline(never)]
fn diamond_square(n: u32) -> Buf2<f32> {
    let mut buf = Buf2::new_with((n + 1, n + 1), |_, _| f32::NAN);
    let rng = &mut DefaultRng::from_time();
    let heights = Uniform(-1.0..1.0);

    buf[[0, 0]] = 0.0; //(heights.sample(rng));
    buf[[0, n]] = 0.0; //(heights.sample(rng));
    buf[[n, 0]] = 0.0; //(heights.sample(rng));
    buf[[n, n]] = 0.0; //(heights.sample(rng));
    let mut s = n;

    let mut scale = 1.0;

    while s > 0 {
        // Diamond phase
        eprintln!("Starting diamond phase {s}...");

        for j in (0..n).step_by(s as usize) {
            for i in (0..n).step_by(s as usize) {
                let p0 = pt2(i, j);
                let p1 = pt2(i + s, j + s);
                let c = pt2(p0.x().midpoint(p1.x()), p0.y().midpoint(p1.y()));

                let mean = (buf[p0]
                    + buf[pt2(p1.x(), p0.y())]
                    + buf[pt2(p0.x(), p1.y())]
                    + buf[p1])
                    / 4.0;

                buf[c] = mean + heights.sample(rng) * scale;
            }
        }

        // eprintln!("After diamond phase {s}:");
        // for row in buf.rows() {
        //     for &c in row {
        //         if c.is_nan() {
        //             eprint!(" ____");
        //         } else {
        //             eprint!(" {c:-4.1}");
        //         }
        //     }
        //     eprintln!();
        // }

        eprintln!("Starting square phase {s}...");
        // Square phase
        for j in (0..n).step_by(s as usize) {
            for i in (0..n).step_by(s as usize) {
                //
                //          / \
                //        /     \
                //    (i,j)---(i+s,j)
                //    / | \     /
                //  /   |   \ /
                //  \   |   /
                //    \ | /
                //   (i,j+s)
                //

                let mean_u = if s / 2 <= j {
                    (buf[[i, j]]
                        + buf[[i + s, j]]
                        + buf[[i + s / 2, j - s / 2]]
                        + buf[[i + s / 2, j + s / 2]])
                        / 4.0
                } else {
                    (buf[[i, j]]
                        + buf[[i + s, j]]
                        + buf[[i + s / 2, j + s / 2]])
                        / 3.0
                };
                let mean_d = if j + s + s / 2 <= n {
                    (buf[[i, j + s]]
                        + buf[[i + s, j + s]]
                        + buf[[i + s / 2, j + s / 2]]
                        + buf[[i + s / 2, j + s + s / 2]])
                        / 4.0
                } else {
                    (buf[[i, j + s]]
                        + buf[[i + s, j + s]]
                        + buf[[i + s / 2, j + s / 2]])
                        / 3.0
                };
                let mean_l = if s / 2 <= i {
                    (buf[[i, j]]
                        + buf[[i, j + s]]
                        + buf[[i - s / 2, j + s / 2]]
                        + buf[[i + s / 2, j + s / 2]])
                        / 4.0
                } else {
                    (buf[[i, j]]
                        + buf[[i, j + s]]
                        + buf[[i + s / 2, j + s / 2]])
                        / 3.0
                };
                let mean_r = if i + s + s / 2 <= n {
                    (buf[[i + s, j]]
                        + buf[[i + s, j + s]]
                        + buf[[i + s / 2, j + s / 2]]
                        + buf[[i + s + s / 2, j + s / 2]])
                        / 4.0
                } else {
                    (buf[[i + s, j]]
                        + buf[[i + s, j + s]]
                        + buf[[i + s / 2, j + s / 2]])
                        / 3.0
                };

                buf[[i, j + s / 2]] = mean_l + heights.sample(rng) * scale;
                buf[[i + s / 2, j]] = mean_u + heights.sample(rng) * scale;
                buf[[i + s, j + s / 2]] = mean_r + heights.sample(rng) * scale;
                buf[[i + s / 2, j + s]] = mean_d + heights.sample(rng) * scale;
            }
        }

        // eprintln!("After square phase {s}:");
        // for row in buf.rows() {
        //     for &c in row {
        //         if c.is_nan() {
        //             eprint!(" ____");
        //         } else {
        //             eprint!(" {c:-4.1}");
        //         }
        //     }
        //     eprintln!();
        // }

        scale /= 2.0;
        s /= 2;
    }

    buf
}

#[inline(never)]
fn floor() -> Obj<Normal3> {
    let mut bld = Mesh::builder();

    let hf = diamond_square(512);

    let size = 256;
    for j in -size..=size {
        for i in -size..=size {
            let i_odd = i & 1;
            let j_odd = j & 1;

            let h = -32.0
                + 64.0
                    * smoothstep(smootherstep(
                        hf[[(i + size) as u32, (j + size) as u32]] * 0.5 + 0.5,
                    ));

            let pos = pt3(i as f32, -1.0 + h, j as f32);
            //let attrib = vec2(i_odd as f32, j_odd as f32);
            bld.push_vert(pos, ());

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
    Obj::new(bld.with_vertex_normals().build())
}
