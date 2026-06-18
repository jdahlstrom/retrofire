use core::ops::ControlFlow::*;
use re::assert_approx_eq;
use re::prelude::*;

use re::math::color::gray;
use re::math::noise::perlin2::noise;
use re::math::noise::perlin3;
use re::render::{
    ModelToProj, cam::FirstPerson, cam::Fov, clip::Status::*, scene::Obj,
};
// Try also Rgb565 or Rgba4444
use re::util::{pixfmt::Rgba8888, pnm::read_pnm};

use re_front::sdl2::Window;
use re_geom::solids::{Build, Cube};

fn main() {
    let mut win = Window::builder()
        .title("retrofire//crates")
        .pixel_fmt(Rgba8888)
        .build()
        .expect("should create window");

    let tex_data = *include_bytes!("../../assets/crate.ppm");
    let tex = Texture::from(read_pnm(&tex_data[..]).expect("data exists"));

    let floor_shader = shader::new(
        |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Normal3>| {
            let d = frag.var.dot(&vec3(0.7, 0.2, 0.0));
            gray(d).to_color4()
        },
    );

    let (w, h) = win.dims;
    let mut cam = Camera::new(win.dims)
        .transform(FirstPerson::default())
        .viewport((10..w - 10, h - 10..10))
        .perspective(Fov::Diagonal(degs(90.0)), 0.1..1000.0);

    let mut floor = terrain();

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
            let t = frame.t.as_secs_f32();

            let Obj { geom, bbox, tf } = &mut floor;

            for v in &mut geom.verts {
                let pt = pt3(v.pos.x() / 4.0, t / 4.0, v.pos.z() / 4.0);

                let d = 0.1;

                let h = perlin3::noise(pt);
                let h_dx = perlin3::noise(pt + d * Vec3::X);
                let h_dz = perlin3::noise(pt + d * Vec3::Y);

                let dy_dx = vec3(d, h - h_dx, 0.0);
                let dy_dz = vec3(0.0, h - h_dz, d);

                let n = dy_dz.cross(&dy_dx).normalize();

                //assert!(ny > 0.0, "{}, {}", dy_dx * dy_dx, dy_dz * dy_dz);

                assert_approx_eq!(n.len_sqr(), 1.0);

                v.pos[1] = h - 10.0;
                v.attrib = n;
            }

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

        Continue(())
    })
    .expect("should run")
}

fn terrain() -> Obj<Normal3> {
    let mut bld = Mesh::builder();

    let size = 100;
    for j in -size..=size {
        for i in -size..=size {
            let i_odd = i & 1;
            let j_odd = j & 1;

            let pos = pt3(i as f32 / 4.0, -1.0, j as f32 / 4.0);
            bld.push_vert(pos, Vec3::zero());

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
