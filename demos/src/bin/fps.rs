use core::ops::ControlFlow::Continue;

use minifb::MouseMode;
use re::math::mat::scale;

use re::prelude::*;

use re::render::scene::{Camera, FirstPerson};
use re::render::ModelToProj;

use re_front::minifb::Window;

fn main() {
    let mut win = Window::builder()
        .title("minifb front demo")
        .size(640, 480)
        .build();

    let shader = Shader::new(
        |v: Vertex<_, _>, (mvp, _): (&Mat4x4<ModelToProj>, ())| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f>| frag.var.to_color4(),
    );

    let mut cam = Camera::with_mode(640, 480, FirstPerson::default())
        //.viewport(vec2(10, 10)..vec2(630, 470));
        .viewport((..400, ..))
        .perspective(1.0, 0.1..1000.0);

    let floor = floor();

    win.run(|frame| {
        // Camera

        let mut cam_vel = Vec3::zero();

        use minifb::Key::*;

        let imp = &frame.win.imp;
        for k in imp.get_keys() {
            match k {
                W => cam_vel[2] += 4.0,
                S => cam_vel[2] -= 2.0,
                D => cam_vel[0] += 3.0,
                A => cam_vel[0] -= 3.0,

                _ => {}
            }
        }
        let (mx, my) = imp.get_mouse_pos(MouseMode::Pass).unwrap();

        cam.mode
            .rotate_to(degs(-0.4 * mx), degs(0.4 * (my - 240.0)));
        cam.mode.translate(cam_vel.mul(frame.dt_secs()));

        let mv = scale(vec3(1.0, -1.0, -1.0)).to();

        // Render

        *frame.stats += cam.render(
            &floor.faces,
            &floor.verts,
            &mv,
            &shader,
            (),
            &mut frame.buf,
        );

        Continue(())
    });
}

fn floor() -> Mesh<Color3f> {
    let (mut faces, mut verts) = (vec![], vec![]);

    let size = 10i32;
    for j in -size..=size {
        for i in -size..=size {
            let even_odd = ((i & 1) ^ (j & 1)) == 1;

            let pos = vec3(i as f32, -1.0, j as f32);
            let col = if even_odd {
                rgb(0.1, 0.1, 0.1)
            } else {
                rgb(0.9, 0.9, 0.9)
            };
            verts.push(vertex(pos.to(), col));

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
                    faces.push(Tri([a, d, b]));
                    faces.push(Tri([a, c, d]))
                } else {
                    faces.push(Tri([b, c, d]));
                    faces.push(Tri([b, a, c]))
                }
            }
        }
    }
    Mesh::new(faces, verts)
}
