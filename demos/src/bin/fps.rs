use core::ops::ControlFlow::Continue;

use minifb::MouseMode;

use re::prelude::*;

use re::render::scene::cam::{
    Camera, FirstPerson, Fov, Mode, Orbit, Resolution,
};
use re::render::scene::{Obj, Scene};
use re::render::shader::normal_to_color;
use re::render::ModelToProj;

use re_front::minifb::Window;
use re_geom::solids;
use re_geom::solids::{Icosahedron, Normal3};

fn main() {
    let res = Resolution::HD;

    let mut win = Window::builder()
        .title("minifb front demo")
        .size(res.0, res.1)
        .build();

    let floor_sh = Shader::new(
        |v: Vertex<_, _>, (mvp, _): (&Mat4x4<ModelToProj>, _)| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f>| rgba(0, 0, 0, 0), //frag.var.to_color4(),
    );
    let oct_sh = Shader::new(
        |v: Vertex<_, _>, (mvp, _): (&Mat4x4<ModelToProj>, _)| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Normal3>| normal_to_color(frag.var),
    );
    let sky_sh = Shader::new(
        |v: Vertex<_, _>, (mvp, _): (&Mat4x4<ModelToProj>, _)| {
            vertex(mvp.apply(&v.pos), v.pos)
        },
        |frag: Frag<Vec3<_>>| {
            let a = frag.var.normalize().y();

            lerp(a, rgba(0.8, 0.8, 1.0, 1.0), rgba(0.1, 0.1, 0.6, 1.0))
                .to_color4()
        },
    );

    let mut cam = Camera::<FirstPerson>::new(res);
    //let mut cam = Camera::<Orbit>::new(res);
    //cam.mode.zoom(5.0);

    let floor = Scene { objs: vec![make_floor()] };

    let mut ico = Scene {
        objs: vec![Obj {
            mesh: Icosahedron.build(),
            transform: Matrix::identity(),
        }],
    };

    let sky = Scene {
        objs: vec![Obj {
            mesh: solids::Box::cube(1.0).build(),
            transform: scale(100.0 * vec3(-1.0, 1.0, 1.0)).to(),
        }],
    };

    win.run(|frame| {
        // Transform

        ico.objs[0].transform = rotate_y(rads(frame.t_secs())).to();

        // Camera

        let imp = &frame.win.imp;
        let mut cam_vel = Vec3::zero();
        for k in imp.get_keys() {
            use minifb::Key::*;
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

        // Render

        let mut cam1 = cam.perspective(Fov::Equiv35mm(16.0), 0.1..1000.0);

        *frame.stats += cam1.render(&floor, &floor_sh, (), &mut frame.buf);
        *frame.stats += cam1.render(&ico, &oct_sh, (), &mut frame.buf);
        *frame.stats += cam1.render(
            &sky,
            &sky_sh,
            cam1.mode.world_to_view(),
            &mut frame.buf,
        );

        Continue(())
    });
}

fn make_floor() -> Obj<Color3f> {
    let (mut faces, mut verts) = (vec![], vec![]);

    let size = 10i32;
    for j in -size..=size {
        for i in -size..=size {
            let even_odd = ((i & 1) ^ (j & 1)) == 1;

            let pos = vec3(i as f32, 0.0, j as f32);
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
    Obj {
        mesh: Mesh::new(faces, verts),
        transform: translate(vec3(0.0, -1.0, 0.0)).to(),
    }
}
