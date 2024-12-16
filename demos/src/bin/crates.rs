use core::ops::ControlFlow::*;

use re::prelude::*;

use re::geom::Vertex3;
use re::math::{color::gray, point::pt3};
use re::render::{
    cam::FirstPerson, raster::Frag, shader::Shader, Batch, Camera, ModelToProj,
};

use re_front::sdl2::Window;
use re_geom::solids::Box;

fn main() {
    let mut win = Window::builder()
        .title("retrofire//crates")
        .build()
        .expect("should create window");

    let floor_shader = Shader::new(
        |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f>| frag.var.to_color4(),
    );
    let crate_shader = Shader::new(
        |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Normal3>| {
            let [x, y, z] = ((frag.var + splat(1.0)) / 2.0).0;
            rgb(x, y, z).to_color4()
        },
    );

    let (w, h) = win.dims;
    let mut cam = Camera::new(win.dims)
        .mode(FirstPerson::default())
        .viewport((10..w - 10, 10..h - 10))
        .perspective(1.0, 0.1..1000.0);

    let floor = floor();
    let crat = Box::cube(2.0).build();

    win.run(|frame| {
        // Camera

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
        let d_az = turns(ms.x() as f32) * -0.001;
        let d_alt = turns(ms.y() as f32) * 0.001;

        cam.mode.rotate(d_az, d_alt);
        cam.mode
            .translate(cam_vel.mul(frame.dt.as_secs_f32()));

        let flip = scale(vec3(1.0, -1.0, -1.0)).to();

        // Render

        let world_to_project = flip.then(&cam.world_to_project());

        let batch = Batch::new()
            .viewport(cam.viewport)
            .context(&frame.ctx);

        batch
            .clone()
            .mesh(&floor)
            .uniform(&world_to_project)
            .shader(floor_shader)
            .target(&mut frame.buf)
            .render();

        let craet = batch.clone().mesh(&crat);

        let n = 30;
        for i in (-n..=n).step_by(5) {
            for j in (-n..=n).step_by(5) {
                let pos = translate(vec3(i as f32, 0.0, j as f32)).to();
                craet
                    // TODO Try to get rid of clone
                    .clone()
                    .uniform(&pos.then(&world_to_project))
                    // TODO Allow setting shader before uniform
                    .shader(crate_shader)
                    // TODO storing &mut target makes Batch not Clone, maybe
                    //      pass to render() instead. OTOH then a Frame::batch
                    //      helper wouldn't be as useful. Maybe just wrap the
                    //      target in a RefCell?
                    .target(&mut frame.buf)
                    .render();
            }
        }
        Continue(())
    })
    .expect("should run")
}

fn floor() -> Mesh<Color3f> {
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
    bld.build()
}
