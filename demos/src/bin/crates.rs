use core::ops::ControlFlow::*;

use minifb::MouseMode;

use re::prelude::*;

use re::math::color::gray;
use re::render::{
    batch::Batch,
    cam::{Camera, FirstPerson},
    ModelToProj,
};
use re_front::minifb::Window;
use re_geom::solids::*;

fn main() {
    const W: u32 = 640;
    const H: u32 = 480;

    let mut win = Window::builder()
        .title("retrofire//crates")
        .size(W, H)
        .build();

    let floor_shader = Shader::new(
        |v: Vertex<_, _>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f>| frag.var.to_color4(),
    );
    let crate_shader = Shader::new(
        |v: Vertex<_, _>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Normal3>| {
            let [x, y, z] = ((frag.var + splat(1.0)) / 2.0).0;
            rgb(x, y, z).to_color4()
        },
    );

    let mut cam = Camera::new(W, H)
        .mode(FirstPerson::default())
        .viewport((10..W - 10, 10..H - 10))
        .perspective(1.0, 0.1..1000.0);

    let floor = floor();
    let crat = Box::cube(2.0).build();

    win.run(|frame| {
        // Camera

        let mut cam_vel = Vec3::zero();

        let imp = &frame.win.imp;

        for key in imp.get_keys() {
            use minifb::Key::*;
            match key {
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
    });
}

fn floor() -> Mesh<Color3f> {
    let mut bld = Mesh::builder();

    let size = 50;
    for j in -size..=size {
        for i in -size..=size {
            let even_odd = ((i & 1) ^ (j & 1)) == 1;

            let pos = vec3(i as f32, -1.0, j as f32);
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
