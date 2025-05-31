use core::ops::ControlFlow::*;

use re::prelude::*;

use re::math::{color::gray, mat::RealToProj};
use re::render::{
    Batch, Camera, ModelToProj,
    cam::{FirstPerson, Fov},
    tex::SamplerRepeatPot,
};
use re::util::pixfmt::Rgba8888;

use re_front::sdl2::Window;
use re_geom::solids::Cube;

fn main() {
    let mut win = Window::builder()
        .title("retrofire//crates")
        // Try also Rgb565 or Rgba4444
        .pixel_fmt(Rgba8888)
        .build()
        .expect("should create window");

    let check = Texture::from(Buf2::new_with((2, 2), |x, y| {
        let even_odd = (x ^ y) as u8 & 1;
        gray(0xCFu8 * even_odd + 0x20).to_rgba()
    }));

    let floor_sampler = SamplerRepeatPot::new(&check);
    let floor_shader = shader::new(
        |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<TexCoord>| floor_sampler.sample(&check, frag.var),
    );
    let crate_shader = shader::new(
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
        .transform(FirstPerson::default())
        .viewport((10..w - 10, 10..h - 10))
        .perspective(Fov::Diagonal(degs(90.0)), 0.1..1000.0);

    let floor = Mesh::new(
        [Tri([0, 2, 1]), Tri([0, 3, 2])],
        [
            vertex(pt3(-1e3, 0.0, -1e3), uv(0.0, 0.0)),
            vertex(pt3(1e3, 0.0, -1e3), uv(1e3, 0.0)),
            vertex(pt3(1e3, 0.0, 1e3), uv(1e3, 1e3)),
            vertex(pt3(-1e3, 0.0, 1e3), uv(0.0, 1e3)),
        ],
    );
    let krate = Cube { side_len: 2.0 }.build();

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

        cam.transform.rotate(d_az, d_alt);
        cam.transform
            .translate(cam_vel.mul(frame.dt.as_secs_f32()));

        let flip = scale3(1.0, -1.0, -1.0).to();

        // Render

        let world_to_project: Mat4x4<RealToProj<World>> =
            flip.then(&cam.world_to_project());

        let batch = Batch::new()
            .viewport(cam.viewport)
            .context(&frame.ctx);

        let mvp = translate3(0.0, -1.0, 0.0)
            .to()
            .then(&world_to_project);
        batch
            .clone()
            .mesh(&floor)
            .uniform(&mvp)
            .shader(floor_shader)
            .target(&mut frame.buf)
            .render();

        let krate = batch.mesh(&krate).shader(crate_shader);

        let n = 30;
        for i in (-n..=n).step_by(5) {
            for j in (-n..=n).step_by(5) {
                let mvp = translate3(i as f32, 0.0, j as f32)
                    .to()
                    .then(&world_to_project);

                krate
                    // TODO Try to get rid of clone
                    .clone()
                    .uniform(&mvp)
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
