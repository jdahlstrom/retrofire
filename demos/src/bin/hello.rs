use std::{env, fmt::Write, ops::ControlFlow::Continue};

use re::prelude::*;

use re::core::{
    math::color::hsl,
    render::{Text, World, tex::Atlas},
    util::{dims, pnm::read_pnm},
};
use re_front::{Frame, minifb::Window};

const FONT: &[u8] = include_bytes!("../../assets/font_16x24.pbm");

fn main() {
    let font = read_pnm(FONT).expect("valid image");
    let font = Atlas::grid(Dims(16, 24), font.into());

    let arg = env::args().nth(1); // Borrow checker...
    let msg = arg
        .as_deref()
        .unwrap_or("   Hello,\nRetrocomputing\n     World!");

    let mut text = Text::new(font.clone());
    write!(text, "{msg}").expect("cannot fail");

    let mut win = Window::builder()
        .title("retrofire//text")
        .dims(dims::SVGA_800_600)
        .build()
        .unwrap();

    win.ctx.face_cull = None;

    let dims = win.dims;
    let world_to_project: ProjMat3<World> = translate(vec3(0.0, 0.0, 15.0))
        .to()
        .then(&perspective(1.0, dims.aspect(), 0.1..1000.0));

    let viewport = viewport(pt2(10, 10)..pt2(dims.0 - 10, dims.1 - 10));

    win.run(|frame: &mut Frame<_, _>| {
        let secs = frame.t.as_secs_f32();

        let model_to_world = scale(0.1)
            .then(&translate((-10.0, -5.0, 5.0 * secs.sin())))
            .then(&rotate_y(rads(secs * 0.59)))
            .then(&rotate_z(rads((secs * 1.13).sin())))
            .to();

        let model_to_project = model_to_world.then(&world_to_project);

        text.color = hsl(secs / 10.0 % 1.0, 0.8, 0.6)
            .to_rgb()
            .to_color3();

        text.batch()
            .uniform(&model_to_project)
            .viewport(viewport)
            .target(&mut frame.buf)
            .context(frame.ctx)
            .render();

        Continue(())
    });
}
