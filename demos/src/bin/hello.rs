use std::{env, fmt::Write, ops::ControlFlow::Continue};

use re::prelude::*;

use re::core::{
    render::{Text, tex::Atlas, tex::Layout},
    util::pnm::parse_pnm,
};

use re_front::{Frame, dims::SVGA_800_600, minifb::Window};

fn main() {
    let font = *include_bytes!("../../assets/font_16x24.pbm");
    let font = parse_pnm(font).expect("valid image");
    let font = Atlas::new(Layout::Grid { sub_dims: (16, 24) }, font.into());

    let msg = env::args().nth(1); // Borrow checker...
    let msg = msg
        .as_deref()
        .unwrap_or("   Hello,\nRetrocomputing\n     World!");

    let mut text = Text::new(font);
    write!(text, "{msg}").expect("cannot fail");

    let mut win = Window::builder()
        .title("retrofire//text")
        .dims(SVGA_800_600)
        .build()
        .unwrap();

    win.ctx.face_cull = None;

    let shader = shader::new(
        |v: Vertex<_, _>, mvp: &ProjMat3<Model>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<TexCoord>| text.sample(frag.var).to_rgba(),
    );

    let vp: ProjMat3<World> = translate(vec3(0.0, 0.0, 15.0))
        .to()
        .then(&perspective(1.0, 4.0 / 3.0, 0.1..1000.0));

    let viewport = viewport(pt2(10, 10)..pt2(790, 590));

    win.run(|frame: &mut Frame<_, _>| {
        let secs = frame.t.as_secs_f32();

        let mvp = scale(splat(0.1))
            .then(&translate(vec3(-10.0, -5.0, 5.0 * secs.sin())))
            .then(&rotate_y(rads(secs * 0.59)))
            .then(&rotate_z(rads((secs * 1.13).sin())))
            .to()
            .then(&vp);

        render(
            &text.geom.faces,
            &text.geom.verts,
            &shader,
            &mvp,
            viewport,
            &mut frame.buf,
            frame.ctx,
        );
        Continue(())
    });
}
