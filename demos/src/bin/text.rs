use std::ops::ControlFlow::Continue;

use re::math::space::Real;
use re::prelude::*;
use re::render::{Model, ModelToProj, render};
use re::render::tex::{Atlas, SamplerClamp, TexCoord, uv};
use re::util::pnm::read_pnm;
use re_front::Frame;
use re_front::minifb::Window;

fn main() {
    let verts: [Vertex<Vec3<Real<3, Model>>, _>; 4] = [
        vertex(vec3(-1.0, -1.0, 0.0).to(), uv(0.0, 0.0)),
        vertex(vec3(-1.0, 1.0, 0.0).to(), uv(0.0, 1.0)),
        vertex(vec3(1.0, -1.0, 0.0).to(), uv(1.0, 0.0)),
        vertex(vec3(1.0, 1.0, 0.0).to(), uv(1.0, 1.0)),
    ];

    let font = *include_bytes!("../../../assets/font_16x24.pbm");
    let font = read_pnm(font).unwrap();
    let (cw, ch) = dbg!((font.width() / 16, font.height() / 16));
    let font = Atlas::new(cw, ch, font);

    let text = b"Hello, World! \x01";
    let mut buf = Buf2::new(cw * text.len() as u32, ch);
    let mut x = 0;
    for &c in text {
        buf.slice_mut((x..x + cw, 0..ch))
            .copy_from(*font.get(c as u32).data());
        x += cw;
    }
    let tex = buf.into();

    let mut win = Window::builder()
        .title("retrofire//text")
        .size(640, 480)
        .build();

    win.ctx.face_cull = None;

    let shader = Shader::new(
        |v: Vertex<_, _>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<TexCoord>| SamplerClamp.sample(&tex, frag.var).to_rgba(),
    );

    let mvp = translate(vec3(0.0, 0.0, 2.0))
        .to()
        .then(&perspective(1.0, 4.0 / 3.0, 0.1..1000.0));

    let viewport = viewport(vec2(10, 10)..vec2(630, 470));

    win.run(|frame: &mut Frame<_>| {
        let secs = frame.t.as_secs_f32();

        let mvp: Mat4x4<ModelToProj> = mvp
            .compose(&rotate_z(rads((secs * 1.13).sin())))
            .compose(&rotate_y(rads(secs * 0.59)))
            .compose(&translate(vec3(0.0, 0.0, secs.sin())))
            .compose(&scale(vec3(1.0, 0.1, 1.0)))
            .to();

        render(
            [Tri([0, 1, 2]), Tri([3, 2, 1])],
            verts,
            &shader,
            &mvp,
            viewport,
            &mut frame.buf,
            &frame.ctx,
        );
        Continue(())
    });
}
