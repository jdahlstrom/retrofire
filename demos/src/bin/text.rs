use std::ops::ControlFlow::Continue;

use re::math::mat::scale;
use re::prelude::*;

use re::math::space::Real;
use re::render::tex::{uv, Atlas, SamplerClamp, TexCoord};
use re::render::{render, Model, ModelToProj, ModelToView};
use re::util::pnm::read_pnm;
use re_front::minifb::Window;
use re_front::Frame;

fn main() {
    let verts: [Vertex<Vec3<Real<3, Model>>, _>; 4] = [
        vertex(vec3(-1.0, -1.0, 0.0).to(), uv(0.0, 0.0)),
        vertex(vec3(-1.0, 1.0, 0.0).to(), uv(0.0, 1.0)),
        vertex(vec3(1.0, -1.0, 0.0).to(), uv(1.0, 0.0)),
        vertex(vec3(1.0, 1.0, 0.0).to(), uv(1.0, 1.0)),
    ];

    let (cw, ch) = (12, 20);
    let font = *include_bytes!("../../../assets/font_12x20.pbm");
    let font = Atlas::new(cw, ch, read_pnm(font).unwrap());

    let text = b"Hello, World!";
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

    //win.ctx.color_clear = Some(rgba(0xff, 0xff, 0xff, 0xff));

    let shader = Shader::new(
        |v: Vertex<_, _>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<TexCoord>| SamplerClamp.sample(&tex, frag.var).to_rgba(),
    );

    let modelview = translate(vec3(0.0, 0.0, 2.0));
    let project = perspective(1.0, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(vec2(10, 10)..vec2(630, 470));

    win.run(|frame: &mut Frame<_>| {
        let secs = frame.t.as_secs_f32();

        let mv: Mat4x4<ModelToView> = modelview
            .compose(&rotate_z(rads(secs)))
            .compose(&rotate_y(rads((secs * 1.3).sin())))
            .compose(&translate(vec3(0.0, 0.0, secs.sin())))
            .compose(&scale(vec3(1.0, 0.1, 1.0)))
            .to();
        let mvp = mv.then(&project);

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
