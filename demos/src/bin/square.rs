use std::ops::ControlFlow::Continue;

use re::prelude::*;

use re::math::space::Real;
use re::render::tex::{uv, SamplerClamp, TexCoord, Texture};
use re::render::{render, Model, ModelToProj, ModelToView};
use re_front::minifb::Window;

fn main() {
    let verts: [Vertex<Vec3<Real<3, Model>>, TexCoord>; 4] = [
        vertex(vec3(-1.0, -1.0, 0.0).to(), uv(0.0, 0.0)),
        vertex(vec3(-1.0, 1.0, 0.0).to(), uv(0.0, 1.0)),
        vertex(vec3(1.0, -1.0, 0.0).to(), uv(1.0, 0.0)),
        vertex(vec3(1.0, 1.0, 0.0).to(), uv(1.0, 1.0)),
    ];

    let mut win = Window::builder()
        .title("retrofire//square")
        .size(640, 480)
        .build();

    win.ctx.face_cull = None;

    let tex = Texture::from(Buf2::new_with(8, 8, |x, y| {
        let xor = (x ^ y) & 1;
        rgba(xor as u8 * 255, 128, 255 - xor as u8 * 128, 0)
    }));

    let shader = Shader::new(
        |v: Vertex<_, TexCoord>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<TexCoord>| SamplerClamp.sample(&tex, frag.var),
    );

    let modelview = translate(vec3(0.0, 0.0, 2.0));
    let project = perspective(1.0, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(vec2(10, 10)..vec2(630, 470));

    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        let mv: Mat4x4<ModelToView> = modelview
            //.compose(&translate(vec3(0.0, 0.0, secs.sin())))
            .compose(&rotate_y(rads(secs)))
            .to();
        let mvp = mv.then(&project);

        render(
            [Tri([0, 1, 2]), Tri([3, 2, 1])],
            verts,
            &shader,
            &mvp,
            viewport,
            &mut frame.buf,
            &frame.win.ctx,
        );
        Continue(())
    });
}
