use std::ops::ControlFlow::*;

use re::prelude::*;

use re::math::space::Real;
use re::render::{ctx::Context, render, tex::SamplerClamp, Model, ModelToProj};
use re_front::sdl2::Window;

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
        .build()
        .expect("window creation");

    win.ctx = Context {
        face_cull: None,
        depth_test: None,
        depth_clear: None,
        depth_write: false,
        ..win.ctx
    };

    let checker = Texture::from(Buf2::new_with(8, 8, |x, y| {
        let xor = (x ^ y) & 1;
        rgba(xor as u8 * 255, 128, 255 - xor as u8 * 128, 0)
    }));

    let shader = Shader::new(
        |v: Vertex<_, _>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<_>| SamplerClamp.sample(&checker, frag.var),
    );

    let project = perspective(1.0, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(vec2(10, 10)..vec2(630, 470));

    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        let mvp = rotate_y(rads(secs))
            .then(&translate(vec3(0.0, 0.0, 3.0 + secs.sin())))
            .to()
            .then(&project);

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
    })
    .expect("error in main loop");
}
