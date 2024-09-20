use std::ops::ControlFlow::*;

use re::prelude::*;

use re::render::tex::{uv, SamplerClamp, TexCoord, Texture};
use re::render::{render, Model, ModelToProj};
use re_front::minifb::Window;

fn main() {
    let verts: [Vertex<Vec3<Model>, TexCoord>; 4] = [
        vertex(vec3(-1.0, -1.0, 0.0), uv(0.0, 0.0)),
        vertex(vec3(-1.0, 1.0, 0.0), uv(0.0, 1.0)),
        vertex(vec3(1.0, -1.0, 0.0), uv(1.0, 0.0)),
        vertex(vec3(1.0, 1.0, 0.0), uv(1.0, 1.0)),
    ];

    let mut win = Window::builder()
        .title("retrofire//square")
        .build();

    win.ctx.face_cull = None;

    let checker = Texture::from(Buf2::new_with(8, 8, |x, y| {
        let xor = (x ^ y) & 1;
        rgba(xor as u8 * 255, 128, 255 - xor as u8 * 128, 0)
    }));

    let shader = Shader::new(
        |v: Vertex<_, TexCoord>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<TexCoord>| SamplerClamp.sample(&checker, frag.var),
    );

    let (w, h) = win.dims;
    let project = perspective(1.0, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(vec2(10, 10)..vec2(w - 10, h - 10));

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
    });
}
