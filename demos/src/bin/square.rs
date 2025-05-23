use std::ops::ControlFlow::*;

use re::prelude::*;

use re::math::{pt2, pt3};
use re::render::{Context, ModelToProj, render, tex::SamplerClamp};

use re_front::minifb::Window;

fn main() {
    let verts: [Vertex3<TexCoord>; 4] = [
        vertex(pt3(-1.0, -1.0, 0.0), uv(0.0, 0.0)),
        vertex(pt3(-1.0, 1.0, 0.0), uv(0.0, 1.0)),
        vertex(pt3(1.0, -1.0, 0.0), uv(1.0, 0.0)),
        vertex(pt3(1.0, 1.0, 0.0), uv(1.0, 1.0)),
    ];

    let mut win = Window::builder()
        .title("retrofire//square")
        .build()
        .expect("should create window");

    win.ctx = Context {
        face_cull: None,
        depth_test: None,
        depth_clear: None,
        depth_write: false,
        ..win.ctx
    };

    let checker = Texture::from(Buf2::new_with((8, 8), |x, y| {
        let xor = (x ^ y) & 1;
        rgba(xor as u8 * 255, 128, 255 - xor as u8 * 128, 0)
    }));

    let shader = shader::new(
        |v: Vertex3<_>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<_>| SamplerClamp.sample(&checker, frag.var),
    );

    let (w, h) = win.dims;
    let project = perspective(1.0, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(pt2(10, 10)..pt2(w - 10, h - 10));

    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        let mvp = rotate_y(rads(secs))
            .then(&translate((3.0 + secs.sin()) * Vec3::Z))
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
        /*render(
            [[0, 1], [1, 2], [2, 3], [1, 3]],
            verts,
            &shader2,
            &mvp,
            viewport,
            &mut frame.buf,
            &frame.ctx,
        );*/

        Continue(())
    });
}
