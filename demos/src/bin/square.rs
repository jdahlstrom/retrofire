use std::ops::ControlFlow::*;

use re::prelude::*;
use re::render::tex::SamplerClamp;

use re_front::minifb::Window;

fn main() {
    // Vertices of a square
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

    // Disable backface culling and depth buffering
    win.ctx = Context {
        face_cull: None,
        depth_test: None,
        depth_clear: None,
        depth_write: false,
        ..win.ctx
    };

    // Texture with a check pattern
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
    let projection = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(10, 10)..pt2(w - 10, h - 10));

    win.run(|frame| {
        let time = frame.t.as_secs_f32();

        let model_view_project = rotate_y(rads(time))
            .then(&translate3(0.0, 0.0, 3.0 + time.sin()))
            .to()
            .then(&projection);

        render(
            [tri(0, 1, 2), tri(3, 2, 1)],
            verts,
            &shader,
            &model_view_project,
            viewport,
            &mut frame.buf,
            &frame.ctx,
        );

        Continue(())
    });
}
