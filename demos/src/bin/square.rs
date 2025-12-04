use core::ops::ControlFlow::*;
use re::core::math::grad::{ColorMap, Gradient2, Interp, Kind};
use re::prelude::*;

use re::core::render::{render, shader, tex::SamplerClamp};
use re::front::minifb::Window;

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
    let grad = Gradient2 {
        //kind: Kind::Linear(pt2(0.0, 0.0), pt2(256.0, 256.0)),
        //kind: Kind::Radial(pt2(64.0, 64.0), 256.0),
        kind: Kind::Conical(pt2(96.0, 96.0)),
        map: ColorMap::new(
            Interp::Smooth,
            [
                (0.1, rgb(0x99, 0x22, 0x11).to_color3f()),
                (0.3, rgb(0xFF, 0xCC, 0x66).to_color3f()),
                (0.6, rgb(0x33, 0x55, 0x11).to_color3f()),
                (1.0, rgb(0x11, 0x22, 0x66).to_color3f()),
            ],
        ),
    };
    let checker = Texture::from(Buf2::new_with((256, 256), |x, y| {
        //let xor = (x ^ y) & 1;
        //rgba(xor as u8 * 255, 128, 255 - xor as u8 * 128, 0)

        grad.eval(pt2(x as f32, y as f32))
    }));

    let shader = shader::new(
        |v: Vertex3<_>, mvp: &ProjMat3<_>| vertex(mvp.apply(&v.pos), v.attrib),
        |frag: Frag<_>| {
            SamplerClamp
                .sample(&checker, frag.var)
                .to_color4()
        },
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
            frame.ctx,
        );

        Continue(())
    });
}
