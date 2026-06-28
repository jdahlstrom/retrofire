use core::ops::ControlFlow::Continue;

use re::prelude::*;

use re::core::math::{
    color::gray,
    rand::{Distrib, PointsInUnitBall, Xorshift64},
};
use re::core::render::{Model, View, cam::*, render_instanced, shader};

use re_front::minifb::Window;

fn main() {
    let tris = [tri(0, 1, 3), tri(0, 3, 2)];

    let verts: [Vertex3<()>; 4] = [
        pt3(-1.0f32, -1.0, 0.0),
        pt3(-1.0, 1.0, 0.0),
        pt3(1.0, -1.0, 0.0),
        pt3(1.0, 1.0, 0.0),
    ]
    .map(|pos| vertex(pos, ()));

    let count = 10000;
    let rng = &mut Xorshift64::default();

    let positions: Vec<Point3> = PointsInUnitBall
        .samples(rng)
        .take(count)
        .collect();

    let mut win = Window::builder()
        .title("retrofire//sprite")
        .build()
        .expect("should create window");

    let shader = shader::new(
        |v: Vertex3<()>,
         (inst_id, (mv, proj)): (
            usize,
            &(Mat4<Model, View>, ProjMat3<View>),
        )| {
            let vertex_pos = 0.008 * v.pos.to_vec().to(); // Model->View
            let pos = positions[inst_id];
            let view_pos = mv.apply(&pos.to()) + vertex_pos;
            vertex(proj.apply(&view_pos), v.pos.to_vec())
        },
        |frag: Frag<Vec3<_>>, _uni: (_, &_)| {
            let d2 = frag.var.len_sqr();
            (d2 < 1.0).then(|| {
                let col = gray(1.0) - d2 * rgb(0.25, 0.5, 1.0);
                col.to_color4()
            })
        },
    );

    let Dims(w, h) = win.dims;
    let cam = Camera::new(win.dims)
        .transform(translate(0.5 * Vec3::Z).to())
        .perspective(Fov::FocalRatio(1.0), 1e-2..1e3)
        .viewport(pt2(10, h - 10)..pt2(w - 10, 10));

    win.run(|frame| {
        let theta = rads(frame.t.as_secs_f32());

        let modelview = rotate_x(theta * 0.2)
            .then(&rotate_z(theta * 0.14))
            .to()
            .then(&cam.world_to_view());

        render_instanced(
            count,
            &tris,
            &verts,
            &shader,
            &(modelview, cam.project),
            cam.viewport,
            &mut frame.buf,
            frame.ctx,
        );
        Continue(())
    });
}
