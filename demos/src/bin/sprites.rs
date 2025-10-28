use core::{array::from_fn, ops::ControlFlow::Continue};

use re::prelude::*;

use re::math::color::gray;
use re::math::mat::{Apply, ProjMat3};
use re::math::rand::{Distrib, PointsInUnitBall, Xorshift64};

use re::render::{Model, cam::*, render};

use re_front::minifb::Window;

fn main() {
    let verts: [Vec2<Model>; 4] = [
        vec2(-1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0),
    ];
    let count = 10000;
    let rng = &mut Xorshift64::default();

    let verts: Vec<Vertex3<Vec2<_>>> = PointsInUnitBall
        .samples(rng)
        .take(count)
        .flat_map(|pos| verts.map(|v| vertex(pos.to(), v)))
        .collect();

    let tris: Vec<_> = (0..count)
        .map(|i| from_fn(|j| 4 * i + j))
        .flat_map(|[a, b, c, d]| [tri(a, b, d), tri(a, d, c)])
        .collect();

    let mut win = Window::builder()
        .title("retrofire//sprite")
        .build()
        .expect("should create window");

    let shader = shader::new(
        |v: Vertex3<Vec2<_>>,
         (mv, proj): (&Mat4<Model, View>, &ProjMat3<View>)| {
            let vertex_pos = 0.008 * v.attrib.to_vec3().to();
            let view_pos = mv.apply(&v.pos) + vertex_pos;
            vertex(proj.apply(&view_pos), v.attrib)
        },
        |frag: Frag<Vec2<_>>| {
            let d2 = frag.var.len_sqr();
            (d2 < 1.0).then(|| {
                let col = gray(1.0) - d2 * rgb(0.25, 0.5, 1.0);
                col.to_color4()
            })
        },
    );

    let (w, h) = win.dims;
    let cam = Camera::new(win.dims)
        .transform(translate(0.5 * Vec3::Z).to())
        .perspective(Fov::FocalRatio(1.0), 1e-2..1e3)
        .viewport(pt2(10, h - 10)..pt2(w - 10, 10));

    win.run(|frame| {
        let theta = rads(frame.t.as_secs_f32());

        let modelview = rotate_x(theta * 0.2)
            .then(&rotate_z(theta * 0.14))
            .to()
            .then(&cam.transform.world_to_view());

        render(
            &tris,
            &verts,
            &shader,
            (&modelview, &cam.project),
            cam.viewport,
            &mut frame.buf,
            frame.ctx,
        );
        Continue(())
    });
}
