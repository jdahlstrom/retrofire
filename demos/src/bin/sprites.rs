use core::array::from_fn;
use core::ops::ControlFlow::Continue;

use re::prelude::*;

use re::math::rand::{UnitBall, Xorshift64};
use re::render::{
    cam::{Camera, Mode},
    render, Model, ModelToView, ViewToProj,
};
use re_front::minifb::Window;

fn main() {
    let verts: [Vec2<Model>; 4] = [
        vec2(-1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0),
    ];
    let count = 10000;
    let rng = Xorshift64::default();
    let verts: Vec<Vertex<Vec3<_>, Vec2<_>>> = UnitBall
        .samples(rng)
        .take(count)
        .flat_map(|pos| verts.map(|v| vertex(pos.to(), v)))
        .collect();

    let tris: Vec<_> = (0..count)
        .map(|i| from_fn(|j| 4 * i + j))
        .flat_map(|[a, b, c, d]| [Tri([a, b, d]), Tri([a, d, c])])
        .collect();

    let mut win = Window::builder()
        .title("retrofire//sprite")
        .size(640, 480)
        .build();

    let shader = Shader::new(
        |v: Vertex<Vec3<_>, Vec2<_>>,
         (mv, proj): (&Mat4x4<ModelToView>, &Mat4x4<ViewToProj>)| {
            let vertex_pos = 0.008 * vec3(v.attrib.x(), v.attrib.y(), 0.0);
            let view_pos = mv.apply(&v.pos) + vertex_pos;
            vertex(proj.apply(&view_pos), v.attrib)
        },
        |frag: Frag<Vec2<_>>| {
            let d2 = frag.var.len_sqr();
            (d2 < 1.0).then_some({
                // TODO ops trait for colors
                let col: Vec3 = splat(1.0) - d2 * vec3(0.25, 0.5, 1.0);
                rgba(col.x(), col.y(), col.z(), 1.0).to_color4()
            })
        },
    );

    let cam = Camera::new(640, 480)
        .mode(translate(vec3(0.0, 0.0, 0.5)).to())
        .perspective(1.0, 1e-2..1e3)
        .viewport(vec2(10, 10)..vec2(630, 470));

    win.run(|frame| {
        let theta = rads(frame.t.as_secs_f32());

        let modelview = rotate_x(theta * 0.2)
            .then(&rotate_z(theta * 0.14))
            .to()
            .then(&cam.mode.world_to_view());

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
