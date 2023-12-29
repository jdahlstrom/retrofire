use std::ops::ControlFlow::Continue;

use re::geom::{vertex, Tri, Vertex};
use re::math::color::rgba;
use re::math::mat::{perspective, rotate_x, rotate_z, translate, viewport};
use re::math::rand::{Distrib, UnitBall, Xorshift64};
use re::math::space::Real;
use re::math::{rads, vec2, vec3, Mat4x4, Vec3};
use re::render::raster::Frag;
use re::render::shader::Shader;
use re::render::{render, Model, ModelToView, View, ViewToProj};
use re_front::minifb::Window;

fn main() {
    let verts: [Vec3<Real<3, Model>>; 4] = [
        vec3(-1.0, -1.0, 0.0).to(),
        vec3(-1.0, 1.0, 0.0).to(),
        vec3(1.0, -1.0, 0.0).to(),
        vec3(1.0, 1.0, 0.0).to(),
    ];
    let count = 1000;
    let rng = Xorshift64(123);
    let verts: Vec<_> = UnitBall(rng)
        .iter()
        .take(count)
        .flat_map(|pos| verts.map(|p| vertex(p, pos.to())))
        .collect();

    let tris: Vec<_> = (0..count)
        .flat_map(|i| {
            let i = 4 * i;
            [Tri([i, i + 1, i + 3]), Tri([i, i + 3, i + 2])]
        })
        .collect();

    let mut win = Window::builder()
        .title("retrofire//sprite")
        .size(640, 480)
        .build();

    let shader = Shader::new(
        |v: Vertex<Vec3<_>, _>,
         (mv, p): (&Mat4x4<ModelToView>, &Mat4x4<ViewToProj>)| {
            let view_pos =
                v.pos.to::<Real<3, View>>() * 0.02 + mv.apply(&v.attrib);
            let screen_pos = p.apply(&view_pos);
            vertex(screen_pos, v.pos)
        },
        |frag: Frag<Vec3<_>>| {
            let d2 = frag.var.dot(&frag.var);
            (d2 < 1.0).then_some(rgba(
                (256.0 - d2 * 64.0) as u8,
                (256.0 - d2 * 128.0) as u8,
                (256.0 - d2 * 256.0) as u8,
                0xff,
            ))
        },
    );

    let project = perspective(1.0, 4.0 / 3.0, 0.05..1000.0);
    let viewport = viewport(vec2(10, 10)..vec2(630, 470));

    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        let mv = rotate_z(rads(0.5 * secs))
            .then(&rotate_x(rads(0.4 * secs)))
            .then(&translate(vec3(0.0, 0.0, 0.5)))
            .to::<ModelToView>();

        *frame.stats += render(
            &tris,
            &verts,
            &shader,
            (&mv, &project),
            viewport,
            &mut frame.buf,
        );
        Continue(())
    });
}
