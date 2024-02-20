use std::ops::ControlFlow::Continue;

use re::prelude::*;

use re::math::space::Real;
use re::render::{render, Model, ModelToProj, ModelToView};
use re_front::minifb::Window;

fn main() {
    let verts: [Vertex<Vec3<Real<3, Model>>, _>; 4] = [
        vertex(vec3(-1.0, -1.0, 0.0).to(), rgb(1.0, 0.3, 0.1)),
        vertex(vec3(-1.0, 1.0, 0.0).to(), rgb(0.2, 0.8, 0.3)),
        vertex(vec3(1.0, -1.0, 0.0).to(), rgb(0.2, 0.5, 1.0)),
        vertex(vec3(1.0, 1.0, 0.0).to(), rgb(1.0, 0.3, 0.1)),
    ];

    let mut win = Window::builder()
        .title("retrofire//square")
        .size(640, 480)
        .build();

    let shader = Shader::new(
        |v: Vertex<_, _>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f>| frag.var.to_color4(),
    );

    let modelview = translate(vec3(0.0, 0.0, 2.0));
    let project = perspective(1.0, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(vec2(10, 10)..vec2(630, 470));

    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        let mv: Mat4x4<ModelToView> = modelview
            .compose(&rotate_z(rads(secs)))
            .compose(&translate(vec3(0.0, 0.0, secs.sin())))
            .to();
        let mvp = mv.then(&project);

        *frame.stats += render(
            [Tri([0, 1, 2]), Tri([3, 2, 1])],
            verts,
            &shader,
            &mvp,
            viewport,
            &mut frame.buf,
        );
        Continue(())
    });
}
