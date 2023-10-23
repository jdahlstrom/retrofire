#![cfg(feature = "minifb")]

use std::ops::ControlFlow::Continue;

use re::geom::{vertex, Tri, Vertex};
use re::math::color::{rgb, Color3f};
use re::math::mat::{perspective, rotate_z, translate, viewport};
use re::math::{rads, vec2, vec3, Mat4x4, Vec3};
use re::render::raster::Frag;
use re::render::shader::Shader;
use re::render::{render, ModelToProjective, ModelToView};
use rf::minifb::Window;

fn main() {
    let verts = [
        vertex(vec3(-1.0, -1.0, 0.0), rgb(1.0, 0.3, 0.1)),
        vertex(vec3(-1.0, 1.0, 0.0), rgb(0.2, 0.8, 0.3)),
        vertex(vec3(1.0, -1.0, 0.0), rgb(0.2, 0.5, 1.0)),
        vertex(vec3(1.0, 1.0, 0.0), rgb(1.0, 0.3, 0.1)),
    ];

    let mut win = Window::builder()
        .title("minifb front demo")
        .size(640, 480)
        .build();

    let shader = Shader::new(
        |Vertex { pos, attrib }: Vertex<Vec3, _>,
         mvp: &Mat4x4<ModelToProjective>| {
            vertex(mvp.apply(&pos.to()), attrib)
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

        render(
            &[Tri([0, 1, 2]), Tri([3, 2, 1])],
            &verts,
            &shader,
            &mvp,
            viewport,
            &mut frame.buf,
        );
        Continue(())
    });
}