use std::ops::ControlFlow::Continue;

use re::geom::{vertex, Sprite, Vertex};
use re::math::color::rgba;
use re::math::mat::{perspective, rotate_x, translate, viewport};
use re::math::vec::Real;
use re::math::{degs, vec2, vec3, Affine, Mat4x4, Vec3};
use re::render::raster::Frag;
use re::render::shader::Shader;
use re::render::tex::{uv, TexCoord};
use re::render::{render_sprites, Model, ModelToView, ViewToProjective};
use rf::minifb::Window;

fn main() {
    let mut win = Window::builder()
        .title("minifb front demo")
        .size(640, 480)
        .build();

    let shader = Shader::new(
        |v: Vertex<Vec3<Real<3, Model>>, _>,
         (sp, (mv, vp)): (
            &Sprite<Vec3<_>, _>,
            &(Mat4x4<ModelToView>, Mat4x4<ViewToProjective>),
        )| {
            let center = mv.apply(&sp.center.to());

            let dx = v.pos.x() * sp.size.x();
            let dy = v.pos.y() * sp.size.y();

            let pos = center.add(&vec3(dx, dy, 0.0).to());
            let pos = vp.apply(&pos);

            vertex(pos, v.attrib)
        },
        |frag: Frag<TexCoord>| {
            let l = frag.var.sub(&uv(0.5, 0.5)).len();
            (l < 0.5).then_some(rgba(0xFF, 0, 0, 0))
        },
    );

    let camera = translate(vec3(0.0, 0.0, 8.0)).compose(&rotate_x(degs(-60.0)));
    let project = perspective(1.0, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(vec2(40, 40)..vec2(600, 440));

    let r = 4i8;
    let d = r as usize * 2 + 1;

    let sprites: Vec<_> = (0..d * d * d)
        .map(|i| Sprite {
            center: vec3(
                (i % d) as f32,
                (i / d % d) as f32,
                (i / d / d) as f32,
            )
            .add(&(-r as f32).into())
            .to(),
            size: 0.2.into(),
            verts: [0, 1, 2, 3],
        })
        .collect();

    let verts: [Vertex<Vec3<Real<3, Model>>, _>; 4] = [
        vertex(vec3(-0.5, -0.5, 0.0).to(), uv(0.0, 0.0)),
        vertex(vec3(0.5, -0.5, 0.0).to(), uv(1.0, 0.0)),
        vertex(vec3(-0.5, 0.5, 0.0).to(), uv(0.0, 1.0)),
        vertex(vec3(0.5, 0.5, 0.0).to(), uv(1.0, 1.0)),
    ];

    win.run(|frame| {
        let t = frame.t.as_secs_f32() * 0.2;

        let mv: Mat4x4<ModelToView> = camera
            //.compose(&translate(vec3(0.0, 0.0, 5.0 * t.sin())))
            //.compose(&rotate_x(rads(t)))
            //.compose(&rotate_z(rads(t)))
            .to();
        let _mvp = mv.then(&project);

        *frame.stats += render_sprites(
            &sprites,
            &verts,
            &shader,
            &(mv, project),
            viewport,
            &mut frame.buf,
        );
        Continue(())
    });
}
