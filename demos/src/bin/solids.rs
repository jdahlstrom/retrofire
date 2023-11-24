use std::ops::ControlFlow::Continue;

use re::geom::{vertex, Vertex};
use re::math::color::rgba;
use re::math::mat::{perspective, rotate_x, rotate_z, translate, viewport};
use re::math::{rads, vec2, vec3, Affine, Linear, Mat4x4, Vec3};
use re::render::raster::Frag;
use re::render::shader::Shader;
use re::render::{render, ModelToProjective, ModelToView};
use re_front::minifb::Window;
use re_geom::solids::*;

fn main() {
    let mut win = Window::builder()
        .title("minifb front demo")
        .size(640, 480)
        .build();

    let shader = Shader::new(
        |v: Vertex<_, _>, mvp: &Mat4x4<ModelToProjective>| {
            vertex(mvp.apply(&v.pos), v.pos.to())
        },
        |frag: Frag<Vec3>| {
            let p = frag.var.add(&1.0.into()).mul(128.0);
            rgba(p.x() as u8, p.y() as u8, p.z() as u8, 0)
        },
    );

    let objs = [
        Box {
            dimensions: vec3(1.8, 0.8, 1.2),
        }
        .build(),
        UnitOctahedron.build(),
        Cone {
            sectors: 12,
            capped: true,
            base_radius: 1.0,
            cap_radius: 0.0,
        }
        .build(),
        Cylinder {
            sectors: 12,
            capped: true,
            radius: 0.8,
        }
        .build(),
        Sphere {
            sectors: 11,
            segments: 5,
            radius: 1.0,
        }
        .build(),
        Capsule {
            sectors: 11,
            segments: 5,
            radius: 0.6,
        }
        .build(),
        Torus {
            major_sectors: 17,
            minor_sectors: 9,
            major_radius: 0.8,
            minor_radius: 0.4,
        }
        .build(),
        teapot(),
    ];

    let modelview = translate(vec3(0.0, 0.0, 4.0));
    let project = perspective(1.5, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(vec2(10, 10)..vec2(630, 470));

    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        let mv: Mat4x4<ModelToView> = modelview
            .compose(&rotate_x(rads(secs)))
            .compose(&rotate_z(rads(secs * 0.7)))
            .to();
        let mvp = mv.then(&project);

        let mesh = &objs[(secs * 0.4) as usize % objs.len()];

        *frame.stats += render(
            &mesh.faces,
            &mesh.verts,
            &shader,
            &mvp,
            viewport,
            &mut frame.buf,
        );
        Continue(())
    });
}
