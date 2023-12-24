use std::ops::ControlFlow::Continue;

use re::geom::{vertex, Vertex};
use re::math::color::rgba;
use re::math::mat::{
    perspective, rotate_x, rotate_z, translate, viewport, RealToReal,
};
use re::math::{rads, vec2, vec3, Affine, Linear, Mat4x4};
use re::render::raster::Frag;
use re::render::shader::Shader;
use re::render::{render, ModelToProjective, ModelToView};
use re_front::minifb::Window;
use re_geom::io::read_obj;
use re_geom::solids::*;

static TEAPOT: &[u8] = include_bytes!("teapot.obj");
static BUNNY: &[u8] = include_bytes!("bunny.obj");

fn main() {
    let w = 640;
    let h = 400;
    let wi = w as i32;
    let hi = h as i32;

    let mut win = Window::builder()
        .title("minifb front demo")
        .size(w, h)
        .build();

    let light_dir = vec3(-1.0, 2.0, -1.0).normalize();

    let shader =
        Shader::new(
            |v: Vertex<_, _>,
             (mvp, rot): (
                &Mat4x4<ModelToProjective>,
                &Mat4x4<RealToReal<3>>,
            )| { vertex(mvp.apply(&v.pos), ()) },
            |frag: Frag<()>| {
                //let p = frag.var.add(&1.0.into()).mul(128.0);
                //rgba(p.x() as u8, p.y() as u8, p.z() as u8, 0)

                let n = frag.var;

                /*let sh =
                ((n.dot(&light_dir).clamp(-0.5, 1.0) + 0.5) * 0.66 * 192.0
                    + 32.0) as u8;*/

                rgba(0xFF, 0xFF, 0xFF, 0xFF)
            },
        );

    let mut teapot = read_obj(TEAPOT.iter().copied()).unwrap();
    teapot.verts = teapot
        .verts
        .into_iter()
        .map(|v| vertex(v.pos.mul(0.4).to(), v.attrib))
        .collect();

    let mut bunny = read_obj(BUNNY.iter().copied()).unwrap();
    bunny.verts = bunny
        .verts
        .into_iter()
        .map(|v| {
            vertex(v.pos.mul(16.0).sub(&vec3(0.0, 1.0, 0.0).to()), v.attrib)
        })
        .collect();

    let objs = [
        teapot,
        //bunny,
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
        //teapot(),
    ]
    .map(|m| m.with_vertex_normals());

    let objs = [bunny.with_vertex_normals()];

    let modelview = translate(vec3(0.0, 0.0, 4.0));
    let project = perspective(1.5, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(vec2(10, 10)..vec2(wi - 10, hi - 10));

    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        let rot = &rotate_x(rads(secs)).compose(&rotate_z(rads(secs * 0.7)));

        let mv: Mat4x4<ModelToView> = modelview.compose(&rot).to();
        let mvp = mv.then(&project);

        let mesh = &objs[(secs * 0.4) as usize % objs.len()];

        *frame.stats += render(
            &mesh.faces,
            &mesh.verts,
            &shader,
            (&mvp, &rot),
            viewport,
            &mut frame.buf,
        );
        Continue(())
    });
}
