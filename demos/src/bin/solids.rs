use minifb::{Key, KeyRepeat};
use std::ops::ControlFlow::Continue;

use re::geom::{vertex, Mesh, Vertex};
use re::math::color::rgba;
use re::math::mat::{perspective, rotate_x, rotate_y, translate, viewport};
use re::math::spline::smootherstep;
use re::math::vec::splat;
use re::math::{rads, turns, vec2, vec3, Mat4x4};
use re::render::raster::Frag;
use re::render::shader::Shader;
use re::render::{render, ModelToProjective};
use re_front::minifb::Window;
use re_geom::solids::*;

fn main() {
    eprintln!("Press Space to cycle between objects...");

    let mut win = Window::builder()
        .title("retrofire//solids")
        .size(640, 480)
        .build();

    let shader = Shader::new(
        |v: Vertex<_, _>, mvp: &Mat4x4<ModelToProjective>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Normal3>| {
            let col = 128.0 * (frag.var + splat(1.1));
            rgba(col.x() as u8, col.y() as u8, col.z() as u8, 0)
        },
    );

    let objs = [
        Box {
            left_bot_near: vec3(-1.0, -0.4, -0.6),
            right_top_far: vec3(1.0, 0.4, 0.6),
        }
        .build(),
        Octahedron.build(),
        Icosahedron.build(),
        /*Lathe {
            pts: vec![vec2(1.0, -0.75), vec2(0.25, 0.0), vec2(1.0, 0.75)],
            sectors: 12,
            capped: true,
        }
        .build(),*/
    ];

    let camera = translate(vec3(0.0, 0.0, 5.0));
    let project = perspective(1.5, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(vec2(10, 10)..vec2(630, 470));

    let mut idx = 2;
    let mut new_idx = idx;
    let mut anim = None;
    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        let carousel = rotate_y(turns(smootherstep(anim.unwrap_or(0.0))));

        let spin = rotate_y(rads(secs)).compose(&rotate_x(rads(secs * 0.7)));

        let mvp: Mat4x4<ModelToProjective> = spin
            .then(&camera)
            .then(&carousel)
            .to()
            .then(&project);

        if frame
            .win
            .imp
            .is_key_pressed(Key::Space, KeyRepeat::No)
        {
            anim = anim.or(Some(0.0));
            new_idx = idx + 1;
        }

        if let Some(a) = &mut anim {
            *a += frame.dt.as_secs_f32();
            if *a >= 0.5 {
                idx = new_idx;
            }
            if *a >= 1.0 {
                anim = None
            }
        }

        let Mesh { faces, verts } = &objs[idx % objs.len()];
        *frame.stats +=
            render(&faces, &verts, &shader, &mvp, viewport, &mut frame.buf);
        Continue(())
    });
}
