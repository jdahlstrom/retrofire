use minifb::{Key, KeyRepeat};
use std::ops::ControlFlow::Continue;

use re::prelude::*;

use re::math::mat::RealToReal;
use re::math::spline::smootherstep;
use re::render::{render, ModelToProj};
use re_front::minifb::Window;
use re_geom::solids::*;

#[derive(Default)]
struct Carousel {
    idx: usize,
    new_idx: usize,
    t: Option<f32>,
}

impl Carousel {
    fn start(&mut self) {
        self.t = Some(0.0);
        self.new_idx = self.idx + 1;
    }
    fn update(&mut self, dt: f32) -> Mat4x4<RealToReal<3>> {
        let Some(t) = self.t.as_mut() else {
            return Mat4x4::identity();
        };
        *t += dt;
        let t = *t;
        if t >= 0.5 {
            self.idx = self.new_idx;
        }
        if t >= 1.0 {
            self.t = None
        }
        rotate_y(turns(smootherstep(t)))
    }
}

fn main() {
    eprintln!("Press Space to cycle between objects...");

    let mut win = Window::builder()
        .title("retrofire//solids")
        .size(640, 480)
        .build();

    let shader = Shader::new(
        |v: Vertex<_, _>, mvp: &Mat4x4<ModelToProj>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Normal3>| {
            let [x, y, z] = (128.0 * (frag.var + splat(1.1))).0;
            rgba(x as u8, y as u8, z as u8, 0)
        },
    );

    let objs = [
        // The five Platonic solids
        Tetrahedron.build(),
        Box::cube(1.25).build(),
        Octahedron.build(),
        Dodecahedron.build(),
        Icosahedron.build(),
    ];

    let camera = translate(vec3(0.0, 0.0, 3.0));
    let project = perspective(1.5, 4.0 / 3.0, 0.1..1000.0);
    let viewport = viewport(vec2(10, 10)..vec2(630, 470));

    let mut carousel = Carousel::default();
    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        let spin = rotate_y(rads(secs * 0.6)) //
            .compose(&rotate_x(rads(secs * 0.7)));
        let carouse = carousel.update(frame.dt.as_secs_f32());

        let mvp: Mat4x4<ModelToProj> = spin
            .then(&camera)
            .then(&carouse)
            .to()
            .then(&project);

        if frame
            .win
            .imp
            .is_key_pressed(Key::Space, KeyRepeat::No)
        {
            carousel.start();
        }

        let Mesh { faces, verts } = &objs[carousel.idx % objs.len()];
        *frame.stats +=
            render(&faces, &verts, &shader, &mvp, viewport, &mut frame.buf);

        Continue(())
    });
}
