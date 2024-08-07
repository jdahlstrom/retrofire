use core::ops::ControlFlow::Continue;

use minifb::{Key, KeyRepeat};

use re::prelude::*;

use re::math::{mat::RealToReal, spline::smootherstep};
use re::render::{cam::Camera, ModelToProj, ModelToWorld};
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
    const W: u32 = 640;
    const H: u32 = 480;

    eprintln!("Press Space to cycle between objects...");

    let mut win = Window::builder()
        .title("retrofire//solids")
        .size(W, H)
        .build();

    let shader = Shader::new(
        |v: Vertex<_, Normal3>, (mvp, _): (&Mat4x4<ModelToProj>, _)| {
            let [x, y, z] = (0.45 * (v.attrib + splat(1.1))).0;
            vertex(mvp.apply(&v.pos), rgb(x, y, z))
        },
        |frag: Frag<Color3f>| frag.var.to_color4(),
    );

    let objs = [
        // The five Platonic solids
        Tetrahedron.build(),
        Box::cube(1.25).build(),
        Octahedron.build(),
        Dodecahedron.build(),
        Icosahedron.build(),
        // Surfaces of revolution
        Lathe::new(
            vec![
                vertex(vec2(0.75, -0.5), vec2(1.0, 1.0)),
                vertex(vec2(0.55, -0.25), vec2(1.0, 0.5)),
                vertex(vec2(0.5, 0.0), vec2(1.0, 0.0)),
                vertex(vec2(0.55, 0.25), vec2(1.0, -0.5)),
                vertex(vec2(0.75, 0.5), vec2(1.0, 1.0)),
            ],
            13,
        )
        .capped(true)
        .build(),
        Sphere {
            sectors: 12,
            segments: 6,
            radius: 1.0,
        }
        .build(),
        Cylinder {
            sectors: 12,
            capped: true,
            radius: 0.8,
        }
        .build(),
        Cone {
            sectors: 12,
            capped: true,
            base_radius: 1.1,
            apex_radius: 0.1,
        }
        .build(),
        Capsule {
            sectors: 10,
            cap_segments: 5,
            radius: 0.5,
        }
        .build(),
        Torus {
            major_radius: 0.9,
            minor_radius: 0.3,
            major_sectors: 16,
            minor_sectors: 8,
        }
        .build(),
    ];

    let camera = Camera::with_mode(W, H, Mat4x4::identity())
        .perspective(1.5, 0.1..1000.0)
        .viewport(vec2(10, 10)..vec2(W - 10, H - 10));

    let translate = translate(vec3(0.0, 0.0, 4.0));

    let mut carousel = Carousel::default();
    win.run(|frame| {
        let secs = frame.t.as_secs_f32();

        let spin = rotate_y(rads(secs * 0.6)) //
            .compose(&rotate_x(rads(secs * 0.7)));
        let carouse = carousel.update(frame.dt.as_secs_f32());

        let model_to_world: Mat4x4<ModelToWorld> =
            spin.then(&translate).then(&carouse).to();

        if frame
            .win
            .imp
            .is_key_pressed(Key::Space, KeyRepeat::No)
        {
            carousel.start();
        }

        let obj = &objs[carousel.idx % objs.len()];
        camera.render(
            &obj.faces,
            &obj.verts,
            &model_to_world,
            &shader,
            (),
            &mut frame.buf,
            &frame.ctx,
        );

        Continue(())
    });
}
