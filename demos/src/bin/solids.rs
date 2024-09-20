use core::ops::ControlFlow::Continue;

use minifb::{Key, KeyRepeat};

use re::prelude::*;

use re::math::{
    color::gray, mat::RealToReal, spline::smootherstep, vec::ProjVec4,
};
use re::render::batch::Batch;
use re::render::{cam::Camera, Model, ModelToProj, ModelToWorld};
use re_front::{minifb::Window, Frame};
use re_geom::{io::parse_obj, solids::*};

// Carousel animation for switching between objects.
#[derive(Default)]
struct Carousel {
    idx: usize,
    new_idx: usize,
    t: Option<f32>,
}

impl Carousel {
    fn start(&mut self) {
        if self.t.is_none() {
            self.t = Some(0.0);
            self.new_idx = self.idx + 1;
        } else {
            // If already started, skip to next
            self.new_idx += 1;
        }
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
        .build();

    win.ctx.color_clear = Some(gray(32).to_rgba());

    let (w, h) = win.dims;
    let cam = Camera::new(win.dims)
        .mode(scale(vec3(1.0, -1.0, -1.0)).to())
        .perspective(1.5, 0.1..1000.0)
        .viewport(vec2(10, 10)..vec2(w - 10, h - 10));

    type VertexIn = Vertex<Vec3<Model>, Normal3>;
    type VertexOut = Vertex<ProjVec4, Color3f>;
    type Uniform<'a> = (&'a Mat4x4<ModelToProj>, &'a Mat4x4<RealToReal<3>>);

    fn vtx_shader(v: VertexIn, (mvp, spin): Uniform) -> VertexOut {
        // Transform vertex normal
        let norm = spin.apply(&v.attrib.to());
        // Calculate diffuse shading
        let diffuse = (norm.z() + 0.2).max(0.2) * 0.8;
        // Visualize normal by mapping to RGB values
        let [r, g, b] = (0.45 * (v.attrib + splat(1.1))).0;
        let col = rgb(r, g, b).mul(diffuse);
        vertex(mvp.apply(&v.pos), col)
    }

    fn frag_shader(f: Frag<Color3f>) -> Color4 {
        f.var.to_color4()
    }

    let shader = Shader::new(vtx_shader, frag_shader);

    let objects = objects(8);

    let translate = translate(vec3(0.0, 0.0, -4.0));
    let mut carousel = Carousel::default();

    win.run(|frame| {
        let Frame { t, dt, win, .. } = frame;

        // Press Space to trigger carousel animation
        if win.imp.is_key_pressed(Key::Space, KeyRepeat::No) {
            carousel.start();
        }

        let theta = rads(t.as_secs_f32());
        let spin = rotate_x(theta * 0.47).then(&rotate_y(theta * 0.61));
        let carouse = carousel.update(dt.as_secs_f32());

        // Compose transform stack
        let model_view_project: Mat4x4<ModelToProj> = spin
            .then(&translate)
            .then(&carouse)
            .to::<ModelToWorld>()
            .then(&cam.world_to_project());

        Batch::new()
            .mesh(&objects[carousel.idx % objects.len()])
            .uniform((&model_view_project, &spin))
            .shader(shader)
            .viewport(cam.viewport)
            .target(&mut frame.buf)
            .context(&*frame.ctx)
            .render();

        Continue(())
    });
}

// Creates the 13 objects exhibited.
#[rustfmt::skip]
fn objects(res: u32) -> [Mesh<Normal3>; 13] {
    let segments = res;
    let sectors = 2 * res;

    let cap_segments = res;

    let major_sectors = 3 * res;
    let minor_sectors = 2 * res;
    [
        // The five Platonic solids
        Tetrahedron.build(),
        Box::cube(1.25).build(),
        Octahedron.build(),
        Dodecahedron.build(),
        Icosahedron.build(),

        // Surfaces of revolution
        lathe(sectors),
        Sphere { sectors, segments, radius: 1.0, }.build(),
        Cylinder { sectors, radius: 0.8, capped: true }.build(),
        Cone { sectors, base_radius: 1.1, apex_radius: 0.3, capped: true, }.build(),
        Capsule { sectors, cap_segments, radius: 0.5, }.build(),
        Torus { major_radius: 0.9, minor_radius: 0.3, major_sectors, minor_sectors, }.build(),

        // Traditional demo models
        teapot(),
        bunny(),
    ]
}

// Creates a Lathe mesh.
fn lathe(secs: u32) -> Mesh<Normal3> {
    Lathe::new(
        vec![
            vertex(vec2(0.75, -0.5), vec2(1.0, 1.0)),
            vertex(vec2(0.55, -0.25), vec2(1.0, 0.5)),
            vertex(vec2(0.5, 0.0), vec2(1.0, 0.0)),
            vertex(vec2(0.55, 0.25), vec2(1.0, -0.5)),
            vertex(vec2(0.75, 0.5), vec2(1.0, 1.0)),
        ],
        secs,
    )
    .capped(true)
    .build()
}

// Loads the Utah teapot model.
fn teapot() -> Mesh<Normal3> {
    parse_obj(*include_bytes!("../../assets/teapot.obj"))
        .unwrap()
        .transform(
            &scale(splat(0.4))
                .then(&translate(vec3(0.0, -0.5, 0.0)))
                .to(),
        )
        .with_vertex_normals()
        .build()
}

// Loads the Stanford bunny model.
fn bunny() -> Mesh<Normal3> {
    parse_obj(*include_bytes!("../../assets/bunny.obj"))
        .unwrap()
        .transform(
            &scale(splat(0.15))
                .then(&translate(vec3(0.0, -1.0, 0.0)))
                .to(),
        )
        .with_vertex_normals()
        .build()
}
