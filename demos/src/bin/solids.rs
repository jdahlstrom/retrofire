use core::ops::ControlFlow::Continue;

use minifb::{Key, KeyRepeat};

use re::prelude::*;

use re::core::geom::Polyline;
use re::core::math::{ProjMat3, ProjVec3, color::gray};
use re::core::render::cam::Fov;

use re::front::{Frame, minifb::Window};
use re::geom::{io::parse_obj, solids::*};

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
    fn update(&mut self, dt: f32) -> Mat4 {
        let Some(t) = self.t.as_mut() else {
            return Mat4::identity();
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
        .build()
        .expect("should create window");

    win.ctx.color_clear = Some(gray(0xFF).to_rgba());

    let (w, h) = win.dims;
    let cam = Camera::new(win.dims)
        .transform(scale3(1.0, -1.0, -1.0).to())
        .perspective(Fov::Equiv35mm(28.0), 0.1..1000.0)
        .viewport(pt2(10, h - 10)..pt2(w - 10, 10));

    type VertexIn = Vertex3<Normal3>;
    type VertexOut = Vertex<ProjVec3, Color3f>;
    type Uniform<'a> = (&'a ProjMat3<Model>, &'a Mat4);

    fn vtx_shader(v: VertexIn, (mvp, spin): Uniform) -> VertexOut {
        // Transform vertex normal
        let norm = spin.apply(&v.attrib.to());
        // Calculate diffuse shading
        let diffuse = (norm.z() + 0.2).max(0.2) * 0.8;
        // Visualize normal by mapping to RGB values
        let [r, g, b] = (0.45 * (v.attrib + splat(1.1))).0;
        let col = diffuse * rgb(r, g, b);
        vertex(mvp.apply(&v.pos), col)
    }

    fn frag_shader(f: Frag<Color3f>) -> Color4 {
        f.var.to_color4()
    }

    let shader = shader::new(vtx_shader, frag_shader);

    let objects = objects_n(8);

    let translate = translate(-4.0 * Vec3::Z);
    let mut carousel = Carousel::default();

    win.run(|frame| {
        let Frame { t, dt, win, .. } = frame;

        // Press Space to trigger carousel animation
        if win.imp.is_key_pressed(Key::Space, KeyRepeat::No) {
            carousel.start();
        }

        let theta = rads(t.as_secs_f32() * 0.0);
        let spin = rotate_x(theta * 0.47).then(&rotate_y(theta * 0.61));
        let carouse = carousel.update(dt.as_secs_f32());

        // Compose transform stack
        let model_view_project: ProjMat3<Model> = spin
            .then(&translate)
            .then(&carouse)
            .to::<ModelToWorld>()
            .then(&cam.world_to_project());

        let object = &objects[carousel.idx % objects.len()];

        Batch::new()
            .mesh(object)
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
fn objects_n(res: u32) -> [Mesh<Normal3>; 14] {
    let segments = res;
    let sectors = 2 * res;

    let cap_segments = res;
    let body_segments = res;

    let major_sectors = 3 * res;
    let minor_sectors = 2 * res;
    [
        dragon(),
        // The five Platonic solids
        Tetrahedron.build(),
        Cube { side_len: 1.25 }.build(),
        Octahedron.build(),
        Dodecahedron.build(),
        Icosahedron.build(),

        // Surfaces of revolution
        lathe(sectors),
        Sphere { radius: 1.0, sectors, segments, }.build(),
        Cylinder { radius: 0.8, sectors, segments, capped: true }.build(),
        Cone { base_radius: 1.1, apex_radius: 0.3, sectors, segments, capped: true }.build(),
        Capsule { radius: 0.5, sectors, body_segments, cap_segments }.build(),
        Torus { major_radius: 0.9, minor_radius: 0.3, major_sectors, minor_sectors }.build(),

        // Traditional demo models
        teapot(),
        bunny(),
    ]
}

// Creates a Lathe mesh.
fn lathe(secs: u32) -> Mesh<Normal3> {
    let pts = [
        vertex(pt2(0.75, -0.5), vec2(1.0, 1.0).normalize()),
        vertex(pt2(0.55, -0.25), vec2(1.0, 0.5).normalize()),
        vertex(pt2(0.5, 0.0), vec2(1.0, 0.0)),
        vertex(pt2(0.55, 0.25), vec2(1.0, -0.5).normalize()),
        vertex(pt2(0.75, 0.5), vec2(1.0, -1.0).normalize()),
    ];
    Lathe::new(Polyline::new(pts), secs, pts.len() as u32)
        .capped(true)
        .build()
}

// Loads the Utah teapot model.
fn teapot() -> Mesh<Normal3> {
    parse_obj(*include_bytes!("../../assets/teapot.obj"))
        .unwrap()
        .transform(
            &scale(splat(0.3))
                .then(&translate(-0.5 * Vec3::Y))
                .to(),
        )
        //.with_vertex_normals()
        .build()
}

// Loads the Stanford bunny model.
fn bunny() -> Mesh<Normal3> {
    parse_obj::<()>(*include_bytes!("../../assets/bunny.obj"))
        .unwrap()
        .transform(&scale(splat(0.15)).then(&translate(-Vec3::Y)).to())
        .with_vertex_normals()
        .build()
}

// Loads the Stanford dragon model.
fn dragon() -> Mesh<Normal3> {
    static DRAGON: &[u8] = include_bytes!("../../assets/dragon.obj");
    parse_obj::<()>(DRAGON.iter().copied())
        .unwrap()
        .with_vertex_normals()
        .transform(
            &scale(splat(0.25))
                .then(&translate(-1.2 * Vec3::Y))
                .to(),
        )
        //.with_vertex_normals()
        .build()
}
