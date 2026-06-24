use core::ops::ControlFlow::Continue;
use std::env::current_dir;
use std::fs::File;
use std::sync::LazyLock;

use minifb::{Key, KeyRepeat};

use re::prelude::*;

use re::core::{
    geom::{Polyline, Ray},
    math::{ProjVec3, color::gray, spline::HermiteSpline},
    render::{Model, ModelToWorld, cam::Fov, shader},
};
use re::front::{Frame, minifb::Window};
use re::geom::io::gltf::load_gltf;
use re::geom::{io::obj::read_obj, solids::*};

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
    eprintln!(
        "Press <space> to cycle between objects, \
        <.> and <,> to adjust level of detail..."
    );

    let mut win = Window::builder()
        .title("retrofire//solids")
        .build()
        .expect("should create window");

    win.ctx.color_clear = Some(gray(0x33).to_rgba());

    let Dims(w, h) = win.dims;
    let cam = Camera::new(win.dims)
        .transform(scale((1.0, -1.0, -1.0)).to())
        .perspective(Fov::Equiv35mm(28.0), 0.1..1000.0)
        .viewport(pt2(10, h - 10)..pt2(w - 10, 10));

    type VertexIn = Vertex3<Normal3>;
    type VertexOut = Vertex<ProjVec3, Color3f>;
    type Uniform<'a> = (&'a ProjMat3<Model>, &'a Mat4);

    fn vtx_shader(v: VertexIn, (mvp, spin): Uniform) -> VertexOut {
        // Transform vertex normal
        let norm = spin.apply(&v.attrib);
        // Calculate diffuse shading
        let diffuse = (norm.z() + 0.2).max(0.2) * 0.8;
        // Visualize normal by mapping to RGB values
        let [r, g, b] = (0.45 * (v.attrib + splat(1.1))).0;
        let col = diffuse * rgb(r, g, b);
        vertex(mvp.apply(&v.pos), col)
    }

    fn frag_shader(f: Frag<Color3f>, _: Uniform) -> Color4 {
        f.var.to_color4()
    }

    let shader = shader::new(vtx_shader, frag_shader);

    let mut lod = 10;
    let mut objects = objects_n(lod);

    let translate = translate(-3.0 * Vec3::Z);
    let mut carousel = Carousel::default();

    win.run(|frame| {
        let Frame { t, dt, win, .. } = frame;

        for key in win.imp.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::Space => carousel.start(),

                Key::Comma | Key::Period => {
                    let (num, denom) =
                        if key == Key::Comma { (3, 4) } else { (4, 3) };
                    lod = (lod * num / denom).clamp(3, 50);
                    objects = objects_n(lod);
                }
                _ => (),
            }
        }

        let theta = rads(t.as_secs_f32());
        let spin = rotate_x(theta * 0.37).then(&rotate_y(theta * 0.51));
        let carouse = carousel.update(dt.as_secs_f32());

        // Compose transform stack
        let model_view_project: ProjMat3<Model> = spin
            .then(&translate)
            .then(&carouse)
            .to::<ModelToWorld>()
            .then(&cam.world_to_project());

        let object = &objects[carousel.idx % objects.len()];

        Batch {
            prims: object.faces.clone(),
            verts: object.verts.clone(),
            uniform: (&model_view_project, &spin),
            shader: shader,
            viewport: cam.viewport,
            target: frame.buf,
            ctx: &*frame.ctx,
        }
        .render();

        Continue(())
    });
}

// Creates the 14 objects exhibited.
#[rustfmt::skip]
fn objects_n(res: u32) -> [Mesh<Normal3>; 14] {
    let segments = res;
    let sectors = 2 * res;

    let cap_segments = res;
    let body_segments = res;

    let major_sectors = 3 * res;
    let minor_sectors = 2 * res;
    [
        suzanne(),


        // The five Platonic solids
        //Tetrahedron.build(),
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
        teapot().clone(),
        bunny().clone(),
        dragon().clone()
    ]
}

// Creates a Lathe mesh.
fn lathe(secs: u32) -> Mesh<Normal3> {
    let spline = HermiteSpline::new([
        Ray(pt2(0.6, -1.0), vec2(0.0, 0.3)),
        Ray(pt2(0.1, -0.8), vec2(0.0, 0.3)),
        Ray(pt2(0.1, -0.3), vec2(0.0, 0.4)),
        Ray(pt2(0.6, 0.2), vec2(0.0, 0.8)),
        Ray(pt2(0.5, 1.0), vec2(0.0, 0.1)),
        Ray(pt2(0.45, 1.0), vec2(0.0, -0.1)),
        Ray(pt2(0.55, 0.2), vec2(0.0, -0.2)),
        Ray(pt2(0.0, -0.2), vec2(-1.0, 0.0)),
    ]);

    let verts = 0.0.vary_to(1.0, 2 * secs).map(|t| {
        vertex(spline.eval(t), -spline.velocity(t).normalize().perp())
    });
    let pl = Polyline::new(verts);
    let segs = pl.0.len() as u32;
    Lathe::new(pl, secs, segs).capped(true).build()
}

fn suzanne() -> Mesh<Normal3> {
    eprintln!("{:?}", current_dir());

    File::open("Suzanne.gltf").unwrap();

    load_gltf("Suzanne.gltf").expect("Suzanne should be found")
}

// Loads the Utah teapot model.
fn teapot() -> &'static Mesh<Normal3> {
    static TEAPOT: LazyLock<Mesh<Normal3>> = LazyLock::new(|| {
        let obj: &[_] = include_bytes!("../../assets/teapot.obj");
        read_obj::<Normal3>(obj)
            .unwrap()
            .transform(&scale(0.4).then(&translate(-0.5 * Vec3::Y)).to())
            .build()
    });
    &TEAPOT
}

// Loads the Stanford bunny model.
fn bunny() -> &'static Mesh<Normal3> {
    static BUNNY: LazyLock<Mesh<Normal3>> = LazyLock::new(|| {
        let obj: &[_] = include_bytes!("../../assets/bunny.obj");
        read_obj::<()>(obj)
            .unwrap()
            .transform(&scale(0.12).then(&translate(-Vec3::Y)).to())
            .with_vertex_normals()
            .build()
    });
    &BUNNY
}

// Loads the Stanford dragon model.
fn dragon() -> &'static Mesh<Normal3> {
    static DRAGON: LazyLock<Mesh<Normal3>> = LazyLock::new(|| {
        let obj: &[_] = include_bytes!("../../assets/dragon.obj");
        read_obj::<()>(obj)
            .unwrap()
            .with_vertex_normals()
            .transform(&scale(0.18).then(&translate(-0.5 * Vec3::Y)).to())
            .build()
    });
    &DRAGON
}
