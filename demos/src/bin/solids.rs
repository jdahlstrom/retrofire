use core::ops::ControlFlow::Continue;
use std::env;

use minifb::{Key, KeyRepeat};

use re::prelude::*;

use re::core::geom::Polyline;
use re::core::math::{ProjMat3, ProjVec3, color::gray};
use re::core::render::{
    Model, View, World, cam::Fov, debug::dir_to_rgb, light::Kind, shader,
};

use re::front::{Frame, minifb::Window};
use re::geom::{io::read_obj, solids::*};

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

#[inline]
fn phong<B>(
    normal: Vec3<B>,
    view_dir: Vec3<B>,
    light_dir: Vec3<B>,
    shininess: i32,
) -> f32 {
    let refl_dir = light_dir.reflect(normal);
    view_dir.dot(&refl_dir).max(0.0).powi(shininess)
}

#[inline]
fn blinn_phong<B>(
    normal: Vec3<B>,
    view_dir: Vec3<B>,
    light_dir: Vec3<B>,
    shininess: i32,
) -> f32 {
    let halfway = (view_dir + light_dir).normalize_approx();
    normal.dot(&halfway).max(0.0).powi(4 * shininess)
}

fn main() {
    eprintln!("Press Space to cycle between objects...");

    let mut win = Window::builder()
        .title("retrofire//solids")
        .build()
        .expect("should create window");

    win.ctx.color_clear = Some(gray(0x33).to_rgba());

    let (w, h) = win.dims;
    let cam = Camera::new(win.dims)
        .transform(Mat4::identity())
        .perspective(Fov::Equiv35mm(28.0), 0.1..1000.0)
        .viewport(pt2(10, h - 10)..pt2(w - 10, 10));

    type Varyings = (Point3<View>, (Color3f, Normal3));
    type VertexIn = Vertex3<Normal3>;
    type VertexOut = Vertex<ProjVec3, Varyings>;
    struct Uniform {
        pub mv: Mat4<Model, View>,
        pub proj: ProjMat3<View>,
        pub norm: Mat4,
        pub light: Light<View>,
    }

    #[inline]
    fn vtx_shader(v: VertexIn, u: &Uniform) -> VertexOut {
        let view_normal = u.norm.apply(&v.attrib);
        let color = dir_to_rgb(v.attrib).to_rgb();
        let view_pos = u.mv.apply(&v.pos);
        let clip_pos = u.proj.apply(&view_pos);

        // (camera_dir, (light_col * color, (view_normal, (light_col, light_dir))),),
        vertex(clip_pos, (view_pos, (color, view_normal)))
    }

    #[inline]
    fn frag_shader(f: Frag<Varyings>, u: &Uniform) -> Color4 {
        //let (camera_dir, (light_x_col, (view_normal, (light_col, light_dir)))) =
        let (view_pos, (color, view_normal)) = f.var;

        let (light_col, light_dir) = u.light.eval(view_pos);
        let camera_dir = -view_pos.to_vec().normalize_approx();
        let view_normal = view_normal.normalize_approx();

        let ambient = rgb(0.15, 0.18, 0.25);
        let diffuse = 0.5 * light_dir.dot(&view_normal.to());
        let specular =
            0.5 * blinn_phong(view_normal.to(), camera_dir, light_dir, 30);

        //(light_x_col * diffuse + light_col * specular + ambient).to_color4()

        (light_col * color * diffuse + light_col * specular + ambient)
            .to_color4()
    }

    let shader = shader::new(vtx_shader, frag_shader);

    let res = env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(24);
    let objects = objects_n(res);

    let translate = translate(3.0 * Vec3::Z);
    let mut carousel = Carousel::default();

    let light = Light::<World>::new(
        rgb(1.0, 0.95, 0.9),
        Kind::Point(pt3(-1.0, 3.0, 1.0)),
    );

    win.run(|frame| {
        let Frame { t, dt, win, .. } = frame;

        // Press Space to trigger carousel animation
        if win.imp.is_key_pressed(Key::Space, KeyRepeat::No) {
            carousel.start();
        }

        let theta = rads(t.as_secs_f32());
        let spin = rotate_x(theta * 0.51).then(&rotate_y(theta * 0.37));
        let carouse = carousel.update(dt.as_secs_f32());

        // Compose transform stack
        let modelview: Mat4<Model, View> =
            spin.then(&translate).then(&carouse).to();

        let object = &objects[carousel.idx % objects.len()];

        let uniform = &Uniform {
            mv: modelview,
            proj: cam.project,
            norm: spin,
            light: light.transform(&cam.world_to_view()),
        };
        Batch {
            prims: object.faces.clone(),
            verts: object.verts.clone(),
            uniform,
            shader,
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
    let segments = res + 2;
    let sectors = res + 3;

    let cap_segments = res + 1;
    let body_segments = res + 1;

    let major_sectors = res + 3;
    let minor_sectors = res.div_ceil(2) + 3;
    [
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
        Cone { base_radius: 1.0, apex_radius: 0.2, sectors, segments, capped: true }.build(),
        Capsule { radius: 0.5, sectors, body_segments, cap_segments }.build(),
        Torus { major_radius: 0.9, minor_radius: 0.25, major_sectors, minor_sectors }.build(),

        // Traditional demo models
        teapot(),
        bunny(),
        dragon()
    ]
}

// Creates a Lathe mesh.
fn lathe(secs: u32) -> Mesh<Normal3> {
    let pts = [
        (pt2(0.75, -0.5), vec2(1.0, 1.0)),
        (pt2(0.55, -0.25), vec2(1.0, 0.5)),
        (pt2(0.5, 0.0), vec2(1.0, 0.0)),
        (pt2(0.55, 0.25), vec2(1.0, -0.5)),
        (pt2(0.75, 0.5), vec2(1.0, -1.0)),
    ]
    .map(|(p, n)| vertex(p, n.normalize()));

    Lathe::new(Polyline::new(pts), secs, pts.len() as u32)
        .capped(true)
        .build()
}

// Loads the Utah teapot model.
fn teapot() -> Mesh<Normal3> {
    static TEAPOT: &[u8] = include_bytes!("../../assets/teapot.obj");
    read_obj(TEAPOT)
        .unwrap()
        .transform(
            &scale(splat(0.4))
                .then(&translate(-0.5 * Vec3::Y))
                .to(),
        )
        .build()
}

// Loads the Stanford bunny model.
fn bunny() -> Mesh<Normal3> {
    static BUNNY: &[u8] = include_bytes!("../../assets/bunny.obj");
    read_obj::<()>(BUNNY)
        .unwrap()
        .transform(&scale(splat(0.12)).then(&translate(-Vec3::Y)).to())
        .with_vertex_normals()
        .build()
}

// Loads the Stanford dragon model.
fn dragon() -> Mesh<Normal3> {
    static DRAGON: &[u8] = include_bytes!("../../assets/dragon.obj");
    read_obj::<()>(DRAGON)
        .unwrap()
        .with_vertex_normals()
        .transform(
            &scale(splat(0.18))
                .then(&translate(-0.5 * Vec3::Y))
                .to(),
        )
        .build()
}
