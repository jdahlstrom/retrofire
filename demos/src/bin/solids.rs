use core::ops::ControlFlow::Continue;
use std::{env, sync::LazyLock};

use minifb::{Key, KeyRepeat};

use re::prelude::*;

use re::core::{
    geom::{Polyline, Ray},
    math::{ProjVec3, color::gray, spline::HermiteSpline},
    render::{Model, cam::Fov, debug, debug::DbgMesh, shader},
};
use re::front::{Frame, minifb::Window};
use re::geom::{io::read_obj, solids::*};

#[derive(Default)]
struct State {
    lod: u32,
    objects: [Mesh<Normal3>; N_OBJS],
    debug_mesh: DbgMesh<Normal3, Model>,
    debug_flags: [bool; 6],

    idx: usize,
    new_idx: usize,
    anim_t: Option<f32>,
}

fn main() {
    eprintln!(
        "[Space]     : cycle between objects\n\
         [.] and [,] : adjust level of detail\n\
         [0]         : toggle mesh rendering\n\
         [1] to [5]  : toggle visualizations"
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

    let translate = translate(-3.0 * Vec3::Z);

    let mut state = State::new();
    if let Some(idx) = env::args().nth(1) {
        let Some(idx) = idx.parse().ok().filter(|&i| i < N_OBJS) else {
            panic!("argument must be integer in range 0..={}", N_OBJS - 1)
        };
        state.idx = idx;
    }

    win.run(|frame| {
        let Frame { t, dt, win, .. } = frame;

        for key in win.imp.get_keys_pressed(KeyRepeat::No) {
            use Key::*;
            match key {
                Space => state.start_carousel(),

                Comma | Period => {
                    let (num, denom) =
                        if key == Comma { (3, 4) } else { (4, 3) };
                    state.set_lod((state.lod * num / denom).clamp(3, 50));
                }

                digit @ (Key0 | Key1 | Key2 | Key3 | Key4 | Key5) => {
                    state.toggle_flag(digit as usize);
                }

                _ => (),
            }
        }

        let theta = rads(t.as_secs_f32());
        let spin = rotate_x(theta * 0.37).then(&rotate_y(theta * 0.51));
        let carouse = state.update(dt.as_secs_f32());

        // Compose transform stack
        let model_view_project: ProjMat3<Model> = spin
            .then(&translate)
            .then(&carouse)
            .to()
            .then(&cam.world_to_project());

        state
            .debug_mesh
            .batch()
            .clone()
            .uniform(&model_view_project)
            .viewport(cam.viewport)
            .target(frame.buf)
            .render();

        if state.debug_flags[0] {
            let object = state.object();
            Batch {
                prims: &object.faces,
                verts: &object.verts,
                uniform: (&model_view_project, &spin),
                shader: shader,
                viewport: cam.viewport,
                target: frame.buf,
                ctx: &*frame.ctx,
            }
            .render();
        }

        Continue(())
    });
}

const N_OBJS: usize = 14;

// Creates the 14 objects exhibited.
#[rustfmt::skip]
fn objects_n(res: u32) -> [Mesh<Normal3>; N_OBJS] {
    let segments = res;
    let sectors = 2 * res;

    let cap_segments = res;
    let body_segments = res;

    let major_sectors = 3 * res;
    let minor_sectors = 2 * res;
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

impl State {
    fn new() -> Self {
        let mut state = State::default();
        state.lod = 10;
        state.objects = objects_n(state.lod);
        state.debug_flags[0] = true;
        state
    }

    fn object(&self) -> &Mesh<Normal3> {
        &self.objects[self.idx]
    }

    fn set_lod(&mut self, lod: u32) {
        self.lod = lod;
        self.objects = objects_n(lod);
        self.build_debug_mesh();
    }

    fn toggle_flag(&mut self, flag: usize) {
        self.debug_flags[flag] ^= true;
        self.build_debug_mesh();
    }

    fn start_carousel(&mut self) {
        if self.anim_t.is_none() {
            self.anim_t = Some(0.0);
            self.new_idx = self.idx + 1;
        } else {
            // If already started, skip to next
            self.new_idx += 1;
        }
        self.new_idx %= self.objects.len();
    }

    fn update(&mut self, dt: f32) -> Mat4 {
        let Some(t) = self.anim_t.as_mut() else {
            return Mat4::identity();
        };
        *t += dt;
        let t = *t;
        if t >= 0.5 && self.idx != self.new_idx {
            self.idx = self.new_idx;
            self.build_debug_mesh();
        }
        if t >= 1.0 {
            self.anim_t = None
        }
        rotate_y(turns(smootherstep(t)))
    }

    fn build_debug_mesh(&mut self) {
        let [_, ed, fns, vns, bb, bas] = self.debug_flags;
        let mut dm = debug::mesh(self.objects[self.idx].clone());
        if ed {
            dm = dm.edges();
        }
        if fns {
            dm = dm.face_normals(0.2);
        }
        if vns {
            dm = dm.vertex_normals(0.2);
        }
        if bb {
            dm = dm.bbox();
        }
        if bas {
            dm = dm.basis();
        }
        self.debug_mesh = dm;
    }
}
