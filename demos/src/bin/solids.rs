use core::ops::ControlFlow::Continue;

use minifb::{Key, KeyRepeat};

use re::prelude::*;

use re::geom::{Edge, Polyline, Ray};
use re::math::{color::gray, mat::Apply, mat::RealToReal, vec::ProjVec3};
use re::render::{ModelToProj, ModelToWorld, cam::Fov};

use re_front::{Frame, minifb::Window};
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
        .build()
        .expect("should create window");

    win.ctx.color_clear = Some(gray(32).to_rgba());

    let (w, h) = win.dims;
    let cam = Camera::new(win.dims)
        .transform(scale3(1.0, -1.0, -1.0).to())
        .perspective(Fov::Equiv35mm(28.0), 0.1..1000.0)
        .viewport(pt2(10, 10)..pt2(w - 10, h - 10));

    type VertexIn = Vertex3<Normal3>;
    type VertexOut = Vertex<ProjVec3, Color4f>;
    type Uniform<'a> = (&'a Mat4x4<ModelToProj>, &'a Mat4x4<RealToReal<3>>);

    fn vtx_shader(v: VertexIn, (mvp, spin): Uniform) -> VertexOut {
        // Transform vertex normal
        let norm = spin.apply(&v.attrib.to());
        // Calculate diffuse shading
        let diffuse = (norm.z() + 0.2).max(0.2) * 0.8;
        // Visualize normal by mapping to RGB values
        let col = diffuse * debug::dir_to_rgb(norm);
        vertex(mvp.apply(&v.pos), col)
    }

    fn frag_shader(f: Frag<Color4f>) -> Color4 {
        f.var.to_color4()
    }

    let shader = shader::new(vtx_shader, frag_shader);

    let objects = objects(8);

    let translate = translate(-3.0 * Vec3::Z);
    let mut carousel = Carousel::default();
    let mut debug: u8 = 0;

    win.run(|frame| {
        let Frame { t, dt, win, .. } = frame;

        // Press Space to trigger carousel animation
        if win.imp.is_key_pressed(Key::Space, KeyRepeat::No) {
            carousel.start();
        }
        if win.imp.is_key_pressed(Key::D, KeyRepeat::No) {
            debug = (debug + 1) % 7;
        }

        let theta = rads(t.as_secs_f32());
        let spin = rotate_x(theta * 0.47).then(&rotate_y(theta * 0.61));
        let carouse = carousel.update(dt.as_secs_f32());

        // Compose transform stack
        let model_to_world: Mat4x4<ModelToWorld> =
            spin.then(&translate).then(&carouse).to();
        let world_to_proj = cam.world_to_project();
        let mvp = model_to_world.then(&world_to_proj);

        let object = &objects[carousel.idx % objects.len()];

        let b = Batch::new()
            .uniform((&mvp, &spin))
            .viewport(cam.viewport)
            .context(&*frame.ctx);

        b.clone()
            .mesh(object)
            .shader(shader)
            .target(&mut frame.buf)
            .render();

        if debug > 0 {
            let bbox = debug::bbox(&object.verts);
            bbox.uniform(&mvp)
                .viewport(cam.viewport)
                .context(&*frame.ctx)
                .target(&mut frame.buf)
                .render();

            let grid = debug::grid::<World>(10);
            let m2w = rotate_x(degs(90.0)).then(&translate).to();
            grid.uniform(&m2w.then(&world_to_proj))
                .viewport(cam.viewport)
                .context(&*frame.ctx)
                .target(&mut frame.buf)
                .render();
        }
        if debug > 1 {
            debug::sphere(pt3::<_, Model>(0.0, 0.0, 0.0), 1.0)
                .uniform(&mvp)
                .viewport(cam.viewport)
                .context(&*frame.ctx)
                .target(&mut frame.buf)
                .render();

            debug::cone(0.8)
                .uniform(&mvp)
                .viewport(cam.viewport)
                .context(&*frame.ctx)
                .target(&mut frame.buf)
                .render();
        }
        if debug > 2 {
            debug::frame::<Model, Model>(Mat4x4::identity())
                .uniform(&mvp)
                .viewport(cam.viewport)
                .context(&*frame.ctx)
                .target(&mut frame.buf)
                .render();
        }
        if debug > 3 {
            // Wireframe faces
            let edges: Vec<_> = object
                .faces
                .iter()
                .flat_map(|tri| tri.edges().map(|Edge(a, b)| Edge(*a, *b)))
                .collect();
            let verts = object
                .verts
                .iter()
                .map(|v| vertex(v.pos, gray(1.0f32).to_rgba()))
                .collect::<Vec<_>>();

            Batch::new()
                .primitives(edges)
                .vertices(verts)
                .uniform(&mvp)
                .shader(debug::Shader)
                .viewport(cam.viewport)
                .context(&*frame.ctx)
                .target(&mut frame.buf)
                .render();
        }
        if debug > 4 {
            debug::frame::<Model, Model>(Mat4x4::identity())
                .uniform(&mvp)
                .viewport(cam.viewport)
                .context(&*frame.ctx)
                .target(&mut frame.buf)
                .render();
        }
        if debug > 5 {
            let norms = object
                .faces
                .iter()
                .map(|Tri(vs)| {
                    debug::face_normal(vs.map(|i| object.verts[i].pos), 0.3)
                })
                .reduce(|mut a, b| {
                    a.append(b);
                    a
                })
                .expect("at least one face");
            norms
                .uniform(&mvp)
                .viewport(cam.viewport)
                .context(&*frame.ctx)
                .target(&mut frame.buf)
                .render();

            /*let norms: Vec<_> = object
                .verts
                .iter()
                .map(|v| (v.pos, v.attrib))
                .collect();

            for (o, dir) in norms {
                debug::ray(o, 0.5 * dir.to())
                    .uniform(&mvp)
                    .viewport(cam.viewport)
                    .context(&*frame.ctx)
                    .target(&mut frame.buf)
                    .render();
            }*/
        }
        Continue(())
    });
}

// Creates the 13 objects exhibited.
#[rustfmt::skip]
fn objects(res: u32) -> [Mesh<Normal3>; 13] {
    let segments = res;
    let sectors = 2 * res;

    let cap_segments = res;
    let body_segments = res;

    let major_sectors = 3 * res;
    let minor_sectors = 2 * res;
    [
        lathe(sectors),
        // The five Platonic solids
        Tetrahedron.build(),
        Cube { side_len: 1.25 }.build(),
        Octahedron.build(),
        Dodecahedron.build(),
        Icosahedron.build(),

        // Surfaces of revolution
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
    let _pts: [Vertex<Point2, Normal2>; _] = [
        // _________
        //         |
        //        /
        //   ____/
        //  /
        // |
        // \______
        // _______\

        // Base
        vertex(pt2(0.5, -0.6), vec2(1.0, 0.0)),
        vertex(pt2(0.45, -0.55), vec2(0.0, 1.0)),
        // Neck
        vertex(pt2(0.15, -0.5), vec2(1.0, 1.0)),
        vertex(pt2(0.1, -0.45), vec2(1.0, 0.0)),
        vertex(pt2(0.1, 0.0), vec2(1.0, 0.0)),
        vertex(pt2(0.15, 0.05), vec2(1.0, -1.0)),
        // Bowl outer
        vertex(pt2(0.4, 0.1), vec2(1.0, -0.5)),
        vertex(pt2(0.5, 0.2), vec2(1.0, 0.0)),
        vertex(pt2(0.5, 0.3), vec2(1.0, 0.0)),
        vertex(pt2(0.4, 0.6), vec2(1.0, 0.1)),
        // Bowl inner
        vertex(pt2(0.35, 0.6), vec2(-1.0, 0.1)),
        vertex(pt2(0.4, 0.25), vec2(-1.0, 0.0)),
        vertex(pt2(0.2, 0.15), vec2(-0.2, 1.0)),
        vertex(pt2(0.0, 0.1), vec2(0.0, 1.0)),
    ];

    let curve = BezierSpline::from_rays(
        [
            Ray(pt2(0.5, -0.6), vec2(0.0, 0.1)),
            Ray(pt2(0.4, -0.55), vec2(-0.1, 0.0)),
            Ray(pt2(0.1, -0.4), vec2(0.0, 0.1)),
            Ray(pt2(0.1, 0.0), vec2(0.0, 0.1)),
            Ray(pt2(0.3, 0.2), vec2(0.1, 0.0)),
            Ray(pt2(0.5, 0.4), vec2(0.0, 0.1)),
            Ray(pt2(0.4, 1.0), vec2(-0.1, 0.0)),
            Ray(pt2(0.48, 0.4), vec2(0.0, -0.1)),
            Ray(pt2(0.0, 0.05), vec2(-0.1, 0.0)),
        ], //.map(|Ray(pos, dir)| Ray(vertex(pos, dir.perp().normalize()))),
    );

    let curve: Vec<_> = curve
        .approximate(0.05)
        .0
        .into_iter()
        .map(|p| vertex(p, vec2(1.0, 1.0)))
        .collect();

    let n = curve.len();
    Lathe::new(Polyline(curve), secs, 2 * n as u32)
        .capped(true)
        .build()
}

// Loads the Utah teapot model.
fn teapot() -> Mesh<Normal3> {
    parse_obj::<()>(*include_bytes!("../../assets/teapot.obj"))
        .unwrap()
        .transform(
            &scale(splat(0.4))
                .then(&translate(-0.5 * Vec3::Y))
                .to(),
        )
        .with_vertex_normals()
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
