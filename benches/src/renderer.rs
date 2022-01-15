use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::Instant;

use criterion::*;

use geom::mesh::Mesh;
use geom::solids::{Torus, UnitCube, UnitSphere};
use math::Angle::Rad;
use math::transform::*;
use math::vec::{dir, Y, Z};
use render::{Framebuf, Render as _, Renderer};
use render::raster::Fragment;
use render::scene::{Obj, Scene};
use render::shade::ShaderImpl;
use util::buf::Buffer;
use util::color::*;
use util::io::{load_pnm, save_ppm};
use util::pixfmt::Identity;
use util::tex::{TexCoord, Texture};

const W: usize = 400;

fn renderer() -> Renderer {
    let mut rdr = Renderer::new();
    rdr.projection = perspective(1.0, 10.0, 1.0, Rad(1.0));
    rdr.viewport = viewport(0.0, 0.0, W as f32, W as f32);
    rdr
}

fn check_hash(buf: &Buffer<Color, &mut [Color]>, expected: u64) {
    let actual = {
        let h = &mut DefaultHasher::new();
        buf.data().hash(h);
        h.finish()
    };
    if actual != expected {
        eprintln!("Hashes differ: actual={} vs expected={}", actual, expected);
    }
}

fn save_screenshot(name: &str, buf: &Buffer<Color, &mut [Color]>) {
    let path = PathBuf::from(name);
    if path.exists() {
        let mut prev = path.clone();
        prev.set_extension("prev.ppm");
        std::fs::rename(path, prev).unwrap();
    }
    save_ppm(name, buf).unwrap();
}

fn torus(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.modelview = scale(4.0);

    let mesh = Torus(0.2, 24, 48).build();

    let mut cb = [BLACK; W * W];
    let mut buf = Framebuf::<'_, Identity> {
        color: Buffer::borrow(W, &mut cb),
        depth: &mut Buffer::new(W, W, f32::INFINITY),
    };
    c.bench_function("torus", |b| {
        b.iter(|| mesh.render(
            &mut rdr,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |_| Some(WHITE),
            },
            &mut buf
        ));
    });
    check_hash(&buf.color, 5667555972845400088);
    save_screenshot("target/torus.ppm", &buf.color);
    eprintln!("Stats/s: {}\n", rdr.stats.avg_per_sec());
}

fn scene(c: &mut Criterion) {
    let mut rdr = renderer();

    let mut objects = vec![];
    let camera = translate(4.0 * Y) * &rotate_x(Rad(0.5));
    for j in -8..=8 {
        for i in -8..=8 {
            objects.push(Obj {
                tf: translate(dir(3.0 * i as f32, 0., 3.0 * j as f32)),
                geom: UnitSphere(5, 9).build(),
            });
        }
    }
    let scene = Scene { objects, camera };

    let mut cb = [BLACK; W * W];
    let mut buf = Framebuf::<'_, Identity> {
        color: Buffer::borrow(W, &mut cb),
        depth: &mut Buffer::new(W, W, f32::INFINITY),
    };
    c.bench_function("scene", |b| {
        b.iter(|| scene.render(
            &mut rdr,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |_| Some(WHITE),
            },
            &mut buf
        ));
    });
    check_hash(&buf.color, 17516720479059830591);
    save_screenshot("target/scene.ppm", &buf.color);
    eprintln!("Stats/s: {}\n", rdr.stats.avg_per_sec());
}

fn gouraud_fillrate(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.modelview = scale(2.0) * &translate(6.0 * Z);

    let mesh = UnitCube.build();
    let mesh = Mesh::<(Color, )> {
        verts: mesh.verts.iter()
            .zip((0..3).cycle())
            .map(|(v, ci)| v.attr(ci))
            .collect(),
        vertex_attrs:  vec![RED, GREEN, BLUE],

        faces: mesh.faces,
        face_attrs: mesh.face_attrs,
        vertex_coords: mesh.vertex_coords,
        bbox: mesh.bbox,
    };

    let mut cb = [BLACK; W * W];
    let mut buf = Framebuf::<'_, Identity> {
        color: Buffer::borrow(W, &mut cb),
        depth: &mut Buffer::new(W, W, f32::INFINITY),
    };
    c.bench_function("gouraud", |b| {
        b.iter(|| mesh.render(
            &mut rdr,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |frag: Fragment<(Color, )>| Some(frag.varying.0),
            },
            &mut buf
        ));
    });
    check_hash(&buf.color, 15890119565587720914);
    save_screenshot("target/gouraud.ppm", &buf.color);
    eprintln!("Stats/frame: {}\n", rdr.stats.avg_per_frame());
}

fn texture_fillrate(c: &mut Criterion) {
    let mut rdr = renderer();

    rdr.modelview = scale(2.0) * &translate(6.0 * Z);
    let mesh = UnitCube.with_texcoords();

    let tex = Texture::from(load_pnm("../examples/sdl/crate.ppm").unwrap());

    let mut cb = [BLACK; W * W];
    let mut buf = Framebuf::<'_, Identity> {
        color: Buffer::borrow(W, &mut cb),
        depth: &mut Buffer::new(W, W, f32::INFINITY),
    };
    c.bench_function("texture", |b| {
        b.iter(|| {
            let clock = Instant::now();
            mesh.render(
                &mut rdr,
                &mut ShaderImpl {
                    vs: |v| v,
                    fs: |f: Fragment<(TexCoord, )>| {
                        Some(tex.sample(f.varying.0))
                    }
                },
                &mut buf,
            );
            rdr.stats.time_used += clock.elapsed();
            rdr.stats.frames += 1;
        });
    });
    check_hash(&buf.color, 10066381871015406825);
    save_screenshot("target/texture.ppm", &buf.color);
    eprintln!("Stats:     {}", rdr.stats);
    eprintln!("Stats/sec: {}\n", rdr.stats.avg_per_sec());
}

criterion_group!(benches,
    torus,
    scene,
    gouraud_fillrate,
    texture_fillrate
);

criterion_main!(benches);
