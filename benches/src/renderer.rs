use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::Instant;

use criterion::*;

use geom::solids;
use math::Angle::Rad;
use math::transform::*;
use math::vec::{dir, Y, Z};
use render::{Render, Renderer, Framebuf};
use render::raster::Fragment;
use render::scene::{Obj, Scene};
use render::shade::ShaderImpl;
use render::tex::{TexCoord, Texture, uv};
use util::Buffer;
use util::color::*;
use util::io::{load_pnm, save_ppm};

const W: usize = 128;

fn renderer() -> Renderer {
    let mut rdr = Renderer::new();
    rdr.projection = perspective(1.0, 10.0, 1.0, Rad(1.0));
    rdr.viewport = viewport(0.0, 0.0, W as f32, W as f32);
    rdr
}

fn check_hash<T: Hash>(buf: &[T], expected: u64) {
    let h = &mut DefaultHasher::new();
    buf.hash(h);
    let actual = h.finish();
    if actual != expected {
        eprintln!("Hashes differ: actual={} vs expected={}", actual, expected);
    }
}

fn save_screenshot(name: &str, buf: &Buffer<u8>) {
    let path = PathBuf::from(name);
    if path.exists() {
        let mut prev = path.clone();
        prev.set_extension("prev.ppm");
        std::fs::rename(path, prev).unwrap();
    }
    let mut data: Vec<_> = buf.data().chunks(4)
        .map(|c| rgb(c[2], c[1], c[0]))
        .collect();

    save_ppm(name, &Buffer::borrow(buf.width(), &mut data)).unwrap();
}

fn torus(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.modelview = scale(4.0);

    let mesh = solids::torus(0.2, 9, 9).build();

    let mut fb = Framebuf {
        color: Buffer::new(W, 4 * W, 0),
        depth: &mut Buffer::new(W, W, f32::INFINITY),
        depth_test: true,
        depth_write: true
    };
    c.bench_function("torus", |b| {
        b.iter(|| mesh.render(
            &mut rdr,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |_| Some(WHITE),
            },
            &mut fb
        ));
    });
    check_hash(fb.color.data(), 12319688252832539604);
    save_screenshot("target/torus.ppm", &fb.color);
    eprintln!("Stats/s: {}\n", rdr.stats.avg_per_sec());
}

fn scene(c: &mut Criterion) {
    let mut rdr = renderer();

    let mut objects = vec![];
    let camera = translate(4.0 * Y) * &rotate_x(Rad(0.5));
    for j in -4..=4 {
        for i in -4..=4 {
            objects.push(Obj {
                tf: translate(dir(4. * i as f32, 0., 4. * j as f32)),
                geom: solids::unit_sphere(9, 9).build(),
            });
        }
    }
    let scene = Scene { objects, camera };

    let mut fb = Framebuf {
        color: Buffer::new(W, 4 * W, 0),
        depth: &mut Buffer::new(W, W, f32::INFINITY),
        depth_test: true,
        depth_write: true
    };

    c.bench_function("scene", |b| {
        b.iter(|| rdr.render_scene(
            &scene,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |_| Some(WHITE),
            },
            &mut fb
        ));
    });
    check_hash(fb.color.data(), 2914773945184235903);
    save_screenshot("target/scene.ppm", &fb.color);
    eprintln!("Stats/s: {}\n", rdr.stats.avg_per_sec());
}

fn gouraud_fillrate(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.modelview = scale(2.0) * &translate(6.0 * Z);
    let mesh = solids::unit_cube()
        .vertex_attrs([RED, GREEN, BLUE].into_iter().cycle())
        .build();

    let mut fb = Framebuf {
        color: Buffer::new(W, 4 * W, 0),
        depth: &mut Buffer::new(W, W, f32::INFINITY),
        depth_test: false,
        depth_write: false
    };
    c.bench_function("gouraud", |b| {
        b.iter(|| mesh.render(
            &mut rdr,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |frag: Fragment<Color>| Some(frag.varying),
            },
            &mut fb
        ));
    });
    check_hash(fb.color.data(), 3313217684519750548);
    save_screenshot("target/gouraud.ppm", &fb.color);
    eprintln!("Stats/frame: {}\n", rdr.stats.avg_per_frame());
}

fn texture_fillrate(c: &mut Criterion) {
    let mut rdr = renderer();

    rdr.modelview = scale(2.0) * &translate(6.0 * Z);
    let mesh = solids::unit_cube()
        .vertex_attrs([
            uv(1.0, 1.0), uv(0.0, 1.0), uv(1.0, 0.0), uv(0.0, 0.0),
            uv(0.0, 1.0), uv(1.0, 1.0), uv(0.0, 0.0), uv(1.0, 0.0),
        ])
        //].iter().map(|&tc| tc.mul(256.0)))
        .build();

    let tex = Texture::from(load_pnm("../examples/sdl/crate.ppm").unwrap());

    let mut fb = Framebuf {
        color: Buffer::new(W, 4 * W, 0),
        depth: &mut Buffer::new(W, W, f32::INFINITY),
        depth_test: false,
        depth_write: false
    };
    c.bench_function("texture", |b| {
        b.iter(|| {
            let clock = Instant::now();
            mesh.render(
                &mut rdr,
                &mut ShaderImpl {
                    vs: |v| v,
                    fs: |f: Fragment<TexCoord>| {
                        Some(tex.sample(f.varying))
                    }
                },
                &mut fb
            );
            rdr.stats.time_used += clock.elapsed();
            rdr.stats.frames += 1;
        });
    });
    check_hash(fb.color.data(), 12904971692947500411);
    save_screenshot("target/texture.ppm", &fb.color);
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
