use std::time::Instant;

use criterion::*;

use geom::solids;
use math::Angle::Rad;
use math::transform::*;
use math::vec::{dir, Y, Z};
use render::{Raster, Render, Renderer};
use render::raster::Fragment;
use render::scene::{Obj, Scene};
use render::shade::ShaderImpl;
use render::tex::{Texture, uv, TexCoord};
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

fn torus(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.modelview = scale(4.0);

    let mesh = solids::torus(0.2, 9, 9).build();

    let mut buf = vec![0u8; 3 * W * W];
    c.bench_function("torus", |b| {
        b.iter(|| mesh.render(
            &mut rdr,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |_| Some(BLACK),
            },
            &mut Raster {
                test: |_| true,
                output: |frag: Fragment<_>| {
                    let (x, y) = frag.coord;
                    let idx = 3 * (W * y + x);
                    buf[idx] = 0xFF;
                    buf[idx + 1] = 0xFF;
                    buf[idx + 2] = 0xFF;
                },
            },
        ))
    });
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

    let mut buf = ['.'; W * W];
    c.bench_function("scene", |b| {
        b.iter(|| rdr.render_scene(
            &scene,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |_| Some(BLACK),
            },
            &mut Raster {
                test: |_| true,
                output: |frag: Fragment<_>| {
                    let (x, y) = frag.coord;
                    buf[W * y + x] = '#'
                },
            },
        ))
    });
    eprintln!("Stats/s: {}\n", rdr.stats.avg_per_sec());
}

fn gouraud_fillrate(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.modelview = scale(2.0) * &translate(6.0 * Z);
    let mesh = solids::unit_cube()
        .vertex_attrs([RED, GREEN, BLUE].iter().copied().cycle())
        .build();

    let mut buf = [BLACK; W * W];
    c.bench_function("gouraud", |b| {
        b.iter(|| mesh.render(
            &mut rdr,
            &mut ShaderImpl {
                vs: |v| v,
                fs: |frag: Fragment<Color>| Some(frag.varying),
            },
            &mut Raster {
                test: |_| true,
                output: |frag: Fragment<(f32, Color)>| {
                    let (x, y) = frag.coord;
                    buf[W * y + x] = frag.varying.1
                },
            },
        ))
    });
    save_ppm("gouraud.ppm", &Buffer::borrow(W, &mut buf)).unwrap();
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

    //let tex = Texture::from(Buffer::from_vec(2, vec![RED, BLUE, BLUE, GREEN]));
    let tex = Texture::from(load_pnm("../examples/sdl/crate.ppm").unwrap());

    let mut buf = [BLACK; W * W];
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
                &mut Raster {
                    test: |_| true,
                    output: |frag: Fragment<(f32, Color)>| {
                        let (x, y) = frag.coord;
                        buf[W * y + x] = frag.varying.1;
                    },
                },
            );
            rdr.stats.time_used += clock.elapsed();
            rdr.stats.frames += 1;
        })
    });
    save_ppm("texture.ppm", &Buffer::borrow(W, &mut buf)).unwrap();
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
