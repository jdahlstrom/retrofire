use criterion::*;

use geom::solids;
use math::Angle::Rad;
use math::transform::*;
use math::vec::dir;
use render::{Raster, Renderer};
use render::raster::Fragment;
use render::scene::{Obj, Scene};
use util::color::*;

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
        b.iter(|| rdr.render(
            &mesh,
            &mut Raster {
                shade: |_, _| BLACK,
                test: |_| true,
                output: |(x, y), _| {
                    let idx = 3 * (W * y + x);
                    buf[idx] = 0xFF;
                    buf[idx + 1] = 0xFF;
                    buf[idx + 2] = 0xFF;
                },
            },
        ))
    });
    eprintln!("Stats/s: {}", rdr.stats.avg_per_sec());
}

fn scene(c: &mut Criterion) {
    let mut rdr = renderer();

    let mut objects = vec![];
    let camera = translate(4.0 * Y) * &rotate_x(Rad(0.5));
    for j in -4..=4 {
        for i in -4..=4 {
            objects.push(Obj {
                tf: translate(dir(4. * i as f32, 0., 4. * j as f32)),
                mesh: solids::unit_sphere(9, 9).build(),
            });
        }
    }
    let scene = Scene { objects, camera };

    let mut buf = ['.'; W * W];
    c.bench_function("scene", |b| {
        b.iter(|| rdr.render_scene(
            &scene,
            &mut Raster {
                shade: |_, _| BLACK,
                test: |_| true,
                output: |(x, y), _| buf[W * y + x] = '#',
            },
        ))
    });
    eprintln!("Stats/s: {}", rdr.stats.avg_per_sec());
}

fn gouraud_fillrate(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.modelview = scale(2.0) * &translate(6.0 * Z);
    let mesh = solids::unit_cube()
        .vertex_attrs([RED, GREEN, BLUE].iter().copied().cycle())
        .build();

    let mut buf = [BLACK; W * W];
    c.bench_function("gouraud", |b| {
        b.iter(|| rdr.render(
            &mesh,
            &mut Raster {
                shade: |frag: Fragment<_>, _| frag.varying,
                test: |_| true,
                output: |(x, y), col| buf[W * y + x] = col,
            },
        ))
    });
    eprintln!("Stats/frame: {}", rdr.stats.avg_per_frame());
}

criterion_group!(benches,
    torus,
    scene,
    gouraud_fillrate);

criterion_main!(benches);