use criterion::*;

use geom::solids;
use math::Angle::Rad;
use math::transform::*;
use render::{Obj, Raster, Renderer, Scene};
use render::color::*;
use render::raster::Fragment;

const W: usize = 128;

fn renderer() -> Renderer {
    let mut rdr = Renderer::new();
    rdr.projection = perspective(1.0, 10.0, 1.0, Rad(1.0));
    rdr.viewport = viewport(0.0, 0.0, W as f32, W as f32);
    rdr
}

fn torus(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.modelview = scale(4., 4., 4.);

    let mesh = solids::torus(0.2, 9, 9);

    let mut buf = ['.'; W * W];
    c.bench_function("torus", |b| {
        b.iter(|| rdr.render(
            &mesh,
            &mut Raster {
                shade: |_, _| BLACK,
                test: |_| true,
                output: |(x, y), _| buf[W * y + x] = '#',
            }
        ))
    });
    eprintln!("Stats/s: {}", rdr.stats.avg_per_sec());
}

fn scene(c: &mut Criterion) {
    let mut rdr = renderer();

    let mut objects = vec![];
    let camera = translate(0., 4., 0.) * &rotate_x(Rad(0.5));
    for j in -4..=4 {
        for i in -4..=4 {
            objects.push(Obj {
                tf: translate(4. * i as f32, 0., 4. * j as f32),
                mesh: solids::unit_sphere(9, 9)
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
                output: |(x, y), _| buf[W * y + x] = '#'
            }
        ))
    });
    eprintln!("Stats/s: {}", rdr.stats.avg_per_sec());
}

fn gouraud_fillrate(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.modelview = scale(2., 2., 2.) * &translate(0.0, 0.0, 6.0);
    let mesh = solids::unit_cube().with_vertex_attrs([
        RED, GREEN, BLUE, RED, GREEN, BLUE, RED, GREEN
    ].iter().copied());

    let mut buf = [BLACK; W * W];
    c.bench_function("gouraud", |b| {
        b.iter(|| rdr.render(
            &mesh,
            &mut Raster {
                shade: |frag: Fragment<_>, _| frag.varying,
                test: |_| true,
                output: |(x, y), col| buf[W * y + x] = col
            }
        ))
    });
    eprintln!("Stats/frame: {}", rdr.stats.avg_per_frame());
}

criterion_group!(benches,
    torus,
    scene,
    gouraud_fillrate);

criterion_main!(benches);