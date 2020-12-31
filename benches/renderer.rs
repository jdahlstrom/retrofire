use criterion::*;

use geom::solids;
use math::transform::*;
use render::{color::*, Obj, Renderer, Scene};

const W: usize = 128;

fn renderer() -> Renderer {
    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(1.0, 10.0, 1., 1.));
    rdr.set_viewport(viewport(0.0, 0.0, W as f32, W as f32));
    rdr
}

fn torus(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.set_transform(scale(4., 4., 4.));

    let mesh = solids::torus(0.2, 9, 9);

    let mut buf = ['.'; W * W];
    c.bench_function("torus", |b| {
        b.iter(|| rdr.render(
            &mesh,
            &|_, _| BLACK,
            &mut |x, y, _| buf[W * y + x] = '#'
        ))
    });
    eprintln!("Stats/s: {}", rdr.stats.avg_per_sec());
}

fn scene(c: &mut Criterion) {
    let mut rdr = renderer();

    let mut objects = vec![];
    let camera = translate(0., 4., 0.) * &rotate_x(0.5);
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
            &|_, _| BLACK,
            &mut |x, y, _| buf[W * y + x] = '#'
        ))
    });
    eprintln!("Stats/s: {}", rdr.stats.avg_per_sec());
}

fn gouraud_fillrate(c: &mut Criterion) {
    let mut rdr = renderer();
    rdr.set_transform(scale(2., 2., 2.) * &translate(0.0, 0.0, 6.0));
    let mesh = solids::unit_cube().with_vertex_attrs([
        RED, GREEN, BLUE, RED, GREEN, BLUE, RED, GREEN
    ].iter().copied());

    let mut buf = [BLACK; W * W];
    c.bench_function("gouraud", |b| {
        b.iter(|| rdr.render(
            &mesh,
            &|frag, _| frag.varying,
            &mut |x, y, col| buf[W * y + x] = col
        ))
    });
    eprintln!("Stats/frame: {}", rdr.stats.avg_per_frame());
}

criterion_group!(benches,
    torus,
    scene,
    gouraud_fillrate);

criterion_main!(benches);