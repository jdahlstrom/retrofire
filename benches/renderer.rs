use criterion::*;

use geom::solids;
use math::transform::*;
use math::vec::ZERO;
use render::{Renderer, Obj, Scene};

fn torus(c: &mut Criterion) {
    const W: usize = 128;

    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(1.0, 10.0, 1., 1.));
    rdr.set_viewport(viewport(0.0, 0.0, W as f32, W as f32));
    rdr.set_transform(scale(4., 4., 4.));

    let mesh = solids::torus(0.2, 9, 9);

    let mut buf = ['.'; W * W];
    c.bench_function(
        "torus",
        |b| {
            b.iter(|| {
                rdr.render(
                    mesh.clone(),
                    &|_, _| ZERO,
                    &mut |x, y, _| buf[W * y + x] = '#'
                );
            })
        });
    eprintln!("Stats/s: {}", rdr.stats.avg_per_sec());
}

fn scene(c: &mut Criterion) {
    const W: usize = 128;

    let mut rdr = Renderer::new();
    rdr.set_projection(perspective(1.0, 10.0, 1., 1.));
    rdr.set_viewport(viewport(0.0, 0.0, W as f32, W as f32));

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
    c.bench_function(
        "scene",
        |b| {
            b.iter(|| {
                rdr.render_scene(
                    scene.clone(),
                    &|_, _| ZERO,
                    &mut |x, y, _| buf[W * y + x] = '#'
                );
            })
        });
    eprintln!("Stats/s: {}", rdr.stats.avg_per_sec());
}

criterion_group!(benches, torus, scene);
criterion_main!(benches);