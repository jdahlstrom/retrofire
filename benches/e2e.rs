//! Benchmarks testing the entire pipeline.

use divan::Bencher;

use retrofire_core::{
    geom::{Normal3, Vertex3, tri, vertex},
    math::{
        Color3f, Color4, Color4f, ProjMat3, perspective, pt2, pt3, rgb, rgba,
        translate, viewport,
    },
    render::{Context, Frag, Model, debug::dir_to_rgb, render, shader},
    util::{Dims, buf::Buf2, pnm},
};
use retrofire_geom::solids::{Build, Sphere};

#[cfg(false)]
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

#[divan::bench(args = [1, 4, 16, 64, 256, 1024], max_time = 2)]
fn triangle(b: Bencher, n: u32) {
    let verts = [
        vertex(pt3(-1.0, 1.0, 0.0), rgb(1.0, 0.0, 0.0)),
        vertex(pt3(1.0, 1.0, 0.0), rgb(0.0, 0.8, 0.0)),
        vertex(pt3(0.0, -1.0, 0.0), rgb(0.4, 0.4, 1.0)),
    ];

    let shader = shader::new(
        |v: Vertex3<Color3f>, mvp: &ProjMat3<Model>| {
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f<_>>, _: &_| frag.var.to_color4(),
    );

    let dims @ Dims(w, h) = Dims(640, 480);
    let modelview = translate((0.0, 0.0, 2.0)).to();
    let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(0, h)..pt2(w, 0));

    let mut framebuf = Buf2::<Color4>::new(dims);

    b.bench_local(|| {
        for _ in 0..n {
            render(
                [tri(0, 1, 2)],
                verts,
                &shader,
                &modelview.then(&project),
                viewport,
                &mut framebuf,
                &Context::default(),
            );
        }
    });

    let center_pixel = framebuf[[w / 2, h / 2]];

    assert_eq!(center_pixel, rgba(151, 128, 187, 255));

    pnm::save_ppm("benches_e2e_triangle.ppm", framebuf).unwrap();
}

#[divan::bench(args=[4, 16, 64, 256, 1024], min_time=5, max_time=5)]
fn sphere(b: Bencher, res: u32) {
    let sphere = Sphere {
        sectors: res,
        segments: 2,
        radius: 1.0,
    }
    .build();

    let shader = shader::new(
        |v: Vertex3<Normal3>, mvp: &ProjMat3<Model>| {
            vertex(mvp.apply(&v.pos), dir_to_rgb(v.attrib))
        },
        |frag: Frag<Color4f>, _: &_| frag.var.to_color4(),
    );

    let dims @ Dims(w, h) = Dims(640, 480);
    let modelview = translate((0.0, 0.0, 2.0)).to();
    let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(0, h)..pt2(w, 0));

    let mut framebuf = Buf2::<Color4>::new(dims);

    b.bench_local(|| {
        render(
            &sphere.faces,
            &sphere.verts,
            &shader,
            &modelview.then(&project),
            viewport,
            &mut framebuf,
            &Context::default(),
        );
    });

    let center_pixel = framebuf[[w / 2, h / 2]];

    assert_eq!(center_pixel, rgba(128, 127, 0, 255));

    pnm::save_ppm("benches_e2e_sphere.ppm", framebuf).unwrap();
}

fn main() {
    divan::main()
}
