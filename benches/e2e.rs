//! Benchmarks testing the entire pipeline.
//!
//! Compile with `stats` feature to enable allocation stats.

use divan::Bencher;

use retrofire_core::{
    geom::{Normal3, Vertex3, tri, vertex},
    math::{
        Color3f, Color4, Color4f, ProjMat3, perspective, pt2, pt3, rgb, rgba,
        translate, viewport,
    },
    render::debug::dir_to_rgb,
    render::{Context, Frag, render, render_, shader},
    util::buf::Buf2,
    util::pnm,
};

use retrofire_geom::solids::{Build, Sphere};

#[cfg(feature = "stats")]
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

#[divan::bench(args = [1, 4, 16, 64, 256, 1024], min_time = 1, max_time = 2)]
fn triangle(b: Bencher, n: u32) {
    let verts = [
        vertex(pt3(-1.0, 1.0, 0.0), rgb(1.0, 0.0, 0.0)),
        vertex(pt3(1.0, 1.0, 0.0), rgb(0.0, 0.8, 0.0)),
        vertex(pt3(0.0, -1.0, 0.0), rgb(0.4, 0.4, 1.0)),
    ];

    let shader = shader::new(
        |v: Vertex3<Color3f>, mvp: &ProjMat3| {
            // Transform vertex position from model to projection space
            // Interpolate vertex colors in linear color space
            vertex(mvp.apply(&v.pos), v.attrib)
        },
        |frag: Frag<Color3f<_>>| frag.var.to_color4(),
    );

    let dims @ (w, h) = (640, 480);
    let modelview = translate((0.0, 0.0, 2.0));
    let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(0, h)..pt2(w, 0));

    let mut framebuf = Buf2::<Color4>::new(dims);

    b.bench_local(|| {
        for _ in 0..n {
            render(
                [tri(0, 1, 2)],
                verts,
                &shader,
                &modelview.then(&project).to(),
                viewport,
                &mut framebuf,
                &Context::default(),
            );
        }
    });

    //let center_pixel = framebuf[[w / 2, h / 2]];

    //assert_eq!(center_pixel, rgba(114, 102, 128, 255));

    //pnm::save_ppm("benches_e2e_triangle.ppm", framebuf).unwrap();
}

#[divan::bench(args=[4, 16, 64, 256, 1024], min_time=1, max_time=2)]
fn sphere_fast(b: Bencher, res: u32) {
    let sphere = Sphere {
        sectors: res,
        segments: 2,
        radius: 1.0,
    }
    .build();

    let shader = shader::new(
        |v: Vertex3<Normal3>, mvp: &ProjMat3| {
            vertex(mvp.apply(&v.pos), dir_to_rgb(v.attrib))
        },
        |frag: Frag<Color4f>| (frag.var).to_color4(),
    );

    let dims @ (w, h) = (640, 480);
    let modelview = translate((0.0, 0.0, 2.0)).to();
    let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(0, h)..pt2(w, 0));

    let mut framebuf = Buf2::<u32>::new(dims);

    b.bench_local(|| {
        render_(
            &sphere.faces,
            &sphere.verts,
            &shader,
            &modelview.then(&project).to(),
            viewport,
            &mut framebuf,
            &Context::default(),
        );
    });

    //let center_pixel = framebuf[[w / 2, h / 2]];
    //assert_eq!(center_pixel, rgba(128, 127, 0, 255));

    let framebuf = Buf2::new_from(
        framebuf.dims(),
        framebuf.iter().map(|u| {
            let [_, r, g, b] = u.to_be_bytes();
            rgba(r, g, b, 0xFF)
        }),
    );

    pnm::save_ppm("benches_e2e_sphere_fast.ppm", framebuf).unwrap();
}

#[divan::bench(args=[4, 16, 64, 256, 1024], min_time=1, max_time=2)]
fn sphere_orig(b: Bencher, res: u32) {
    let sphere = Sphere {
        sectors: res,
        segments: 2,
        radius: 1.0,
    }
    .build();

    let shader = shader::new(
        |v: Vertex3<Normal3>, mvp: &ProjMat3| {
            vertex(mvp.apply(&v.pos), dir_to_rgb(v.attrib))
        },
        |frag: Frag<Color4f>| frag.var.to_color4(),
    );

    let dims @ (w, h) = (640, 480);
    let modelview = translate((0.0, 0.0, 2.0)).to();
    let project = perspective(1.0, w as f32 / h as f32, 0.1..1000.0);
    let viewport = viewport(pt2(0, h)..pt2(w, 0));

    let mut framebuf = Buf2::<Color4>::new(dims);

    b.bench_local(|| {
        render(
            &sphere.faces,
            &sphere.verts,
            &shader,
            &modelview.then(&project).to(),
            viewport,
            &mut framebuf,
            &Context::default(),
        );
    });

    //let center_pixel = framebuf[[w / 2, h / 2]];
    //assert_eq!(center_pixel, rgba(128, 127, 0, 255));

    pnm::save_ppm("benches_e2e_sphere_orig.ppm", framebuf).unwrap();
}

fn main() {
    divan::main()
}
