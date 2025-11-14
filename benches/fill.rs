//! Fillrate benchmarks.

use core::iter::zip;

use divan::{Bencher, counter::ItemsCount};

use retrofire_core::{
    geom::{Tri, vertex},
    math::{Color3, Color3f, color::gray, pt3, rgb},
    render::{
        Texture, raster::ScreenPt, raster::tri_fill, tex::SamplerRepeatPot, uv,
    },
    util::{buf::Buf2, pnm::save_ppm},
};

const SIZES: [f32; 5] = [4.0, 16.0, 64.0, 256.0, 1024.0];

const VERTS: [ScreenPt; 3] =
    [pt3(0.1, 0.1, 0.0), pt3(0.9, 0.3, 0.5), pt3(0.4, 0.9, 1.0)];

#[divan::bench(args = SIZES)]
fn flat(b: Bencher, sz: f32) {
    let mut buf: Buf2<Color3> = Buf2::new((1024, 1024));

    b.with_inputs(|| VERTS.map(|p| vertex(p * sz, ())))
        .input_counter(move |vs| ItemsCount::new(Tri(*vs).area() as usize))
        .bench_local_values(|vs| {
            tri_fill(vs, |sl| {
                buf[sl.y][sl.xs].fill(gray(0xCC));
            });
        });

    save_ppm("benches_fill_flat.ppm", buf).unwrap();
}
#[divan::bench(args = SIZES)]
fn gouraud(b: Bencher, sz: f32) {
    let mut buf: Buf2<Color3f> = Buf2::new((1024, 1024));

    b.with_inputs(|| {
        [
            vertex(VERTS[0] * sz, rgb(0.9, 0.1, 0.0)),
            vertex(VERTS[1] * sz, rgb(0.1, 0.8, 0.1)),
            vertex(VERTS[2] * sz, rgb(0.2, 0.3, 1.0)),
        ]
    })
    .input_counter(move |vs| ItemsCount::new(Tri(*vs).area() as usize))
    .bench_local_values(|vs| {
        tri_fill(vs, |sl| {
            let y = sl.y;
            let xs = sl.xs.clone();
            let span = &mut buf[y][xs];

            for ((_, col), pix) in zip(sl.vs, span) {
                *pix = col;
            }
        });
    });

    let buf = Buf2::new_from(
        (1024, 1024),
        buf.data().into_iter().map(|c| c.to_color3()),
    );
    save_ppm("benches_fill_color.ppm", buf).unwrap();
}

#[divan::bench(args = SIZES)]
fn texture(b: Bencher, sz: f32) {
    let mut buf: Buf2<Color3> = Buf2::new((1024, 1024));

    let tex = Texture::from(Buf2::<Color3>::new_from(
        (2, 2),
        [gray(0xFF), gray(0x33), gray(0x33), gray(0xFF)],
    ));
    let sampler = SamplerRepeatPot::new(&tex);

    b.with_inputs(|| {
        [
            vertex(VERTS[0] * sz, uv(0.0, 0.0)),
            vertex(VERTS[1] * sz, uv(4.0, 0.0)),
            vertex(VERTS[2] * sz, uv(0.0, 4.0)),
        ]
    })
    .input_counter(move |vs| ItemsCount::new(Tri(*vs).area() as usize))
    .bench_local_values(|vs| {
        tri_fill(vs, |sl| {
            let y = sl.y;
            let xs = sl.xs.clone();
            let span = &mut buf[y][xs];

            for ((_, uv), pix) in zip(sl.vs, span) {
                *pix = sampler.sample(&tex, uv);
            }
        });
    });

    save_ppm("benches_fill_tex.ppm", buf).unwrap();
}

fn main() {
    divan::main()
}
