//! Fillrate benchmarks.
use core::iter::zip;

use divan::{Bencher, counter::ItemsCount};

use retrofire_core::render::raster::tri_fill_;
use retrofire_core::render::{Context, Frag};
use retrofire_core::util::Dims;
use retrofire_core::util::pixfmt::{IntoPixel, Rgb888, Xrgb8888};
use retrofire_core::{
    geom::{Tri, vertex},
    math::{Color3, Color3f, color::gray, pt3, rgb},
    render::{
        Texture, raster::ScreenPt, raster::tri_fill, tex::SamplerRepeatPot, uv,
    },
    util::{buf::Buf2, pnm::save_ppm},
};

const SIZES: [f32; 6] = [4.0, 16.0, 64.0, 256.0, 1024.0, 2048.0];

const VERTS: [ScreenPt; 3] =
    [pt3(0.1, 0.1, 0.0), pt3(0.9, 0.3, 0.5), pt3(0.4, 0.9, 1.0)];

const BUF_SIZE: Dims = (2048, 2048);

#[divan::bench(args = SIZES, min_time=1)]
fn flat(b: Bencher, sz: f32) {
    let mut buf: Buf2<Color3> = Buf2::new(BUF_SIZE);

    b.with_inputs(|| VERTS.map(|p| vertex(p * sz, ())))
        .input_counter(move |vs| ItemsCount::new(Tri(*vs).area() as usize))
        .bench_local_values(|vs| {
            tri_fill(vs, |sl| {
                buf[sl.y][sl.xs].fill(gray(0xCC));
            });
        });

    save_ppm(format!("benches_fill_flat_orig_{sz}.ppm"), buf).unwrap();
}

#[divan::bench(args = SIZES, min_time = 1)]
fn flat_fast(b: Bencher, sz: f32) {
    let mut buf: Buf2<u32> = Buf2::new(BUF_SIZE);
    let ctx = Context::default();

    b.with_inputs(|| VERTS.map(|p| vertex(p * sz, ())))
        .input_counter(move |vs| ItemsCount::new(Tri(*vs).area() as usize))
        .bench_local_values(|vs| {
            tri_fill_(
                vs,
                &|_| gray(0xCCu8).into_pixel_fmt(Rgb888),
                &mut buf.as_mut_slice2(),
                &ctx,
            );
        });

    let buf = Buf2::new_from(
        BUF_SIZE,
        buf.data().into_iter().map(|c| {
            let [_, r, g, b] = c.to_be_bytes();
            rgb(r, g, b)
        }),
    );
    save_ppm(format!("benches_fill_flat_fast_{sz}.ppm"), buf).unwrap();
}

#[divan::bench(args = SIZES)]
fn gouraud_orig(b: Bencher, sz: f32) {
    let mut buf: Buf2<u32> = Buf2::new(BUF_SIZE);

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
                *pix = col.to_color3().into_pixel();
            }
        });
    });

    let buf = Buf2::new_from(
        BUF_SIZE,
        buf.data().into_iter().map(|c| {
            let [_, r, g, b] = c.to_be_bytes();
            rgb(r, g, b)
        }),
    );
    save_ppm(format!("benches_fill_gouraud_orig_{sz}.ppm"), buf).unwrap();
}

#[divan::bench(args = SIZES)]
fn gouraud_fast(b: Bencher, sz: f32) {
    let mut buf: Buf2<u32> = Buf2::new(BUF_SIZE);
    let ctx = Context::default();

    b.with_inputs(|| {
        [
            vertex(VERTS[0] * sz, rgb(0.9, 0.1, 0.0)),
            vertex(VERTS[1] * sz, rgb(0.1, 0.8, 0.1)),
            vertex(VERTS[2] * sz, rgb(0.2, 0.3, 1.0)),
        ]
    })
    .input_counter(move |vs| ItemsCount::new(Tri(*vs).area() as usize))
    .bench_local_values(|vs| {
        tri_fill_(
            vs,
            &|frag: Frag<Color3f>| frag.var.to_color3().into_pixel_fmt(Rgb888),
            &mut buf.as_mut_slice2(),
            &ctx,
        );
    });

    let buf = Buf2::new_from(
        BUF_SIZE,
        buf.data().into_iter().map(|c| {
            let [_, r, g, b] = c.to_be_bytes();
            rgb(r, g, b)
        }),
    );
    save_ppm(format!("benches_fill_gouraud_fast_{sz}.ppm"), buf).unwrap();
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
