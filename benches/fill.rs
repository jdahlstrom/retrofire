use std::iter::zip;

use divan::{Bencher, counter::ItemsCount};

use retrofire_core::{
    geom::{Tri, vertex},
    math::{Color3, Color3f, color::gray, pt3, rand::DEFAULT_RNG, rgb},
    render::{Texture, raster::tri_fill, tex::SamplerRepeatPot, uv},
    util::{buf::Buf2, pnm::save_ppm},
};

#[divan::bench(args = [0.5, 2.0, 8.0, 32.0, 100.0])]
fn flat(b: Bencher, z: f32) {
    let mut _rng = DEFAULT_RNG;
    let mut buf: Buf2<Color3> = Buf2::new((1000, 1000));

    b.with_inputs(|| {
        [
            vertex(pt3(1.0 * z, 1.0 * z, 0.0), ()),
            vertex(pt3(9.0 * z, 3.0 * z, 0.0), ()),
            vertex(pt3(4.0 * z, 9.0 * z, 0.0), ()),
        ]
    })
    .input_counter(move |vs| ItemsCount::new(Tri(*vs).area() as usize))
    .bench_local_values(|vs| {
        tri_fill(vs, |sl| {
            buf[sl.y][sl.xs].fill(gray(0xCC));
        });
    });

    save_ppm("benches_fill_flat.ppm", buf).unwrap();
}
#[divan::bench(args = [0.5, 2.0, 8.0, 32.0, 100.0])]
fn color(b: Bencher, z: f32) {
    let mut _rng = DEFAULT_RNG;
    let mut buf: Buf2<Color3f> = Buf2::new((1000, 1000));

    b.with_inputs(|| {
        [
            vertex(pt3(1.0 * z, 1.0 * z, 0.0), rgb(0.9, 0.1, 0.0)),
            vertex(pt3(9.0 * z, 3.0 * z, 0.0), rgb(0.1, 0.8, 0.1)),
            vertex(pt3(4.0 * z, 9.0 * z, 0.0), rgb(0.2, 0.3, 1.0)),
        ]
    })
    .input_counter(move |vs| ItemsCount::new(Tri(*vs).area() as usize))
    .bench_local_values(|vs| {
        tri_fill(vs, |sl| {
            //black_box(sl);
            //let Some((_, col)) = sl.vs.next() else { return };
            //buf[sl.y][sl.xs].fill(col);

            let y = sl.y;
            let xs = sl.xs.clone();
            let span = &mut buf[y][xs];

            for ((_, col), pix) in zip(sl.vs, span) {
                *pix = col;
            }
        });
    });

    let buf = Buf2::new_from(
        (1000, 1000),
        buf.data().into_iter().map(|c| c.to_color3()),
    );
    save_ppm("benches_fill_color.ppm", buf).unwrap();
}

#[divan::bench(args = [0.5, 2.0, 8.0, 32.0, 100.0])]
fn texture(b: Bencher, z: f32) {
    let mut _rng = DEFAULT_RNG;
    let mut buf: Buf2<Color3> = Buf2::new((1000, 1000));

    let tex = Texture::from(Buf2::<Color3>::new_from(
        (2, 2),
        [gray(0xFF), gray(0x33), gray(0x33), gray(0xFF)],
    ));
    let sampler = SamplerRepeatPot::new(&tex);

    b.with_inputs(|| {
        [
            vertex(pt3(1.0 * z, 1.0 * z, 0.0), uv(0.0, 0.0)),
            vertex(pt3(9.0 * z, 3.0 * z, 0.0), uv(4.0, 0.0)),
            vertex(pt3(4.0 * z, 9.0 * z, 0.0), uv(0.0, 4.0)),
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
