//! Fillrate benchmarks.

use divan::Bencher;

use retrofire_core::{
    geom::{Tri, vertex},
    math::{Color3, Color3f, Vary, color::gray, pt3, rgb},
    render::raster::{ScreenPt, tri_fill},
    render::tex::SamplerRepeatPot,
    render::{Texture, uv},
    util::{Buf2, Dims, pnm::save_ppm},
};

const SIZES: [f32; 5] = [4.0, 16.0, 64.0, 256.0, 1024.0];
const BUF_SIZE: Dims = Dims(1024, 1024);

const VERTS: [ScreenPt; 3] =
    [pt3(0.1, 0.1, 0.0), pt3(0.4, 0.9, 1.0), pt3(0.9, 0.3, 0.5)];

#[divan::bench(args = SIZES)]
fn flat(b: Bencher, sz: f32) {
    let mut buf: Buf2<Color3> = Buf2::new(BUF_SIZE);

    b.with_inputs(|| VERTS.map(|p| vertex(p * sz, ())))
        .input_counter(move |vs| Tri(*vs).area() as usize)
        .bench_local_values(|vs| {
            tri_fill(vs, |sl| {
                buf[sl.y][sl.xs].fill(gray(0xCC));
            });
        });

    save_ppm("benches/out/fill_flat.ppm", buf).unwrap();
}

#[divan::bench(args = SIZES)]
fn gouraud(b: Bencher, sz: f32) {
    let mut buf: Buf2<Color3f> = Buf2::new(BUF_SIZE);

    b.with_inputs(|| {
        [
            vertex(VERTS[0] * sz, rgb(0.9, 0.1, 0.0)),
            vertex(VERTS[1] * sz, rgb(0.1, 0.8, 0.1)),
            vertex(VERTS[2] * sz, rgb(0.2, 0.3, 1.0)),
        ]
    })
    .input_counter(move |vs| Tri(*vs).area() as usize)
    .bench_local_values(|vs| {
        tri_fill(vs, |sl| {
            let y = sl.y;
            let xs = sl.xs.clone();
            let span = &mut buf[y][xs];

            let mut col = sl.left.attrib;
            for pix in span {
                *pix = col;
                col = col.step(&sl.d_dx.attrib);
            }
        });
    });

    let buf =
        Buf2::new_from(buf.dims(), buf.data().iter().map(|c| c.to_color3()));
    save_ppm("benches/out/fill_color.ppm", buf).unwrap();
}

#[divan::bench(args = SIZES)]
fn texture(b: Bencher, sz: f32) {
    let mut buf: Buf2<Color3> = Buf2::new(BUF_SIZE);

    let tex = Texture::from(Buf2::<Color3>::new_from(
        Dims(2, 2),
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
    .input_counter(move |vs| Tri(*vs).area() as usize)
    .bench_local_values(|vs| {
        tri_fill(vs, |sl| {
            let y = sl.y;
            let xs = sl.xs.clone();
            let span = &mut buf[y][xs];

            // TODO doesn't do perspective correction!
            let mut uv = sl.left.attrib;
            for pix in span {
                *pix = sampler.sample(&tex, uv);
                uv = uv.step(&sl.d_dx.attrib);
            }
        });
    });

    save_ppm("benches/out/fill_tex.ppm", buf).unwrap();
}

fn main() {
    divan::main()
}
