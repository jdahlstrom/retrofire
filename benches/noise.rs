//! Noise generation benchmarks.

use core::hint::black_box;

use divan::Bencher;

use retrofire_core::math::{
    noise::{Perlin2, Perlin3},
    pt2, pt3,
};

const SIZES: [u32; 5] = [1, 4, 16, 64, 256];

#[divan::bench(args = SIZES)]
fn perlin2(b: Bencher, sz: u32) {
    let noise = Perlin2::default();

    b.counter(sz * sz).bench_local(|| {
        for i in 0..sz {
            for j in 0..sz {
                black_box(noise.eval(pt2(i as f32, j as f32) / 64.0));
            }
        }
    });
}

#[divan::bench(args = SIZES)]
fn cache_hits(b: Bencher, sz: u32) {
    let noise = Perlin2::default();

    b.counter(sz * sz)
        .counter(256 * 256u32)
        .bench_local(|| {
            for i in 0..256 {
                for j in 0..256 {
                    black_box(noise.eval(pt2(i as f32, j as f32) / sz as f32));
                }
            }
        });
}

#[divan::bench(args = SIZES)]
fn perlin3(b: Bencher, sz: u32) {
    let noise = Perlin3::default();

    b.counter(sz * sz).bench_local(|| {
        for i in 0..sz {
            for j in 0..sz {
                black_box(noise.eval(pt3(i as f32, j as f32, 0.0) / 64.0));
            }
        }
    });
}

fn main() {
    divan::main()
}
