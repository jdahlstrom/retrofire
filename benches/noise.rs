//! Noise generation benchmarks.

use core::hint::black_box;

use divan::Bencher;

use retrofire_core::math::{noise::Perlin2, pt2};

const SIZES: [u32; 5] = [1, 4, 16, 64, 256];

#[divan::bench(args = SIZES)]
fn perlin2(b: Bencher, sz: u32) {
    let mut p = Perlin2::default();

    b.counter(sz * sz).bench_local(|| {
        for i in 0..sz {
            for j in 0..sz {
                black_box(p.eval(pt2(i as f32 / 256.0, j as f32 / 256.0)));
            }
        }
    });
}

fn main() {
    divan::main()
}
