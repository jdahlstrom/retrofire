//! Vector operations benchmarks.

use divan::Bencher;

use retrofire_core::{
    math::rand::{DefaultRng, Distrib},
    math::{Vec3, splat},
};

#[divan::bench]
fn normalize_exact(b: Bencher) {
    let rng = &mut DefaultRng::default();
    let vecs = splat(-1e6)..splat(1e6);

    b.with_inputs(|| vecs.sample(rng))
        .counter(1u32)
        .bench_local_values(|v: Vec3| v.normalize());
}

#[divan::bench]
fn normalize_approx(b: Bencher) {
    let rng = &mut DefaultRng::default();
    let vecs = splat(-1e6)..splat(1e6);

    b.with_inputs(|| vecs.sample(rng))
        .counter(1u32)
        .bench_local_values(|v: Vec3| v.normalize_approx());
}

fn main() {
    divan::main()
}
