//! Random number generation benchmarks.

use divan::Bencher;

use retrofire_core::math::{rand::*, splat};

#[divan::bench]
fn raw_bits(b: Bencher) {
    let rng = &mut DefaultRng::default();
    b.counter(1u32).bench_local(|| rng.next_bits());
}

#[divan::bench]
fn uniform_i32_in_range(b: Bencher) {
    bench_distrib(b, -1_000_000..1_000_000);
}

#[divan::bench]
fn uniform_i32_iter(b: Bencher) {
    let rng = &mut DefaultRng::default();
    let mut ints = rng.iter(-1_000_000..1_000_000);
    b.counter(1u32).bench_local(|| ints.next());
}

#[divan::bench]
fn uniform_f32(b: Bencher) {
    bench_distrib(b, -1.0..1.0)
}

#[divan::bench]
fn uniform_vec2(b: Bencher) {
    bench_distrib(b, splat::<(), f32, 2>(-1e3)..splat(1e3))
}

#[divan::bench]
fn uniform_vec3(b: Bencher) {
    bench_distrib(b, splat::<(), f32, 3>(-1e3)..splat(1e3))
}

#[divan::bench]
fn bernoulli(b: Bencher) {
    bench_distrib(b, Bernoulli(0.25));
}

#[divan::bench]
fn unit_circle(b: Bencher) {
    bench_distrib(b, UnitCircle);
}

#[divan::bench]
fn unit_sphere(b: Bencher) {
    bench_distrib(b, UnitSphere);
}

#[divan::bench]
fn unit_ball(b: Bencher) {
    bench_distrib(b, PointsInUnitBall);
}

#[divan::bench]
fn unit_disk(b: Bencher) {
    bench_distrib(b, PointsOnUnitDisk);
}

fn bench_distrib(b: Bencher, d: impl Distrib) {
    let rng = &mut DefaultRng::default();
    b.counter(1u32).bench_local(|| d.sample(rng));
}

fn main() {
    divan::main()
}
