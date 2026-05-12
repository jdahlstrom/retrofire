//! Vector operation benchmarks.

use divan::Bencher;

use retrofire_core::math::{
    rand::{DefaultRng, Distrib},
    vec::dot,
    {Vec3, acos, splat},
};

#[divan::bench]
fn angle(b: Bencher) {
    let rng = &mut DefaultRng::default();
    let vecs = splat(-1e6)..splat(1e6);

    b.with_inputs(|| (vecs.sample(rng), vecs.sample(rng)))
        .counter(1u32)
        .bench_local_values(|(v, u): (Vec3, Vec3)| {
            acos(dot(&v.0, &u.0) / (v.len() * u.len()))
        });
}

#[divan::bench]
fn angle_faster(b: Bencher) {
    let rng = &mut DefaultRng::default();
    let vecs = splat(-1e6)..splat(1e6);

    b.with_inputs(|| (vecs.sample(rng), vecs.sample(rng)))
        .counter(1u32)
        .bench_local_values(|(v, u): (Vec3, Vec3)| {
            acos(dot(&v.0, &u.0) / (dot(&v.0, &v.0) * dot(&v.0, &v.0)).sqrt())
        });
}

#[divan::bench]
fn angle_approx(b: Bencher) {
    let rng = &mut DefaultRng::default();
    let vecs = splat(-1e6)..splat(1e6);

    b.with_inputs(|| (vecs.sample(rng), vecs.sample(rng)))
        .counter(1u32)
        .bench_local_values(|(v, u): (Vec3, Vec3)| {
            //let dot_v = v.x() * v.x() + v.y() * v.y() + v.z() * v.z();
            //let dot_u = u.x() * u.x() + u.y() * u.y() + u.z() * u.z();
            //let inv_len = fast_recip_sqrt(dot_v) * fast_recip_sqrt(dot_u);
            acos(dot(&v.normalize_approx().0, &u.normalize_approx().0))
        });
}

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
