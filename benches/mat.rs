//! Matrix manipulation benchmarks.

use divan::Bencher;

use retrofire_core::{
    math::rand::{DefaultRng, Distrib},
    math::{Mat2, Mat3, Mat4, Vec2, Vec3, splat},
};

#[divan::bench]
fn invert2(b: Bencher) {
    let rng = &mut DefaultRng::default();
    let vecs = splat(-1e3)..splat(1e3);

    b.with_inputs(|| {
        let x: Vec2 = vecs.sample(rng);
        let y = x.perp();
        Mat2::new([x.0, y.0])
    })
    .counter(1u32)
    .bench_local_values(|m: Mat2| m.inverse());
}

#[divan::bench]
fn invert3_lin(b: Bencher) {
    let rng = &mut DefaultRng::default();
    let vecs = splat(-1e3)..splat(1e3);

    b.with_inputs(|| {
        let x: Vec3 = vecs.sample(rng);
        let y = Vec3::Z.cross(&x);
        let z = x.cross(&y);

        Mat3::new([x.0, y.0, z.0])
    })
    .counter(1u32)
    .bench_local_values(|m: Mat3<(), (), 3>| m.inverse());
}

#[divan::bench]
fn invert3_aff(b: Bencher) {
    let rng = &mut DefaultRng::default();
    let vecs = splat(-1e3)..splat(1e3);

    b.with_inputs(|| {
        let x: Vec2 = vecs.sample(rng);
        let y = x.perp();
        let o: Vec2 = vecs.sample(rng);

        Mat3::from_affine(x, y, o.to_pt())
    })
    .counter(1u32)
    .bench_local_values(|m: Mat3| m.inverse());
}

#[divan::bench]
fn invert4(b: Bencher) {
    let rng = &mut DefaultRng::default();
    let vecs = splat(-1e3)..splat(1e3);

    b.with_inputs(|| {
        let x: Vec3 = vecs.sample(rng);
        let y = Vec3::Z.cross(&x);
        let z = x.cross(&y);
        let o: Vec3 = vecs.sample(rng);

        Mat4::from_affine(x, y, z, o.to_pt())
    })
    .counter(1u32)
    .bench_local_values(|m: Mat4| m.inverse());
}

fn main() {
    divan::main()
}
