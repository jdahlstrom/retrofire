//! Matrix manipulation benchmarks.

use core::ops::Range;

use divan::Bencher;

use retrofire_core::math::{
    Mat2, Mat3, Mat4, Vec2, Vec3,
    rand::{DefaultRng, Distrib},
    splat,
};

mod application {
    use super::*;
    #[divan::bench]
    fn apply2(b: Bencher) {
        let rng = DefaultRng::default();
        let v = VEC2S.sample(&rng);

        b.with_inputs(|| random_mat2(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat2| m.apply(&v));
    }

    #[divan::bench]
    fn apply3_lin(b: Bencher) {
        let rng = DefaultRng::default();
        let v = VEC3S.sample(&rng);

        b.with_inputs(|| random_mat3_lin(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat3<(), (), 3>| m.apply(&v));
    }

    #[divan::bench]
    fn apply3_aff(b: Bencher) {
        let rng = DefaultRng::default();
        let v = VEC2S.sample(&rng);

        b.with_inputs(|| random_mat3_aff(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat3| m.apply(&v));
    }

    #[divan::bench]
    fn apply4(b: Bencher) {
        let rng = DefaultRng::default();
        let v = VEC3S.sample(&rng);

        b.with_inputs(|| random_mat4(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat4| m.apply(&v));
    }
}

mod inverse_application {
    use super::*;

    #[divan::bench]
    fn apply_inv2(b: Bencher) {
        let rng = DefaultRng::default();
        let v = VEC2S.sample(&rng);

        b.with_inputs(|| random_mat2(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat2| m.inverse().apply(&v));
    }

    #[divan::bench]
    fn apply_inv3_lin(b: Bencher) {
        let rng = DefaultRng::default();
        let v = VEC3S.sample(&rng);

        b.with_inputs(|| random_mat3_lin(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat3<(), (), 3>| m.inverse().apply(&v));
    }

    #[divan::bench]
    fn apply_inv3_aff(b: Bencher) {
        let rng = DefaultRng::default();
        let v = VEC2S.sample(&rng);

        b.with_inputs(|| random_mat3_aff(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat3| m.inverse().apply(&v));
    }

    #[divan::bench]
    fn apply_inv4(b: Bencher) {
        let rng = DefaultRng::default();
        let v = VEC3S.sample(&rng);

        b.with_inputs(|| random_mat4(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat4| m.inverse().apply(&v));
    }
}

mod composition {
    use super::*;
    #[divan::bench]
    fn compose2(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| (random_mat2(&rng), random_mat2(&rng)))
            .counter(1u32)
            .bench_local_values(|(m, n)| m.compose(&n));
    }

    #[divan::bench]
    fn compose3_lin(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| (random_mat3_lin(&rng), random_mat3_lin(&rng)))
            .counter(1u32)
            .bench_local_values(|(m, n)| m.compose(&n));
    }

    #[divan::bench]
    fn compose3_aff(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| (random_mat3_aff(&rng), random_mat3_aff(&rng)))
            .counter(1u32)
            .bench_local_values(|(m, n)| m.compose(&n));
    }

    #[divan::bench]
    fn compose4(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| (random_mat4(&rng), random_mat4(&rng)))
            .counter(1u32)
            .bench_local_values(|(m, n)| m.compose(&n));
    }
}

mod solving {
    use super::*;
    #[divan::bench]
    fn solve2(b: Bencher) {
        let rng = DefaultRng::default();
        let v = VEC2S.sample(&rng);

        b.with_inputs(|| random_mat2(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat2| m.solve(v));
    }

    #[divan::bench]
    fn solve3(b: Bencher) {
        let rng = DefaultRng::default();
        let v = VEC3S.sample(&rng);

        b.with_inputs(|| random_mat3_lin(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat3<(), (), 3>| m.solve(v));
    }
}

mod inversion {
    use super::*;

    #[divan::bench]
    fn invert2(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| random_mat2(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat2| m.inverse());
    }

    #[divan::bench]
    fn invert3_lin(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| random_mat3_lin(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat3<(), (), 3>| m.inverse());
    }

    #[divan::bench]
    fn invert3_aff(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| random_mat3_aff(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat3| m.inverse());
    }

    #[divan::bench]
    fn invert4_aff(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| random_mat4(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat4| m.inverse());
    }
    #[divan::bench]

    fn invert4_lin(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| {
            let mut mat = random_mat4(&rng);
            mat.0[3][1] = 0.5; // Make non-affine (simulate projection matrix)
            mat
        })
        .counter(1u32)
        .bench_local_values(|m: Mat4| m.inverse());
    }
}

mod determinant {
    use super::*;

    #[divan::bench]
    fn det2(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| random_mat2(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat2| m.determinant());
    }

    #[divan::bench]
    fn det3_lin(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| random_mat3_lin(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat3<(), (), 3>| m.determinant());
    }

    #[divan::bench]
    fn det3_aff(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| random_mat3_aff(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat3| m.determinant());
    }

    #[divan::bench]
    fn det4_aff(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| random_mat4(&rng))
            .counter(1u32)
            .bench_local_values(|m: Mat4| m.determinant());
    }

    #[divan::bench]
    fn det4_lin(b: Bencher) {
        let rng = DefaultRng::default();

        b.with_inputs(|| {
            let mut mat = random_mat4(&rng);
            mat.0[3][1] = 0.5; // Make non-affine (simulate projection matrix)
            mat
        })
        .counter(1u32)
        .bench_local_values(|m: Mat4| m.determinant());
    }
}

const VEC2S: Range<Vec2> = splat(-1e3)..splat(1e3);
const VEC3S: Range<Vec3> = splat(-1e3)..splat(1e3);

fn random_mat2(rng: &DefaultRng) -> Mat2 {
    let x = VEC2S.sample(rng);
    let y = x.perp();
    Mat2::new([x.0, y.0])
}
fn random_mat3_lin(rng: &DefaultRng) -> Mat3<(), (), 3> {
    let x = VEC3S.sample(&rng);
    let y = Vec3::Z.cross(&x);
    let z = x.cross(&y);
    Mat3::new([x.0, y.0, z.0])
}
fn random_mat3_aff(rng: &DefaultRng) -> Mat3 {
    let x = VEC2S.sample(rng);
    let y = x.perp();
    let o = VEC2S.sample(rng).to_pt();
    Mat3::from_affine(x, y, o)
}
fn random_mat4(rng: &DefaultRng) -> Mat4 {
    let x = VEC3S.sample(rng);
    let y = Vec3::Z.cross(&x);
    let z = x.cross(&y);
    let o = VEC3S.sample(rng).to_pt();
    Mat4::from_affine(x, y, z, o)
}

fn main() {
    divan::main()
}
