//! Intersection testing benchmarks.

use core::hint::black_box;

use divan::Bencher;

use retrofire_core::{
    geom::{Plane3, Ray, Sphere},
    math::{Point3, degs, pt3, rand::*, spherical, splat, vec3},
    render::scene::BBox,
};
use retrofire_geom::Intersect;

#[divan::bench]
fn ray_plane_hit(b: Bencher) {
    let rng = DefaultRng::default();
    let plane = <Plane3>::from_point_and_normal(
        pt3(0.0, 1.0, 0.0),
        vec3(1.0, 1.0, 1.0),
    );

    b.with_inputs(|| {
        let v = rng.next(splat(-1.0)..splat(0.0));
        Ray(pt3(0.0, 10.0, 0.0), 100.0 * (v - vec3(1.0, 1.0, 1.0)))
    })
    .counter(1u32)
    .bench_local_values(|ray| ray.intersect(&black_box(plane)));
}
#[divan::bench]
fn ray_plane_miss(b: Bencher) {
    let rng = DefaultRng::default();
    let plane = <Plane3>::from_point_and_normal(
        pt3(0.0, 1.0, 0.0),
        vec3(1.0, 1.0, 1.0),
    );

    b.with_inputs(|| {
        let v = rng.next(splat(0.0)..splat(1.0));
        Ray(pt3(0.0, 10.0, 0.0), 100.0 * v)
    })
    .counter(1u32)
    .bench_local_values(|ray| ray.intersect(&black_box(plane)));
}
#[divan::bench]
fn ray_plane_mixed(b: Bencher) {
    let rng = DefaultRng::default();
    let plane = <Plane3>::from_point_and_normal(
        pt3(0.0, 1.0, 0.0),
        vec3(1.0, 1.0, 1.0),
    );

    b.with_inputs(|| {
        let v = rng.next(VectorsInUnitBall);
        Ray(pt3(0.0, 10.0, 0.0), 100.0 * v)
    })
    .counter(1u32)
    .bench_local_values(|ray| ray.intersect(&black_box(plane)));
}

#[divan::bench]
fn ray_bbox_hit(b: Bencher) {
    let rng = DefaultRng::default();
    let bbox = BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    b.with_inputs(|| {
        let v = rng.next(VectorsInUnitBall);
        Ray(v.to_pt(), 100.0 * v)
    })
    .counter(1u32)
    .bench_local_values(|ray| {
        assert!(ray.intersect(&black_box(bbox)).is_some())
    });
}
#[divan::bench]
fn ray_bbox_hit_2(b: Bencher) {
    let rng = DefaultRng::default();
    let bbox = BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    let min = spherical(0.0, degs(-180.0), degs(-90.0));
    let max = spherical(10.0, degs(180.0), degs(-45.0));
    b.with_inputs(|| {
        let v = (min..max).sample(&mut rng);
        Ray(pt3(0.0, 2.0, 0.0), v.to_cart())
    })
    .counter(1u32)
    .bench_local_values(|ray| {
        assert!(ray.intersect(&black_box(bbox)).is_some())
    });
}
#[divan::bench]
fn ray_bbox_inside(b: Bencher) {
    let rng = DefaultRng::default();
    let bbox = BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    b.with_inputs(|| {
        let pt = rng.next(PointsInUnitBall);
        let dir = rng.next(VectorsInUnitBall);
        Ray(pt, dir)
    })
    .counter(1u32)
    .bench_local_values(|ray| {
        assert!(ray.intersect(&black_box(bbox)).is_some())
    });
}

#[divan::bench]
fn ray_bbox_miss(b: Bencher) {
    let rng = DefaultRng::default();
    let bbox = BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    let min = spherical(0.0, degs(-180.0), degs(-45.0));
    let max = spherical(10.0, degs(180.0), degs(90.0));
    b.with_inputs(|| {
        let v = rng.next(min..max);
        Ray(pt3(0.0, 3.0, 0.0), v.to_cart())
    })
    .counter(1u32)
    .bench_local_values(|ray| {
        assert!(ray.intersect(&black_box(bbox)).is_none())
    });
}

#[divan::bench]
fn ray_bbox_mixed(b: Bencher) {
    let rng = DefaultRng::default();
    let (p, q) = (pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));
    let bbox = BBox::<()>(p, q);

    b.with_inputs(|| {
        // Approximately one third of the rays hits the box
        let orig = rng.next(p..q);
        let dir = rng.next(p..q);
        Ray(2.0 * orig, 100.0 * dir.to_vec())
    })
    .counter(1u32)
    .bench_local_values(|ray| ray.intersect(&black_box(bbox)));
}

#[divan::bench]
fn ray_sphere_miss(b: Bencher) {
    let rng = DefaultRng::default();

    let sphere = Sphere(<Point3>::origin(), 1.0);

    let min = spherical(0.0, degs(-180.0), degs(-45.0));
    let max = spherical(10.0, degs(180.0), degs(90.0));
    b.with_inputs(|| {
        let v = rng.next(min..max);
        Ray(pt3(0.0, 3.0, 0.0), v.to_cart())
    })
    .counter(1u32)
    .bench_local_values(|ray| {
        let ip = ray.intersect(&black_box(sphere));
        assert!(ip.is_none());
        ip
    });
}

#[divan::bench]
fn ray_sphere_hit(b: Bencher) {
    let rng = DefaultRng::default();

    let sphere = Sphere(<Point3>::origin(), 1.0);

    b.with_inputs(|| {
        let v = (spherical(0.0, degs(-180.0), degs(-90.0))
            ..spherical(10.0, degs(180.0), degs(-45.0)))
            .sample(&mut rng);
        Ray(pt3(0.0, 2.0f32.sqrt(), 0.0), v.to_cart())
    })
    .counter(1u32)
    .bench_local_values(|ray| {
        let ip = ray.intersect(&black_box(sphere));
        assert!(ip.is_some());
        ip
    });
}

#[divan::bench]
fn ray_sphere_mixed(b: Bencher) {
    let rng = DefaultRng::default();

    let sphere = Sphere(<Point3>::origin(), 1.0);

    b.with_inputs(|| {
        let v = VectorsInUnitBall.sample(&mut rng);
        Ray(pt3(0.0, 2.0, 0.0), v)
    })
    .counter(1u32)
    .bench_local_values(|ray| ray.intersect(&black_box(sphere)));
}

fn main() {
    divan::main();
}
