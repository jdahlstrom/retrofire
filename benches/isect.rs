use std::hint::black_box;

use divan::{Bencher, counter::ItemsCount};

use retrofire::core::{
    geom::{Ray, Sphere},
    math::rand::*,
    math::{Point3, degs, pt3, spherical},
    render::scene::BBox,
};
use retrofire::geom::Intersect;

#[divan::bench]
fn ray_bbox_hit(b: Bencher) {
    let mut rng = DefaultRng::default();
    let bbox = BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    b.with_inputs(|| {
        let v = 100.0 * UnitSphere.sample(&mut rng);
        Ray(v.to_pt(), -v * (0.0..10.0).sample(&mut rng))
    })
    .counter(ItemsCount::new(1usize))
    .bench_local_values(|ray| {
        assert!(ray.intersect(&black_box(bbox)).is_some())
    });
}
#[divan::bench]
fn ray_bbox_hit_2(b: Bencher) {
    let mut rng = DefaultRng::default();
    let bbox = BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    b.with_inputs(|| {
        let v = (spherical(0.0, degs(-180.0), degs(-90.0))
            ..spherical(10.0, degs(180.0), degs(-45.0)))
            .sample(&mut rng);

        Ray(pt3(0.0, 2.0, 0.0), v.to_cart())
    })
    .counter(ItemsCount::new(1usize))
    .bench_local_values(|ray| {
        assert!(ray.intersect(&black_box(bbox)).is_some())
    });
}
#[divan::bench]
fn ray_bbox_inside(b: Bencher) {
    let mut rng = DefaultRng::default();
    let bbox = BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    b.with_inputs(|| {
        let pt = PointsInUnitBall.sample(&mut rng);
        let dir = VectorsInUnitBall.sample(&mut rng);
        Ray(pt, dir)
    })
    .counter(ItemsCount::new(1usize))
    .bench_local_values(|ray| {
        assert!(ray.intersect(&black_box(bbox)).is_some())
    });
}

#[divan::bench]
fn ray_bbox_miss(b: Bencher) {
    let mut rng = DefaultRng::default();
    let bbox = BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    b.with_inputs(|| {
        let v = (spherical(0.0, degs(-180.0), degs(-45.0))
            ..spherical(10.0, degs(180.0), degs(90.0)))
            .sample(&mut rng);

        Ray(pt3(0.0, 3.0, 0.0), v.to_cart())
    })
    .counter(ItemsCount::new(1usize))
    .bench_local_values(|ray| {
        assert!(ray.intersect(&black_box(bbox)).is_none())
    });
}

#[divan::bench]
fn ray_bbox_mixed(b: Bencher) {
    let mut rng = DefaultRng::default();
    let (p, q) = (pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));
    let bbox = BBox::<()>(p, q);

    b.with_inputs(|| {
        // Approximately one third of the rays hits the box
        let orig = (p..q).sample(&mut rng);
        let dir = (p..q).sample(&mut rng);
        Ray(2.0 * orig, 100.0 * dir.to_vec())
    })
    .counter(ItemsCount::new(1usize))
    .bench_local_values(|ray| ray.intersect(&black_box(bbox)));
}

#[divan::bench]
fn ray_sphere_miss(b: Bencher) {
    let mut rng = DefaultRng::default();

    let sphere = Sphere::<()>(Point3::origin(), 1.0);

    b.with_inputs(|| {
        let v = (spherical(0.0, degs(-180.0), degs(-45.0))
            ..spherical(10.0, degs(180.0), degs(90.0)))
            .sample(&mut rng);
        Ray(pt3(0.0, 3.0, 0.0), v.to_cart())
    })
    .counter(ItemsCount::new(1usize))
    .bench_local_values(|ray| {
        let ip = ray.intersect(&black_box(sphere));
        assert!(ip.is_none());
        ip
    });
}

#[divan::bench]
fn ray_sphere_hit(b: Bencher) {
    let mut rng = DefaultRng::default();

    let sphere = Sphere::<()>(Point3::origin(), 1.0);

    b.with_inputs(|| {
        let v = (spherical(0.0, degs(-180.0), degs(-90.0))
            ..spherical(10.0, degs(180.0), degs(-45.0)))
            .sample(&mut rng);
        Ray(pt3(0.0, 2.0f32.sqrt(), 0.0), v.to_cart())
    })
    .counter(ItemsCount::new(1usize))
    .bench_local_values(|ray| {
        let ip = ray.intersect(&black_box(sphere));
        assert!(ip.is_some());
        ip
    });
}

#[divan::bench]
fn ray_sphere_mixed(b: Bencher) {
    let mut rng = DefaultRng::default();

    let sphere = Sphere::<()>(Point3::origin(), 1.0);

    b.with_inputs(|| {
        let v = VectorsInUnitBall.sample(&mut rng);
        Ray(pt3(0.0, 2.0, 0.0), v)
    })
    .counter(ItemsCount::new(1usize))
    .bench_local_values(|ray| ray.intersect(&black_box(sphere)));
}

fn main() {
    divan::main();
}
