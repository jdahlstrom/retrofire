use std::hint::black_box;

use divan::{Bencher, counter::ItemsCount};

use retrofire::core::{
    geom::{Ray, Sphere},
    math::rand::{DefaultRng, Distrib, UnitSphere, VectorsInUnitBall},
    math::{Point3, degs, pt3, spherical},
    render::scene::BBox,
};
use retrofire::geom::Intersect;

#[divan::bench]
fn ray_bbox_hit(b: Bencher) {
    let mut rng = DefaultRng::default();
    let bbox = BBox::<()>(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    b.with_inputs(|| {
        let v = 4.0 * UnitSphere.sample(&mut rng);
        Ray(v.to_pt(), -v)
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
        let v = 4.0 * UnitSphere.sample(&mut rng);
        Ray(v.to_pt(), v)
    })
    .counter(ItemsCount::new(1usize))
    .bench_local_values(|ray| {
        assert!(ray.intersect(&black_box(bbox)).is_none())
    });
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
