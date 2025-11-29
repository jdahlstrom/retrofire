use divan::Bencher;
use divan::counter::ItemsCount;
use retrofire::core::{
    geom::Ray,
    math::rand::{DefaultRng, Distrib, VectorsInUnitBall},
    math::{pt3, vec3},
    render::scene::BBox,
};
use retrofire::geom::Intersect;
use retrofire_core::math::rand::UnitSphere;
use std::hint::black_box;

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

fn main() {
    divan::main();
}
