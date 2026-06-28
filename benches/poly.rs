//! Polygon operations benchmarks.

use core::iter::chain;

use divan::Bencher;

use retrofire_core::{
    geom::Polygon,
    math::{Point2, Vary, polar, pt2, turns},
};
use retrofire_geom::triangulate;

static SIDES: [u32; 5] = [6, 10, 20, 100, 1000];

#[divan::bench(sample_size = 100)]
fn triangulate_triangle(b: Bencher) {
    let make_poly = || {
        let p = Polygon::new(
            turns(0.0)
                .vary_to(turns(0.7), 3)
                .map(|a| polar(1.0, a).to_cart().to_pt()),
        );
        assert_eq!(p.0.len(), 3);
        p
    };

    b.with_inputs(make_poly)
        .counter(3u32)
        .bench_local_values(|poly: Polygon<Point2>| triangulate(&poly));
}

#[divan::bench(args = SIDES, sample_size = 100)]
fn triangulate_convex(b: Bencher, sides: u32) {
    b.with_inputs(|| {
        let p = Polygon::new(
            turns(0.0)
                .vary_to(turns(1.0), sides + 1)
                .take(sides as usize)
                .map(|a| polar(1.0, a).to_cart().to_pt()),
        );
        assert_eq!(p.0.len(), sides as usize);
        p
    })
    .counter(sides)
    .bench_local_values(|poly: Polygon<Point2>| triangulate(&poly));
}

#[divan::bench(args = SIDES, sample_size = 100, max_time=2)]
fn triangulate_almost_convex(b: Bencher, sides: u32) {
    let make_poly = || {
        let p = Polygon::new(
            turns(0.1)
                .vary_to(turns(0.9), sides - 1)
                .map(|a| polar(1.0, a).to_cart().to_pt())
                .chain([pt2(0.0, 0.0)]),
        );
        assert!(!p.is_convex());
        assert_eq!(p.0.len(), sides as usize);
        p
    };

    b.with_inputs(make_poly)
        .counter(sides)
        .bench_local_values(|poly: Polygon<Point2>| triangulate(&poly));
}

#[divan::bench(args = SIDES, sample_size = 100, max_time=2)]
fn triangulate_concave(b: Bencher, sides: u32) {
    let make_poly = || {
        let p = Polygon::new(
            turns(0.0)
                .vary_to(turns(1.0), sides + 1)
                .take(sides as usize)
                .map(|a| {
                    let r = 1.0 - 0.8 * (1.5 * a).sin().abs().powf(0.5);
                    polar(r, a + turns(0.1 * r)).to_cart().to_pt()
                }),
        );
        assert!(!p.is_convex());
        assert_eq!(p.0.len(), sides as usize);
        p
    };

    b.with_inputs(make_poly)
        .counter(sides)
        .bench_local_values(|poly: Polygon<Point2>| triangulate(&poly));
}

fn main() {
    divan::main()
}
