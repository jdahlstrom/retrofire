//! Triangle clipping benchmarks.

use core::{array, iter::repeat_with};

use divan::{Bencher, counter::ItemsCount};

use retrofire_core::{
    geom::{Tri, vertex},
    math::rand::{DEFAULT_RNG, DefaultRng, Distrib},
    math::{orthographic, pt3},
    render::clip::{ClipVert, view_frustum},
};

//#[global_allocator]
//static ALLOC: AllocProfiler = AllocProfiler::system();

#[divan::bench(args = [1, 10, 100, 1000, 10_000])]
fn clip_mixed(b: Bencher, n: usize) {
    let rng = &mut DefaultRng::default();
    let pts = pt3(-10.0, -10.0, -10.0)..pt3(10.0, 10.0, 10.0);
    let proj = orthographic(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    b.with_inputs(|| {
        repeat_with(|| {
            let vs = array::from_fn(|_| {
                ClipVert::new(vertex(proj.apply(&pts.sample(rng)), ()))
            });
            Tri(vs)
        })
        .take(n)
        .collect::<Vec<_>>()
    })
    .input_counter(|tris| ItemsCount::of_iter(tris))
    .bench_local_values(|tris| {
        let mut out = Vec::new();
        view_frustum::clip(tris.as_slice(), &mut out);
        out
    })
}

#[divan::bench(args = [1, 10, 100, 1000, 10_000])]
fn clip_all_inside(b: Bencher, n: usize) {
    let rng = &mut DefaultRng::default();
    let pts = pt3(-1.0, -1.0, -1.0)..pt3(1.0, 1.0, 1.0);
    let proj = orthographic(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    b.with_inputs(|| {
        repeat_with(|| {
            let vs = array::from_fn(|_| {
                ClipVert::new(vertex(proj.apply(&pts.sample(rng)), ()))
            });
            Tri(vs)
        })
        .take(n)
        .collect::<Vec<_>>()
    })
    .input_counter(|tris| ItemsCount::of_iter(tris))
    .bench_local_values(|tris| {
        let mut out = Vec::new();
        view_frustum::clip(tris.as_slice(), &mut out);
        out
    })
}

#[divan::bench(args = [1, 10, 100, 1000, 10_000])]
fn clip_all_outside(b: Bencher, n: usize) {
    let mut rng = DEFAULT_RNG;
    let pts = pt3(2.0, -10.0, -10.0)..pt3(10.0, 10.0, 10.0);
    let proj = orthographic(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

    b.with_inputs(|| {
        repeat_with(|| {
            let vs = ([pts.start; 3]..[pts.end; 3])
                .sample(&mut rng)
                .map(|pt| ClipVert::new(vertex(proj.apply(&pt), ())));
            Tri(vs)
        })
        .take(n)
        .collect::<Vec<_>>()
    })
    .input_counter(|tris| ItemsCount::of_iter(tris))
    .bench_local_values(|tris| {
        let mut out = Vec::with_capacity(tris.len());
        view_frustum::clip(tris.as_slice(), &mut out);
        out
    })
}

fn main() {
    divan::main()
}
