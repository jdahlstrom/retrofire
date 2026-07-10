//! Triangle clipping benchmarks.

use core::{array, iter::repeat_with};

use divan::{AllocProfiler, Bencher};

use retrofire_core::{
    geom::{Tri, vertex},
    math::rand::{DefaultRng, Distrib},
    math::{ProjMat3, orthographic, pt3},
    render::View,
    render::clip::{Clip, ClipVert, view_frustum},
};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

static PROJ: ProjMat3<View> =
    orthographic(pt3(-1.0, -1.0, -1.0), pt3(1.0, 1.0, 1.0));

#[divan::bench(args = [1, 10, 100, 1000, 10_000])]
fn clip_mixed(b: Bencher, n: usize) {
    let rng = &mut DefaultRng::default();
    let pts = pt3(-10.0, -10.0, -10.0)..pt3(10.0, 10.0, 10.0);

    b.with_inputs(|| {
        repeat_with(|| {
            Tri(array::from_fn(|_| {
                ClipVert::new(vertex(PROJ.apply(&pts.sample(rng)), ()))
            }))
        })
        .take(n)
        .collect::<Vec<_>>()
    })
    .counter(n)
    .bench_local_values(|tris| Tri::clip(tris, &view_frustum::PLANES))
}

#[divan::bench(args = [1, 10, 100, 1000, 10_000])]
fn clip_all_inside(b: Bencher, n: usize) {
    let rng = &mut DefaultRng::default();
    let pts = pt3(-1.0, -1.0, -1.0)..pt3(1.0, 1.0, 1.0);

    b.with_inputs(|| {
        repeat_with(|| {
            Tri(array::from_fn(|_| {
                ClipVert::new(vertex(PROJ.apply(&pts.sample(rng)), ()))
            }))
        })
        .take(n)
        .collect::<Vec<_>>()
    })
    .counter(n)
    .bench_local_values(|tris| Tri::clip(tris, &view_frustum::PLANES))
}

#[divan::bench(args = [1, 10, 100, 1000, 10_000])]
fn clip_all_outside(b: Bencher, n: usize) {
    let rng = &mut DefaultRng::default();
    let pts = pt3(2.0, -10.0, -10.0)..pt3(10.0, 10.0, 10.0);

    b.with_inputs(|| {
        repeat_with(|| {
            Tri(([pts.start; 3]..[pts.end; 3])
                .sample(rng)
                .map(|pt| ClipVert::new(vertex(PROJ.apply(&pt), ()))))
        })
        .take(n)
        .collect::<Vec<_>>()
    })
    .counter(n)
    .bench_local_values(|tris| Tri::clip(tris, &view_frustum::PLANES))
}

fn main() {
    divan::main()
}
