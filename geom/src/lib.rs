#![no_std]

extern crate alloc;
extern crate core;
#[cfg(feature = "std")]
extern crate std;

use alloc::vec::Vec;
use core::fmt::Debug;

use std::collections::HashMap;
use std::{eprintln, print, time::Instant};

pub mod io;
pub mod isect;
pub mod solids;

pub use isect::Intersect;

use retrofire_core::geom::{Mesh, Polygon, Tri, tri, vertex};
use retrofire_core::math::{Point2, pt3};

/// Converts the given polygon into a list of triangles.
///
/// Uses the so-called ear-clipping algorithm [TODO ref]
pub fn triangulate<B: Debug + Default>(
    poly: &Polygon<Point2<B>>,
) -> Mesh<(), B> {
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    enum Kind {
        Concave,
        Contains,
        Ear,
        Clipped,
    }

    let start = Instant::now();

    let pts = poly.0.clone();

    let Some(winding) = poly.winding() else {
        return Mesh::default(); // Degenerate, no triangles
    };

    let verts: Vec<_> = pts
        .iter()
        .map(|p| vertex(pt3(p.x(), 0.0, p.y()), ()))
        .collect();

    let mut faces = Vec::with_capacity(verts.len() - 2);

    let mut ixs: Vec<_> = (0..pts.len()).collect();

    let next = |i: usize| if i + 1 < pts.len() { i + 1 } else { 0 };
    let _prev = |i: usize| if i > 0 { i - 1 } else { pts.len() };

    let mut kinds = HashMap::<[usize; 3], Kind>::new();

    let mut pt_checks = 0;
    let mut kind_calls = 0;
    let mut cache_hits = 0;
    let mut kind = |ijk: [usize; 3]| {
        kind_calls += 1;

        if let Some(&kind) = kinds.get(&ijk) {
            cache_hits += 1;
            return kind;
        }

        // 25641866

        let tri = Tri(ijk).map(|i| vertex(pts[i], ()));
        if tri.winding() != winding {
            kinds.insert(ijk, Kind::Concave);
            // AB-BC is a concave pair of edges, the diagonal AC is outside
            // the polygon -> cannot be an ear
            return Kind::Concave;
        }

        for l in 0..pts.len() {
            if ijk.contains(&l) {
                continue;
            }
            pt_checks += 1;
            if tri.contains(pts[l]) {
                kinds.insert(ijk, Kind::Contains);
                //eprintln!("{:?} contains point {l} = {:?}", (i, j, k),
                // pts[l]);
                return Kind::Contains;
            }
        }
        kinds.insert(ijk, Kind::Ear);
        Kind::Ear
    };

    /*let kinds: Vec<([usize; 3], Kind)> = (0..pts.len())
        .map(|i| {
            let j = next(i);
            let k = next(j);
            ([i, j, k], kind([i, j, k]))
        })
        .collect();

    eprintln!("{:?}", kinds);*/

    let mut n = 0;
    let mut hits = 0;
    let mut misses = 0;
    while faces.len() < verts.len() - 2 {
        debug_assert!(
            n < poly.0.len().pow(2),
            "triangulation stuck, polygon may be self-intersecting"
        );

        let il = ixs.len();

        let i = (n + 0) % il;
        let j = (n + 1) % il;
        let k = (n + 2) % il;

        let pts_ijk = [ixs[i], ixs[j], ixs[k]];
        if kind(pts_ijk) == Kind::Ear {
            // Valid ear -> clip it
            faces.push(Tri(pts_ijk));
            //kinds.remove(&pts_ijk); // borrowck
            ixs.remove(j);
            hits += 1;
        } else {
            misses += 1;
        }

        n += 1;
    }

    eprintln!(
        "Triangulated {}-gon -> {} triangles in {n} iters, \
        {kind_calls} kind calls, {cache_hits} cache hits, {pt_checks} pt cont \
        checks \
        ({hits} ears, {misses} non-ears {:.2}%, {:.3} ms elapsed)",
        poly.0.len(),
        faces.len(),
        hits as f32 / (hits + misses) as f32 * 100.0,
        start.elapsed().as_secs_f32() * 1000.0
    );

    Mesh { faces, verts }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrofire_core::math::pt2;

    #[test]
    fn triangulate_tri() {
        let pts = [pt2(0.0, 0.0), pt2(1.0, 0.0), pt2(1.0, 1.0)];
        let p = Polygon::new(pts);
        let t = triangulate(&p);
        assert_eq!(t, [Tri(pts)]);
    }
    #[test]
    fn triangulate_quad() {
        let [a, b, c, d] =
            [pt2(0.0, 0.0), pt2(1.0, 0.0), pt2(1.0, 1.0), pt2(0.0, 1.0)];
        let p = Polygon::new([a, b, c, d]);
        let t = triangulate(&p);
        assert_eq!(t, [tri(a, b, c), tri(a, c, d)]);
    }
    #[test]
    fn triangulate_concave_quad() {
        let [a, b, c, d] =
            [pt2(0.0, 0.0), pt2(0.5, 0.2), pt2(1.0, 0.0), pt2(0.5, 1.0)];
        let p = Polygon::new([a, b, c, d]);
        let t = triangulate(&p);
        assert_eq!(t, [tri(b, c, d), tri(a, b, d)]);
    }
    #[test]
    fn triangulate_concave_6gon() {
        let [a, b, c, d, e, f] = [
            pt2(0.0, 0.0),
            pt2(1.0, 0.0),
            pt2(0.4, 0.8),
            pt2(1.0, 1.0),
            pt2(0.0, 1.0),
            pt2(0.6, 0.2),
        ];
        let p = Polygon::new([a, b, c, d, e, f]);
        let t = triangulate(&p);
        assert_eq!(t, [tri(c, d, e), tri(f, a, b), tri(c, e, f), tri(b, c, f)]);
    }
    #[test]
    fn triangulate_self_intersecting() {
        let [a, b, c, d] =
            [pt2(0.0, 0.0), pt2(1.0, 0.0), pt2(0.0, 1.0), pt2(1.0, 1.0)];
        let p = Polygon::new([a, b, c, d]);
        let t = triangulate(&p);
        assert_eq!(t, []);
    }
}
