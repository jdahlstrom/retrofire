#![no_std]

extern crate alloc;
extern crate core;
#[cfg(feature = "std")]
extern crate std;

use alloc::vec::Vec;
use core::fmt::Debug;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::time::Instant;
use std::{eprint, eprintln};

pub mod io;
pub mod isect;
pub mod solids;

pub use isect::Intersect;

use retrofire_core::geom::{Mesh, Polygon, Tri, Vertex, Vertex3, tri, vertex};
use retrofire_core::math::{Point2, pt2, pt3};

/// Converts the given polygon into a list of triangles.
///
/// Uses the so-called ear-clipping algorithm [TODO ref]
pub fn triangulate<B: Debug + Default>(
    poly: &Polygon<Point2<B>>,
) -> Mesh<(), B> {
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

    // indices refer to verts, not to ixs!
    //let mut non_ears = BTreeSet::<[usize; 3]>::new();

    let mut is_ear = |tri: Tri<usize>, ixs: &Vec<_>| {
        // if non_ears.contains(&tri.0) {
        //     return false;
        // }

        let tri_vi = tri.map(|i| ixs[i]);
        let tri_v = tri_vi
            .map(|i| verts[i])
            .map(|v: Vertex3<_, _>| vertex(v.pos.xz(), ()));

        if tri_v.winding() != winding {
            // AB-BC is a concave pair of edges, the diagonal AC outside
            // the polygon -> cannot be an ear
            // non_ears.insert(tri_vi.0);
            return false;
        }

        for i in 0..tri.0[0] {
            //count += 1;
            if tri_v.contains(pts[ixs[i]]) {
                // non_ears.insert(tri_vi.0);
                return false;
            }
        }
        for i in tri.0[2] + 1..ixs.len() {
            //count += 1;
            if tri_v.contains(pts[ixs[i]]) {
                // non_ears.insert(tri_vi.0);
                return false;
            }
        }

        true
    };

    let mut n = 0;
    let mut count = 0;
    loop {
        count += 1;

        debug_assert!(
            n < poly.0.len().pow(2),
            "triangulation stuck, polygon may be self-intersecting"
        );

        let il = ixs.len();
        if il == 3 {
            // Only one triangle left, we're done
            faces.push(tri(ixs[0], ixs[1], ixs[2]));
            break;
        }

        let i = (n + 0) % il;
        let j = (n + 1) % il;
        let k = (n + 2) % il;

        n += 1;

        let ai = ixs[i];
        let bi = ixs[j];
        let ci = ixs[k];

        if is_ear(tri(i, j, k), &ixs) {
            // Valid ear -> clip it
            ixs.remove(j);
            faces.push(tri(ai, bi, ci));
            //eprint!("!");
        }
    }

    eprintln!(
        "Triangulated {}-gon -> {} triangles in {count} iters ({:.3} ms)",
        poly.0.len(),
        faces.len(),
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
